# Z-Image 架构说明

## 0. 文档目标

本文面向工程师，说明 Z-Image 的**系统架构**与**推理数据流**（architecture-level data flow）。

范围：

- 讲清楚模型由哪些模块组成、数据如何在模块间流动。
- 说明 S3-DiT 的设计特点，以及与同类 Diffusion Transformer 的异同。

不在范围：

- 源码文件路径、函数调用顺序、ONNX 导出细节。
- 训练数据管线、蒸馏算法推导、评测指标。

参考来源：Z-Image 论文（`Z-Image/docs/Z-Image_paper.md`）、diffusers `ZImagePipeline` / `ZImageTransformer2DModel` 公开实现。

---

## 1. 系统定位

Z-Image 是阿里通义开源的文生图(**Text-to-Image**) 基础模型，参数量约 **6B** ，核心生成骨干为 **S3-DiT**（Scalable Single-Stream Diffusion Transformer）, 它不是 FLUX/Qwen-Image 那类显式双流结构，而是把不同模态 token 早融合到同一条序列里，让每一层 self-attention 都能做跨模态交互。

同赛道常见方案（Qwen-Image、FLUX、Hunyuan-Image 等）参数量多在 20B–80B。Z-Image 的设计目标是：在较小参数量下，通过架构与训练流程优化，达到可接受的生成质量与推理成本。

公开变体：

| 变体 | 用途 | 推理特点 |
|------|------|----------|
| Z-Image | 标准文生图 | 多步 Flow Matching 去噪 |
| Z-Image-Turbo | 加速文生图 | 蒸馏后约 **8 NFE**（Number of Function Evaluations），通常 `guidance_scale=0` |
| Z-Image-Edit | 指令编辑 | 在 omni 架构上扩展，支持 reference image + 编辑指令 |

---

## 2. 整体组成

Z-Image 推理系统由三个独立权重模块串联：

```text
┌─────────────┐     ┌──────────────────────┐     ┌─────────────┐
│ Text Encoder│     │ S3-DiT (Transformer) │     │  VAE Decoder│
│  Qwen3-4B   │────►│  去噪 / 预测 velocity │────►│  Flux VAE   │
└─────────────┘     └──────────────────────┘     └─────────────┘
       ▲                       ▲
   Prompt 文本            噪声 Latent + Timestep
```

| 模块 | 角色 | 输入 | 输出 |
|------|------|------|------|
| **Text Encoder** | 将自然语言 prompt 编码为数值条件向量 | token 序列 | caption 特征矩阵（hidden states） |
| **S3-DiT** | 在 latent 空间迭代去噪 | 噪声 latent、timestep、caption 特征 | 每步对 latent 的修正量 |
| **VAE Decoder** | 将 latent 解码为像素图像 | 去噪完成的 latent | RGB 图像 |

> 三者 checkpoint 分开存放，推理时可分阶段加载.

---

## 3. Text Encoder（文本编码器）

Text Encoder 的作用是把自然语言 prompt 转成一组条件 token。Z-Image 使用 **Qwen3-4B** 作为 Text Encoder，利用它的中英文理解能力和指令理解能力。

Text Encoder 不直接生成图像，只输出 caption 特征，供后面的 S3-DiT 使用。

```text
Prompt
  → Tokenizer / Chat Template
  → Qwen3-4B Text Encoder
  → caption features
```

![Qwen3 dense Text Encoder](./Qwen3_dense.png)

上图可以作为 Z-Image Text Encoder 的结构参考。Prompt token 先经过 embedding 和 RotaryEmbedding（旋转位置编码），再进入多层 decoder layer。每层包含 self-attention 和 MLP，并使用 RMSNorm、残差连接和 gated MLP。Z-Image 使用这一路前向计算得到文本 token 的 hidden states。

这里的 Text Encoder 只做编码。它不使用图中最上方的语言模型输出去预测下一个 token，也不在推理时生成一段新文本。Z-Image 取 Qwen3-4B 的 hidden states 作为 caption features，再交给 S3-DiT。

### 3.1 输入输出

输入是用户 prompt。推理时 prompt 会先被格式化成对话模板，再进入 Qwen3-4B。

输出是每个文本 token 对应的 hidden state。Z-Image 不使用语言模型最后的词表 logits，也不让 Qwen 自回归生成文本。它取中间层 hidden states 作为图像生成条件。

这些 caption features 进入 S3-DiT 前，会先被投影到 S3-DiT 的 hidden dimension。公开实现中 caption feature dimension 是 2560，S3-DiT hidden dimension 是 3840。

### 3.2 为什么使用外置 Text Encoder

文生图模型需要理解长 prompt、物体关系、风格描述、文字渲染要求和中英文混合指令。把这部分交给一个成熟的 Large Language Model（大语言模型）可以减少 DiT 主干承担的语言理解压力。

这种设计和 Stable Diffusion / FLUX / Qwen-Image 类似：文本理解模块和图像生成模块分开。区别在于 Z-Image 选择 Qwen3-4B 作为轻量但能力较强的文本编码器，并通过 Prompt Enhancer（提示词增强器）补足复杂世界知识和推理型 prompt 的理解。

### 3.3 和其他模型的关系

Stable Diffusion 系列常用 CLIP Text Encoder 或 T5 Text Encoder。CLIP 对短文本和图文对齐有效，但长指令能力有限；T5 更适合长文本语义，但模型体积和推理成本较高。

Z-Image 使用 Qwen3-4B，重点是中英文 prompt、复杂指令和文字渲染场景。它不是把 LLM 融入 DiT 内部，而是把 LLM 当作独立条件编码器。这样可以保持模块边界清晰：Text Encoder 负责理解文本，S3-DiT 负责生成图像 latent。

---

## 4. S3-DiT（Scalable Single-Stream Diffusion Transformer）

S3-DiT 是 Z-Image 的生成主干。它在 latent 空间工作，每一步接收 noisy latent、timestep 和 caption features，输出对 latent 的预测修正量。

Z-Image 的主要架构特点集中在 S3-DiT：单流融合、多模态 token 统一建模、30 层 Transformer backbone，以及面向稳定训练的归一化和条件注入设计。

### 4.1 Single-Stream 的含义

Single-Stream（单流）指文本 token、图像 latent token，以及编辑场景中的参考图像 token，会被拼接成一条统一序列，然后进入同一组 Transformer layers。

```text
caption tokens ─┐
image tokens   ├─ concat → unified tokens → Transformer layers
semantic tokens┘
```

这里没有长期独立的 text stream 和 image stream。所有 token 在同一个 self-attention 中交互。

这和一些 Dual-Stream（双流）架构不同。双流架构通常先让文本和图像分别经过自己的分支，再通过 cross-attention 或联合层交互。双流的优点是模态边界清楚；缺点是参数和计算会分散到多条路径。Z-Image 选择单流结构，目标是在 6B 左右参数量下提高跨模态参数复用率。

### 4.2 S3-DiT 的输入序列

普通文生图场景中，S3-DiT 处理两类 token：

- Caption token：来自 Qwen3-4B Text Encoder。
- Image token：来自当前 noisy latent 的 patch。

图像不是以像素进入 S3-DiT。图像先在 VAE latent 空间表示，再被 patchify（切 patch）成 image tokens。以 1024×1024 图像为例，VAE latent 通常约为 128×128，patch size 为 2 时，image token 数约为 64×64，即 4096 个。

编辑模型 Z-Image-Edit 还会加入 reference image 的 VAE token 和 SigLIP2 semantic token。它们也会进入同一条 unified sequence。这说明 S3-DiT 的单流设计不是只服务文生图，而是用于统一 text-to-image 和 image-to-image/editing。

### 4.3 Modality Processor

S3-DiT 不是把所有 token 直接送入 30 层主干。不同模态会先经过轻量的 Modality Processor（模态处理器）。

```text
noisy latent → patchify → image processor ×2
caption features → projection → text processor ×2
reference image semantic features → semantic processor ×2（编辑模型）
```

这些 processor 都由 Transformer blocks 组成。它们的作用是让不同来源的 token 在进入主干前先完成基本对齐。

这个设计介于“完全分流”和“直接拼接”之间。它保留了少量模态专用处理能力，但把主要参数留给统一的 single-stream backbone。

### 4.4 Backbone：30 层单流 Transformer

经过模态处理后，所有 token 被拼接成 unified sequence，进入 30 层 single-stream Transformer backbone。

![S3-DiT Transformer Block](./S3-DiT-Transformer_Block.jpg)

上图展示的是 single-stream backbone 中一个 Transformer block 的两部分。下半部分是 Single-Stream Attention Block，上半部分是 Single-Stream FFN Block。Z-Image 的 30 层主干可以理解为重复执行这类 block。

单层 block 包含两个子块：

```text
Single-Stream Attention Block
  RMSNorm → Scale → Q/K/V → QK-Norm → U-RoPE
  → Multi-Head Self-Attention
  → RMSNorm → Gate → Residual Add

Single-Stream FFN Block
  RMSNorm → Scale → Feed Forward
  → RMSNorm → Gate → Residual Add
```

Attention 负责 token 之间的信息交换。因为文本 token 和图像 token 已在同一序列中，self-attention 可以直接建模“某个文本描述”和“某个图像区域 token”之间的关系。

Feed Forward（前馈网络）负责对每个 token 的 hidden state 做非线性变换。它不做 token 间通信。Z-Image 使用 gated FFN，形态类似 SwiGLU：先升维到 10240，再通过门控激活，最后投回 3840。

### 4.5 Timestep 条件注入

Diffusion / Flow Matching 模型每一步的噪声强度不同，所以 S3-DiT 必须知道当前 timestep。

Z-Image 使用 timestep embedding 生成条件向量，并把条件注入到 Attention 和 FFN 子块中。注入方式是生成 scale 和 gate：

- scale 调制归一化后的输入。
- gate 控制子块输出写回 residual 的强度。

这类设计接近 AdaLN-Zero（Adaptive Layer Normalization Zero）风格。它让模型在不同噪声阶段使用不同的特征变换方式。

在编辑模型中，clean reference image 和 noisy target image 会使用不同的 time-conditioning。clean token 表示条件图像，noisy token 表示正在生成的目标图像。这样模型可以在同一个序列中区分“参考内容”和“待生成内容”。

### 4.6 3D Unified RoPE

Z-Image 使用 **3D Unified RoPE**（统一三维旋转位置编码）给 unified sequence 提供位置信息。

Image token 使用空间坐标。它们的 RoPE 位置对应 latent patch 的 height 和 width，也包含一个类似 frame/time 的轴。

Text token 沿序列维递增。它们和 image token 使用同一套 RoPE 框架，但 position id 的语义不同。

编辑场景中，reference image token 和 target image token 可以共享空间坐标，但在时间轴上错开。这样模型知道两者在空间上对齐，但角色不同。

### 4.7 稳定性设计

S3-DiT 使用几类常见但重要的稳定化设计：

- QK-Norm：对 Query / Key 做归一化，控制 attention score 的幅度。
- Sandwich-Norm：在 Attention / FFN 的输入和输出附近使用 RMSNorm，限制激活幅度。
- Gate：通过条件控制每个子块写回 residual 的强度。
- Full Self-Attention：没有改成 linear attention，也没有使用 MoE。模型主要依赖单流结构和训练流程控制成本。

这些设计的目标不是改变生成范式，而是让 6B 级别的 DiT 在大分辨率、多模态 token 和多阶段训练下保持稳定。

### 4.8 与同类架构的区别

Z-Image 和 SD3 / FLUX 一样属于 Diffusion Transformer 路线。它们都不再使用传统 U-Net 作为主干，而是在 latent token 上运行 Transformer。

Z-Image 的特别之处是：

- 参数量控制在约 6.15B，而不是 20B 到 80B。
- 采用 single-stream early fusion，让 text / image token 在主干每层直接交互。
- 使用轻量 Modality Processor，再进入共享 backbone。
- 统一考虑 text-to-image 和 image editing，而不是为编辑任务完全另建一套架构。
- Turbo 版本通过 few-step distillation 把推理步数降到约 8 NFE。

---

## 5. VAE Decoder（变分自编码器解码器）

VAE Decoder 负责把 S3-DiT 生成完成的 latent 还原成 RGB 图像。Z-Image 使用的是 **Flux VAE**。

S3-DiT 不直接生成像素。它生成的是 VAE latent。VAE Decoder 是最后一步。

```text
denoised latent
  → inverse scaling / shift
  → Flux VAE Decoder
  → RGB image
```

### 5.1 为什么在 latent 空间生成

如果直接在像素空间生成 1024×1024 图像，token 数和计算成本会很高。Latent Diffusion（潜空间扩散）的做法是先用 VAE 把图像压缩到更小的 latent 空间，再让生成模型在 latent 上工作。

这样 S3-DiT 需要处理的空间分辨率更低，但仍能通过 VAE Decoder 还原出高分辨率图像。

### 5.2 VAE 在推理中的角色

文生图推理时，VAE Encoder 通常不参与。流程从随机 latent 噪声开始，经过 S3-DiT 多步去噪，最后由 VAE Decoder 解码。

VAE Decoder 只运行一次。它不参与每一步 denoising loop。

图生图和编辑场景会使用 VAE Encoder，把输入图像或参考图像编码成 latent 条件。普通 text-to-image 场景只需要 Decoder。

---

## 6. 端到端推理流程

从架构视角看，Z-Image 的文生图推理可以分为四步：

```text
1. Text Encode
   Prompt → Qwen3-4B → caption features

2. Latent Init
   Gaussian noise → noisy latent

3. Denoising Loop
   noisy latent + caption features + timestep
     → S3-DiT
     → velocity / latent update
     → scheduler step
   重复 N 步

4. Decode
   denoised latent → Flux VAE Decoder → image
```

其中最重的部分是第 3 步。S3-DiT 会在每个 denoising step 运行一次。Base 模型通常需要数十步；Z-Image-Turbo 通过 distillation 把步数降低到约 8 NFE。

Z-Image 使用 **Flow Matching**（流匹配）目标。模型学习从噪声 latent 到干净 latent 的速度场。推理时 scheduler 根据模型预测更新 latent。

这和传统 DDPM 的 epsilon prediction 表达方式不同，但在工程理解上仍可以看成：每一步模型读取当前 noisy latent 和文本条件，预测如何把 latent 往干净图像方向推进。

### 6.1 Classifier-Free Guidance

Classifier-Free Guidance（无分类器引导，CFG）是一种推理时增强 prompt 约束的方法。启用 CFG 时，同一个 timestep 会计算两路结果：

- conditional：带 prompt 条件。
- unconditional：空 prompt 或 negative prompt 条件。

两路预测在线性组合后再交给 scheduler。CFG 会增强文本遵循，但会增加计算量，也可能损失自然度。

### 6.1 Z-Image 的整体特点

Z-Image 的架构可以概括为：

- Qwen3-4B 负责文本理解。
- S3-DiT 负责 latent 空间中的跨模态融合和去噪。
- Flux VAE 负责 latent 到图像的解码。
- Single-stream backbone 让 text / image token 在每层直接交互。

这个设计和同类 latent diffusion / diffusion transformer 模型共享基本范式，但在参数效率和单流融合上更激进。它的目标是在较小参数量下保留足够的 prompt understanding、图像质量和推理速度。
