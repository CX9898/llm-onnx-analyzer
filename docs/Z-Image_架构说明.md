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

Z-Image 是阿里通义开源的文生图(**Text-to-Image, T2I**) 基础模型，参数量约 **6B** ，核心生成骨干为 **S3-DiT**（Scalable Single-Stream Diffusion Transformer）。

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

## 3. 端到端推理数据流

### 3.1 三阶段划分

```text
[Phase 1: Text Encode — 每个 prompt 一次]
  Prompt → Tokenizer → Text Encoder → caption 特征

[Phase 2: Denoise — scheduler 循环 × N 步(Turbo 常用 8，Base 常用 28–50)]
  随机噪声 latent + caption 特征 + timestep
      → S3-DiT forward
      → noise / velocity 预测
      → Scheduler 更新 latent

[Phase 3: VAE Decode — 去噪结束后一次]
  干净 latent → 反 scaling → VAE decode → 图像
```

### 3.2 Denoise 循环（架构视角）

Z-Image 使用 **Flow Matching** 调度（diffusers 中为 `FlowMatchEulerDiscreteScheduler`），不是 DDPM 的 ε-prediction 范式。架构上仍表现为：

- 每步输入：当前 noisy latent、归一化 timestep、caption 条件。
- 每步输出：模型预测量（经 scheduler 转换为 latent 更新）。
- 步数：Base 模型通常数十步；Turbo 约 8 步。

**Classifier-Free Guidance（CFG）** 在架构层体现为：同一步内对 cond / uncond 两路 caption 各跑一次 DiT forward，在 host 侧线性组合预测结果。Turbo 默认关闭 CFG（`guidance_scale=0`）。

### 3.3 Latent 与 Token 的关系

- 图像经 **VAE Encoder**（训练阶段）压缩到 latent，通道数 **16**。
- DiT 不直接在 `[C,H,W]` 张量上做全局 Attention，而是先 **patchify**：将 latent 切为 patch token 序列。
- 512×512 像素对应 latent 约 64×64，patch 后 image token 数约 **1024**（与分辨率、patch 配置相关）。

---

## 4. S3-DiT 核心架构

S3-DiT 是 Z-Image 的生成骨干，名称中的 **Single-Stream** 指：多种模态 token 在序列维 **concat** 后，进入**同一套** Transformer layer 做 self-attention，而不是 text / image 各走独立双流再交叉。

### 4.1 与前段 MM-DiT 的关系

S3-DiT 继承 **MM-DiT**（Multi-Modal Diffusion Transformer）的 single-stream concat 思路（与 Stable Diffusion 3 同族），但 Z-Image 将其扩展为可 scaling 的 6B 实现，并统一支持 T2I 与 Edit。

### 4.2 内部四段结构

单次 DiT forward（Basic 文生图）在架构上分为四段：

```text
(1) Timestep 条件
    t → TimestepEmbedder → adaln 向量（供后续 AdaLN 使用）

(2) Patchify + 模态预处理（Modality Processor）
    noisy latent → patch tokens ──► noise_refiner ×2  ──┐
    caption 特征 → cap tokens   ──► context_refiner ×2 ──┤
                                                          │
(3) 序列融合                                          ◄───┘
    image tokens + caption tokens → 统一序列 [S_x + S_cap, D]

(4) 单流 Backbone + 输出头
    unified layers ×30 → FinalLayer → unpatchify → 预测量
```

**Modality Processor**（各模态前的 refiner）是 Z-Image 相对「直接 concat 进 backbone」的额外设计：每个模态先经 **2 层** 轻量 Transformer block 对齐，再进入 30 层主 stack。论文动机是在 concat 前完成模态对齐，提高参数效率。

### 4.3 关键规格（S3-DiT）

| 配置项 | 值 |
|--------|-----|
| 总参数量 | ~6.15B |
| Backbone 层数 | 30 |
| Hidden dim | 3840 |
| Attention heads | 32 |
| FFN 中间维 | 10240 |
| Refiner 层数（每模态） | 2 |
| 3D RoPE 轴维度 (d_t, d_h, d_w) | (32, 48, 48) |
| Latent 通道 | 16 |
| Caption 特征维 | 2560（来自 Qwen3 Text Encoder） |

---

## 5. Transformer Block 结构

S3-DiT 的基本重复单元为 **ZImageTransformerBlock**，在 refiner 与 backbone 中复用，差异在于是否启用 **AdaLN**（Adaptive Layer Normalization）：

| 位置 | AdaLN | Timestep 条件 |
|------|-------|---------------|
| `noise_refiner` | 有 | 有 |
| `context_refiner` | 无 | 无 |
| `layers`（backbone ×30） | 有 | 有 |

单个 block 的数据路径：

```text
hidden [B, S, D]
    │
    ├─ AdaLN(timestep) → scale/gate（refiner/backbone 有 AdaLN 的 block）
    │
    ├─ Attention 支：
    │     RMSNorm → ×scale → Q/K/V → QK-Norm → 3D U-RoPE
    │     → Full Self-Attention → Out → RMSNorm → ×gate → +residual
    │
    └─ FFN 支：
          RMSNorm → ×scale → FFN(D→10240→D) → RMSNorm → ×gate → +residual
```

稳定化设计（与同代 DiT 共性 + Z-Image 采用项）：

- **QK-Norm**：约束 attention 中 Q/K 幅度。
- **Sandwich-Norm**：Attention / FFN 的输入、输出两侧各一层 RMSNorm。
- **AdaLN-Zero 风格 gate**：scale / gate 来自 timestep 条件；gate 经 tanh，scale 为 `1 + Δ`。
- **AdaLN 低秩分解**：共享 down-projection + 逐层 up-projection，降低条件注入参数量。

Attention 为 **Full Self-Attention**（非 linear attention，非 cross-attention 双流）。text token 与 image token 在同一序列内互相 attend。

---

## 6. 位置编码：3D Unified RoPE

Z-Image 对混合序列使用 **3D Unified RoPE**（Unified Rotary Position Embedding）：

- **Image token**：RoPE 坐标扩展在 **空间维**（height、width）及帧维（F patch）。
- **Text token**：RoPE 坐标沿 **时间 / 序列维** 递增。
- 不同模态 token 在同一 self-attention 中共享 RoPE 框架，但 position id 语义按模态区分。

512² 文生图代表场景：unified 序列长度约 **1152**（image ~1024 + caption ~128），hidden **3840**。

---

## 7. Basic 文生图 vs Omni 编辑

同一 S3-DiT 骨架支持两种运行模式：

### 7.1 Basic（文生图）

- 输入：单张 noisy latent + caption 特征。
- Timestep：全局单一 embedding，所有 token 共享。
- 序列：image tokens + caption tokens。

### 7.2 Omni（Z-Image-Edit）

- 输入：多张图像 latent（如 reference 干净图 + 待编辑 noisy 图）+ caption + 可选 **SigLIP** semantic 特征。
- Timestep：reference 与 target 可用 **不同** 条件（区分 clean / noisy）；omni 模式下有 `t_noisy` / `t_clean` 双路 AdaLN。
- RoPE：reference 与 target 空间坐标对齐，**时间维** 加 unit offset 区分。
- 额外模态：**SigLIP-2** embedding + `siglip_refiner ×2`（仅 Edit 配置启用）。

Basic 是 Omni 的子集。文生图推理走 Basic；编辑任务走 Omni。

---

## 8. Text Encoder 与 VAE 的架构角色

### 8.1 Text Encoder（Qwen3-4B）

- 独立 Decoder-only LLM，**不参与** DiT 的 layer 堆叠。
- 推理时取 **倒数第二层** hidden states 作为 caption 特征（非最后一层 LM head 输出）。
- 支持中英文 prompt；pipeline 侧通过 chat template 格式化输入。
- Caption 特征维 **2560**，经 DiT 内 `cap_embedder` 投影到 **3840** 后进入 refiner / backbone。

### 8.2 VAE（Flux VAE）

- 负责 pixel ↔ latent 转换；DiT 在 latent 空间工作。
- Latent 通道 16；decode 前需按 `scaling_factor` / `shift_factor` 反变换。
- VAE 与 DiT 分离训练、分离加载；推理时 VAE 只运行 decode 一次。

---

## 9. 与同类型架构的对比

### 9.1 对比维度

| 维度 | Z-Image (S3-DiT) | 典型 Dual-Stream DiT（如早期 SD3/FLUX 部分实现） | 超大单体（如 Qwen-Image 20B+） |
|------|------------------|--------------------------------------------------|--------------------------------|
| Text/Image 交互 | Single-stream concat + self-attn | Text / Image 分离 stream，通过 cross-attn 或 joint block 交互 | 常为 dual-stream 或更大 single-stream |
| 参数量 | ~6B | 10B–30B 常见 | 20B–80B |
| 模态预处理 | 每模态 refiner ×2 再 concat | 部分模型直接 embed 后 concat | 因模型而异 |
| 条件注入 | AdaLN(timestep) | AdaLN / cross-attn | 类似 AdaLN 或额外 cond stream |
| Attention | Full self-attn，3D U-RoPE | Full 或 mixed | Full，规模更大 |
| FFN | Dense | Dense | 部分采用 MoE |
| Text Encoder | 外挂 Qwen3-4B | 外挂 T5/CLIP/LLM | 外挂大 LLM |

### 9.2 与 LLM（如 Qwen3.5-MoE）的结构差异

两者都使用 Transformer stack，但任务与数据形态不同：

| | Z-Image S3-DiT | Qwen3.5-MoE 等 LLM |
|--|----------------|---------------------|
| 任务 | 迭代去噪（扩散） | 自回归 next-token |
| 序列内容 | image patch + text token 混合 | 纯 text token |
| 循环方式 | 外循环 scheduler × N 步 | 内循环每步 1 token |
| 条件 | timestep AdaLN | 无扩散 timestep |
| FFN | Dense | MoE（稀疏路由） |
| 位置编码 | 3D U-RoPE（多模态） | 1D RoPE |
| KV Cache | 每步全序列重算（无 decode cache 语义） | decode 阶段依赖 KV cache |

### 9.3 Z-Image 的架构特点（归纳）

1. **Single-stream early fusion**：text / image 早期 concat，每层 self-attn 全交互；参数效率高于长期维持独立双流。
2. **两阶段模态对齐**：refiner（×2）+ backbone（×30），concat 前先做模态内预处理。
3. **6B 规模下的 Full Attention**：未采用 linear attention 或 MoE 换规模；依靠单流与训练效率优化控制成本。
4. **统一骨架覆盖 T2I / Edit**：Omni 模式扩展输入模态与双 timestep，不重训整套独立编辑模型结构。
5. **外挂 Text Encoder**：DiT 本体不含 LLM 层；caption 以 token 特征注入，而非在 DiT 内嵌整段 LLM forward。

---

## 10. 512×512 代表场景的序列规模

以下数值用于理解 tensor 形状，具体以 config 与分辨率为准：

| 阶段 | 关键 shape（batch=1） |
|------|------------------------|
| Text Encoder 输出 | caption `[128, 2560]`（padding 后；有效 token 数可变） |
| DiT latent 输入 | `[16, 1, 64, 64]` bf16（含 frame 维） |
| Image tokens | `[1024, 3840]` |
| Caption tokens | `[128, 3840]` |
| Unified 序列 | `[1152, 3840]` |
| Denoise 输出 | noise pred `[16, 64, 64]` f32 |
| VAE 输出 | image `[3, 512, 512]` |

---

## 11. 架构总览图

布局参考 Qwen3.5-MoE 架构图：**上方放大最重重复 block，底行为端到端主链**。

```text
┌────────────────────────────────────────────────────────────────────────────┐
│  【放大】S3-DiT Block（backbone ×30）                                       │
│   AdaLN(t) → RMSNorm → Single-Stream Attn(U-RoPE, QK-Norm) → FFN → residual│
└────────────────────────────────────────────────────────────────────────────┘

Prompt ──► [Qwen3 Text Encoder] ──► caption 特征 ────────────────┐
                                                                 ├──► [Patchify]
Noise Latent ────────────────────────────────────────────────────┘
       │                              ┌──► X: embed + noise_refiner×2
Timestep ──► [TimestepEmbed] ──► AdaLN│──► Cap: embed + context_refiner×2
                                       │              │
                                       └──────► [Concat S=1152]
                                                      │
                                              ┌─ S3-DiT ×30 ─┐
                                              └──────────────┘
                                                      │
                                              [Final + Unpatchify]
                                                      │
                                              scheduler × N ──► clean latent
                                                      │
                                              [VAE Decode] ──► Image
```

---

## 12. 进一步阅读

| 资料 | 内容 |
|------|------|
| `Z-Image/docs/Z-Image_paper.md` | 论文原文：S3-DiT、训练流程、Turbo 蒸馏 |
| diffusers `pipeline_z_image.py` | 推理三阶段与 CFG / scheduler 行为 |
| diffusers `transformer_z_image.py` | S3-DiT forward 结构与 Basic / Omni 分支 |
| `output/Z_Image_ONNX_512/README.md` | 本仓库 ONNX 子图与架构块映射（分析用） |

---

## 13. 术语表

| 术语 | 含义 |
|------|------|
| **S3-DiT** | Scalable Single-Stream Diffusion Transformer，Z-Image 生成骨干 |
| **MM-DiT** | Multi-Modal DiT，多模态 token concat + transformer 范式 |
| **Flow Matching** | 连续归一化流式生成框架；Z-Image 去噪调度基于此 |
| **NFE** | Number of Function Evaluations，扩散推理中 DiT forward 次数 |
| **AdaLN** | Adaptive Layer Normalization，用 timestep 等条件生成 scale / shift / gate |
| **U-RoPE** | Unified RoPE，Z-Image 对混合序列的 3D 旋转位置编码 |
| **CFG** | Classifier-Free Guidance，cond / uncond 预测线性组合 |
| **Latent** | VAE 压缩空间中的图像表示，DiT 操作对象 |
| **Patchify** | 将 latent 切分为 patch token 序列 |
| **Omni** | 支持多图 + 编辑条件的运行模式（含 Edit） |
