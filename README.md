# onnx_export_analysis

面向真实大模型权重的 ONNX 导出与流程分析工具集：把目标模型按 canonical 路径切成一组贴近真实执行语义的代表层 merged ONNX 子图，并对其做逐节点 MACs / 显存 / 参数占比统计。

当前已接入：`Qwen3.5-MoE`（`35B-A3B`），同时支持纯文本子集导出（`qwen3.5moe`）和完整多模态导出（`qwen3.5moe-vl`，含 vision tower + 多模态注入）。`LLaDA-2.1` 占位中。

## 项目初衷

面向大模型的**定量分析**，关注模型**真实推理路径**上的：

- 真实算子流程（与 `transformers` 源码同构的拓扑 / 控制流）
- 每个张量的 **shape**
- 每个算子的输入 / 输出 **dtype**

最终交付的 ONNX 子图必须能作为分析依据，逐节点反推 MACs、激活显存、权重占比。

## 约束

为保证导出结果"等价于真实推理"，所有导出实现必须满足：

1. **与 `transformers` 源码对齐**
   导出图的算子拓扑、调用顺序、分支条件必须能在 HuggingFace `transformers` 官方实现里逐处对应；不允许出现源码里不存在的"等价改写"。

2. **dtype 必须是真实权重的 dtype**
   - 权重 dtype 取自 HuggingFace 仓库下载下来的**实际 safetensors 权重**（例如 `bf16` / `fp8` 量化权重就保持原样）。
   - `transformers` 在 forward 过程中显式做的 cast / upcast / downcast（如 RMSNorm 内部 fp32 累加、softmax 上提精度、MoE expert 输出回落 dtype 等），都必须在 ONNX 图里**原样保留并显式可见**，不做"理想化"简化或全图统一精度。

3. **按语义模块拆分，重复结构只导一份**
   完整 forward 按 `transformers` 的语义层级切成代表性子图（Embedding / DecoderLayer / Attention / MoE-MLP / RMSNorm / LM Head 等）；对同构重复堆叠的结构（如 N 层 DecoderLayer），只导出**一份代表层**作为该模块的真实流程样本。

4. **完整真实数据流可拼接**
   所有子图按真实推理顺序衔接后，相邻子图的输出 / 输入 **shape 与 dtype 必须严格首尾对接**，整组子图等价于一条可追溯的完整 forward 数据流，不允许出现"分析意义上的孤岛"。

5. **避免单图体积爆炸 / 越过 ONNX 2 GB 上限**
   分析依赖的核心信息是**算子流 / shape / dtype / 参数数量与占比**，**不依赖**权重的具体数值。基于此目标可以接受两种处理方式，按实际权重体量自动选择：
   - **超大权重**（典型如 MoE 的 N 个 expert 矩阵堆叠、embedding / lm_head 的 vocab × hidden 等 ≥ 数百 MB / 易冲撞 ONNX 单文件 2 GB protobuf 上限的张量）：**必须作为 graph input 暴露**，仅保留 shape 与 dtype，**不**内嵌任何数值，确保单 onnx 文件本身保持轻量、不被权重撑爆。
   - **小到中等权重**（如 LayerNorm scale/bias、q/k/v/o_proj、router、vision tower 内部的 Conv3d / Linear 等单张 ≪ 100 MB 的张量）：允许直接以 initializer 形式 inline 真实数值，避免给每张代表层子图额外挂一长串 weight input、影响图的可读性；图体积仍然受控（单文件保持在百 MB 级以内），不会触发 protobuf 上限。

   无论走哪种方式，**任何 initializer / input 的 shape 与 dtype 都必须与真实加载后的权重一致**，这是 MACs / 显存 / 参数占比等定量分析得以成立的前提。

6. **难以中间切断的强耦合重复结构 → 包装成自定义节点 + 单独子图**
   对于一个模块内部存在大量重复且**不便从中间切断拆分**的结构（如 `RecurrentGatedDeltaRule`、`ChunkGatedDeltaRule` 这类带状态 / 带分块循环的算子），整体作为**一个自定义算子节点**出现在父图中，仅暴露其外部 I/O 的 shape / dtype；该节点内部实现**单独导出为一份独立的 ONNX 子图**，需要下钻分析时再打开对应子图查看，避免父图被内部重复结构淹没。

## 目录结构

```text
onnx_export_analysis/
├── export_model.py     # 统一入口：选模型 + 给权重路径，一键产出完整子图
├── export_common/      # 跨模型复用的导出底盘（pipeline / shape / manifest / 算子注册）
├── modes/<model>/      # 各模型的导出实现 + 主入口 + 模型 README
├── scripts/            # 模型无关的 ONNX 分析脚本
└── output/             # 默认导出根目录（每个 phase 一个独立子目录 + 子目录 README）
```

## 一键导出

只需告诉入口要导哪个模型 + 真实权重在哪：

```bash
# 纯文本子集（加载 Qwen3_5MoeForCausalLM）
python export_model.py qwen3.5moe \
    --model_path /mnt/data8t/share/models/Qwen/Qwen3.5-35B-A3B
# -> output/Qwen3_5_35B_A3B_ONNX_Prefill_8k/
# -> output/Qwen3_5_35B_A3B_ONNX_Decode_8k/

# 完整多模态（加载 Qwen3_5MoeForConditionalGeneration，含 vision tower）
python export_model.py qwen3.5moe-vl \
    --model_path /mnt/data8t/share/models/Qwen/Qwen3.5-35B-A3B
# -> output/Qwen3_5_35B_A3B_VL_ONNX_Prefill_8k/
# -> output/Qwen3_5_35B_A3B_VL_ONNX_Decode_8k/
```

两个入口默认都跑 8k 上下文，自动同时导出 prefill + decode 两套。`qwen3.5moe-vl` 在 text 子图基础上**额外**导出 4 个 vision/MM 子图（`vision_patch_embed` / `vision_block_<idx>_repr` / `vision_patch_merger` / `mm_inject`），并且这 4 个图**只出现在 prefill 输出目录**——图像 token 在 prefill 阶段一次性写入 `inputs_embeds`，decode 阶段沿用 KV-cache 不再触发 vision tower / mm_inject，因此 vl decode 输出目录与纯文本 decode 等价（同 4 张代表层 text 主图）。

其它高级旋钮（`export_scope` / `batch_size` / 代表层索引 / `vision_token_seq_len` / `mm_image_token_count` 等）走对应模型主入口的内置默认值，统一入口里不暴露。

可选：`--output_dir <dir>` 改导出根目录（默认 `./output`）。

## 高级用法 / 产物语义

`export_model.py` 是薄壳，细调请直接走 `modes/<model>/` 自己的主入口。各模型的参数清单与产物逐文件说明见：

- 模型主入口 README：[`modes/qwen_3_5_MoE/README.md`](modes/qwen_3_5_MoE/README.md)
- 输出语义 README：
  [`output/Qwen3_5_35B_A3B_ONNX_Prefill_8k/README.md`](output/Qwen3_5_35B_A3B_ONNX_Prefill_8k/README.md)
  ｜
  [`output/Qwen3_5_35B_A3B_ONNX_Decode_8k/README.md`](output/Qwen3_5_35B_A3B_ONNX_Decode_8k/README.md)
  ｜
  [`output/Qwen3_5_35B_A3B_VL_ONNX_Prefill_8k/README.md`](output/Qwen3_5_35B_A3B_VL_ONNX_Prefill_8k/README.md)
  ｜
  [`output/Qwen3_5_35B_A3B_VL_ONNX_Decode_8k/README.md`](output/Qwen3_5_35B_A3B_VL_ONNX_Decode_8k/README.md)

输出目录命名约定：
- 纯文本：`Qwen3_5_<model_tag>_ONNX_<Phase>_<context>`，例如 `Qwen3_5_35B_A3B_ONNX_Prefill_8k`。
- 多模态：`Qwen3_5_<model_tag>_VL_ONNX_<Phase>_<context>`，例如 `Qwen3_5_35B_A3B_VL_ONNX_Prefill_8k`。

## ONNX 流量分析

`scripts/` 下两个脚本与具体模型无关，对任意 ONNX 都能跑。统计 `Forward_MACs`（仅 MatMul/Gemm/Conv/常见 Einsum）/ 输出激活字节数 / 参数 initializer 元素数。默认输出落在输入文件/输入目录的同一位置，无需手填路径。

```bash
# 单图逐节点 -> 与输入同目录的 <stem>.flow_stats.tsv / <stem>.flow_stats.summary.json
python scripts/analyze_onnx_flow_stats_single.py <xx.onnx>

# 目录批量 -> <dir>/onnx_flow_stats_multi.xlsx（summary + 每图一 sheet）
python scripts/analyze_onnx_flow_stats_batch.py <dir>
```

## 依赖

```bash
pip install torch transformers onnx onnxsim safetensors openpyxl
```

`transformers` 需要带 `qwen3_5_moe` 实现的版本；权重使用 HuggingFace 格式的本地真实模型目录。

## 添加新模型

1. 在 `modes/<model_name>/` 下放该模型的导出实现与一个独立主入口
2. 在 `export_model.py` 的 `_DISPATCH` 字典里加一行，对应 `_export_<model_name>` 函数

顶层入口只暴露"哪个模型 + 权重在哪 + 输出去哪"三件事，**不**透传细粒度旋钮——细旋钮留在该模型的主入口里维护。
