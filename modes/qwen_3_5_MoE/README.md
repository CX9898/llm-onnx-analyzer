# Qwen ONNX Export Scripts

## 推荐入口

- `export_qwen_onnx_main.py`: 当前 canonical 主入口

### 主入口用法

```bash
python export_qwen_onnx/export_qwen_onnx_main.py model_path --phase {prefill|decode} [options]
```

常用参数：

- `model_path`: 本地真实模型目录
- `--phase`: 必填，`prefill` 或 `decode`
- `--variant`: `text` 或 `vl`，默认 `text`。
  - `text` ⇒ 加载 `Qwen3_5MoeForCausalLM`，只导出文本侧子图。
  - `vl` ⇒ 加载 `Qwen3_5MoeForConditionalGeneration`；**`--phase prefill` 时**在文本侧子图基础上**额外**导出 4 个 vision/MM 子图（patch_embed、代表 ViT block、patch_merger、mm_inject），**`--phase decode` 时**只导文本侧子图（图像 token 已在 prefill 阶段写入 KV-cache，decode 不再触发 vision/mm_inject）。
- `--export_scope`: `representative` 或 `full`，默认 `representative`
- `--seq_len`: `prefill` 导出长度，默认 `8192`
- `--decode_context_len`: `decode` 时 full attention 的历史上下文长度，默认 `8192`
- `--linear_layer`: 代表性 linear attention 层，默认 `0`
- `--full_layer`: 代表性 full attention 层，默认 `3`
- `--batch_size`: 导出场景 batch size，默认 `1`
- `--linear_prefill_chunk_size`: prefill linear attention 的 chunk size，默认 `64`
- `--vision_block_layer`: 代表性 ViT block 层 index，仅 `--variant vl` 生效，默认 `0`
- `--vision_token_seq_len`: vision tower 静态导出 seq_len（即一次推理的 patch token 数），仅 `--variant vl` 生效，默认 `1024`（对应 32×32 grid）
- `--mm_image_token_count`: mm_inject 导出时假设注入的 image token 数，仅 `--variant vl` 生效，默认 `256`（= `vision_token_seq_len // spatial_merge_size**2`）
- `--opset`: ONNX opset，默认 `20`
- `--no_simplify`: 跳过 `onnxsim` 简化
- `--output_dir`: 输出目录；不填时按 phase + variant 自动给默认目录

常见示例：

```bash
# 1) 真实模型导出 representative decode（纯文本子集）
python export_qwen_onnx/export_qwen_onnx_main.py \
  /mnt/data8t/share/models/Qwen/Qwen3.5-35B-A3B \
  --phase decode \
  --decode_context_len 8192

# 2) 导出 full scope（所有 canonical 层）
python export_qwen_onnx/export_qwen_onnx_main.py \
  /mnt/data8t/share/models/Qwen/Qwen3.5-35B-A3B \
  --phase prefill \
  --export_scope full \
  --seq_len 8192

python export_qwen_onnx/export_qwen_onnx_main.py   \
  /mnt/data8t/share/models/Qwen/Qwen3.5-35B-A3B   \
  --phase decode  \
  --decode_context_len 8192 \
  --output_dir Qwen3_5_35B_A3B_Merged_ONNX_Decode_8k_new

python export_qwen_onnx/export_qwen_onnx_main.py \
  /mnt/data8t/share/models/Qwen/Qwen3.5-35B-A3B \
  --phase prefill \
  --seq_len 8192 \
  --output_dir Qwen3_5_35B_A3B_Merged_ONNX_Prefill_8k_new

# 3) 完整多模态导出：vision tower + 多模态注入 + 全套文本子图
python export_qwen_onnx/export_qwen_onnx_main.py \
  /mnt/data8t/share/models/Qwen/Qwen3.5-35B-A3B \
  --phase prefill \
  --variant vl \
  --seq_len 8192 \
  --vision_token_seq_len 1024 \
  --mm_image_token_count 256
```

默认只支持基于真实本地权重的 canonical 导出；不再提供 `dummy`、`dtype override`、`abstract export`、裁剪 experts 等会偏离真实模型语义的模式。

## 默认输出

所有导出都会生成：

- `embedding_<seq>.onnx`
- `layer_XX_linear_attn_block...`
- `layer_XX_moe_block_<seq>.onnx`
- `layer_XX_full_attn_block...`
- `norm_<seq>.onnx`
- `lm_head_<seq>.onnx`
- `onnx_stats.json`

`decode` 默认 linear attention 还会导出：

- `layer_XX_linear_attn_block_RecurrentGatedDeltaRule.onnx`
- `layer_XX_linear_attn_block.json`

`prefill` 默认 linear attention 还会导出：

- `layer_XX_linear_attn_block_<seq>.json`
- `layer_XX_linear_attn_block_ChunkGatedDeltaRule_chunk<chunk>_<seq>.onnx`
- `layer_XX_linear_attn_block_ChunkGatedDeltaRule_DeltaNetChunkStep_chunk<chunk>.onnx`

`--variant vl --phase prefill` 在以上文本子图基础上额外生成：

- `vision_patch_embed_<vseq>.onnx`：Conv3d patchify
- `vision_block_<idx>_repr_<vseq>.onnx`：单个代表性 ViT block（norm1 → attn → 残差 → norm2 → mlp → 残差），attention 走单 segment 路径（见下方"未导出的 vision 控制流"）
- `vision_patch_merger_<vseq>.onnx`：spatial merge + LayerNorm + 2-Linear MLP（vision hidden → text hidden）
- `mm_inject_<seq>.onnx`：`inputs_embeds.masked_scatter(image_mask, image_embeds)`，把 vision tower 的输出写回 text 序列里的 image-token 位置

> 其中 `<vseq>` 取自 `--vision_token_seq_len`（默认 `1024`）；`<seq>` 同 text 路径（prefill 取 `--seq_len`）。

`--variant vl --phase decode` **不会**额外导出上述 4 个 vision/MM 子图——decode 阶段是单 token 自回归（`seq_len=1`），图像 token 在 prefill 阶段就已通过 `mm_inject` 写入 `inputs_embeds` 并落入 KV-cache，后续步骤不会再触发 vision tower 或 `mm_inject`，因此 vl decode 输出目录与纯文本 decode 等价（同一组代表层 text 主图）。

未导出的 vision 控制流（无可学习权重，由推理引擎在 CPU 端按 `grid_thw` 计算并以 `cos/sin` 等张量喂入子图）：

- `Qwen3_5MoeVisionModel.fast_pos_embed_interpolate(grid_thw)`：基于 48×48 `pos_embed` 表的双线性插值
- `Qwen3_5MoeVisionModel.rot_pos_emb(grid_thw)`：2D（H, W）旋转位置索引构造
- `Qwen3_5MoeModel.get_rope_index(...)`：M-RoPE 的 3D `position_ids: [3, B, S]` 构造（含 `itertools.groupby` 与 Python list 操作）

注意：默认 **不会** 额外导出独立的 `DeltaNetTriangularSolve` 参考子图文件；但主图和 `ChunkGatedDeltaRule` 子图中的 `qwen_onnx::DeltaNetTriangularSolve` 语义仍然保留。

## Qwen 导出实现

- `qwen_export_shared.py`: 主线复用的共享导出 helper / 基础 block 导出 + variant accessor（`_text_model` / `_text_config` / `_visual_model` / `_vision_config`）
- `qwen_shape_propagation.py`: Qwen 专用静态 shape propagation 规则
- `qwen_merged_block_export.py`: merged block 导出实现（含 `_load_model(... , variant=...)`）
- `qwen_onnx_blocks.py`: 文本侧 ONNX wrapper / block 定义
- `qwen_onnx_blocks_vision.py`: vision tower 与多模态注入的 ONNX wrapper / block 定义
- `qwen_vision_export.py`: vision/MM 子图 export wrapper
- `qwen_export_semantics.py`: Qwen 导出语义与 custom op 规则

## 校验与分析

- `audit_qwen_dtype_semantics.py`
- `analyze_onnx_flow_stats_single.py`
- `analyze_onnx_flow_stats_batch.py`
- `compare_onnx_graphs.py`

### 分析脚本用法

```bash
# 1) 单个 ONNX 的逐节点流量统计
python export_qwen_onnx/analyze_onnx_flow_stats_single.py \
  Qwen3_5_35B_A3B_Merged_ONNX_Decode_8k_new/layer_00_linear_attn_block.onnx \
  --out_tsv Qwen3_5_35B_A3B_Merged_ONNX_Decode_8k_new/layer_00_linear_attn_block.flow.tsv \
  --out_json Qwen3_5_35B_A3B_Merged_ONNX_Decode_8k_new/layer_00_linear_attn_block.flow.json

# 2) 一个目录里所有 ONNX 的批量统计（输出 Excel 多 sheet）
python export_qwen_onnx/analyze_onnx_flow_stats_batch.py \
  Qwen3_5_35B_A3B_Merged_ONNX_Decode_8k_new \
  --out_xlsx Qwen3_5_35B_A3B_Merged_ONNX_Decode_8k_new/
   \
  --out_json Qwen3_5_35B_A3B_Merged_ONNX_Decode_8k_new/onnx_flow_stats_multi.json

# 3) 对比两份 ONNX 图的结构、输入输出 shape 和算子差异
python export_qwen_onnx/compare_onnx_graphs.py \
  Qwen3_5_35B_A3B_Merged_ONNX_Decode_8k_new/layer_00_linear_attn_block.onnx \
  Qwen3_5_35B_A3B_Merged_ONNX_Decode_8k/layer_00_linear_attn_block.onnx

# 4) 生成 Qwen 导出与 Transformers 源码的 dtype 语义审计
python export_qwen_onnx/audit_qwen_dtype_semantics.py \
  --out export_qwen_onnx/qwen_dtype_audit.md
```

说明：

- 主导出 CLI 默认只生成 ONNX、结构 manifest（`layer_00_linear_attn_block*.json`）和 `onnx_stats.json`
- `onnx_flow_stats_multi.xlsx` / `onnx_flow_stats_multi.json` 需要通过 `analyze_onnx_flow_stats_batch.py` 额外生成

这些脚本的典型用途：

- `analyze_onnx_flow_stats_single.py`: 看单个 ONNX 每个节点的 MACs / 输出内存 / 参数占比
- `analyze_onnx_flow_stats_batch.py`: 汇总一个目录下所有 ONNX，适合看整个导出集合
- `compare_onnx_graphs.py`: 比较新旧导出图、重构前后结构差异
- `audit_qwen_dtype_semantics.py`: 检查导出 wrapper 与 Transformers 源码的 dtype 处理是否一致

## 当前原则

目录只保留 canonical 主线与必要共享模块；不再保留旧 split-export / 旧子图 CLI 入口。

当前主线强调两点：

- 导出接口中的 `dtype` / `shape` 应与真实权重加载后的实际执行语义一致
- 默认路径优先保留主图与必要子图，不额外导出高耗时的参考图产物

