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
  - `vl` ⇒ 加载 `Qwen3_5MoeForConditionalGeneration`；**`--phase prefill` 时**在文本侧子图基础上**额外**导出 9 个 vision/multimodal 子图（patch_embed、pos_embed_interp、rot_pos_emb、cu_seqlens、代表 ViT block、patch_merger、image_mask_build、mm_inject、mrope_position_ids_prefill），**`--phase decode` 时**导出文本侧子图 + 1 个 multimodal 子图（mrope_position_ids_decode）——图像 token 在 prefill 阶段已写入 KV-cache，但 M-RoPE 的 3D `position_ids` 在每一步 decode 都要重新构造。
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
- `--mrope_text_pre_len`: 代表性 prefill 请求中位于唯一 image segment 之前的 text token 数，仅 `--variant vl --phase prefill` 生效，默认 `64`。剩余 text token 自动落到 image 之后的后缀里，使 `text_pre + image_token_count + text_post == seq_len`。
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

`--variant vl --phase prefill` 在以上文本子图基础上额外生成 vision tower **+ multimodal flow** 共 9 张子图，按数据流顺序：

- `vision_patch_embed_<vseq>.onnx`：Conv3d patchify（对应 `Qwen3_5MoeVisionModel.forward` 行 1239）
- `vision_pos_embed_interp_<vseq>.onnx`：`hidden_states + fast_pos_embed_interpolate(grid_thw)`，含 **学习参数 `pos_embed.weight: [num_position_embeddings, hidden_size]`**（对应源码行 1241–1242，`fast_pos_embed_interpolate` 行 1163）
- `vision_rot_pos_emb_<vseq>.onnx`：`rot_pos_emb(grid_thw) → cat → cos/sin`（对应源码行 1244–1250，`rot_pos_emb` 行 1123；含 `Qwen3_5MoeVisionRotaryEmbedding.inv_freq` 内联 buffer）
- `vision_cu_seqlens_<vseq>.onnx`：`repeat_interleave/cumsum/pad` 构造可变长度打包前缀和（对应源码行 1252–1260）
- `vision_block_<idx>_repr_<vseq>.onnx`：代表 ViT block；attention **保留 `cu_seqlens` 输入并发出源码侧 `tensor_split → 逐段 SDPA → cat` 的拓扑**（对应源码行 1054–1086 + `Qwen3_5MoeVisionAttention` eager 分支 1026–1047）
- `vision_patch_merger_<vseq>.onnx`：spatial merge + LayerNorm + 2-Linear MLP（vision hidden → text hidden）
- `image_mask_build_<seq>.onnx`：`Equal(input_ids, image_token_id) → Unsqueeze → Expand`，构造 image-token 位置布尔掩码（对应 `Qwen3_5MoeModel.get_placeholder_mask` 行 1666–1671）；作为源码对齐参考保留
- `mm_inject_<seq>.onnx`：源码 `inputs_embeds.masked_scatter(image_mask, image_embeds)`（行 1773）的 ONNX 友好等价——行级 `ScatterND(inputs_embeds, image_position_indices, image_embeds)`。避免 `masked_scatter → NonZero` 引入的 data-dependent `unk__N` shape，全图静态便于 MAC/内存分析。等价依据：源码 `image_mask` 是按 `unsqueeze(-1).expand_as(...)` 的整行 mask，每个 image token 占满整个 `H_text` 行，因此与按行 `ScatterND` 数学一致
- `mrope_position_ids_prefill_<seq>.onnx`：M-RoPE 的 `[3, B, S]` 位置索引 + `[B, 1]` rope_deltas，按代表场景静态展开 `[text_pre | image | text_post]` 段布局（对应源码行 1707 `if` 分支 + `get_rope_index` 行 1511 + `get_vision_position_ids` 行 1455）
- 此外 **`vl` 变体下的 `layer_XX_full_attn_block...` 在 prefill / decode 两侧均使用 3D `[3, B, S]` 形状的 dummy `position_ids`** 做 trace，对应锁住 `RotaryEmbeddingBlockMoE.forward` 的 `position_ids.ndim==3` 分支，使 IO 契约直接接收 `mrope_position_ids_*.onnx` 的输出（参数实现：`qwen_merged_block_export.export_full_attn_block(position_ids_ndim=3)`）。纯文本 `base` 变体仍保留 1D `[B, S]` 契约；T=H=W 时两条路径数值完全等价，已通过 unit-level smoke 验证

> 其中 `<vseq>` 取自 `--vision_token_seq_len`（默认 `1024`）；`<seq>` 同 text 路径（prefill 取 `--seq_len`）。

`--variant vl --phase decode` 在文本侧子图基础上**额外**导出 1 张 multimodal 子图：

- `mrope_position_ids_decode_ctx<N>.onnx`：每一步 decode 都要重算的 3D `position_ids: [3, B, ctx+1]`（对应 `compute_3d_position_ids` 行 1720 `elif` 分支 + `attention_mask is not None` 子分支）

> Decode 阶段不导 vision tower / `mm_inject`：图像 token 在 prefill 阶段已通过 `mm_inject` 写入 `inputs_embeds` 并落入 KV-cache，单 token 自回归不会再触发它们；但 M-RoPE 的位置索引必须每步重算，因此独占一份 decode 子图。

源码侧仍然不导出的纯 Python 控制流（无可学习权重；表现为 `itertools.groupby` / `.tolist()` / Python list 操作；与代表层导出策略不兼容）：

- `Qwen3_5MoeModel.get_rope_index` 内的 batch loop + `groupby` 段切分：对每一批的多模态 segment layout 而言，**导出选定的代表场景**（单 image，T=1，``[text_pre | image | text_post]``）已经把 segment 静态展开。要分析其他段布局，重导一份。
- `Qwen3_5MoeModel.compute_3d_position_ids` 的 fallback `else` 分支（`position_ids = None`）：纯调度，下游模型自己重新构造，没有可分析算子。

注意：默认 **不会** 额外导出独立的 `DeltaNetTriangularSolve` 参考子图文件；但主图和 `ChunkGatedDeltaRule` 子图中的 `qwen_onnx::DeltaNetTriangularSolve` 语义仍然保留。

## Qwen 导出实现

- `qwen_export_shared.py`: 主线复用的共享导出 helper / 基础 block 导出 + variant accessor（`_text_model` / `_text_config` / `_visual_model` / `_vision_config`）
- `qwen_shape_propagation.py`: Qwen 专用静态 shape propagation 规则
- `qwen_merged_block_export.py`: merged block 导出实现（含 `_load_model(... , variant=...)`）
- `qwen_onnx_blocks.py`: 文本侧 ONNX wrapper / block 定义
- `qwen_onnx_blocks_vision.py`: vision tower 内部的 ONNX wrapper / block 定义（patch_embed / pos_embed_interp / rot_pos_emb / cu_seqlens / cu_seqlens-aware ViT block / patch_merger）
- `qwen_onnx_blocks_mm.py`: 多模态 flow 边界（`Qwen3_5MoeModel.forward` 内位于 `inputs_embeds` 与 language_model 之间的部分）的 ONNX wrapper / block 定义（image_mask_build / mm_inject / M-RoPE position_ids prefill+decode）
- `qwen_vision_export.py`: vision tower 子图 export wrapper
- `qwen_mm_export.py`: 多模态 flow 子图 export wrapper
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

