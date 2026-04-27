# Qwen3.5-MoE-VL Merged ONNX 说明：`prefill`

这个目录保存的是 `Qwen3.5-35B-A3B`（多模态完整版）一套面向 `8k prefill` 的代表层 merged ONNX 子图，以及配套的 vision tower / 多模态注入子图与统计文件。

## 完整数据流（按这个顺序看 onnx）

下图覆盖本目录所有主链 onnx 文件，箭头方向就是 forward 时数据流的方向；列在右侧的 `*.onnx` 就是每一步对应的产物文件名。

```text
[Vision tower — 每张图/视频片段一次性计算]

  pixel_values_flat:[N,1536]bf16       (N = patch token 数；默认 N=1024 = 32×32 grid)
        │
        ▼  vision_patch_embed_1k.onnx
  patch_embeds:[N,1152]bf16
        │  (host CPU 加 fast_pos_embed_interpolate, 不进 ONNX 图)
        ▼  vision_block_00_repr_1k.onnx        ×27 (block-0 作为 27 层 ViT 的代表层)
  vision_features:[N,1152]bf16
        │
        ▼  vision_patch_merger_1k.onnx
  image_embeds:[N/4, 2048]bf16  ──────────────────────────────┐
                                                               │
[Text 主链 — 整段 8k 一次性灌入]                               │
                                                               │
  input_ids:[1,8192]i64                                        │
        │                                                      │
        ▼  embedding_8k.onnx                                   │
  inputs_embeds:[1,8192,2048]bf16                              │
        │                                                      ▼
        ▼  mm_inject_8k.onnx ◄── image_mask:[1,8192,2048]bool (host CPU 由 input_ids==image_token_id 构造)
  hidden_states:[1,8192,2048]bf16
        │
        │   ×40 层 decoder stack (layer_idx % 4 == 3 → full attention, 否则 linear attention)
        │   本目录只导出两类代表层:
        │
        │     layer 0 (linear attn + MoE):
        │       ─► layer_00_linear_attn_block_8k.onnx
        │       ─► layer_00_moe_block_8k.onnx
        │
        │     layer 3 (full attn + MoE):
        │       ─► layer_03_full_attn_block_8k.onnx
        │       ─► layer_03_moe_block_8k.onnx
        │
        ▼  norm_8k.onnx
  hidden_states:[1,8192,2048]bf16
        │
        ▼  lm_head_8k.onnx
  logits:[1,8192,248320]bf16
```

主链之外，本目录还导出两张 **custom op 展开参考子图**（不是独立调用步骤，只是 `layer_00_linear_attn_block_8k.onnx` 顶层图里 `qwen_onnx::ChunkGatedDeltaRule` / `DeltaNetChunkStep` 的内部展开，方便结构对照）：

- `layer_00_linear_attn_block_ChunkGatedDeltaRule_chunk64_8k.onnx`
- `layer_00_linear_attn_block_ChunkGatedDeltaRule_DeltaNetChunkStep_chunk64.onnx`

> 注：这两类代表层是 **并列样本**，不是 `layer_00_*` 跑完接到 `layer_03_*` 的顺序链路。真实的 40 层 decoder 是 `layer_idx=0,1,2(linear) → 3(full) → 4,5,6(linear) → 7(full) → ...` 这样不断重复，本目录只各取一种代表层导出。

## 先看结论

- `prefill` 的意思是：一次输入整段上下文（含图像/视频 token），文本侧固定 `seq_len=8192`，vision 侧 `vseq=1024`（默认 1024 个 patch token）。
- 本目录最值得关注的图分两块：
  - **vision / 多模态接入**（这一目录与纯文本目录的差异点）：
    - `vision_patch_embed_1k.onnx`
    - `vision_block_00_repr_1k.onnx`
    - `vision_patch_merger_1k.onnx`
    - `mm_inject_8k.onnx`
  - **文本主链**（与 `Qwen3_5_35B_A3B_ONNX_Prefill_8k` 完全等价的代表层四张主图）：
    - `layer_00_linear_attn_block_8k.onnx`
    - `layer_00_moe_block_8k.onnx`
    - `layer_03_full_attn_block_8k.onnx`
    - `layer_03_moe_block_8k.onnx`
- 顶层加载类是 `Qwen3_5MoeForConditionalGeneration`；text 侧子图通过 `model.model.language_model` 取 backbone，与纯文本路径 `Qwen3_5MoeForCausalLM.model` 同构，只是路径被多包了一层。
- `embedding_8k.onnx`、`norm_8k.onnx`、`lm_head_8k.onnx` 主要补齐完整首尾链路，不是这个目录最核心的差异点。
- prefill linear attention 的 `ChunkGatedDeltaRule` / `DeltaNetChunkStep` 参考子图与纯文本目录完全一致，本目录不重复说明。
- **vision tower 与 `mm_inject` 仅在本目录（prefill）导出**：图像 token 在 prefill 阶段一次性写进 `inputs_embeds` 并经 KV-cache 落到缓存里，decode 阶段不会再触发任何 vision/mm 图，因此 [`Qwen3_5_35B_A3B_VL_ONNX_Decode_8k/`](../Qwen3_5_35B_A3B_VL_ONNX_Decode_8k/README.md) 只保留代表层 text 主链 4 张图。

## 目录在整模型里代表什么

Qwen3.5-MoE 多模态完整版的 forward 由四部分组成：

1. **Vision tower**（27 层 ViT）：
   `pixel_values → patch_embed (Conv3d) → 27 × vision_block → patch_merger → image_embeds`
   produced once per multimodal request；和文本侧 KV-cache 没有任何耦合。
2. **Text embedding**：
   `input_ids → embedding → inputs_embeds`
3. **多模态注入**：
   `inputs_embeds.masked_scatter(image_mask, image_embeds)` 把 vision tower 的输出写回 text 序列里 `image_token_id` 对应的位置。
4. **Text decoder stack（40 层）**：与纯文本路径完全一致。

本目录沿用代表层导出口径：

- ViT 一侧只导**第 0 层**作为 27 层 Transformer 块的代表样本（`--vision_block_layer 0`，可改）；
- text 一侧仍是 `layer 0`（linear attention + MoE）+ `layer 3`（full attention + MoE）两条并列样本路径。

## `prefill` 在这里具体是什么意思

- 文本主链激活张量 `seq_len=8192`，整段上下文一次性灌入；
- vision tower 的 patch token 数（vseq）是另一条独立维度，默认 `vseq=1024`（对应一张 32×32 grid 的图）；
- 两条数据流通过 `mm_inject` 汇合到 text 主链上。

## vision / 多模态接入主图

### `vision_patch_embed_1k.onnx`

- 作用：Conv3d 把已 unfold 的 patch tensor 投到 vision hidden
- 输入：`pixel_values_flat:bf16:[1024, 3*2*16*16]`（即 `[N_patches, in_channels * temporal_patch_size * patch_size * patch_size]`）
- 输出：`patch_embeds:bf16:[1024, 1152]`
- 说明：
  - 严格对齐 `Qwen3_5MoeVisionPatchEmbed.forward` —— `view → Conv3d → view(-1, hidden)`
  - `temporal_patch_size=2`，单图复制为长度 2 的时间轴；视频天然带时间维
  - `pos_embed` 双线性插值（`fast_pos_embed_interpolate`）由推理引擎计算后**加到** patch_embeds 上，**不**进 ONNX 图（Python 控制流，无可学习参数）

### `vision_block_00_repr_1k.onnx`

- 作用：单个 ViT block 的代表样本 `norm1 → attn → +residual → norm2 → mlp → +residual`
- 输入：
  - `hidden_states:bf16:[1024, 1152]`
  - `cos:bf16:[1024, 72]`（head_dim = 1152 / 16 = 72）
  - `sin:bf16:[1024, 72]`
- 输出：`hidden_states_out:bf16:[1024, 1152]`
- 说明：
  - 对齐 `Qwen3_5MoeVisionBlock.forward`，但 attention 走**单 segment 路径**（`cu_seqlens` 简化为 `[0, vseq]`）
  - 单 segment 路径与上游 `eager_attention_forward` 在 `cu_seqlens.size(0)==2` 时数学等价；切换到 packed 多图/多视频场景时，外部以多次调用本图（每次喂一段）即可
  - rotary 由 `apply_rotary_pos_emb_vision` 直接调用 transformers 源码符号，避免漂移
  - softmax 在 fp32 计算后回落到主 dtype，与文本侧一致

### `vision_patch_merger_1k.onnx`

- 作用：spatial merge + LayerNorm + 2-Linear MLP，把 vision hidden（1152）映射到 text hidden（2048）
- 输入：`vision_features:bf16:[1024, 1152]`
- 输出：`image_embeds:bf16:[256, 2048]`
  - `256 = 1024 / spatial_merge_size**2 = 1024 / 4`
- 说明：
  - 对齐 `Qwen3_5MoeVisionPatchMerger.forward`
  - 先 `view(-1, hidden_size * spatial_merge**2)`（spatial concat），再 LayerNorm + Linear（`4*1152→4*1152`）+ `GELU` + Linear（`4*1152→2048`）
  - `use_postshuffle_norm=False`（与默认配置一致）

### `mm_inject_8k.onnx`

- 作用：把 `image_embeds` 写回 `inputs_embeds` 中 image-token 占位
- 输入：
  - `inputs_embeds:bf16:[1, 8192, 2048]`
  - `image_mask:bool:[1, 8192, 2048]`（按 hidden 维 broadcast 后的布尔张量；`True` 表示该位置是 image token 占位）
  - `image_embeds:bf16:[256, 2048]`
- 输出：`inputs_embeds_out:bf16:[1, 8192, 2048]`
- 说明：
  - 严格对应源码 `inputs_embeds.masked_scatter(image_mask, image_embeds)`
  - `image_mask` 由外部按 `input_ids == image_token_id` 构造（保留同样的 `unsqueeze(-1).expand_as` 语义）；ONNX 图本身不绑定 `image_token_id`，便于复用到不同 vocab 配置
  - 静态导出时假设 `mask.sum() == image_embeds.shape[0] * H_text`；运行时 image token 数可变化

## 文本主链主图

### `layer 0` 样本路径

`embedding_8k → mm_inject_8k → layer_00_linear_attn_block_8k → layer_00_moe_block_8k`

### `layer 3` 样本路径

`layer_03_full_attn_block_8k → layer_03_moe_block_8k`

> `mm_inject` 在数据流上紧跟 embedding，把 image token 占位换成真正的 visual feature，**之后整段 text decoder 的输入** `hidden_states:[1, 8192, 2048]` **与纯文本路径完全一致**。这是当前文本侧子图无需任何改动就能直接接入多模态的关键。

四张代表层 text 主图的语义、shape、dtype 与 [`Qwen3_5_35B_A3B_ONNX_Prefill_8k/README.md`](../ori/Qwen3_5_35B_A3B_ONNX_Prefill_8k/README.md) 完全相同（同一份导出代码、同一份 backbone 权重，仅经过 transformers 的 key 透明 remap），不在本目录重复说明。

## 你真正需要关心的多模态差异

- 多模态情况下 `position_ids` 的语义从 1D `[B, S]` 变为 3D `[3, B, S]`（M-RoPE 的 `T/H/W` 三轴），`Qwen3_5MoeTextRotaryEmbedding`（已封装为 `RotaryEmbeddingBlockMoE`）会按 `mrope_section=[11,11,10]` 做 interleave。
- 文本侧的 `layer_03_full_attn_block_8k.onnx` **本身已对 M-RoPE 做兼容**（`mrope_interleaved=true` 来自 config，导出时已生效）；纯文本场景下外部喂 1D `position_ids`（被 `RotaryEmbeddingBlockMoE` 内部静态展开为三路相同），多模态场景下喂真实 3D `position_ids`，**不需要重新导出**。
- M-RoPE 的 3D `position_ids` 由 `Qwen3_5MoeModel.get_rope_index` 计算（含 `itertools.groupby` 与 Python list 操作，不进 ONNX 图），由推理引擎在 CPU 端完成。
- vision tower 一侧的 `cos/sin`（2D 旋转）由 `rot_pos_emb(grid_thw)` + `fast_pos_embed_interpolate(grid_thw)` 在 CPU 端计算并喂入子图，同样不进 ONNX。

## 目录里的辅助文件

- `onnx_stats.json`：每个导出文件的基础统计
- linear attention 的 `ChunkGatedDeltaRule` / `DeltaNetChunkStep` 参考子图与纯文本目录完全一致，按需查阅 [`Qwen3_5_35B_A3B_ONNX_Prefill_8k/README.md`](../ori/Qwen3_5_35B_A3B_ONNX_Prefill_8k/README.md) "custom op 参考子图" 一节
