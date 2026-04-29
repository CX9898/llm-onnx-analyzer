# Qwen3.5-MoE-VL Merged ONNX 说明：`prefill`

这个目录保存的是 `Qwen3.5-35B-A3B`（多模态完整版）一套面向 `8k prefill` 的代表层 merged ONNX 子图，以及配套的 vision tower / 多模态注入子图与统计文件。

## 完整数据流（按这个顺序看 onnx）

下图覆盖本目录所有 onnx 文件，箭头方向就是 forward 时数据流的方向；列在右侧的 `*.onnx` 就是每一步对应的产物文件名。本目录的 9 张 vision/multimodal 子图与文本侧子图首尾对接，**无任何"分析意义上的孤岛"**——所有源码侧带可学习权重 / 真实张量算子的步骤都已落到 ONNX。

```text
[Vision tower — 每张图/视频片段一次性计算]

  pixel_values_flat:[N,1536]bf16        (N = patch token 数；默认 N=1024 = 32×32 grid)
        │
        ▼  vision_patch_embed_1k.onnx
  patch_embeds:[N,1152]bf16
        │
        │   grid_thw:[1,3]i64 ──────────────────────────────────┐
        ▼  vision_pos_embed_interp_1k.onnx                       │
                  ◄── grid_thw                                   │
                  含 nn.Embedding(2304,1152) ≈2.65M 学习参数     │
  hidden_states:[N,1152]bf16                                     │
        │                                                        │
        │   ┌──── vision_rot_pos_emb_1k.onnx ◄── grid_thw ─────┘ │
        │   │      cos:[N,72]bf16 / sin:[N,72]bf16             │ │
        │   │   含 Qwen3_5MoeVisionRotaryEmbedding.inv_freq    │ │
        │   │                                                  │ │
        │   ├──── vision_cu_seqlens_1k.onnx ◄── grid_thw ──────┘ │
        │   │      cu_seqlens:[num_seg+1]i32                     │
        │   │                                                    │
        ▼   ▼                                                    │
  vision_block_00_repr_1k.onnx                                   │
        ◄── hidden_states, cos, sin, cu_seqlens                  │
        ×27 (block-0 作为 27 层 ViT 的代表层)                    │
  vision_features:[N,1152]bf16                                   │
        │                                                        │
        ▼  vision_patch_merger_1k.onnx                           │
  image_embeds:[N/4, 2048]bf16  ─────────────────────────────────┤
                                                                 │
[Text 主链 — 整段 8k 一次性灌入]                                 │
                                                                 │
  input_ids:[1,8192]i64                                          │
        │                                                        │
        ├──► image_mask_build_8k.onnx ◄── image_token_id:i64     │
        │      image_mask:[1,8192,2048]bool                      │
        │      (源码对齐参考；分析用，与 mm_inject 输入解耦)     │
        │                                                        │
        ▼  embedding_8k.onnx                                     │
  inputs_embeds:[1,8192,2048]bf16                                │
        │                                                        │
        │   image_position_indices:[256,2]i64                    │
        │     (host: input_ids==image_token_id 的 nonzero(),     │
        │      与 image_mask 数学等价)                           │
        │                                                        ▼
        ▼  mm_inject_8k.onnx ◄── image_position_indices, image_embeds
  hidden_states:[1,8192,2048]bf16
        │
        │   mm_token_type_ids:[1,8192]i32, image_grid_thw:[1,3]i64
        │           │
        │           ▼  mrope_position_ids_prefill_8k.onnx
        │      position_ids:[3,1,8192]i64       (喂入 layer_03_full_attn_block)
        │      mrope_position_deltas:[1,1]i64   (host 缓存，供 decode 重用)
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
- 本目录最值得关注的图分三块：
  - **vision tower 内部**（27 层 ViT 一次性出 image_embeds）：
    - `vision_patch_embed_1k.onnx`
    - `vision_pos_embed_interp_1k.onnx`（**含 nn.Embedding 2304×1152 ≈ 2.65 M 学习参数**）
    - `vision_rot_pos_emb_1k.onnx`
    - `vision_cu_seqlens_1k.onnx`
    - `vision_block_00_repr_1k.onnx`（cu_seqlens-aware attention，源码侧 `tensor_split → 逐段 SDPA → cat` 拓扑全部保留）
    - `vision_patch_merger_1k.onnx`
  - **多模态 flow 边界**（介于 vision tower 与 text decoder 之间的源码 bookkeeping）：
    - `image_mask_build_8k.onnx`
    - `mm_inject_8k.onnx`
    - `mrope_position_ids_prefill_8k.onnx`
  - **文本主链**（与 `Qwen3_5_35B_A3B_ONNX_Prefill_8k` 完全等价的代表层四张主图）：
    - `layer_00_linear_attn_block_8k.onnx`
    - `layer_00_moe_block_8k.onnx`
    - `layer_03_full_attn_block_8k.onnx`
    - `layer_03_moe_block_8k.onnx`
- 顶层加载类是 `Qwen3_5MoeForConditionalGeneration`；text 侧子图通过 `model.model.language_model` 取 backbone，与纯文本路径 `Qwen3_5MoeForCausalLM.model` 同构，只是路径被多包了一层。
- `embedding_8k.onnx`、`norm_8k.onnx`、`lm_head_8k.onnx` 主要补齐完整首尾链路，不是这个目录最核心的差异点。
- prefill linear attention 的 `ChunkGatedDeltaRule` / `DeltaNetChunkStep` 参考子图与纯文本目录完全一致，本目录不重复说明。
- **vision tower / `image_mask_build` / `mm_inject` / `mrope_position_ids_prefill` 仅在本目录（prefill）导出**：图像 token 在 prefill 阶段一次性写进 `inputs_embeds` 并经 KV-cache 落到缓存里。Decode 阶段不会再触发它们，但 M-RoPE 的 3D `position_ids` **每步都要重算**，所以 [`Qwen3_5_35B_A3B_VL_ONNX_Decode_8k/`](../Qwen3_5_35B_A3B_VL_ONNX_Decode_8k/README.md) 在代表层 text 主链 4 张图之外，保留 1 张 `mrope_position_ids_decode_ctx<N>.onnx`。

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

### `vision_pos_embed_interp_1k.onnx`

- 作用：基于学习参数 `pos_embed: nn.Embedding(num_position_embeddings=2304, hidden_size=1152)` 的双线性插值，并把结果残差加到 patch_embed 输出上
- 输入：`hidden_states_pre:bf16:[1024, 1152]`
- 输出：`hidden_states_post:bf16:[1024, 1152]`
- 说明：
  - 严格对应 `Qwen3_5MoeVisionModel.forward` 行 1241–1242 与 `fast_pos_embed_interpolate` 行 1163；表层 4 个 `Gather + Mul + Sum`（双线性 4 个角的查表加权求和）后接 spatial-merge 重排（`view + permute + flatten`）
  - **本张图是补完原本"留在 host CPU 计算"的关键缺口**——`pos_embed.weight` 共 `2304 × 1152 ≈ 2.65 M` 学习参数会落到 ONNX initializer 里，此前完全不可见
  - 代表场景：`grid_thw = (1, 32, 32)`，由 `--vision_token_seq_len` 决定（同时驱动其它 vision 子图）

### `vision_rot_pos_emb_1k.onnx`

- 作用：从 `grid_thw` 构造 vision tower 的 2D（H/W）旋转位置编码 `cos / sin` 表
- 输入：`grid_thw:i64:[num_images, 3]`（代表场景 `[[1, 32, 32]]`）
- 输出：
  - `cos:bf16:[1024, 72]`
  - `sin:bf16:[1024, 72]`
- 说明：
  - 严格对应 `Qwen3_5MoeVisionModel.rot_pos_emb` 行 1123–1161 + `forward` 行 1244–1250 的 `cat → cos/sin`
  - `Qwen3_5MoeVisionRotaryEmbedding.inv_freq` 作为 `register_buffer(persistent=False)` 由 PyTorch tracer 内联为 Constant（与源码语义一致）；图内可见 `Range / Outer (Einsum) / Mul / Add / Expand / Reshape / Concat / Gather / Cat / Cos / Sin / Cast` 全套算子
  - `H`/`W` 取自 `grid_thw[0,1]/[0,2]` 的 tensor scalar，`Range` 等真实消费 `grid_thw`，输入端不会被 tracer 折叠成纯常量

### `vision_cu_seqlens_1k.onnx`

- 作用：从 `grid_thw` 构造可变长度打包的前缀和张量 `cu_seqlens`
- 输入：`grid_thw:i64:[num_images, 3]`
- 输出：`cu_seqlens:i32:[num_images + 1]`
- 说明：
  - 严格对应 `Qwen3_5MoeVisionModel.forward` 行 1252–1260：`repeat_interleave(grid_thw[:,1]*grid_thw[:,2], grid_thw[:,0]).cumsum(0, int32) → pad((1,0))`
  - 输出 dtype 与源码非-jit 分支一致（`int32`，Flash-Attention 要求）
  - 多图/多视频时 `cu_seqlens` 形状随 `grid_thw` 动态变化，本图同样适用

### `vision_block_00_repr_1k.onnx`

- 作用：单个 ViT block 的代表样本 `norm1 → attn → +residual → norm2 → mlp → +residual`
- 输入：
  - `hidden_states:bf16:[1024, 1152]`
  - `cos:bf16:[1024, 72]`（head_dim = 1152 / 16 = 72）
  - `sin:bf16:[1024, 72]`
  - `cu_seqlens:i32:[num_segments + 1]`
- 输出：`hidden_states_out:bf16:[1024, 1152]`
- 说明：
  - 对齐 `Qwen3_5MoeVisionBlock.forward`，attention 走**源码同构的 cu_seqlens 路径**：`tensor_split(q/k/v, cu_seqlens[1:-1], dim=2) → 逐段 SDPA → cat`，与 `Qwen3_5MoeVisionAttention` eager 分支（行 1026–1047）按位等价
  - 代表场景下默认喂 `cu_seqlens=[0, 1024]`（单 segment），`tensor_split` 退化为身份；分析图时仍能看到 `Sub / Slice / Cast / Split / Concat` 节点；多 segment 时无需重导，`cu_seqlens` 自然驱动 split/concat 数
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

### `image_mask_build_8k.onnx`

- 作用：构造 image-token 位置布尔掩码（源码侧 `Qwen3_5MoeModel.get_placeholder_mask` 的 `input_ids` 分支）
- 输入：
  - `input_ids:i64:[1, 8192]`
  - `image_token_id:i64:scalar`
- 输出：`image_mask:bool:[1, 8192, 2048]`
- 说明：
  - 严格对应源码行 1666–1671：`Equal(input_ids, image_token_id) → Unsqueeze(-1) → Expand(-1,-1,H_text)`
  - `image_token_id` 作为 graph input（不烘焙到常量），同一份图可跨 vocab 复用
  - 注：`image_mask` 与 `mm_inject_8k.onnx` 的输入 `image_position_indices` 是**数学等价**的两种表示——前者是源码原生形态，后者是 ONNX 友好形态。本图作为源码对齐参考保留，便于核对 `get_placeholder_mask` 的语义；推理引擎实际驱动 `mm_inject` 时通常直接用 `(input_ids == image_token_id).nonzero()` 算 indices

### `mm_inject_8k.onnx`

- 作用：把 `image_embeds` 写回 `inputs_embeds` 中 image-token 占位行
- 输入：
  - `inputs_embeds:bf16:[1, 8192, 2048]`
  - `image_position_indices:i64:[256, 2]`：每行是 `(batch_idx, seq_idx)`，由 caller 在 ONNX 之外计算（与 `image_mask_build_8k.onnx` 输出的布尔位置一一对应）
  - `image_embeds:bf16:[256, 2048]`
- 输出：`inputs_embeds_out:bf16:[1, 8192, 2048]`
- 说明：
  - 与源码 `Qwen3_5MoeModel.forward` 行 1773 `inputs_embeds.masked_scatter(image_mask, image_embeds)` **数学等价**
  - 等价证明：源码侧 `image_mask` 是按行扩展的（`(input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)`），每个 image token 在 `[B, S, H_text]` 中占满整个 `[H_text]` 一整行，所以 `masked_scatter` 等同于行级 `ScatterND(inputs_embeds, indices=[N,2]_(b,s), updates=[N,H_text])`
  - **为什么不直接用 `masked_scatter`**：PyTorch 的 `masked_scatter` 在 ONNX 里会被降级成 `NonZero → Transpose → ScatterND`，`NonZero` 输出的 second axis 是运行时布尔张量里 True 的个数，是 data-dependent shape，ONNX 静态 shape inference 会标成 `unk__1`，破坏 MAC/内存的静态分析。改用 indices 输入后 `N` 在 export 时就是已知静态值（`--mm_image_token_count` 默认 256），全图静态
  - 简化后拓扑（6 节点）：`Gather × 2（取 b/s 列）→ Unsqueeze × 2 → Concat（拼成 [N,2]）→ ScatterND(inputs_embeds, indices, image_embeds)`

### `mrope_position_ids_prefill_8k.onnx`

- 作用：构造 prefill 阶段 M-RoPE 用的 3D `position_ids: [3, B, S]` 与 `mrope_position_deltas: [B, 1]`
- 输入：
  - `input_ids:i64:[1, 8192]`
  - `mm_token_type_ids:i32:[1, 8192]`
  - `image_grid_thw:i64:[num_images, 3]`
- 输出：
  - `position_ids:i64:[3, 1, 8192]`
  - `mrope_position_deltas:i64:[1, 1]`
- 说明：
  - 严格对应 `Qwen3_5MoeModel.compute_3d_position_ids` 行 1707 `if` 分支 → `get_rope_index` 行 1511 → `get_vision_position_ids` 行 1455
  - 源码使用 `itertools.groupby(mm_token_type_ids.tolist())` 做段切分，非 ONNX 友好；本子图按代表场景**静态展开** `[text_pre | image | text_post]` 的 segment layout（`text_pre` 长度由 `--mrope_text_pre_len` 控制，默认 64；`image_token_count` 由 `--mm_image_token_count` 控制，默认 256；后缀 text 自动补齐到 `seq_len`）
  - 真实场景含多图 / 多 segment 时按相同公式重导一份
  - `mrope_position_deltas` 由 host 缓存，供 decode 阶段的 `mrope_position_ids_decode_ctx<N>.onnx` 重用

## 文本主链主图

### `layer 0` 样本路径

`embedding_8k → mm_inject_8k → layer_00_linear_attn_block_8k → layer_00_moe_block_8k`

### `layer 3` 样本路径

`layer_03_full_attn_block_8k → layer_03_moe_block_8k`

> `mm_inject` 在数据流上紧跟 embedding，把 image token 占位换成真正的 visual feature，**之后整段 text decoder 的输入** `hidden_states:[1, 8192, 2048]` **与纯文本路径完全一致**。这是当前文本侧子图无需任何改动就能直接接入多模态的关键。

四张代表层 text 主图的语义、shape、dtype 与 [`Qwen3_5_35B_A3B_ONNX_Prefill_8k/README.md`](../ori/Qwen3_5_35B_A3B_ONNX_Prefill_8k/README.md) 完全相同（同一份导出代码、同一份 backbone 权重，仅经过 transformers 的 key 透明 remap），不在本目录重复说明。

## 你真正需要关心的多模态差异

- 多模态情况下 `position_ids` 的语义从 1D `[B, S]` 变为 3D `[3, B, S]`（M-RoPE 的 `T/H/W` 三轴），`Qwen3_5MoeTextRotaryEmbedding`（已封装为 `RotaryEmbeddingBlockMoE`）会按 `mrope_section=[11,11,10]` 做 interleave。
- **文本侧 `layer_03_full_attn_block_8k.onnx` 在本目录（`vl` 变体）下导出时使用的 dummy `position_ids` 形状为 `[3, B, S]` 而不是 `[B, S]`**：PyTorch tracer 据此把 `RotaryEmbeddingBlockMoE.forward` 的 `position_ids.ndim==3` 分支固化进图，IO 契约直接接收 `mrope_position_ids_prefill_8k.onnx` 的输出 `position_ids:[3, 1, 8192]i64`，**无需任何 host 端 reshape**。
  - 纯文本目录 `Qwen3_5_35B_A3B_ONNX_Prefill_8k/` 仍保留 1D `[B, S]` 契约（同份代码用 `--variant base` 时 `position_ids_ndim=2`）；当 T=H=W 时 3D 路径与 2D 路径数值完全等价（已通过 unit-level smoke 验证），二选一不影响纯文本结果。
  - 实际改动：`qwen_merged_block_export.export_full_attn_block` 新增 `position_ids_ndim` 参数，由 `export_qwen_onnx_main.py` 在 `args.variant == "vl"` 时设为 3。
- M-RoPE 的 3D `position_ids` **本目录已有专门的子图**：prefill 用 `mrope_position_ids_prefill_8k.onnx`，decode 用 `mrope_position_ids_decode_ctx<N>.onnx`（在 [`Qwen3_5_35B_A3B_VL_ONNX_Decode_8k/`](../Qwen3_5_35B_A3B_VL_ONNX_Decode_8k/README.md) 目录）。
- vision tower 一侧的 `cos/sin`（2D 旋转）与基于 `pos_embed` 表的 `fast_pos_embed_interpolate` **本目录也已经有对应子图**：`vision_rot_pos_emb_1k.onnx` 与 `vision_pos_embed_interp_1k.onnx`，源码 forward 拓扑完整可见，`pos_embed.weight` 作为 initializer 计入参数总量。

## 端到端连通性自检

按上面"完整数据流"图自上而下逐条核对每一处生产者→消费者的 shape/dtype 契约，本目录现状：

| 生产者 | 张量 | shape / dtype | 消费者 | 状态 |
| --- | --- | --- | --- | --- |
| host | `pixel_values_flat` | `bf16[1024,1536]` | `vision_patch_embed_1k` | ✓ |
| `vision_patch_embed_1k` | `patch_embeds` | `bf16[1024,1152]` | `vision_pos_embed_interp_1k.hidden_states_pre` | ✓ |
| `vision_pos_embed_interp_1k` | `hidden_states_post` | `bf16[1024,1152]` | `vision_block_00_repr_1k.hidden_states` | ✓ |
| host | `grid_thw` | `i64[1,3]` | `vision_rot_pos_emb_1k.grid_thw` | ✓ |
| host | `grid_thw` | `i64[1,3]` | `vision_cu_seqlens_1k.grid_thw` | ✓ |
| `vision_rot_pos_emb_1k` | `cos / sin` | `bf16[1024,72]` | `vision_block_00_repr_1k.cos / .sin` | ✓ |
| `vision_cu_seqlens_1k` | `cu_seqlens` | `i32[num_seg+1]` | `vision_block_00_repr_1k.cu_seqlens` | ✓ |
| `vision_block_00_repr_1k` | `hidden_states_out` | `bf16[1024,1152]` | `vision_patch_merger_1k.vision_features` | ✓ |
| `vision_patch_merger_1k` | `image_embeds` | `bf16[256,2048]` | `mm_inject_8k.image_embeds` | ✓ |
| host | `input_ids` | `i64[1,8192]` | `embedding_8k / image_mask_build_8k / mrope_position_ids_prefill_8k` | ✓ |
| `embedding_8k` | `hidden_states` (重命名 `inputs_embeds`) | `bf16[1,8192,2048]` | `mm_inject_8k.inputs_embeds` | ✓ |
| host | `image_position_indices` | `i64[256,2]` | `mm_inject_8k.image_position_indices` | ⚠ 见下 |
| `mm_inject_8k` | `inputs_embeds_out` | `bf16[1,8192,2048]` | `layer_00_linear_attn_block_8k.hidden_states.3` | ✓ |
| `mrope_position_ids_prefill_8k` | `position_ids` | `i64[3,1,8192]` | `layer_03_full_attn_block_8k.position_ids` | ✓（本次刚修） |
| `mrope_position_ids_prefill_8k` | `mrope_position_deltas` | `i64[1,1]` | host 缓存 → decode 阶段 `rope_deltas` | ✓ |

**唯一需要 host 桥接的一步**是 `image_mask_build_8k → mm_inject_8k`：源码 `masked_scatter` 走 mask 形态，但 ONNX 静态分析需要静态 `N`，所以 `mm_inject_8k` 改用 `image_position_indices: i64[256,2]`，由 host 在 ONNX 之外做一次 `(input_ids == image_token_id).nonzero()` 算出。`image_mask_build_8k.onnx` 仍单独导出作为**源码对齐参考图**——两者数学等价，`(b, s)` 集合 ↔ 行级布尔位置一一对应（参见 `mm_inject_8k.onnx` 等价证明）。这是工程上为换取静态 shape 必须接受的 host 切口，不属于"分析意义上的孤岛"。

## 目录里的辅助文件

- `onnx_stats.json`：每个导出文件的基础统计
- linear attention 的 `ChunkGatedDeltaRule` / `DeltaNetChunkStep` 参考子图与纯文本目录完全一致，按需查阅 [`Qwen3_5_35B_A3B_ONNX_Prefill_8k/README.md`](../ori/Qwen3_5_35B_A3B_ONNX_Prefill_8k/README.md) "custom op 参考子图" 一节
