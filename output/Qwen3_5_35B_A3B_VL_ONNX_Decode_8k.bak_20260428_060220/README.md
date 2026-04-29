# Qwen3.5-MoE-VL Merged ONNX 说明：`decode`

这个目录保存的是 `Qwen3.5-35B-A3B`（多模态完整版）一套面向 `8k decode` 的代表层 merged ONNX 子图与统计文件。

## 完整数据流（按这个顺序看 onnx）

下图覆盖本目录所有主链 onnx 文件，箭头方向就是 decode 单步 forward 时数据流的方向；列在右侧的 `*.onnx` 就是每一步对应的产物文件名。

```text
[Decode — 每步只前进 1 个新 token，历史信息全部走 KV-cache / recurrent state]

  attention_mask:[1, ctx+1]i64,  rope_deltas:[1,1]i64  (host 缓存自 prefill)
        │                              │
        ▼  mrope_position_ids_decode_ctx8k.onnx
  position_ids:[3, 1, ctx+1]i64       (供 layer_03_full_attn_block 使用)
        │
        ▼ (host 取最后一列作为 decode 当前 step 的位置)

  input_ids:[1,1]i64
        │
        ▼  embedding_1.onnx
  inputs_embeds:[1,1,2048]bf16
        │
        │   ×40 层 decoder stack (layer_idx % 4 == 3 → full attention, 否则 linear attention)
        │   本目录只导出两类代表层:
        │
        │     layer 0 (linear attn + MoE):
        │       ─► layer_00_linear_attn_block.onnx
        │            states (in/out): conv_state:[1,8192,4]bf16,
        │                             recurrent_state:[1,32,128,128]float32
        │       ─► layer_00_moe_block_1.onnx
        │
        │     layer 3 (full attn + MoE):
        │       ─► layer_03_full_attn_block_decode_ctx8k.onnx
        │            cache (in/out): past_key/past_value:[1,2,8192,256]bf16
        │            attention_mask:[1,1,1,8193]bf16
        │            position_ids:[3,1,1]i64  (来自 mrope_position_ids_decode_ctx8k.onnx)
        │       ─► layer_03_moe_block_1.onnx
        │
        ▼  norm_1.onnx
  hidden_states:[1,1,2048]bf16
        │
        ▼  lm_head_1.onnx
  logits:[1,1,248320]bf16
```

主链之外，本目录还导出一张 **custom op 展开参考子图**（不是独立调用步骤，只是 `layer_00_linear_attn_block.onnx` 顶层图里 `qwen_onnx::RecurrentGatedDeltaRule` 的内部展开，方便结构对照）：

- `layer_00_linear_attn_block_RecurrentGatedDeltaRule.onnx`

> 注：vision tower 与 `mm_inject` **不在本目录**。多模态输入的视觉特征已在 prefill 阶段经 `mm_inject_8k.onnx` 一次性写进 `inputs_embeds` 并通过 KV-cache 落到缓存里，decode 阶段不会再触发任何 vision tower / `mm_inject` / `image_mask_build` 子图。但 M-RoPE 的 3D `position_ids` 每步都要重算，因此本目录额外保留 `mrope_position_ids_decode_ctx8k.onnx`。完整的 vision 数据流请查阅 [`Qwen3_5_35B_A3B_VL_ONNX_Prefill_8k/README.md`](../Qwen3_5_35B_A3B_VL_ONNX_Prefill_8k/README.md)。

## 先看结论

- `decode` 的意思是：每一步只处理 `1` 个新 token，但携带历史状态继续递推；多模态输入（图像 / 视频）的 visual feature 已在 prefill 阶段一次性计算并通过 KV-cache 隐式带入。
- **decode 阶段不导出 vision tower / `image_mask_build` / `mm_inject` 子图**：图像 token 早在 prefill 阶段就被 `masked_scatter` 写入了 `inputs_embeds`，并通过 KV-cache 把对应位置的 K/V 落在缓存里；decode 步是逐 token 自回归，不会再触发它们。完整的 vision 数据流请查阅 prefill 目录 [`Qwen3_5_35B_A3B_VL_ONNX_Prefill_8k/README.md`](../Qwen3_5_35B_A3B_VL_ONNX_Prefill_8k/README.md)。
- **decode 阶段确实需要 1 张多模态子图**：`mrope_position_ids_decode_ctx8k.onnx`。M-RoPE 的 3D `position_ids` 每步都要从 `attention_mask` 与 `rope_deltas` 重新构造（对应 `Qwen3_5MoeModel.compute_3d_position_ids` 行 1720 的 `elif` 分支）。本图把这部分源码张量算子从 host 收回到 ONNX，与 prefill 目录的 `mrope_position_ids_prefill_8k.onnx` 配对，使整条 M-RoPE 数据流端到端可见。
- 此外本目录最值得关注的是 **代表层 text 主链 4 张图**（与 `Qwen3_5_35B_A3B_ONNX_Decode_8k` 在算子和权重层面完全等价）：
  - `layer_00_linear_attn_block.onnx`
  - `layer_00_moe_block_1.onnx`
  - `layer_03_full_attn_block_decode_ctx8k.onnx`
  - `layer_03_moe_block_1.onnx`
- 顶层加载类是 `Qwen3_5MoeForConditionalGeneration`；text 侧子图通过 `model.model.language_model` 取 backbone，与纯文本路径同构。

## 目录在整模型里代表什么

完整 forward 由四部分组成：

1. **Vision tower**（27 层 ViT）：跑在 prefill 阶段，结果作为 image_embeds 通过 `mm_inject` 注入文本序列；decode 阶段不再跑，相关 ONNX 仅在 prefill 目录导出。
2. **Text embedding**：单 token `[B, 1]` → `[B, 1, H]`。
3. **多模态注入**：发生在 prefill 阶段；decode 阶段不会再出现 image-token，因此**不导出** `mm_inject_*.onnx`。
4. **Text decoder stack（40 层）**：携带 `past_key/past_value`（full attention）或 `conv_state/recurrent_state`（linear attention）的单步递推，与纯文本路径完全一致。

代表层口径同纯文本目录：

- text 一侧仍是 `layer 0`（linear attention + MoE）+ `layer 3`（full attention + MoE）两条并列样本路径。

## `decode` 在这里具体是什么意思

- 文本主链激活张量 `seq_len=1`；历史信息通过 KV-cache / 递推状态保留。
- 与纯文本 decode 目录的两个差异：
  1. text 主链里的 `position_ids` 在 vl 推理中是 3D `[3, B, 1]`（M-RoPE 的 `T/H/W` 三轴），来自 `mrope_position_ids_decode_ctx8k.onnx` 的 `[3, B, ctx+1]` 输出取最后一列。**本目录（`vl` 变体）下 `layer_03_full_attn_block_decode_ctx8k.onnx` 导出 dummy `position_ids` 形状为 `[3, B, 1]`**：tracer 据此固化 `RotaryEmbeddingBlockMoE.forward` 的 `position_ids.ndim==3` 分支，IO 契约直接接受 3D 输入，无需 host 对 1D/3D 做额外 reshape；纯文本目录仍保留 1D 契约（同份代码用 `--variant base` 时 `position_ids_ndim=2`，T=H=W 时两条路径数值完全等价）。
  2. 多导一张 `mrope_position_ids_decode_ctx8k.onnx` 把 M-RoPE 索引重算搬进 ONNX。

## 多模态主图

### `mrope_position_ids_decode_ctx8k.onnx`

- 作用：构造每步 decode 用的 3D `position_ids: [3, B, ctx+1]`
- 输入：
  - `attention_mask:i64:[1, 8193]`
  - `rope_deltas:i64:[1, 1]`（host 缓存自 prefill 阶段的 `mrope_position_ids_prefill_8k.onnx` 输出）
- 输出：`position_ids:i64:[3, 1, 8193]`
- 说明：
  - 严格对应 `Qwen3_5MoeModel.compute_3d_position_ids` 行 1720 的 `elif` 分支（`attention_mask is not None` 子分支）：`Cumsum(attention_mask)-1 → MaskedFill → View(1,B,-1) → Repeat(3,1,1) → + rope_deltas`
  - decode 时 host 取 `position_ids[:, :, -1:]` 作为当前 step 的 3D `position_ids`，喂入 `layer_03_full_attn_block_decode_ctx8k.onnx`
  - 上下文长度由 `--decode_context_len` 控制（与 full-attention KV-cache 长度一致）

## 文本主链主图

### `layer 0` 样本路径

`embedding_1 → layer_00_linear_attn_block → layer_00_moe_block_1`

### `layer 3` 样本路径

`layer_03_full_attn_block_decode_ctx8k → layer_03_moe_block_1`

四张代表层 text 主图的语义、shape、dtype 与 [`Qwen3_5_35B_A3B_ONNX_Decode_8k/README.md`](../ori/Qwen3_5_35B_A3B_ONNX_Decode_8k/README.md) 完全相同（同一份导出代码、同一份 backbone 权重，仅经过 transformers 的 key 透明 remap），不在本目录重复说明。

## 你真正需要关心的多模态差异

- 多模态情况下 `position_ids` 的语义从 1D `[B, 1]` 变为 3D `[3, B, 1]`（M-RoPE 的 `T/H/W` 三轴），由本目录的 `mrope_position_ids_decode_ctx8k.onnx` 在 ONNX 内部完成构造，不再依赖 host CPU 隐式计算。
- **文本侧 `layer_03_full_attn_block_decode_ctx8k.onnx` 在本目录（`vl` 变体）下使用 3D `[3, B, 1]` 的 dummy `position_ids` 导出**，IO 契约直接接收 `mrope_position_ids_decode_ctx8k.onnx` 输出取最后一列后的形状；与 prefill 目录的 `layer_03_full_attn_block_8k.onnx` 改动同步（实现见 `qwen_merged_block_export.export_full_attn_block` 的 `position_ids_ndim` 参数）。
- vision tower / `image_mask_build` / `mm_inject` 一侧在 decode 阶段不会触发，相关 ONNX 子图全部归档到 prefill 目录，避免与 prefill 重复冗余。

## 端到端连通性自检（decode 单步）

| 生产者 | 张量 | shape / dtype | 消费者 | 状态 |
| --- | --- | --- | --- | --- |
| host（缓存自 prefill） | `rope_deltas` | `i64[1,1]` | `mrope_position_ids_decode_ctx8k.rope_deltas` | ✓ |
| host | `attention_mask` | `i64[1,8193]` | `mrope_position_ids_decode_ctx8k.attention_mask` | ✓ |
| `mrope_position_ids_decode_ctx8k` | `position_ids` | `i64[3,1,8193]` | host 取 `[:,:,-1:]` | ✓ |
| host | `position_ids[:,:,-1:]` | `i64[3,1,1]` | `layer_03_full_attn_block_decode_ctx8k.position_ids` | ✓（本次刚修） |
| host | `input_ids` | `i64[1,1]` | `embedding_1.input_ids` | ✓ |
| `embedding_1` | `hidden_states` | `bf16[1,1,2048]` | `layer_00_linear_attn_block.hidden_states.1` | ✓ |
| `layer_*_*_block` | `hidden_states` | `bf16[1,1,2048]` | 下一层 / `norm_1` | ✓ |
| `norm_1` | `output` | `bf16[1,1,2048]` | `lm_head_1.hidden_states` | ✓ |
| `lm_head_1` | `logits` | `bf16[1,1,248320]` | host 采样 | ✓ |

唯一的 host 切口是 `mrope_position_ids_decode_ctx8k → host 取最后一列 → full_attn_block_decode`，源码 `compute_3d_position_ids` 在 elif 分支里产出的 `[3, B, ctx+1]` 也是同样的语义（host 之后会按当前 token 数取最后一列），属于源码原生行为而非引入的工程切口。

## 目录里的辅助文件

- `onnx_stats.json`：每个导出文件的基础统计
- linear attention 的 `RecurrentGatedDeltaRule` 参考子图与纯文本目录完全一致，按需查阅 [`Qwen3_5_35B_A3B_ONNX_Decode_8k/README.md`](../ori/Qwen3_5_35B_A3B_ONNX_Decode_8k/README.md) "custom op 参考子图" 一节
