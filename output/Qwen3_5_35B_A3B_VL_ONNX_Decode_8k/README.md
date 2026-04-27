# Qwen3.5-MoE-VL Merged ONNX 说明：`decode`

这个目录保存的是 `Qwen3.5-35B-A3B`（多模态完整版）一套面向 `8k decode` 的代表层 merged ONNX 子图与统计文件。

## 完整数据流（按这个顺序看 onnx）

下图覆盖本目录所有主链 onnx 文件，箭头方向就是 decode 单步 forward 时数据流的方向；列在右侧的 `*.onnx` 就是每一步对应的产物文件名。

```text
[Decode — 每步只前进 1 个新 token，历史信息全部走 KV-cache / recurrent state]

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

> 注：vision tower 与 `mm_inject` **不在本目录**。多模态输入的视觉特征已在 prefill 阶段经 `mm_inject_8k.onnx` 一次性写进 `inputs_embeds` 并通过 KV-cache 落到缓存里，decode 阶段不会再触发任何 vision/mm 图。完整的 vision 数据流请查阅 [`Qwen3_5_35B_A3B_VL_ONNX_Prefill_8k/README.md`](../Qwen3_5_35B_A3B_VL_ONNX_Prefill_8k/README.md)。

## 先看结论

- `decode` 的意思是：每一步只处理 `1` 个新 token，但携带历史状态继续递推；多模态输入（图像 / 视频）的 visual feature 已在 prefill 阶段一次性计算并通过 KV-cache 隐式带入。
- **decode 阶段不导出 vision tower 与 mm_inject 子图**：图像 token 早在 prefill 阶段就被 `masked_scatter` 写入了 `inputs_embeds`，并通过 KV-cache 把对应位置的 K/V 落在缓存里；decode 步是逐 token 自回归，不会再触发 `vision_patch_embed` / `vision_block` / `vision_patch_merger` / `mm_inject` 中的任何一个图。完整的 vision 数据流请查阅 prefill 目录 [`Qwen3_5_35B_A3B_VL_ONNX_Prefill_8k/README.md`](../Qwen3_5_35B_A3B_VL_ONNX_Prefill_8k/README.md)。
- 因此本目录最值得关注的就是 **代表层 text 主链 4 张图**（与 `Qwen3_5_35B_A3B_ONNX_Decode_8k` 在算子和权重层面完全等价）：
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
- 与文本目录唯一的差异是：text 主链里的 `position_ids` 在 vl 推理中是 3D `[3, B, 1]`（M-RoPE 的 `T/H/W` 三轴），由 `Qwen3_5MoeModel.get_rope_index` 在 CPU 端构造、每步自增；现有 `RotaryEmbeddingBlockMoE` 已支持 3D `position_ids`，相关 full-attention 子图导出时同步打开了 `mrope_interleaved`，无需重新导出。

## 文本主链主图

### `layer 0` 样本路径

`embedding_1 → layer_00_linear_attn_block → layer_00_moe_block_1`

### `layer 3` 样本路径

`layer_03_full_attn_block_decode_ctx8k → layer_03_moe_block_1`

四张代表层 text 主图的语义、shape、dtype 与 [`Qwen3_5_35B_A3B_ONNX_Decode_8k/README.md`](../ori/Qwen3_5_35B_A3B_ONNX_Decode_8k/README.md) 完全相同（同一份导出代码、同一份 backbone 权重，仅经过 transformers 的 key 透明 remap），不在本目录重复说明。

## 你真正需要关心的多模态差异

- 多模态情况下 `position_ids` 的语义从 1D `[B, 1]` 变为 3D `[3, B, 1]`（M-RoPE 的 `T/H/W` 三轴）。
- 文本侧的 `layer_03_full_attn_block_decode_ctx8k.onnx` 已对 M-RoPE 做兼容（`mrope_interleaved=true` 来自 config，导出时已生效），不需要重新导出。
- M-RoPE 的 3D `position_ids` 由 `Qwen3_5MoeModel.get_rope_index` 计算（不进 ONNX），由推理引擎在 CPU 端完成，每步 decode 自增。
- vision tower / mm_inject 一侧在 decode 阶段不会触发，相关 ONNX 子图全部归档到 prefill 目录，避免与 prefill 重复冗余。

## 目录里的辅助文件

- `onnx_stats.json`：每个导出文件的基础统计
- linear attention 的 `RecurrentGatedDeltaRule` 参考子图与纯文本目录完全一致，按需查阅 [`Qwen3_5_35B_A3B_ONNX_Decode_8k/README.md`](../ori/Qwen3_5_35B_A3B_ONNX_Decode_8k/README.md) "custom op 参考子图" 一节
