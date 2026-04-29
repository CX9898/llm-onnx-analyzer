# Qwen3.5-35B-A3B VL — Decode 8K KV ONNX 图清单与阅读顺序

本目录是 `Qwen3.5-35B-A3B-VL` **decode 阶段（单 token 自回归）** 的代表性 ONNX 导出。
源码对应：`transformers/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`。

- 单步 token：`seq_len = 1`
- KV 历史长度：`decode_context_len = 8192`
- 文件总数：**9**；总 `unk__N`：**0**（decode 阶段无视觉/`masked_scatter` 路径，无源码本征数据依赖）

---

## 1. 阅读顺序（按源码 `forward` pass 自上而下）

decode 比 prefill 简单——视觉塔已在 prefill 时跑完并把 image embeddings 写入 KV，每一步 decode 只需要：
1. embed 当前新 token（1 个）
2. 重新构造 M-RoPE 3D `position_ids`（因为 KV 历史在每步都增长）
3. 走文本解码层（吃 `past_key/past_value`，吐 `new_key/new_value`）
4. 输出 logits

```
input_ids[1, 1] ──→ embedding_1 ──→ hidden_states[1, 1, 2048] ─┐
                                                                  │
attention_mask, rope_deltas ──→ mrope_position_ids_decode_ctx8k   │
                              ──→ position_ids[3, 1, 1] ───┐      │
                                                            │      │
            ┌───────────────────────────────────────────────┴──────┴────┐
            │ Stage: 文本解码层（代表 linear@layer 0、full@layer 3）       │
            │   layer_00_linear_attn_block          ←─ 代表 linear-attn   │
            │     └─ 内部子图：layer_00_..._RecurrentGatedDeltaRule        │
            │   layer_00_moe_block_1                                      │
            │   ... 真实 47 + 1 层 ...                                     │
            │   layer_03_full_attn_block_decode_ctx8k    ←─ 代表 full-attn │
            │   layer_03_moe_block_1                                       │
            └─────────────────────────────────────────────────────────────┘
                                       ↓
                               norm_1 → lm_head_1 → logits[1, 1, 248320]
```

---

## 2. 完整文件清单（按读法顺序）

| # | 文件 | 算子分组 | 输入 | 输出 | 节点 | unk |
|---|---|---|---|---|---|---|
| **Stage 1：embedding 与 M-RoPE** ||||||||
| 1 | `embedding_1.onnx` | `get_input_embeddings()(input_ids)`（line 1762） | `input_ids: i64[1, 1]`、`embedding_weight: bf16[248320, 2048]` | `hidden_states: bf16[1, 1, 2048]` | 1 | 0 |
| 2 | `mrope_position_ids_decode_ctx8k.onnx` | `compute_3d_position_ids` elif 分支（line 1720-1726） | `attention_mask: i64[1, 8193]`、`rope_deltas: i64[1, 1]` | `position_ids: i64[3, 1, 8193]` | 11 | 0 |
| **Stage 2：解码层（代表性）** ||||||||
| 3 | `layer_00_linear_attn_block.onnx` | linear-attn 层包装（含 q/k/v/g/beta + recurrent core） | `hidden_states.1: bf16[1, 1, 2048]`、`conv_state`、`recurrent_state`、`padding_mask` | `hidden_states: bf16[1, 1, 2048]`、`new_conv_state`、`new_recurrent_state` | 93 | 0 |
| 4 | `layer_00_linear_attn_block_RecurrentGatedDeltaRule.onnx` | linear-attn 主体（recurrent step） | q/k/v/g/beta + recurrent_state | `core_out`、`new_recurrent_state` | 47 | 0 |
| 5 | `layer_00_moe_block_1.onnx` | layer 0 配对 MoE | hidden + experts + shared | `hidden_states: bf16[1, 1, 2048]` | 50 | 0 |
| 6 | `layer_03_full_attn_block_decode_ctx8k.onnx` | full-attn 层（M-RoPE 3D + KV-cache） | `hidden_states.1`、**`position_ids: i64[3, 1, 1]`**、`attention_mask: bf16[1, 1, 1, 8193]`、`past_key/past_value: bf16[1, 2, 8192, 256]` | `hidden_states`、`new_key/new_value: bf16[1, 2, 8193, 256]` | 136 | 0 |
| 7 | `layer_03_moe_block_1.onnx` | layer 3 配对 MoE | 同 #5 | `hidden_states` | 50 | 0 |
| **Stage 3：输出头** ||||||||
| 8 | `norm_1.onnx` | `Qwen3_5MoeRMSNorm` | `hidden_states: bf16[1, 1, 2048]` | `output: bf16[1, 1, 2048]` | 11 | 0 |
| 9 | `lm_head_1.onnx` | `lm_head` 线性投影 | `hidden_states`、`lm_head_weight: bf16[248320, 2048]` | `logits: bf16[1, 1, 248320]` | 2 | 0 |

---

## 3. 数据流连接边（生产者 → 消费者）

| 张量 | dtype × shape | 生产者 | 消费者 |
|---|---|---|---|
| `hidden_states` | bf16 [1, 1, 2048] | `embedding_1` | 第一个 layer block（`hidden_states.1`） |
| `position_ids` | **i64 [3, 1, 1]**（M-RoPE 3D） | `mrope_position_ids_decode_ctx8k`（取 `[:, :, -1:]`） | 所有 `layer_*_full_attn_block_decode_ctx8k` |
| `hidden_states`（每层输出） | bf16 [1, 1, 2048] | layer N attn / moe | layer N+1 |
| 末层 `hidden_states` | bf16 [1, 1, 2048] | 最后一个 moe block | `norm_1` |
| `output` | bf16 [1, 1, 2048] | `norm_1` | `lm_head_1` |
| `new_key/new_value` | bf16 [1, 2, 8193, 256] | `layer_*_full_attn_block_decode_ctx8k` | 下一步的 `past_key/past_value` |
| `new_recurrent_state` | f32 [1, 32, 128, 128] | `layer_*_linear_attn_block` | 下一步的 `recurrent_state` |

注意：

- `mrope_position_ids_decode_ctx8k` 输出是 **8193 长度**的 3D positions（覆盖 8192 历史 + 1 新 token），但每一步只把**最后一列** `[:, :, -1:]` 传给 full-attn block（其 input shape 是 `[3, 1, 1]`）
- `attention_mask: bf16[1, 1, 1, 8193]` 给 full-attn block，对应 KV 长 8192 + 当前 1 个新 token
- `past_key/past_value`：上一步的 KV 缓存（长度 8192）；`new_key/new_value`：拼接后的 KV 缓存（长度 8193）

---

## 4. `unk__N` 状况

decode 阶段所有 9 张图 **0 个 `unk__N`**。原因：

- 没有 `vision_cu_seqlens` 路径（视觉塔只在 prefill 跑）
- 没有 `mm_inject` / `masked_scatter` 路径（image embeddings 已在 prefill 写入 KV）
- 解码层全部静态形状（`seq_len=1` 已知、KV 长度 = `decode_context_len + 1` 也已知）

---

## 5. 运行时如何把这些 ONNX 串起来

伪代码（一步 decode）：

```python
import onnxruntime as ort

# 输入：单 token、上一步的 KV / state、累计 attention_mask、rope_deltas
# Stage 1: embedding + M-RoPE
hidden = run("embedding_1.onnx",
             input_ids=new_token_id,            # i64 [1, 1]
             embedding_weight=W_embed)["hidden_states"]

position_ids_full = run("mrope_position_ids_decode_ctx8k.onnx",
                        attention_mask=cumul_attention_mask,    # i64 [1, 8193]
                        rope_deltas=rope_deltas)["position_ids"]  # i64 [3, 1, 8193]
position_ids_step = position_ids_full[:, :, -1:]                  # i64 [3, 1, 1]

# Stage 2: 文本解码层
for layer_idx, layer_kind in enumerate(real_layer_layout):
    if layer_kind == "linear":
        out = run("layer_00_linear_attn_block.onnx",
                  **{"hidden_states.1": hidden,
                     "conv_state": conv_state[layer_idx],
                     "recurrent_state": recurrent_state[layer_idx],
                     "padding_mask": padding_mask})
        conv_state[layer_idx]      = out["new_conv_state"]
        recurrent_state[layer_idx] = out["new_recurrent_state"]
        hidden = run("layer_00_moe_block_1.onnx",
                     **{"hidden_states.1": out["hidden_states"], "experts_gate_up": ..., ...})
    elif layer_kind == "full":
        out = run("layer_03_full_attn_block_decode_ctx8k.onnx",
                  **{"hidden_states.1": hidden,
                     "position_ids": position_ids_step,
                     "attention_mask": cumul_attn_mask_4d,
                     "past_key": kv_cache_k[layer_idx],
                     "past_value": kv_cache_v[layer_idx]})
        kv_cache_k[layer_idx] = out["new_key"]
        kv_cache_v[layer_idx] = out["new_value"]
        hidden = run("layer_03_moe_block_1.onnx",
                     **{"hidden_states.1": out["hidden_states"], ...})

# Stage 3: 输出
hidden = run("norm_1.onnx", hidden_states=hidden)["output"]
logits = run("lm_head_1.onnx",
             hidden_states=hidden, lm_head_weight=W_lmhead)["logits"]
# logits: bf16 [1, 1, 248320]

# 采样得到下一个 token，更新 cumul_attention_mask（拼一个 1）、rope_deltas，循环
```

---

## 6. 与 prefill 的关系

| 阶段 | 文件 | vl-only | 文本侧 |
|---|---|---|---|
| prefill | 9 个 vision/multimodal 文件 + 9 个文本文件 | ✓ | ✓ |
| decode | **0** 个 vision/multimodal 文件 + 9 个文本文件 | — | ✓ |

decode 不再涉及视觉塔与 `mm_inject`，因为：
1. 视觉特征在 prefill 跑过一次，结果已经混入 `inputs_embeds_out` 并写入了 KV-cache
2. M-RoPE 3D `position_ids` 在每一步重算（看 `mrope_position_ids_decode_ctx8k.onnx`）

decode 与 prefill **共享文本权重**——所有 `layer_*` / `embedding_weight` / `lm_head_weight` 是同一份；只是为了支持单 token 推理，文件后缀从 `_8k.onnx` 变为 `_1.onnx` 并且 full-attn 文件名加了 `_decode_ctx8k` 区分前缀和长度。
