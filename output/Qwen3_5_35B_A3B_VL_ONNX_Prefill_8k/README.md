# Qwen3.5-35B-A3B VL — Prefill 8K ONNX 图清单（dynamic 模式）

本目录是 `Qwen3.5-35B-A3B-VL` **prefill 阶段**的代表性 ONNX 导出，用 `--vision_grid_mode dynamic`（默认）生成。
源码对应：`transformers/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`。

- 上下文长度：`seq_len = 8192`（文本侧）
- 视觉 token：`vision_token_seq_len = 1024`（32×32 grid，单图代表性场景）
- 注入 image token：`mm_image_token_count = 256`（= `1024 / spatial_merge_size² = 1024 / 4`）
- 文本前缀长度：`mrope_text_pre_len = 4096`
- 文件总数：**18**；总 `unk__N`：**145**
- 设计选择：**dynamic 模式**——3 个对 H/W 敏感的子图（`vision_rot_pos_emb` / `vision_pos_embed_interp` / `mrope_position_ids_prefill`）让 H/W 通过 `grid_thw` 张量真输入，源码每个算子（`Range / Mul / Cos / Sin / Tile / OneHot / Max / ...`）都在 ONNX 图里可见，代价是 145 个 `unk__N`。同一份 ONNX 可以喂不同分辨率（动态形状），quant/编译工具加载时 hint 一下 `grid_thw=[[T,H,W]]` 即可解析所有 unk。

> **如果你需要每个分辨率一份完全静态的 ONNX**（0 unk），用 `--vision_grid_mode static` 重导一遍——会输出到 `Qwen3_5_35B_A3B_VL_ONNX_Prefill_8k_static/`。详见末尾 §7 "两种模式的差异"。

---

## 1. 阅读顺序（按源码 `forward` pass 自上而下）

整体数据流分 **4 个 stage**——视觉塔、多模态簿记、解码层、输出头。每个 stage 内部再按算子先后展开。

```
┌─────────────────────────── Stage 1 ───────────────────────────┐
│   pixel_values (image) ──→ image_embeds [256, 2048]           │
│                                                                │
│   vision_patch_embed_1k    ──┐                                 │
│                              ├─→ vision_pos_embed_interp_1k    │
│   vision_rot_pos_emb_1k    ──┤      (cos, sin, hidden 流入下方) │
│   vision_cu_seqlens_1k     ──┘                                 │
│                                                                │
│             ↓  hidden_states [1024, 1152] bf16                 │
│   vision_block_00_repr_1k                                      │
│   (代表性 ViT block；真实模型有 27 层，逐层串联)                  │
│             ↓                                                  │
│   vision_patch_merger_1k                                       │
│             ↓                                                  │
│        image_embeds [256, 2048] bf16   ──────────────┐          │
└────────────────────────────────────────────────────────┼──────┘
                                                         │
┌─────────────────────────── Stage 2 ───────────────────┼──────┐
│   input_ids ──→ embedding_8k ──→ inputs_embeds        │       │
│   input_ids ──→ image_mask_build_8k ──→ image_mask    │       │
│                                          ↓            │       │
│   inputs_embeds ─────┐                                │       │
│   image_mask  ───────┼──→ mm_inject_8k  ←─────────────┘       │
│   image_embeds ──────┘         ↓                              │
│                          inputs_embeds_out [1, 8192, 2048]    │
│                                ↓                              │
│   input_ids, mm_token_type_ids, image_grid_thw                │
│                ──→ mrope_position_ids_prefill_8k              │
│                       → position_ids [3, 1, 8192] (M-RoPE 3D) │
│                       → mrope_position_deltas [1, 1]          │
└───────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────── Stage 3 ───────────────────────────┐
│   layer_00_linear_attn_block_8k        ←─ 代表 linear-attn 层  │
│      └─ 内部子图：                                             │
│         layer_00_linear_attn_block_ChunkGatedDeltaRule_chunk64_8k│
│             └─ 内部子图：                                       │
│                layer_00_linear_attn_block_ChunkGatedDeltaRule_  │
│                DeltaNetChunkStep_chunk64                        │
│   layer_00_moe_block_8k                ←─ 配对 MoE 块           │
│   ... ... 真实模型 47 层 linear + 1 层 full（layer 3）...        │
│   layer_03_full_attn_block_8k          ←─ 代表 full-attn 层     │
│   layer_03_moe_block_8k                ←─ 配对 MoE 块           │
└───────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────── Stage 4 ───────────────────────────┐
│   norm_8k    →  lm_head_8k  →  logits [1, 8192, 248320]        │
└───────────────────────────────────────────────────────────────┘
```

---

## 2. 完整文件清单（按读法顺序）

| # | 文件 | 算子分组 | 输入 | 输出 | 节点 | unk |
|---|---|---|---|---|---|---|
| **Stage 1：视觉塔（line 1191-1272）** ||||||||
| 1 | `vision_patch_embed_1k.onnx` | `Qwen3_5MoeVisionPatchEmbed.forward` | `pixel_values_flat: bf16[1024, 1536]` | `patch_embeds: bf16[1024, 1152]` | 3 | 0 |
| 2 | `vision_pos_embed_interp_1k.onnx` | `fast_pos_embed_interpolate`（line 1163-1224）+ residual add（line 1242）| `hidden_states_pre: bf16[1024, 1152]`、`grid_thw: i64[1, 3]` | `hidden_states_post: bf16[1024, 1152]` | **84** | **82** ★ |
| 3 | `vision_rot_pos_emb_1k.onnx` | `rot_pos_emb`（line 1123-1161）+ `cos/sin`（line 1244-1250）| `grid_thw: i64[1, 3]` | `cos: bf16[*, 72]`、`sin: bf16[*, 72]` | **36** | **35** ★ |
| 4 | `vision_cu_seqlens_1k.onnx` | line 1252-1260 源码原版 `repeat_interleave + cumsum + pad` | `grid_thw: i64[1, 3]` | `cu_seqlens: i64[Padcu_seqlens_dim_0]` | 12 | 4 |
| 5 | `vision_block_00_repr_1k.onnx` | 代表性 ViT block（27 层中第 0 层；line 1262-1268） | `hidden_states: bf16[1024, 1152]`、`cos/sin: bf16[1024, 72]`、`cu_seqlens: i64[2]` | `hidden_states_out: bf16[1024, 1152]` | 60 | 0 |
| 6 | `vision_patch_merger_1k.onnx` | `Qwen3_5MoeVisionPatchMerger.forward` | `vision_features: bf16[1024, 1152]` | `image_embeds: bf16[256, 2048]` | 5 | 0 |
| **Stage 2：多模态簿记（line 1758-1815）** ||||||||
| 7 | `embedding_8k.onnx` | `get_input_embeddings()(input_ids)`（line 1762） | `input_ids: i64[1, 8192]`、`embedding_weight: bf16[248320, 2048]` | `hidden_states: bf16[1, 8192, 2048]` | 1 | 0 |
| 8 | `image_mask_build_8k.onnx` | `get_placeholder_mask` 图像分支（line 1666-1671） | `input_ids: i64[1, 8192]`、`image_token_id: i64[]` | `image_mask: bool[1, 8192, 2048]` | 3 | 0 |
| 9 | `mm_inject_8k.onnx` | line 1773 源码原版 `inputs_embeds.masked_scatter` | `inputs_embeds: bf16[1, 8192, 2048]`、`image_mask: bool[1, 8192, 2048]`、`image_embeds: bf16[256, 2048]` | `inputs_embeds_out: bf16[1, 8192, 2048]` | 8 | 3 |
| 10 | `mrope_position_ids_prefill_8k.onnx` | `compute_3d_position_ids` if 分支 + `get_rope_index`（line 1511-1707） | `input_ids: i64[1, 8192]`、`mm_token_type_ids: i32[1, 8192]`、`image_grid_thw: i64[1, 3]` | `position_ids: i64[3, 1, *]`、`mrope_position_deltas: i64[1, 1]` | **47** | **21** ★ |
| **Stage 3：解码层（代表性，line 1336-1430 等）** ||||||||
| 11 | `layer_00_linear_attn_block_8k.onnx` | linear-attn 层包装（含 q/k/v/g/beta + chunk 主体 + 输出投影） | `hidden_states.3: bf16[1, 8192, 2048]`、`conv_state`、`recurrent_state`、`padding_mask` | `hidden_states: bf16[1, 8192, 2048]`、`new_conv_state`、`new_recurrent_state` | 93 | 0 |
| 12 | `layer_00_linear_attn_block_ChunkGatedDeltaRule_chunk64_8k.onnx` | linear-attn 主体（chunk=64） | q/k/v/g/beta + recurrent_state | `core_out`、`new_recurrent_state` | 1223 | 0 |
| 13 | `layer_00_linear_attn_block_ChunkGatedDeltaRule_DeltaNetChunkStep_chunk64.onnx` | 单步 chunk 子图（chunk=64） | 9 个张量（含 upper_mask 等） | `core_out`、`new_recurrent_state` | 23 | 0 |
| 14 | `layer_00_moe_block_8k.onnx` | layer 0（linear-attn）配对的 MoE | hidden + experts + shared | `hidden_states: bf16[1, 8192, 2048]` | 50 | 0 |
| 15 | `layer_03_full_attn_block_8k.onnx` | full-attn 层（M-RoPE 3D） | `hidden_states.1`、**`position_ids: i64[3, 1, 8192]`**、`attention_mask`、`past_key/past_value` | `hidden_states`、`new_key`、`new_value` | 136 | 0 |
| 16 | `layer_03_moe_block_8k.onnx` | layer 3（full-attn）配对的 MoE | 同 #14 | `hidden_states` | 50 | 0 |
| **Stage 4：输出头** ||||||||
| 17 | `norm_8k.onnx` | `Qwen3_5MoeRMSNorm` | `hidden_states: bf16[1, 8192, 2048]` | `output: bf16[1, 8192, 2048]` | 11 | 0 |
| 18 | `lm_head_8k.onnx` | `lm_head` 线性投影 | `hidden_states`、`lm_head_weight: bf16[248320, 2048]` | `logits: bf16[1, 8192, 248320]` | 2 | 0 |

---

## 3. 数据流连接边（生产者 → 消费者）

只列**跨文件**的张量；权重 / KV / 状态等仅在单文件内部出现的不计。

| 张量 | dtype × shape | 生产者 | 消费者 |
|---|---|---|---|
| `patch_embeds` | bf16 [1024, 1152] | `vision_patch_embed_1k` | `vision_pos_embed_interp_1k` |
| `hidden_states_post` | bf16 [1024, 1152] | `vision_pos_embed_interp_1k` | `vision_block_00_repr_1k` |
| `cos`, `sin` | bf16 [1024, 72] | `vision_rot_pos_emb_1k` | `vision_block_00_repr_1k` |
| `cu_seqlens` | i64 *动态长度* | `vision_cu_seqlens_1k` | `vision_block_00_repr_1k`（代表场景下传入 [2]） |
| `hidden_states_out` | bf16 [1024, 1152] | `vision_block_00_repr_1k` | `vision_patch_merger_1k`（实际跑时所有 27 层串联后才给 merger） |
| `image_embeds` | bf16 [256, 2048] | `vision_patch_merger_1k` | `mm_inject_8k` |
| `hidden_states` | bf16 [1, 8192, 2048] | `embedding_8k` | `mm_inject_8k`（作为 `inputs_embeds`） |
| `image_mask` | **bool [1, 8192, 2048]** | `image_mask_build_8k` | `mm_inject_8k` |
| `inputs_embeds_out` | bf16 [1, 8192, 2048] | `mm_inject_8k` | 第一个 layer block（`hidden_states.1` / `hidden_states.3`） |
| `position_ids` | **i64 [3, 1, 8192]**（M-RoPE 3D） | `mrope_position_ids_prefill_8k` | 所有 `layer_*_full_attn_block_8k`（vl 变体 3D 接口） |
| `hidden_states`（每层输出） | bf16 [1, 8192, 2048] | layer N attn / moe | layer N+1 |
| 末层 `hidden_states` | bf16 [1, 8192, 2048] | 最后一个 moe block | `norm_8k` |
| `output` | bf16 [1, 8192, 2048] | `norm_8k` | `lm_head_8k` |

---

## 4. 145 个 `unk__N` 的来源（按文件分类）

★ 标记的 3 个文件采用 **option B（H/W 真张量输入 + 算子全可见）**：用更多 `unk__N` 换在 ONNX 图里把源码每个张量算子都呈现出来；其余文件保持静态形状。

### 4.1 ★ Option B 类（3 个文件，138 个 unk）

| 文件 | unk | 节点 | 源码算子覆盖 |
|---|---|---|---|
| `vision_pos_embed_interp_1k.onnx` | 82 | 84 | `Range×2`（line 1175-1176 `linspace`，等价展开为 `step = (ngs-1)/(n-1); out = arange(n)*step`）、`Mul×11`、`Add×10`、`Sub×6`、`Clip×2`（line 1182-1183）、`Reciprocal×2`、`Div×2`、`Gather×8`（4 个角的 `pos_embed` 查表）、`Reshape×10` |
| `vision_rot_pos_emb_1k.onnx` | 35 | 36 | `Range×3`（line 1138-1141 的 3 个 `arange`）、`Einsum`（line 1128 `outer(seq, inv_freq)`）、`Max`（line 1127 `max(h, w)`）、`Cos / Sin`（line 1250）、`Gather×4`、`Mul×2`、`Add×4`、`Expand×2`、`Reshape×2`、`Concat×2` |
| `mrope_position_ids_prefill_8k.onnx` | 21 | 47 | `Range×3`（pre / image / post 3 段 `arange`）、`Tile×2`（line 1493 `repeat`）、`OneHot×1`（line 1494 `repeat_interleave` 的下放）、`ConstantOfShape×1`（line 1500 `torch.full`）、`Max`（line 1594 `max(grid_h, grid_w)`）、`ReduceMax`（line 1600 `.max() + 1`）、`Sub×2`、`Mul×2`、`ReduceSum×2`、`Stack/Concat`、`Expand×2` |

### 4.2 源码本征 unk（2 个文件，7 个 unk）

| 文件 | unk | 算子 | 数据依赖原因 |
|---|---|---|---|
| `vision_cu_seqlens_1k.onnx` | 4 | `Tile`（`repeat_interleave` 下放）+ `CumSum`/`Pad`/`Reshape` | `repeat_interleave(values, repeats=grid_thw[:,0])` 输出长度 = `repeats.sum()` |
| `mm_inject_8k.onnx` | 3 | `NonZero`（`masked_scatter` 下放）+ `Transpose`/`Slice`/`Gather` | `NonZero(image_mask)` 输出尺寸 = mask 中 True 的个数 |

### 4.3 unk 实际运行时形状

**所有 145 个 `unk__N` 在运行时都会解析为静态值**（前提是 `grid_thw` 给定）：

- `vision_rot_pos_emb` 输出：`[4*u1*u2, 72]` = `4 * (H/merge) * (W/merge) * 72` = `[1024, 72]`（当 grid=`[1,32,32]`）
- `vision_pos_embed_interp` 输出：仍是 `[1024, 1152]`（hidden_states_pre 锁住 shape）
- `mrope_position_ids_prefill` 输出：`[3, 1, L1+image_seq+L2]` = `[3, 1, 8192]`

下游 quant/编译工具加载时 hint `grid_thw = [[1,32,32]]` 即可静态化。

---

## 5. 运行时如何把这些 ONNX 串起来

伪代码（单图、单 batch、prefill 一次）：

```python
import onnxruntime as ort

# Stage 1: vision tower
patch_embeds = run("vision_patch_embed_1k.onnx",
                   pixel_values_flat=pixel_values.reshape(1024, 1536))
hidden_post = run("vision_pos_embed_interp_1k.onnx",
                  hidden_states_pre=patch_embeds,
                  grid_thw=image_grid_thw)

cos, sin   = run("vision_rot_pos_emb_1k.onnx", grid_thw=image_grid_thw)
cu_seqlens = run("vision_cu_seqlens_1k.onnx",   grid_thw=image_grid_thw)

vision_hidden = hidden_post
for layer_idx in range(27):
    vision_hidden = run("vision_block_00_repr_1k.onnx",
                        hidden_states=vision_hidden,
                        cos=cos, sin=sin, cu_seqlens=cu_seqlens)

image_embeds = run("vision_patch_merger_1k.onnx",
                   vision_features=vision_hidden)  # [256, 2048]

# Stage 2: 多模态簿记
inputs_embeds = run("embedding_8k.onnx",
                    input_ids=input_ids, embedding_weight=W_embed)
image_mask    = run("image_mask_build_8k.onnx",
                    input_ids=input_ids, image_token_id=image_token_id)
inputs_embeds = run("mm_inject_8k.onnx",
                    inputs_embeds=inputs_embeds,
                    image_mask=image_mask,
                    image_embeds=image_embeds)
position_ids, rope_deltas = run("mrope_position_ids_prefill_8k.onnx",
                                input_ids=input_ids,
                                mm_token_type_ids=mm_token_type_ids,
                                image_grid_thw=image_grid_thw)

# Stage 3: 文本解码层
hidden = inputs_embeds
for layer_idx, layer_kind in enumerate(real_layer_layout):
    if layer_kind == "linear":
        out = run("layer_00_linear_attn_block_8k.onnx", ...)
        hidden = run("layer_00_moe_block_8k.onnx", ...)
    elif layer_kind == "full":
        out = run("layer_03_full_attn_block_8k.onnx",
                  position_ids=position_ids, ...)
        hidden = run("layer_03_moe_block_8k.onnx", ...)

# Stage 4: 输出
hidden = run("norm_8k.onnx", hidden_states=hidden)["output"]
logits = run("lm_head_8k.onnx",
             hidden_states=hidden, lm_head_weight=W_lmhead)["logits"]
```

---

## 6. 已知导出注释

- `layer_00_linear_attn_block_8k.onnx` 与 `..._ChunkGatedDeltaRule_chunk64_8k.onnx`：`onnxsim` 在大节点数（>1k）上偶发死循环，跳过简化（`onnxsim=not_run`），shape inference 正常。
- `vision_rot_pos_emb_1k.onnx` 输出 `cos / sin` 第一维为符号 `4*u1*u2`（即 `(H/merge) * (W/merge) * merge² = H*W`，运行时 = 1024）；上层 ONNX 量化/编译工具如果不支持符号表达式，可在加载时把第一维 hint 为 1024。
- `mrope_position_ids_prefill_8k.onnx` 输出 `position_ids` 第三维为符号表达式 `L1 + image_seq_length + L2`（运行时 = 8192）；同上，工具如果只接受静态 shape，在加载时 hint 为 8192 即可。
- `vision_pos_embed_interp_1k.onnx` 输出 `hidden_states_post` 仍是静态 `[1024, 1152]`，因为最后一行 `hidden_states_pre + patch_pos_embeds` 的左操作数是静态形状，broadcast 把右侧的符号 shape 锁回静态。

---

## 7. 两种模式的差异（`--vision_grid_mode dynamic` vs `static`）

CLI 参数 `--vision_grid_mode {dynamic, static}` 决定 3 个 ★ 子图的导出方式：

| 比较项 | `dynamic`（本目录） | `static`（→ `Prefill_8k_static/`） |
|---|---|---|
| 子图算子可见性 | **全可见**（Range/Cos/Sin/Tile/OneHot/...） | 折成查找表 + 少量末端算子 |
| `vision_rot_pos_emb_1k` 节点 | 36 | 5（cos/sin 折成 288 KB initializer） |
| `vision_pos_embed_interp_1k` 节点 | 84 | 21（4 角索引/权重折成常量） |
| `mrope_position_ids_prefill_8k` 节点 | 47 | 10（position_ids 折成 192 KB initializer） |
| 总 `unk__N` | 145 | 7（仅源码本征 unk：cu_seqlens 4 + mm_inject 3） |
| 同一份 ONNX 喂多种 grid_thw | ✓ 支持，加载时 hint shape 即可 | ✗ 必须每个分辨率重导一份 |
| 源码算子拓扑 1:1 对应 | ✓ | 部分折叠 |
| 适合的下游用途 | 源码分析、Netron 可视化、量化工具加载时 hint shape | 必须 ONNX 端就静态形状的工具（老 NPU 编译器） |

切换示例：

```bash
# dynamic（默认）：本目录
python modes/qwen_3_5_MoE/export_qwen_onnx_main.py MODEL_PATH \
  --variant vl --phase prefill --seq_len 8192 \
  --vision_token_seq_len 1024 \
  --vision_grid_mode dynamic        # 可省略

# static：覆盖一组分辨率，每个分辨率重导一遍
python modes/qwen_3_5_MoE/export_qwen_onnx_main.py MODEL_PATH \
  --variant vl --phase prefill --seq_len 8192 \
  --vision_token_seq_len 1024 \
  --vision_grid_mode static
# 输出 → Qwen3_5_35B_A3B_VL_ONNX_Prefill_8k_static/
```
