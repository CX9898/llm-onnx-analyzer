# Qwen3.5-35B-A3B VL — Prefill 8K ONNX（**static** 模式：32×32 grid）

本目录是用 `--vision_grid_mode static` 重导的版本——**3 个对 H/W 敏感的子图全部按 `grid_thw=[1,32,32]` 焊死成 0 unk__N 的静态 ONNX**。整张数据流形状全部锁定。

> **想要源码每个算子可见 / 一份 ONNX 喂多个分辨率** → 用同级目录 [`../Qwen3_5_35B_A3B_VL_ONNX_Prefill_8k/`](../Qwen3_5_35B_A3B_VL_ONNX_Prefill_8k/)（dynamic 模式，默认）。两个目录的"非 ★"文件（共 15 个）完全相同。

---

## 1. 目录与生成命令

```bash
python modes/qwen_3_5_MoE/export_qwen_onnx_main.py \
  /path/to/Qwen3.5-35B-A3B \
  --variant vl --phase prefill --seq_len 8192 \
  --vision_token_seq_len 1024 \
  --mm_image_token_count 256 \
  --mrope_text_pre_len 4096 \
  --vision_grid_mode static
```

参数解析：
- `--vision_token_seq_len 1024` → `_representative_grid_thw` 推出 `(grid_t=1, grid_h=32, grid_w=32)`
- `--vision_grid_mode static` → 把 (1, 32, 32) 在 `__init__` 时锁进 wrapper，整张图按这个 grid 折叠
- 想换分辨率（如 64×48）：再跑一次，把 `--vision_token_seq_len` 改成 `1*64*48 = 3072`，`--mm_image_token_count` 改成 `64/2 * 48/2 = 384`，输出会到 `Prefill_8k_static/`（会覆盖；你可以加 `--output_dir` 自定义路径）

---

## 2. 文件清单（与 dynamic 模式对照）

★ 标的 3 个文件是两种模式唯一不同的文件：

| # | 文件 | dynamic | static（本目录） | 差异点 |
|---|---|---|---|---|
| 1 | `vision_patch_embed_1k.onnx` | 3 / 0 unk | 3 / 0 unk | 无 |
| 2 ★ | `vision_pos_embed_interp_1k.onnx` | 84 / 82 unk | **21 / 0 unk** | linspace + 4 角双线性查表折成常量；只剩 4 个 `Gather` + Add + Reshape |
| 3 ★ | `vision_rot_pos_emb_1k.onnx` | 36 / 35 unk | **5 / 0 unk** | 整张 cos/sin 表折成 288 KB initializer；只剩 1 个 Mul（topology anchor）+ 几个 Cast |
| 4 | `vision_cu_seqlens_1k.onnx` | 12 / 4 unk | 12 / 4 unk | 无（4 unk 是源码 `repeat_interleave` 本征语义，与 grid_thw 无关） |
| 5 | `vision_block_00_repr_1k.onnx` | 60 / 0 unk | 60 / 0 unk | 无 |
| 6 | `vision_patch_merger_1k.onnx` | 5 / 0 unk | 5 / 0 unk | 无 |
| 7 | `embedding_8k.onnx` | 1 / 0 unk | 1 / 0 unk | 无 |
| 8 | `image_mask_build_8k.onnx` | 3 / 0 unk | 3 / 0 unk | 无 |
| 9 | `mm_inject_8k.onnx` | 8 / 3 unk | 8 / 3 unk | 无（3 unk 是源码 `masked_scatter` 本征语义） |
| 10 ★ | `mrope_position_ids_prefill_8k.onnx` | 47 / 21 unk | **10 / 0 unk** | 完整 [3, 8192] position_ids 折成 192 KB initializer；只剩 ReduceSum + Add（topology anchor）|
| 11-16 | layer_00 / layer_03 系列 | 同 | 同 | 无 |
| 17-18 | norm_8k / lm_head_8k | 同 | 同 | 无 |

总和：**18 文件 / 7 unk__N**（dynamic 是 18 文件 / 145 unk）。

---

## 3. ★ 3 个子图 static 化后的算子清单

### `vision_rot_pos_emb_1k.onnx`（5 节点）

```
inputs : grid_thw[1, 3] (i64)
outputs: cos[1024, 72] (bf16)、sin[1024, 72] (bf16)

ops    : Cast×1 + Mul×1 + ReduceSum×1 + Add×2  ← 全是 topology anchor
initializers: 144 KB cos 表 + 144 KB sin 表 = 288 KB
```

源码 `rot_pos_emb`（line 1123-1161）的 `Range / Cast / Mul / Add / Einsum / Cos / Sin` 全部被 trace 时算成了固定 `[1024, 72]` 的 cos/sin 值表。

### `vision_pos_embed_interp_1k.onnx`（21 节点）

```
inputs : hidden_states_pre[1024, 1152] (bf16)、grid_thw[1, 3] (i64)
outputs: hidden_states_post[1024, 1152] (bf16)

ops    : Gather×5 + Add×5 + Constant×3 + Mul×2 + Reshape×2 + Cast×2 + Transpose×1 + ReduceSum×1
initializers: 5184 KB （pos_embed.weight 4096×1152×bf16 ≈ 4.6 MB + 4 角索引/权重 ≈ 0.5 MB）
```

源码 `fast_pos_embed_interpolate`（line 1163-1224）的 `linspace / clip / floor / ceil / 4 角索引/权重` 全部折成常量；运行时只需要 4 个 `Gather`（pos_embed 查表）+ 加权求和（Mul/Add）+ 末端 spatial-merge `Reshape/Transpose`。

### `mrope_position_ids_prefill_8k.onnx`（10 节点）

```
inputs : input_ids[1, 8192] (i64)、mm_token_type_ids[1, 8192] (i32)、image_grid_thw[1, 3] (i64)
outputs: position_ids[3, 1, 8192] (i64)、mrope_position_deltas[1, 1] (i64)

ops    : Add + ReduceSum + Cast + Mul ... ← 全是 topology anchor
initializers: 192 KB position_ids 完整表 [3, 8192]
```

源码 `get_rope_index` + `get_vision_position_ids`（line 1455-1645）的 `arange / repeat / repeat_interleave / full / max / cumsum` 全部折成单个 `[3, 8192]` int64 lookup table。

---

## 4. 数据流形状（全部静态）

| 张量 | dtype × shape | 备注 |
|---|---|---|
| `pixel_values_flat` | bf16 [1024, 1536] | 32×32 grid × 2 temporal × 16² × 3ch = 1536 |
| `patch_embeds` | bf16 [1024, 1152] | hidden_size = 1152 |
| `cos / sin` | **bf16 [1024, 72]** ← 静态 | dynamic 版是 `[4*u1*u2, 72]` |
| `cu_seqlens` | i64 [Padcu_seqlens_dim_0] | 仍带 1 个本征 unk（运行时 [2]） |
| `image_embeds` | bf16 [256, 2048] | 256 = 1024 / merge² |
| `image_mask` | bool [1, 8192, 2048] | |
| `inputs_embeds_out` | bf16 [1, 8192, 2048] | mm_inject 仍含 3 个本征 unk（masked_scatter 语义）|
| `position_ids` | **i64 [3, 1, 8192]** ← 静态 | dynamic 版是 `[3, 1, L1+image_seq+L2]` |
| `mrope_position_deltas` | i64 [1, 1] | |
| `logits` | bf16 [1, 8192, 248320] | |

唯一 7 个 unk 都集中在 `vision_cu_seqlens_1k.onnx`（4）+ `mm_inject_8k.onnx`（3），都是 ONNX 标准下放 `repeat_interleave / masked_scatter` 时的数据依赖输出，与 `grid_thw` 是否静态无关。

---

## 5. 多分辨率 / 多 bucket 用法

如果业务有几种常见分辨率，建议按桶导出：

```bash
# 32×32 grid（512×512 输入图） → 1024 vision tokens / 256 LLM tokens
python ... --vision_grid_mode static --vision_token_seq_len 1024 \
  --output_dir output/static_32x32

# 48×32 grid（768×512 输入图） → 1536 / 384
python ... --vision_grid_mode static --vision_token_seq_len 1536 \
  --mm_image_token_count 384 \
  --output_dir output/static_48x32

# 64×64 grid（1024×1024 输入图） → 4096 / 1024
python ... --vision_grid_mode static --vision_token_seq_len 4096 \
  --mm_image_token_count 1024 \
  --output_dir output/static_64x64
```

> 注意：`_representative_grid_thw` 假设 `H == W`（要求 `vision_token_seq_len` 是平方数）。如果你想要非正方形 grid（如 64×48），需要扩 `_representative_grid_thw` 接受 `(t, h, w)` 三参数（目前未做，可以提需求）。

每个 bucket 跑一次 → 18 个完全静态的 ONNX。运行时按 `smart_resize` 算出的 `grid_thw` 路由到最近的 bucket。
