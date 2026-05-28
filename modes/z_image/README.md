# Z-Image ONNX Export

面向 `/mnt/data8t/share/models/Tongyi-MAI/Z-Image` 的 ONNX 代表层导出。

## 设计原则（不为修而修）

1. **源码**：`diffusers/.../transformer_z_image.py` 的 `forward` 是 DiT 算子拓扑的唯一定义。
2. **权重**：`/Tongyi-MAI/Z-Image/{transformer,text_encoder,vae}/` 下 safetensors 的真实 dtype/shape（当前全部为 **bf16**）。
3. **契约**：导出前用 `z_image_source_profile.py` 加载真实权重、逐步执行 forward，生成 `source_profile.json`；所有 I/O 契约从该 profile 推导，而非手写猜测。
4. **分解**：ONNX 子图是分析用的**语义切片**，允许在 batch 维等边界做静态化，但 **dtype 与源码 cast 路径必须一致**。

## 真实权重（Tongyi-MAI/Z-Image）

| 组件 | config 关键点 | 权重 dtype |
|------|--------------|------------|
| transformer | dim=3840, in_channels=16, cap_feat_dim=2560, n_layers=30, n_refiner_layers=2, t_scale=1000, axes_dims=[32,48,48] | bf16 |
| text_encoder | hidden_size=2560 | bf16 |
| vae | latent_channels=16, scaling_factor≈0.3611 | bf16 |

512² 代表场景：latent `[1,16,64,64]` f32（scheduler）→ DiT 输入 `[1,16,1,64,64]` bf16 → image tokens S_x=1024。

## 源码 forward 顺序（basic 模式）

见 `transformer_z_image.py:938-1068`，trace 结果写入 `source_profile.json`：

```text
T1  t_embedder(t*t_scale)           → adaln [1,256] bf16
T2  patchify_and_embed               → x patches [1024,64], cap [128,2560] bf16
T3  x_embedder + _prepare_sequence   → x [1,1024,3840], freqs.real f32
T4  noise_refiner ×2
T5  cap_embedder + _prepare_sequence
T6  context_refiner ×2
T7  _build_unified_sequence          → [1,1152,3840]
T8  layers ×30
T9  final_layer → unpatchify         → [16,1,64,64] bf16
T11 pipeline: -sample.float().squeeze(2) → noise_pred [1,16,64,64] f32
```

## 文件后缀约定

后缀格式 `<prefix><value>`，表示该子图绑定的代表场景轴：

| 前缀 | 示例 | 含义 |
|------|------|------|
| `img` | `img512` | 像素分辨率场景（512×512） |
| `cap` | `cap128` | caption token 序列长 |
| `seq` | `seq1k` / `seq1152` | DiT token 序列长（image 或 unified） |

## ONNX 子图映射（6 张 denoise 主干 + timestep + text + vae）

```text
text_encode_cap128 → patchify_and_embed_img512
  ├─ x_branch_seq1k (embed + noise_refiner×2 in-graph)
  └─ cap_branch_cap128 (embed + context_refiner×2 in-graph)
→ sequence_concat_seq1152 → main_layer×30 → final_output_img512
timestep_embed → adaln（x_branch / main / final）
```

导出完成后自动运行 `scripts/analyze_onnx_flow_stats_batch.py`（各 phase）与 `z_image_e2e_rollup.py`（端到端 MACs）。

## 目录

```text
modes/z_image/
├── z_image_source_profile.py   # 真实权重 + forward trace → 契约源头
├── z_image_export_semantics.py # 从 profile 生成 GraphContract
├── z_image_text_onnx_blocks.py # Qwen3 text encoder ONNX wrapper（实数 RoPE / eager attn）
├── z_image_boundary_validate.py
├── z_image_dit_export.py / z_image_text_export.py / vae
└── export_z_image_onnx_main.py
```

### text_encode 导出（拆分子图，对齐 denoise 代表层模式）

- **不**整图导出 36 层 Qwen3，拆为可拼接子图：
  1. `text_embed_prepare_cap128.onnx` — embed + RoPE + attn mask
  2. `text_decoder_layer_repr_cap128.onnx` — **layers[0] 代表**，host 串联 **×35**（对应 `hidden_states[-2]`，见 `text_encode_layer_manifest.json`）
- RoPE：`cos/sin` + `rotate_half`，无 `COMPLEX128`
- Attention：显式 `MatMul + Softmax + MatMul`，无 SDPA / hub kernel
- 静态 shape + post-process 传播，无 `unk__`
- 权重：initializer 绑算子；导出后丢弃权重字节，无 `.data` sidecar

## 一键导出

```bash
python export_model.py z-image \
    --model_path /mnt/data8t/share/models/Tongyi-MAI/Z-Image
```

产物含 `source_profile.json`（ground truth）与 `denoise/` 等 ONNX 子图；导出结束自动校验相邻 I/O。
