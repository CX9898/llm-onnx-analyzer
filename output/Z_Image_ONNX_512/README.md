# Z-Image ONNX 说明：`Z_Image_ONNX_512`

代表场景：Basic 文生图，`512×512` 输入，`cap_seq=128`，`batch=1`。
共 **10 张 ONNX**（text 2 + denoise 7 + vae 1）。

## ONNX 子图清单

| Phase | 文件 | 源码步骤 | host repeat |
|-------|------|----------|-------------|
| text_encode | `text_embed_prepare_cap128.onnx` | TE embed + RoPE + mask | ×1 |
| text_encode | `text_decoder_layer_repr_cap128.onnx` | Qwen3 layers[i]（repr: layers[0]） | **×35** |
| denoise | `patchify_and_embed_img512.onnx` | T2 patchify_and_embed | ×1 |
| denoise | `timestep_embed.onnx` | T1 t_embedder | ×1 |
| denoise | `x_branch_seq1k.onnx` | T3 embed + T4 noise_refiner×2（图内） | ×1 |
| denoise | `cap_branch_cap128.onnx` | T5 embed + T6 context_refiner×2（图内） | ×1 |
| denoise | `sequence_concat_basic_seq1152.onnx` | T7 _build_unified_sequence | ×1 |
| denoise | `main_layer_repr_seq1152.onnx` | T8 layers[i]（repr: layers[0]） | **×30** |
| denoise | `final_output_img512.onnx` | T9 final + unpatchify + cast f32 | ×1 |
| vae_decode | `vae_decode_img512.onnx` | pipeline VAE decode | ×1 |

Denoise 采用 **6 张主干 + timestep** 方案：`x_branch` / `cap_branch` 已将 refiner×2 展开在单图内，不再单独导出 `noise_refiner_block_repr` / `context_refiner_block_repr`。

## 目录结构

```text
Z_Image_ONNX_512/
├── source_profile.json
├── README.md
├── z_image_e2e_rollup.{md,json}
├── text_encode/
│   ├── text_embed_prepare_cap128.onnx
│   ├── text_decoder_layer_repr_cap128.onnx
│   └── text_encode_layer_manifest.json
├── denoise/
│   ├── patchify_and_embed_img512.onnx
│   ├── timestep_embed.onnx
│   ├── x_branch_seq1k.onnx
│   ├── cap_branch_cap128.onnx
│   ├── sequence_concat_basic_seq1152.onnx
│   ├── main_layer_repr_seq1152.onnx
│   ├── final_output_img512.onnx
│   ├── denoise_layer_manifest.json
│   └── source_alignment_audit.md
└── vae_decode/
    └── vae_decode_img512.onnx
```

## 完整推理数据流

```text
[Phase: text_encode — 每 prompt 一次]

  input_ids:[1,128]i64 + attention_mask:[1,128]i64
        │
        ▼  text_embed_prepare_cap128.onnx
  hidden + rope + attn_mask
        │
        ▼  text_decoder_layer_repr_cap128.onnx  (×35)
  prompt_hidden:[1,128,2560]bf16
        │  (pipeline gather → patchify_and_embed cap 支，见 source_profile.json T2)
        │
[Phase: denoise — 源码 forward 顺序，见 transformer_z_image.py:938-1068]

  T1 adaln ← timestep_embed (t×t_scale=1000.0)
  T2 patchify_and_embed (latent + cap_feats → x/cap patches + pos/pad)
  T3+T4 x_branch (embed + noise_refiner×2)
  T5+T6 cap_branch (embed + context_refiner×2)
  T7 concat → main_layer×30 → final_output → noise_pred f32

  latent:[1,16,1,64,64]bf16
  + prompt_hidden:[1,128,2560]bf16
        │
        ▼  patchify_and_embed_img512.onnx
        ├─► x_branch_seq1k.onnx  (embed + noise_refiner×2 in-graph)
        └─► cap_branch_cap128.onnx  (embed + context_refiner×2 in-graph)
  timestep → adaln → x_branch / main / final_output_img512.onnx
        │
        ▼  sequence_concat_basic_seq1152.onnx
  unified:[1,1152,3840]bf16 + unified_rope
        │
        ▼  main_layer_repr_seq1152.onnx  (×30)
  noise_pred:[1,16,64,64]f32  → scheduler.step

[Phase: vae_decode — 去噪结束后一次，latents f32]

  latents:[1,16,64,64]f32
        │
        ▼  vae_decode_img512.onnx
  image:[1,3,512,512]bf16
```

## 约束符合性

| # | 约束 | 状态 |
|---|------|------|
| 1 | 源码对齐（diffusers DiT + pipeline + transformers TE） | block 主体走源码模块；RoPE 用实数 Mul/Add 等价于源码 complex 路径 |
| 2 | 真实 dtype | 与 pipeline/transformer 源码 cast 路径一致；边界自动校验 |
| 3 | 语义拆分 + 代表层 | x/cap branch 图内 unroll refiner×2；main/text decoder 各导 layer[0]，host repeat 见 manifest |
| 4 | 完整可拼接数据流 | denoise 主链 + text→cap 边界校验；scheduler 步间由 host 衔接 |
| 5 | 权重 initializer 绑定算子；超大走 external data | 禁止默认 strip 为 graph input |
| 6 | Custom op + 子图 | 不适用（无 RecurrentGatedDeltaRule 类强耦合循环结构） |

## 代表层 repeat（manifest）

| 子图 stem | 图内 unroll | host repeat | 权重来源 |
|-----------|-------------|-------------|----------|
| `text_decoder_layer_repr_cap128` | — | ×35 | `Qwen3Model.layers[0]` |
| `x_branch_seq1k` | refiner ×2 | ×1 | `noise_refiner[0]` |
| `cap_branch_cap128` | refiner ×2 | ×1 | `context_refiner[0]` |
| `main_layer_repr_seq1152` | — | ×30 | `ZImageTransformer2DModel.layers[0]` |

详见 `text_encode/text_encode_layer_manifest.json` 与 `denoise/denoise_layer_manifest.json`。

## Host 侧职责（不在 ONNX 内）

- Tokenizer → `input_ids` / `attention_mask`
- Scheduler 循环（默认 28 步）、`timestep` 调度、`scheduler.step`
- CFG：cond / uncond 两路 denoise（rollup 默认 ×2）
- text→cap 对齐：pipeline gather / 截断到有效 caption token
- VAE 后处理：normalize、转 uint8 等

## 辅助产物

| 文件 | 用途 |
|------|------|
| `source_profile.json` | 真实权重 forward trace，I/O 契约 ground truth |
| `*/onnx_flow_stats_multi.{xlsx,json}` | 各 phase 算子/MACs 统计 |
| `*/*.flow_stats.summary.json` | 单图 MACs 摘要（e2e rollup 输入） |
| `z_image_e2e_rollup.{md,json}` | text + denoise×步×CFG + vae 端到端 MACs |
| `denoise/source_alignment_audit.md` | 导出 wrapper 与 diffusers 源码对照 |

## 子图 I/O 契约

- **text_embed_prepare_cap128.onnx**: pipeline_z_image.py:_encode_prompt (embed + RoPE + mask)
  - text_embed_prepare_cap128.onnx  (input_ids:[1, 128]int64, attention_mask:[1, 128]int64) -> (hidden_states:[1, 128, 2560]bfloat16, rope_cos:[1, 128, 128]bfloat16, rope_sin:[1, 128, 128]bfloat16, attn_mask:[1, 1, 128, 128]bfloat16)
- **text_decoder_layer_repr_cap128.onnx** (×35): Qwen3Model.layers[i] (repr: layers[0]) — Host 串联 ×repeat；末段输出即 prompt_hidden（对应 hidden_states[-2]）。
  - text_decoder_layer_repr_cap128.onnx ×35  (hidden_states:[1, 128, 2560]bfloat16, rope_cos:[1, 128, 128]bfloat16, rope_sin:[1, 128, 128]bfloat16, attn_mask:[1, 1, 128, 128]bfloat16) -> (hidden_states_out:[1, 128, 2560]bfloat16)
- **patchify_and_embed_img512.onnx**: transformer_z_image.py:588 patchify_and_embed — T2：含 _patchify_image + cap _pad_with_ids。
  - patchify_and_embed_img512.onnx  (latent:[1, 16, 1, 64, 64]bfloat16, cap_feats:[1, 128, 2560]bfloat16) -> (x_patch_feats:[1, 1024, 64]bfloat16, cap_feats_padded:[1, 128, 2560]bfloat16, x_pos_ids:[1, 1024, 3]int32, cap_pos_ids:[1, 128, 3]int32, x_pad_mask:[1, 1024]bool, cap_pad_mask:[1, 128]bool)
- **timestep_embed.onnx**: transformer_z_image.py:945
  - timestep_embed.onnx  (timestep:[1]float32) -> (adaln_input:[1, 256]bfloat16)
- **x_branch_seq1k.onnx**: transformer_z_image.py:980-992 — T3+T4 merged: x_embed_prepare + noise_refiner[0]×2 in-graph.
  - x_branch_seq1k.onnx  (x_patch_feats:[1, 1024, 64]bfloat16, x_pos_ids:[1, 1024, 3]int32, x_pad_mask:[1, 1024]bool, adaln_input:[1, 256]bfloat16) -> (x_tokens:[1, 1024, 3840]bfloat16, x_rope_cos:[1, 1024, 64]float32, x_rope_sin:[1, 1024, 64]float32, x_attn_mask:[1, 1024]bool)
- **cap_branch_cap128.onnx**: transformer_z_image.py:996-1006 — T5+T6 merged: cap_embed_prepare + context_refiner[0]×2 in-graph.
  - cap_branch_cap128.onnx  (cap_feats_padded:[1, 128, 2560]bfloat16, cap_pos_ids:[1, 128, 3]int32, cap_pad_mask:[1, 128]bool) -> (cap_tokens:[1, 128, 3840]bfloat16, cap_rope_cos:[1, 128, 64]float32, cap_rope_sin:[1, 128, 64]float32, cap_attn_mask:[1, 128]bool)
- **sequence_concat_basic_seq1152.onnx**: transformer_z_image.py:859
  - sequence_concat_basic_seq1152.onnx  (x_tokens:[1, 1024, 3840]bfloat16, cap_tokens:[1, 128, 3840]bfloat16, x_rope_cos:[1, 1024, 64]float32, x_rope_sin:[1, 1024, 64]float32, cap_rope_cos:[1, 128, 64]float32, cap_rope_sin:[1, 128, 64]float32) -> (unified_tokens:[1, 1152, 3840]bfloat16, unified_rope_cos:[1, 1152, 64]float32, unified_rope_sin:[1, 1152, 64]float32)
- **main_layer_repr_seq1152.onnx** (×30): transformer_z_image.py:1048 layers[0]
  - main_layer_repr_seq1152.onnx ×30  (hidden_states:[1, 1152, 3840]bfloat16, attn_mask:[1, 1152]bool, rope_cos:[1, 1152, 64]float32, rope_sin:[1, 1152, 64]float32, adaln_input:[1, 256]bfloat16) -> (output:[1, 1152, 3840]bfloat16)
- **final_output_img512.onnx**: transformer_z_image.py:1064-1068 + pipeline_z_image.py:556-558 — T9 merged: final_layer + slice image tokens + unpatchify.
  - final_output_img512.onnx  (hidden_states:[1, 1152, 3840]bfloat16, adaln_input:[1, 256]bfloat16) -> (noise_pred:[1, 16, 64, 64]float32)
- **vae_decode_img512.onnx**: pipeline_z_image.py:583-586
  - vae_decode_img512.onnx  (latents:[1, 16, 64, 64]float32) -> (image:[1, 3, 512, 512]bfloat16)