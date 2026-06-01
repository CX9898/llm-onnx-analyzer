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

## 架构总览（对照 Qwen3.5-MoE 架构图）

Z-Image 是扩散三段式（text → denoise 循环 → vae），denoise 内为 **双支预处理 → concat → 单流 backbone**。

### 上方放大：S3-DiT Block（`main_layer_repr_seq1152.onnx`，host repeat ×30）

类比 Qwen 图里的 MoE Block 放大图；Z-Image 无 Expert 路由，核心是 **AdaLN + Single-Stream Self-Attn + dense FFN**。

```text
hidden [1,1152,3840] bf16          adaln_input [1,256] bf16
         │                                    │
         │         AdaLN-Linear → chunk×4      │
         │         (scale_msa, gate_msa,       │
         │          scale_mlp, gate_mlp)       │
         ▼                                    ▼
    ┌─ Attention 支 ─────────────────────────────────┐
    │  RMSNorm → ×scale_msa                          │
    │       → Q/K/V Linear  (30 heads, head_dim=128)     │
    │       → QK-Norm (RMSNorm)                      │
    │       → 3D U-RoPE (axes [32, 48, 48], rope_dim=64)   │
    │       → Softmax Self-Attn  (S=1152 全序列)           │
    │       → Out-Linear                             │
    │       → RMSNorm → ×gate_msa → Add(residual)    │
    └────────────────────────────────────────────────┘
         │
    ┌─ FFN 支 ───────────────────────────────────────┐
    │  RMSNorm → ×scale_mlp                          │
    │       → FFN: Linear 3840→10240 → GELU → 3840   │
    │       → RMSNorm → ×gate_mlp → Add(residual)    │
    └────────────────────────────────────────────────┘
         │
         ▼  output [1,1152,3840] bf16
```

**与 Qwen MoE Block 的差异：** Qwen 用 Router + Top-K Experts；Z-Image 用 **dense FFN**，条件注入走 **AdaLN(timestep)**，Attention 为 **全序列 Full Self-Attn + 3D U-RoPE**（非 linear/full 交替）。

### 底行：端到端主链（含 ONNX 文件名映射）

```text
Prompt Token
     │
     ▼
┌──────────────────────────────────────┐
│  Text Encoder (Qwen3)                │  text_embed_prepare_cap128.onnx
│  Embed → DecoderLayer ×35             │  text_decoder_layer_repr_cap128.onnx ×35
└──────────────────────────────────────┘
     │ prompt_hidden [1,128,2560] bf16
     │ (host: mask gather → cap_feats)
     │
Random Latent [1,16,64,64] f32 ─────────────────┐
     │                                           │
     │  ┌── denoise loop ×N (rollup 默认 28 步; Turbo 9 步; CFG host ×2) ─┐
     │  │                                                                 │
Timestep ──► TimestepEmbed ──► adaln [1,256]          timestep_embed.onnx
     │  │                                                                 │
     │  │  patchify_and_embed ──┬──► X-Branch (embed+Refiner×2)   patchify + x_branch_seq1k
     │  │                       └──► Cap-Branch (embed+Refiner×2)  cap_branch_cap128
     │  │                                    │                            │
     │  │                                    ▼ Concat [1,1152,3840]   sequence_concat_basic_seq1152
     │  │                                    │
     │  │                         ┌─ S3-DiT Block ×30 ─┐  main_layer_repr_seq1152 ×30
     │  │                         └──────────────────────────────────┘
     │  │                                    │
     │  │                         FinalLayer + unpatchify              final_output_img512
     │  │                                    ▼ noise_pred [1,16,64,64] f32
     │  └──────── scheduler.step (host) ◄───────────────────────────────┘
     │
     └──────── latents f32 ──► VAE Decode ──► image [1,3,512,512]   vae_decode_img512.onnx
```

### 模态预处理器 Refiner（图内 unroll，不单独导出 ONNX）

```text
noise_refiner (有 AdaLN, S=1024)     context_refiner (无 AdaLN, S=128)
  ZImageTransformerBlock ×2 in x_branch_seq1k    ZImageTransformerBlock ×2 in cap_branch_cap128
```

### 底行模块 ↔ ONNX 对照

| 底行模块 | ONNX 文件 | repeat |
|----------|-----------|--------|
| TE Embed + RoPE + Mask | `text_embed_prepare_cap128.onnx` | ×1 |
| TE DecoderLayer | `text_decoder_layer_repr_cap128.onnx` | **×35** |
| TimestepEmbed | `timestep_embed.onnx` | ×1/步 |
| Patchify | `patchify_and_embed_img512.onnx` | ×1/步 |
| X-Branch (refiner×2 图内) | `x_branch_seq1k.onnx` | ×1/步 |
| Cap-Branch (refiner×2 图内) | `cap_branch_cap128.onnx` | ×1/步 |
| Concat unified sequence | `sequence_concat_basic_seq1152.onnx` | ×1/步 |
| S3-DiT Block | `main_layer_repr_seq1152.onnx` | **×30/步** |
| Output Head | `final_output_img512.onnx` | ×1/步 |
| VAE Decode | `vae_decode_img512.onnx` | ×1 |

Host 不在 ONNX 图内：tokenizer、scheduler 循环、`scheduler.step`、CFG 双路、text mask gather、VAE 后处理。

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