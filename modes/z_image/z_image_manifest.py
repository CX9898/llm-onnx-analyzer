"""Generate output README with full Z-Image data-flow diagram."""

from __future__ import annotations

from pathlib import Path

from z_image_export_semantics import (
    ExportScene,
    build_denoise_graph_contracts,
    build_text_encode_graph_contracts,
    build_vae_decode_contract,
    load_source_profile,
)
from z_image_export_shared import cap_scene_tag, img_scene_tag, seq_scene_tag
from z_image_shape_propagation import contract_io_summary


def render_output_readme(
    out_root: Path,
    scene: ExportScene,
    profile,
) -> str:
    b = scene.batch_size
    s_x = profile.image_token_seq
    s_cap = scene.cap_seq
    s_u = s_x + s_cap
    h_lat, w_lat = profile.latent_hw
    patch_dim = profile.patch_dim
    dim = profile.dim
    out_channels = profile.in_channels
    cap_feat_dim = profile.cap_feat_dim
    text_hidden = profile.cap_feat_dim
    latent_channels = profile.in_channels
    wdtype = profile.weight_dtype

    text_contracts = build_text_encode_graph_contracts(scene, profile)
    vae_contract = build_vae_decode_contract(scene, profile)
    denoise_contracts = build_denoise_graph_contracts(scene, profile)
    text_tag = cap_scene_tag(scene.cap_seq)
    img_tag = img_scene_tag(scene.image_size)
    tag_x = seq_scene_tag(s_x)
    tag_u = seq_scene_tag(s_u)
    n_text_fwd = profile.n_text_forward_layers
    n_denoise_onnx = len(denoise_contracts)
    n_total_onnx = len(text_contracts) + n_denoise_onnx + 1

    lines = [
        f"# Z-Image ONNX 说明：`{out_root.name}`",
        "",
        f"代表场景：Basic 文生图，`{scene.image_size}×{scene.image_size}` 输入，`cap_seq={scene.cap_seq}`，`batch={b}`。",
        f"共 **{n_total_onnx} 张 ONNX**（text {len(text_contracts)} + denoise {n_denoise_onnx} + vae 1）。",
        "",
        "## ONNX 子图清单",
        "",
        "| Phase | 文件 | 源码步骤 | host repeat |",
        "|-------|------|----------|-------------|",
        f"| text_encode | `text_embed_prepare_{text_tag}.onnx` | TE embed + RoPE + mask | ×1 |",
        f"| text_encode | `text_decoder_layer_repr_{text_tag}.onnx` | Qwen3 layers[i]（repr: layers[0]） | **×{n_text_fwd}** |",
        f"| denoise | `patchify_and_embed_{img_tag}.onnx` | T2 patchify_and_embed | ×1 |",
        f"| denoise | `timestep_embed.onnx` | T1 t_embedder | ×1 |",
        f"| denoise | `x_branch_{tag_x}.onnx` | T3 embed + T4 noise_refiner×{profile.n_refiner_layers}（图内） | ×1 |",
        f"| denoise | `cap_branch_{text_tag}.onnx` | T5 embed + T6 context_refiner×{profile.n_refiner_layers}（图内） | ×1 |",
        f"| denoise | `sequence_concat_basic_{tag_u}.onnx` | T7 _build_unified_sequence | ×1 |",
        f"| denoise | `main_layer_repr_{tag_u}.onnx` | T8 layers[i]（repr: layers[0]） | **×{profile.n_layers}** |",
        f"| denoise | `final_output_{img_tag}.onnx` | T9 final + unpatchify + cast f32 | ×1 |",
        f"| vae_decode | `{vae_contract.file_name}` | pipeline VAE decode | ×1 |",
        "",
        "Denoise 采用 **6 张主干 + timestep** 方案：`x_branch` / `cap_branch` 已将 refiner×2 展开在单图内，不再单独导出 `noise_refiner_block_repr` / `context_refiner_block_repr`。",
        "",
        "## 目录结构",
        "",
        "```text",
        f"{out_root.name}/",
        "├── source_profile.json",
        "├── README.md",
        "├── z_image_e2e_rollup.{md,json}",
        "├── text_encode/",
        f"│   ├── text_embed_prepare_{text_tag}.onnx",
        f"│   ├── text_decoder_layer_repr_{text_tag}.onnx",
        "│   └── text_encode_layer_manifest.json",
        "├── denoise/",
        f"│   ├── patchify_and_embed_{img_tag}.onnx",
        "│   ├── timestep_embed.onnx",
        f"│   ├── x_branch_{tag_x}.onnx",
        f"│   ├── cap_branch_{text_tag}.onnx",
        f"│   ├── sequence_concat_basic_{tag_u}.onnx",
        f"│   ├── main_layer_repr_{tag_u}.onnx",
        f"│   ├── final_output_{img_tag}.onnx",
        "│   ├── denoise_layer_manifest.json",
        "│   └── source_alignment_audit.md",
        "└── vae_decode/",
        f"    └── {vae_contract.file_name}",
        "```",
        "",
        "## 完整推理数据流",
        "",
        "```text",
        "[Phase: text_encode — 每 prompt 一次]",
        "",
        f"  input_ids:[{b},{s_cap}]i64 + attention_mask:[{b},{s_cap}]i64",
        "        │",
        f"        ▼  text_embed_prepare_{text_tag}.onnx",
        f"  hidden + rope + attn_mask",
        "        │",
        f"        ▼  text_decoder_layer_repr_{text_tag}.onnx  (×{n_text_fwd})",
        f"  prompt_hidden:[{b},{s_cap},{text_hidden}]{wdtype.replace('bfloat16','bf16')}",
        "        │  (pipeline gather → patchify_and_embed cap 支，见 source_profile.json T2)",
        "        │",
        "[Phase: denoise — 源码 forward 顺序，见 transformer_z_image.py:938-1068]",
        "",
        f"  T1 adaln ← timestep_embed (t×t_scale={profile.t_scale})",
        f"  T2 patchify_and_embed (latent + cap_feats → x/cap patches + pos/pad)",
        f"  T3+T4 x_branch (embed + noise_refiner×{profile.n_refiner_layers})",
        f"  T5+T6 cap_branch (embed + context_refiner×{profile.n_refiner_layers})",
        f"  T7 concat → main_layer×{profile.n_layers} → final_output → noise_pred f32",
        "",
        f"  latent:[{b},{out_channels},1,{h_lat},{w_lat}]{wdtype.replace('bfloat16','bf16')}",
        f"  + prompt_hidden:[{b},{s_cap},{cap_feat_dim}]bf16",
        "        │",
        f"        ▼  patchify_and_embed_{img_tag}.onnx",
        f"        ├─► x_branch_{tag_x}.onnx  (embed + noise_refiner×{profile.n_refiner_layers} in-graph)",
        f"        └─► cap_branch_{text_tag}.onnx  (embed + context_refiner×{profile.n_refiner_layers} in-graph)",
        f"  timestep → adaln → x_branch / main / final_output_{img_tag}.onnx",
        "        │",
        f"        ▼  sequence_concat_basic_{tag_u}.onnx",
        f"  unified:[{b},{s_u},{dim}]bf16 + unified_rope",
        "        │",
        f"        ▼  main_layer_repr_{tag_u}.onnx  (×{profile.n_layers})",
        f"  noise_pred:[{b},{out_channels},{h_lat},{w_lat}]f32  → scheduler.step",
        "",
        "[Phase: vae_decode — 去噪结束后一次，latents f32]",
        "",
        f"  latents:[{b},{latent_channels},{h_lat},{w_lat}]f32",
        "        │",
        f"        ▼  {vae_contract.file_name}",
        f"  image:[{b},3,{scene.image_size},{scene.image_size}]bf16",
        "```",
        "",
        "## 约束符合性",
        "",
        "| # | 约束 | 状态 |",
        "|---|------|------|",
        "| 1 | 源码对齐（diffusers DiT + pipeline + transformers TE） | block 主体走源码模块；RoPE 用实数 Mul/Add 等价于源码 complex 路径 |",
        "| 2 | 真实 dtype | 与 pipeline/transformer 源码 cast 路径一致；边界自动校验 |",
        "| 3 | 语义拆分 + 代表层 | x/cap branch 图内 unroll refiner×2；main/text decoder 各导 layer[0]，host repeat 见 manifest |",
        "| 4 | 完整可拼接数据流 | denoise 主链 + text→cap 边界校验；scheduler 步间由 host 衔接 |",
        "| 5 | 权重 initializer 绑定算子；超大走 external data | 禁止默认 strip 为 graph input |",
        "| 6 | Custom op + 子图 | 不适用（无 RecurrentGatedDeltaRule 类强耦合循环结构） |",
        "",
        "## 代表层 repeat（manifest）",
        "",
        "| 子图 stem | 图内 unroll | host repeat | 权重来源 |",
        "|-----------|-------------|-------------|----------|",
        f"| `text_decoder_layer_repr_{text_tag}` | — | ×{n_text_fwd} | `Qwen3Model.layers[0]` |",
        f"| `x_branch_{tag_x}` | refiner ×{profile.n_refiner_layers} | ×1 | `noise_refiner[0]` |",
        f"| `cap_branch_{text_tag}` | refiner ×{profile.n_refiner_layers} | ×1 | `context_refiner[0]` |",
        f"| `main_layer_repr_{tag_u}` | — | ×{profile.n_layers} | `ZImageTransformer2DModel.layers[0]` |",
        "",
        "详见 `text_encode/text_encode_layer_manifest.json` 与 `denoise/denoise_layer_manifest.json`。",
        "",
        "## Host 侧职责（不在 ONNX 内）",
        "",
        "- Tokenizer → `input_ids` / `attention_mask`",
        "- Scheduler 循环（默认 28 步）、`timestep` 调度、`scheduler.step`",
        "- CFG：cond / uncond 两路 denoise（rollup 默认 ×2）",
        "- text→cap 对齐：pipeline gather / 截断到有效 caption token",
        "- VAE 后处理：normalize、转 uint8 等",
        "",
        "## 辅助产物",
        "",
        "| 文件 | 用途 |",
        "|------|------|",
        "| `source_profile.json` | 真实权重 forward trace，I/O 契约 ground truth |",
        "| `*/onnx_flow_stats_multi.{xlsx,json}` | 各 phase 算子/MACs 统计 |",
        "| `*/*.flow_stats.summary.json` | 单图 MACs 摘要（e2e rollup 输入） |",
        "| `z_image_e2e_rollup.{md,json}` | text + denoise×步×CFG + vae 端到端 MACs |",
        "| `denoise/source_alignment_audit.md` | 导出 wrapper 与 diffusers 源码对照 |",
        "",
        "## 子图 I/O 契约",
        "",
    ]

    for contract in text_contracts:
        repeat = f" (×{contract.repeat})" if contract.repeat > 1 else ""
        notes = f" — {contract.notes}" if contract.notes else ""
        lines.append(f"- **{contract.file_name}**{repeat}: {contract.source_symbol}{notes}")
        lines.append(f"  - {contract_io_summary(contract)}")

    for contract in denoise_contracts:
        repeat = f" (×{contract.repeat})" if contract.repeat > 1 else ""
        notes = f" — {contract.notes}" if contract.notes else ""
        lines.append(f"- **{contract.file_name}**{repeat}: {contract.source_symbol}{notes}")
        lines.append(f"  - {contract_io_summary(contract)}")

    lines.append(f"- **{vae_contract.file_name}**: {vae_contract.source_symbol}")
    lines.append(f"  - {contract_io_summary(vae_contract)}")

    content = "\n".join(lines)
    readme_path = out_root / "README.md"
    readme_path.write_text(content, encoding="utf-8")
    return str(readme_path)
