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
    n_text_fwd = profile.n_text_forward_layers

    lines = [
        f"# Z-Image ONNX 说明：`{out_root.name}`",
        "",
        "代表场景：Basic 文生图，`512×512` 输入，`cap_seq=128`，`batch=1`。",
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
        "        ▼  patchify_and_embed_img512.onnx",
        "        ├─► x_branch_seq1k.onnx  (embed + noise_refiner×2 in-graph)",
        "        └─► cap_branch_cap128.onnx  (embed + context_refiner×2 in-graph)",
        "  timestep → adaln → x_branch / main / final_output_img512.onnx",
        "        │",
        "        ▼  sequence_concat_basic_seq1152.onnx",
        f"  unified:[{b},{s_u},{dim}]bf16 + unified_rope",
        "        │",
        f"        ▼  main_layer_repr_seq1152.onnx  (×{profile.n_layers})",
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
        "| 3 | 语义拆分 + 代表层 | refiner/main 各导 1 份，repeat 见 manifest |",
        "| 4 | 完整可拼接数据流 | denoise 主链 + text→cap 边界校验；scheduler 步间由 host 衔接 |",
        "| 5 | 权重 initializer 绑定算子；超大走 external data | 禁止默认 strip 为 graph input |",
        "| 6 | Custom op + 子图 | 不适用（无 RecurrentGatedDeltaRule 类强耦合循环结构） |",
        "",
        "## 子图 I/O 契约",
        "",
    ]

    for contract in text_contracts:
        repeat = f" (×{contract.repeat})" if contract.repeat > 1 else ""
        lines.append(f"- **{contract.file_name}**{repeat}: {contract.source_symbol}")
        lines.append(f"  - {contract_io_summary(contract)}")

    for contract in denoise_contracts:
        repeat = f" (×{contract.repeat})" if contract.repeat > 1 else ""
        lines.append(f"- **{contract.file_name}**{repeat}: {contract.source_symbol}")
        lines.append(f"  - {contract_io_summary(contract)}")

    content = "\n".join(lines)
    readme_path = out_root / "README.md"
    readme_path.write_text(content, encoding="utf-8")
    return str(readme_path)
