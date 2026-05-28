"""Export Z-Image DiT denoise subgraphs with full data-flow coverage."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from z_image_export_semantics import (
    ADALN_EMBED_DIM,
    PATCH_KEY,
    ExportScene,
    default_strip_initializers,
    load_source_profile,
)
from z_image_export_shared import (
    cap_scene_tag,
    img_scene_tag,
    load_transformer,
    make_attn_mask,
    make_rope_cos_sin,
    model_dtype,
    onnx_export,
    patch_feature_dim,
    seq_scene_tag,
)
from z_image_onnx_rope import AttentionStaticShape
from z_image_onnx_blocks import (
    CapBranchBlock,
    FinalOutputBlock,
    PatchifyAndEmbedBasicBlock,
    SequenceConcatBasicBlock,
    XBranchBlock,
    ZImageTransformerBlockExportBlock,
    TimestepEmbedBlock,
)
from z_image_shape_propagation import validate_denoise_chain


def _export(
    module,
    dummy_inputs,
    save_path: str,
    *,
    input_names: list[str],
    output_names: list[str],
    kind: str,
    opset: int,
    simplify: bool,
    strip_initializers: bool | None,
) -> str:
    strip = default_strip_initializers(kind, cli_override=strip_initializers)
    onnx_export(
        module,
        dummy_inputs,
        save_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={},
        opset=opset,
        simplify=simplify,
        strip_initializers=strip,
    )
    return save_path


def _make_pos_ids(transformer, seq_len: int, device: torch.device) -> torch.Tensor:
    grid_side = int(seq_len**0.5)
    if grid_side * grid_side < seq_len:
        grid_side += 1
    pos_ids = transformer.create_coordinate_grid(
        size=(1, grid_side, grid_side),
        start=(0, 0, 0),
        device=device,
    ).flatten(0, 2)
    if pos_ids.shape[0] < seq_len:
        pad = (
            transformer.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device)
            .flatten(0, 2)
            .repeat(seq_len - pos_ids.shape[0], 1)
        )
        pos_ids = torch.cat([pos_ids, pad], dim=0)
    return pos_ids[:seq_len]


def export_patchify_and_embed(
    transformer,
    out_dir: str,
    scene: ExportScene,
    *,
    opset: int = 20,
    simplify: bool = True,
    strip_initializers: bool | None = None,
) -> str:
    block = PatchifyAndEmbedBasicBlock(transformer, scene.patch_size, scene.f_patch_size)
    dtype = model_dtype(transformer)
    device = next(transformer.parameters()).device
    h_lat, w_lat = scene.latent_hw()
    c = transformer.config.in_channels
    b = scene.batch_size
    latent = torch.randn(b, c, 1, h_lat, w_lat, device=device, dtype=dtype)
    cap = torch.randn(b, scene.cap_seq, transformer.config.cap_feat_dim, device=device, dtype=dtype)
    save_path = os.path.join(out_dir, f"patchify_and_embed_{img_scene_tag(scene.image_size)}.onnx")
    return _export(
        block,
        (latent, cap),
        save_path,
        input_names=["latent", "cap_feats"],
        output_names=[
            "x_patch_feats",
            "cap_feats_padded",
            "x_pos_ids",
            "cap_pos_ids",
            "x_pad_mask",
            "cap_pad_mask",
        ],
        kind="patchify_and_embed",
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
    )


def export_timestep_embed(
    transformer,
    out_dir: str,
    scene: ExportScene,
    *,
    opset: int = 20,
    simplify: bool = True,
    strip_initializers: bool | None = None,
) -> str:
    dtype = model_dtype(transformer)
    block = TimestepEmbedBlock(transformer.t_embedder, transformer.t_scale, dtype)
    device = next(transformer.parameters()).device
    t = torch.linspace(0.9, 0.1, scene.batch_size, device=device, dtype=torch.float32)
    save_path = os.path.join(out_dir, "timestep_embed.onnx")
    return _export(
        block,
        (t,),
        save_path,
        input_names=["timestep"],
        output_names=["adaln_input"],
        kind="timestep_embed",
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
    )


def export_x_branch(
    transformer,
    out_dir: str,
    scene: ExportScene,
    *,
    opset: int = 20,
    simplify: bool = True,
    strip_initializers: bool | None = None,
) -> str:
    x_embedder = transformer.all_x_embedder[PATCH_KEY]
    s_x = scene.image_token_seq_len()
    block = XBranchBlock(
        transformer,
        x_embedder,
        batch_size=scene.batch_size,
        seq_len=s_x,
        n_refiner_layers=transformer.config.n_refiner_layers,
    )
    dtype = model_dtype(transformer)
    device = next(transformer.parameters()).device
    patch_dim = patch_feature_dim(transformer.config.in_channels)
    patches = torch.randn(scene.batch_size, s_x, patch_dim, device=device, dtype=dtype)
    pos_ids = _make_pos_ids(transformer, s_x, device).unsqueeze(0).expand(scene.batch_size, -1, -1)
    pad_mask = torch.zeros(scene.batch_size, s_x, dtype=torch.bool, device=device)
    adaln = torch.randn(scene.batch_size, ADALN_EMBED_DIM, device=device, dtype=dtype)
    tag = seq_scene_tag(s_x)
    save_path = os.path.join(out_dir, f"x_branch_{tag}.onnx")
    return _export(
        block,
        (patches, pos_ids, pad_mask, adaln),
        save_path,
        input_names=["x_patch_feats", "x_pos_ids", "x_pad_mask", "adaln_input"],
        output_names=["x_tokens", "x_rope_cos", "x_rope_sin", "x_attn_mask"],
        kind="x_branch",
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
    )


def export_cap_branch(
    transformer,
    out_dir: str,
    scene: ExportScene,
    *,
    opset: int = 20,
    simplify: bool = True,
    strip_initializers: bool | None = None,
) -> str:
    block = CapBranchBlock(
        transformer,
        transformer.cap_embedder,
        batch_size=scene.batch_size,
        seq_len=scene.cap_seq,
        n_refiner_layers=transformer.config.n_refiner_layers,
    )
    dtype = model_dtype(transformer)
    device = next(transformer.parameters()).device
    cap = torch.randn(scene.batch_size, scene.cap_seq, transformer.config.cap_feat_dim, device=device, dtype=dtype)
    pos_ids = torch.zeros(scene.batch_size, scene.cap_seq, 3, dtype=torch.int32, device=device)
    pad_mask = torch.zeros(scene.batch_size, scene.cap_seq, dtype=torch.bool, device=device)
    tag = cap_scene_tag(scene.cap_seq)
    save_path = os.path.join(out_dir, f"cap_branch_{tag}.onnx")
    return _export(
        block,
        (cap, pos_ids, pad_mask),
        save_path,
        input_names=["cap_feats_padded", "cap_pos_ids", "cap_pad_mask"],
        output_names=["cap_tokens", "cap_rope_cos", "cap_rope_sin", "cap_attn_mask"],
        kind="cap_branch",
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
    )


def export_final_output(
    transformer,
    out_dir: str,
    scene: ExportScene,
    *,
    opset: int = 20,
    simplify: bool = True,
    strip_initializers: bool | None = None,
) -> str:
    final = transformer.all_final_layer[PATCH_KEY]
    h_lat, w_lat = scene.latent_hw()
    block = FinalOutputBlock(
        final,
        batch_size=scene.batch_size,
        unified_seq_len=scene.unified_seq_len(),
        image_seq_len=scene.image_token_seq_len(),
        patch_dim=patch_feature_dim(transformer.config.in_channels),
        out_channels=transformer.config.in_channels,
        f_size=1,
        h_size=h_lat,
        w_size=w_lat,
        patch_size=scene.patch_size,
        f_patch_size=scene.f_patch_size,
    )
    dtype = model_dtype(transformer)
    device = next(transformer.parameters()).device
    s_u = scene.unified_seq_len()
    hidden = torch.randn(scene.batch_size, s_u, transformer.dim, device=device, dtype=dtype)
    adaln = torch.randn(scene.batch_size, ADALN_EMBED_DIM, device=device, dtype=dtype)
    save_path = os.path.join(out_dir, f"final_output_{img_scene_tag(scene.image_size)}.onnx")
    return _export(
        block,
        (hidden, adaln),
        save_path,
        input_names=["hidden_states", "adaln_input"],
        output_names=["noise_pred"],
        kind="final_output",
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
    )


def _prune_stale_denoise_artifacts(out_dir: str, keep_basenames: list[str]) -> None:
    """Remove ONNX + flow-stats sidecars not in the current export set."""
    keep = set(keep_basenames)
    root = Path(out_dir)
    for path in root.iterdir():
        if not path.is_file():
            continue
        if path.suffix == ".onnx" and path.name not in keep:
            path.unlink(missing_ok=True)
            for suffix in (".flow_stats.summary.json", ".flow_stats.tsv"):
                root.joinpath(f"{path.stem}{suffix}").unlink(missing_ok=True)


def export_transformer_block_repr(
    transformer,
    out_dir: str,
    scene: ExportScene,
    *,
    block_kind: str,
    layer_idx: int = 0,
    seq_len: int | None = None,
    with_adaln: bool = True,
    opset: int = 20,
    simplify: bool = True,
    strip_initializers: bool | None = None,
) -> str:
    if block_kind == "noise_refiner":
        block_src = transformer.noise_refiner[layer_idx]
        prefix = "noise_refiner_block_repr"
    elif block_kind == "context_refiner":
        block_src = transformer.context_refiner[layer_idx]
        prefix = "context_refiner_block_repr"
    elif block_kind == "main_layer":
        block_src = transformer.layers[layer_idx]
        prefix = "main_layer_repr"
    else:
        raise ValueError(f"unknown block_kind {block_kind!r}")

    s = seq_len or (scene.unified_seq_len() if block_kind == "main_layer" else scene.image_token_seq_len())
    if block_kind == "context_refiner":
        s = seq_len or scene.cap_seq

    wrapper = ZImageTransformerBlockExportBlock(
        block_src,
        AttentionStaticShape.from_transformer(transformer, scene.batch_size, s),
    )
    dtype = model_dtype(transformer)
    device = next(transformer.parameters()).device
    hidden = torch.randn(scene.batch_size, s, transformer.dim, device=device, dtype=dtype)
    attn_mask = make_attn_mask(scene.batch_size, s, device)
    rope_cos, rope_sin = make_rope_cos_sin(transformer, scene.batch_size, s, device)
    if block_kind == "context_refiner":
        tag = cap_scene_tag(s)
    elif block_kind == "main_layer":
        tag = seq_scene_tag(s)
    else:
        tag = seq_scene_tag(s)
    save_path = os.path.join(out_dir, f"{prefix}_{tag}.onnx")

    if with_adaln and block_src.modulation:
        adaln = torch.randn(scene.batch_size, ADALN_EMBED_DIM, device=device, dtype=dtype)
        inputs = (hidden, attn_mask, rope_cos, rope_sin, adaln)
        input_names = ["hidden_states", "attn_mask", "rope_cos", "rope_sin", "adaln_input"]
    else:
        inputs = (hidden, attn_mask, rope_cos, rope_sin)
        input_names = ["hidden_states", "attn_mask", "rope_cos", "rope_sin"]

    kind: str = {
        "noise_refiner": "noise_refiner_block",
        "context_refiner": "context_refiner_block",
        "main_layer": "main_layer_repr",
    }[block_kind]

    return _export(
        wrapper,
        inputs,
        save_path,
        input_names=input_names,
        output_names=["output"],
        kind=kind,
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
    )


def export_sequence_concat(
    transformer,
    out_dir: str,
    scene: ExportScene,
    *,
    opset: int = 20,
    simplify: bool = True,
    strip_initializers: bool | None = None,
) -> str:
    block = SequenceConcatBasicBlock()
    dtype = model_dtype(transformer)
    device = next(transformer.parameters()).device
    b = scene.batch_size
    s_x = scene.image_token_seq_len()
    s_cap = scene.cap_seq
    dim = transformer.dim
    x_tokens = torch.randn(b, s_x, dim, device=device, dtype=dtype)
    cap_tokens = torch.randn(b, s_cap, dim, device=device, dtype=dtype)
    x_cos, x_sin = make_rope_cos_sin(transformer, b, s_x, device)
    cap_cos, cap_sin = make_rope_cos_sin(transformer, b, s_cap, device)
    tag = seq_scene_tag(scene.unified_seq_len())
    save_path = os.path.join(out_dir, f"sequence_concat_basic_{tag}.onnx")
    return _export(
        block,
        (x_tokens, cap_tokens, x_cos, x_sin, cap_cos, cap_sin),
        save_path,
        input_names=[
            "x_tokens",
            "cap_tokens",
            "x_rope_cos",
            "x_rope_sin",
            "cap_rope_cos",
            "cap_rope_sin",
        ],
        output_names=["unified_tokens", "unified_rope_cos", "unified_rope_sin"],
        kind="sequence_concat",
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
    )


def export_denoise_bundle(
    model_path: str,
    out_dir: str,
    *,
    scene: ExportScene | None = None,
    opset: int = 20,
    simplify: bool = True,
    strip_initializers: bool | None = None,
) -> list[str]:
    scene = scene or ExportScene()
    os.makedirs(out_dir, exist_ok=True)
    profile = load_source_profile(model_path, scene)
    transformer = load_transformer(model_path)

    validation = validate_denoise_chain(scene, profile)
    if not validation.ok:
        raise RuntimeError("Denoise chain validation failed:\n  " + "\n  ".join(validation.errors))

    paths = [
        export_patchify_and_embed(transformer, out_dir, scene, opset=opset, simplify=simplify, strip_initializers=strip_initializers),
        export_timestep_embed(transformer, out_dir, scene, opset=opset, simplify=simplify, strip_initializers=strip_initializers),
        export_x_branch(transformer, out_dir, scene, opset=opset, simplify=simplify, strip_initializers=strip_initializers),
        export_cap_branch(transformer, out_dir, scene, opset=opset, simplify=simplify, strip_initializers=strip_initializers),
        export_sequence_concat(transformer, out_dir, scene, opset=opset, simplify=simplify, strip_initializers=strip_initializers),
        export_transformer_block_repr(
            transformer, out_dir, scene, block_kind="main_layer", opset=opset, simplify=simplify, strip_initializers=strip_initializers
        ),
        export_final_output(transformer, out_dir, scene, opset=opset, simplify=simplify, strip_initializers=strip_initializers),
    ]

    export_names = [os.path.basename(p) for p in paths]
    _prune_stale_denoise_artifacts(out_dir, export_names)

    meta = {
        "scene": {
            "batch_size": scene.batch_size,
            "image_size": scene.image_size,
            "cap_seq": scene.cap_seq,
            "image_seq_len": scene.image_token_seq_len(),
            "unified_seq_len": scene.unified_seq_len(),
        },
        "source_alignment": {
            "dit": "diffusers/models/transformers/transformer_z_image.py",
            "pipeline": "diffusers/pipelines/z_image/pipeline_z_image.py",
        },
        "exports": [os.path.basename(p) for p in paths],
        "chain_validation": "passed",
    }
    with open(os.path.join(out_dir, "denoise_export_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    n_ref = transformer.config.n_refiner_layers
    manifest = {
        "main_layer_repr": {
            "repeat": transformer.config.n_layers,
            "source": "ZImageTransformer2DModel.layers[0]",
        },
        "x_branch": {
            "repeat": 1,
            "refiner_unrolled_in_graph": n_ref,
            "source": f"embed_prepare + noise_refiner[0] ×{n_ref} (in-graph)",
        },
        "cap_branch": {
            "repeat": 1,
            "refiner_unrolled_in_graph": n_ref,
            "source": f"embed_prepare + context_refiner[0] ×{n_ref} (in-graph)",
        },
    }
    with open(os.path.join(out_dir, "denoise_layer_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return paths
