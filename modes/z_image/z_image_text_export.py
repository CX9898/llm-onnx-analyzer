"""Export Z-Image text encoder as stitchable ONNX subgraphs."""

from __future__ import annotations

import json
import os

import torch

from z_image_export_semantics import ExportScene, default_strip_initializers
from z_image_export_shared import cap_scene_tag, load_text_encoder, onnx_export
from z_image_text_onnx_blocks import (
    TextDecoderLayerExportBlock,
    TextEmbedPrepareBlock,
    TextEncodeStaticShape,
)


def _export(
    module,
    dummy_inputs: tuple,
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


def export_text_embed_prepare(
    text_encoder,
    out_dir: str,
    scene: ExportScene,
    static: TextEncodeStaticShape,
    *,
    opset: int = 20,
    simplify: bool = True,
    strip_initializers: bool | None = None,
) -> str:
    block = TextEmbedPrepareBlock(text_encoder, static)
    device = next(text_encoder.parameters()).device
    vocab = text_encoder.config.vocab_size
    input_ids = torch.randint(1, vocab, (scene.batch_size, scene.cap_seq), device=device, dtype=torch.long)
    attention_mask = torch.ones(scene.batch_size, scene.cap_seq, device=device, dtype=torch.long)
    tag = cap_scene_tag(scene.cap_seq)
    save_path = os.path.join(out_dir, f"text_embed_prepare_{tag}.onnx")
    return _export(
        block,
        (input_ids, attention_mask),
        save_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["hidden_states", "rope_cos", "rope_sin", "attn_mask"],
        kind="text_embed_prepare",
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
    )


def export_text_decoder_layer_repr(
    text_encoder,
    out_dir: str,
    scene: ExportScene,
    static: TextEncodeStaticShape,
    *,
    layer_index: int = 0,
    opset: int = 20,
    simplify: bool = True,
    strip_initializers: bool | None = None,
) -> str:
    layer = text_encoder.layers[layer_index]
    block = TextDecoderLayerExportBlock(layer, static)
    device = next(text_encoder.parameters()).device
    dtype = next(text_encoder.parameters()).dtype
    b, s, h = scene.batch_size, scene.cap_seq, static.hidden_size
    rope_dim = static.head_dim
    hidden_states = torch.randn(b, s, h, device=device, dtype=dtype)
    rope_cos = torch.randn(b, s, rope_dim, device=device, dtype=dtype)
    rope_sin = torch.randn(b, s, rope_dim, device=device, dtype=dtype)
    attn_mask = torch.zeros(b, 1, s, s, device=device, dtype=dtype)
    tag = cap_scene_tag(scene.cap_seq)
    save_path = os.path.join(out_dir, f"text_decoder_layer_repr_{tag}.onnx")
    return _export(
        block,
        (hidden_states, rope_cos, rope_sin, attn_mask),
        save_path,
        input_names=["hidden_states", "rope_cos", "rope_sin", "attn_mask"],
        output_names=["hidden_states_out"],
        kind="text_decoder_layer_repr",
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
    )


def _prune_stale_text_artifacts(out_dir: str, keep_names: set[str]) -> None:
    root = os.path.join(out_dir)
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        if name.endswith(".onnx") and name not in keep_names:
            os.remove(path)


def export_text_encode(
    model_path: str,
    out_dir: str,
    scene: ExportScene,
    *,
    opset: int = 20,
    simplify: bool = True,
    strip_initializers: bool | None = None,
) -> list[str]:
    """Export text_encode as embed_prepare + decoder_layer_repr (host 拼接 ×N)。"""
    os.makedirs(out_dir, exist_ok=True)
    text_encoder = load_text_encoder(model_path)
    static = TextEncodeStaticShape.from_model(text_encoder, scene.batch_size, scene.cap_seq)
    n_hidden = int(text_encoder.config.num_hidden_layers)
    n_forward = n_hidden - 1  # pipeline: hidden_states[-2] → layers[0..n_hidden-2]

    paths = [
        export_text_embed_prepare(
            text_encoder,
            out_dir,
            scene,
            static,
            opset=opset,
            simplify=False,
            strip_initializers=strip_initializers,
        ),
        export_text_decoder_layer_repr(
            text_encoder,
            out_dir,
            scene,
            static,
            layer_index=0,
            opset=opset,
            simplify=simplify,
            strip_initializers=strip_initializers,
        ),
    ]

    export_names = [os.path.basename(p) for p in paths]
    _prune_stale_text_artifacts(out_dir, set(export_names))

    meta = {
        "scene": {
            "batch_size": scene.batch_size,
            "cap_seq": scene.cap_seq,
        },
        "source_alignment": {
            "text_encoder": "pipeline_z_image.py:_encode_prompt → hidden_states[-2]",
            "block": "TextEmbedPrepareBlock + TextDecoderLayerExportBlock (layer 0 repr)",
        },
        "exports": export_names,
        "chain": [
            "text_embed_prepare: input_ids, attention_mask → hidden_states, rope_cos, rope_sin, attn_mask",
            f"text_decoder_layer_repr ×{n_forward}: hidden_states, rope_* , attn_mask → hidden_states_out",
            "final prompt_hidden = hidden_states_out after last layer repeat",
        ],
    }
    with open(os.path.join(out_dir, "text_encode_export_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    manifest = {
        "text_decoder_layer_repr": {
            "repeat": n_forward,
            "source": "Qwen3Model.layers[0] (structure repr; MACs rollup ×repeat)",
            "notes": f"Run layers 0..{n_forward - 1} in source; export uses layer 0 weights for topology.",
        },
    }
    with open(os.path.join(out_dir, "text_encode_layer_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return paths
