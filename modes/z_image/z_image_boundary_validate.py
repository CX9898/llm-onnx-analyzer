"""Validate adjacent exported ONNX graphs: shape + dtype must match source boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import onnx

_TYPE = {1: "float32", 6: "int32", 7: "int64", 9: "bool", 10: "float16", 16: "bfloat16"}


@dataclass(frozen=True)
class EdgeSpec:
    upstream: str
    downstream: str
    out_idx: int = 0
    in_idx: int = 0
    note: str = ""


def _io_tensor(path: Path, *, is_output: bool, index: int) -> tuple[str, list, str]:
    model = onnx.load(str(path))
    item = model.graph.output[index] if is_output else model.graph.input[index]
    shape = [d.dim_value if d.dim_value else "?" for d in item.type.tensor_type.shape.dim]
    dtype = _TYPE.get(item.type.tensor_type.elem_type, str(item.type.tensor_type.elem_type))
    return item.name, shape, dtype


DEFAULT_EDGES: tuple[EdgeSpec, ...] = (
    EdgeSpec("text_encode/text_decoder_layer_repr_cap128.onnx", "denoise/patchify_and_embed_img512.onnx", 0, 1, "text→patchify cap"),
    EdgeSpec("text_encode/text_embed_prepare_cap128.onnx", "text_encode/text_decoder_layer_repr_cap128.onnx", 0, 0, "embed→layer hidden"),
    EdgeSpec("text_encode/text_embed_prepare_cap128.onnx", "text_encode/text_decoder_layer_repr_cap128.onnx", 1, 1, "embed→layer rope_cos"),
    EdgeSpec("text_encode/text_embed_prepare_cap128.onnx", "text_encode/text_decoder_layer_repr_cap128.onnx", 2, 2, "embed→layer rope_sin"),
    EdgeSpec("text_encode/text_embed_prepare_cap128.onnx", "text_encode/text_decoder_layer_repr_cap128.onnx", 3, 3, "embed→layer attn_mask"),
    EdgeSpec("denoise/patchify_and_embed_img512.onnx", "denoise/x_branch_seq1k.onnx", 0, 0, "patchify→x_branch"),
    EdgeSpec("denoise/patchify_and_embed_img512.onnx", "denoise/x_branch_seq1k.onnx", 2, 1, "x_pos_ids"),
    EdgeSpec("denoise/patchify_and_embed_img512.onnx", "denoise/x_branch_seq1k.onnx", 4, 2, "x_pad_mask"),
    EdgeSpec("denoise/patchify_and_embed_img512.onnx", "denoise/cap_branch_cap128.onnx", 1, 0, "patchify→cap_branch"),
    EdgeSpec("denoise/patchify_and_embed_img512.onnx", "denoise/cap_branch_cap128.onnx", 3, 1, "cap_pos_ids"),
    EdgeSpec("denoise/patchify_and_embed_img512.onnx", "denoise/cap_branch_cap128.onnx", 5, 2, "cap_pad_mask"),
    EdgeSpec("denoise/timestep_embed.onnx", "denoise/x_branch_seq1k.onnx", 0, 3, "adaln→x_branch"),
    EdgeSpec("denoise/x_branch_seq1k.onnx", "denoise/sequence_concat_basic_seq1152.onnx", 0, 0, "x_tokens→concat"),
    EdgeSpec("denoise/x_branch_seq1k.onnx", "denoise/sequence_concat_basic_seq1152.onnx", 1, 2, "x_rope_cos→concat"),
    EdgeSpec("denoise/x_branch_seq1k.onnx", "denoise/sequence_concat_basic_seq1152.onnx", 2, 3, "x_rope_sin→concat"),
    EdgeSpec("denoise/cap_branch_cap128.onnx", "denoise/sequence_concat_basic_seq1152.onnx", 0, 1, "cap_tokens→concat"),
    EdgeSpec("denoise/cap_branch_cap128.onnx", "denoise/sequence_concat_basic_seq1152.onnx", 1, 4, "cap_rope_cos→concat"),
    EdgeSpec("denoise/cap_branch_cap128.onnx", "denoise/sequence_concat_basic_seq1152.onnx", 2, 5, "cap_rope_sin→concat"),
    EdgeSpec("denoise/sequence_concat_basic_seq1152.onnx", "denoise/main_layer_repr_seq1152.onnx", 0, 0),
    EdgeSpec("denoise/main_layer_repr_seq1152.onnx", "denoise/final_output_img512.onnx", 0, 0),
    EdgeSpec("denoise/timestep_embed.onnx", "denoise/main_layer_repr_seq1152.onnx", 0, 4),
    EdgeSpec("denoise/timestep_embed.onnx", "denoise/final_output_img512.onnx", 0, 1),
)


def validate_onnx_directory(root: Path, edges: tuple[EdgeSpec, ...] = DEFAULT_EDGES) -> list[str]:
    errors: list[str] = []
    for edge in edges:
        up = root / edge.upstream
        down = root / edge.downstream
        if not up.is_file():
            errors.append(f"missing upstream {up}")
            continue
        if not down.is_file():
            errors.append(f"missing downstream {down}")
            continue
        on, os_, ot = _io_tensor(up, is_output=True, index=edge.out_idx)
        inn, is_, it = _io_tensor(down, is_output=False, index=edge.in_idx)
        if os_ != is_ or ot != it:
            label = edge.note or f"{edge.upstream} → {edge.downstream}"
            errors.append(f"{label}: {on}{os_}{ot} -> {inn}{is_}{it}")

    final_out = root / "denoise/final_output_img512.onnx"
    if final_out.is_file():
        _, shape, dtype = _io_tensor(final_out, is_output=True, index=0)
        if dtype != "float32":
            errors.append(f"final_output noise_pred must be float32, got {dtype}")
        if len(shape) != 4:
            errors.append(f"final_output noise_pred must be [B,C,H,W], got {shape}")

    return errors
