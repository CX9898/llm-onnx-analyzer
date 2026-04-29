"""
ONNX export wrappers for the Qwen3.5-MoE *multimodal flow* — the bookkeeping
that lives in ``Qwen3_5MoeModel.forward`` (``modeling_qwen3_5_moe.py`` lines
1738-1815) between ``inputs_embeds`` and the language-model entry, plus
the M-RoPE 3D ``position_ids`` constructor that drives the text decoder's
rotary embedding.

Exported files
--------------

  - ``image_mask_build_<seq>.onnx``                  ``input_ids ==
                                                    image_token_id`` →
                                                    image_mask broadcast
                                                    to ``inputs_embeds``
                                                    shape (source line
                                                    1667 + 1671).
  - ``mm_inject_<seq>.onnx``                         Source-verbatim
                                                    ``inputs_embeds.masked_scatter(
                                                    image_mask, image_embeds)``
                                                    (source line 1773).
                                                    Directly consumes the
                                                    ``image_mask`` produced
                                                    by ``image_mask_build_*``,
                                                    forming a fully connected
                                                    ONNX-level data flow.
                                                    Carries a few ``unk__N``
                                                    placeholders from the
                                                    exporter's
                                                    ``NonZero -> Transpose
                                                    -> ScatterND`` lowering,
                                                    matching the source's
                                                    inherent
                                                    ``masked_scatter``
                                                    semantics.
  - ``mrope_position_ids_prefill_<seq>.onnx``        Statically-unrolled
                                                    M-RoPE 3D ``position_ids``
                                                    constructor for the
                                                    representative
                                                    ``[text_pre | image |
                                                    text_post]`` layout
                                                    (source line 1707
                                                    ``if`` branch +
                                                    ``get_rope_index``).
  - ``mrope_position_ids_decode_ctx<N>.onnx``        Decode-step M-RoPE
                                                    ``position_ids`` builder
                                                    (source line 1720
                                                    ``elif`` branch).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_EXPORT_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_EXPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPORT_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qwen_export_shared import (  # noqa: E402
    Qwen3_5MoeModelLike,
    _model_float_dtype,
    _onnx_export,
    _seq_tag,
    _text_config,
    _vision_config,
)
from qwen_onnx_blocks_mm import (  # noqa: E402
    ImageMaskBuildBlock,
    MMInjectBlock,
    MRoPEPositionIdsDecodeBlock,
    MRoPEPositionIdsPrefillBlock,
)


# ---------------------------------------------------------------------------
# 1. image_mask_build (Qwen3_5MoeModel.get_placeholder_mask, image branch)
# ---------------------------------------------------------------------------

def export_image_mask_build(
    model: Qwen3_5MoeModelLike,
    out_dir: str,
    batch_size: int,
    opset: int,
    simplify: bool,
    seq_len: int,
    *,
    fold_pure_shape_chains: bool = False,
    strip_initializers: bool = False,
) -> None:
    text_cfg = _text_config(model)
    text_hidden = int(text_cfg.hidden_size)
    block = ImageMaskBuildBlock(hidden_size=text_hidden)

    image_token_id_default = int(getattr(model.config, "image_token_id", 0))
    sample_inputs = (
        torch.zeros(batch_size, seq_len, dtype=torch.int64),
        torch.tensor(image_token_id_default, dtype=torch.int64),
    )
    save_path = os.path.join(
        out_dir,
        f"image_mask_build_{_seq_tag(seq_len)}.onnx",
    )
    _onnx_export(
        block,
        sample_inputs,
        save_path,
        input_names=["input_ids", "image_token_id"],
        output_names=["image_mask"],
        dynamic_axes={},
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
        fold_pure_shape_chains=fold_pure_shape_chains,
    )


# ---------------------------------------------------------------------------
# 2. mm_inject (masked_scatter image_embeds into inputs_embeds)
# ---------------------------------------------------------------------------

def export_mm_inject(
    model: Qwen3_5MoeModelLike,
    out_dir: str,
    batch_size: int,
    opset: int,
    simplify: bool,
    seq_len: int,
    *,
    image_token_count: int,
    fold_pure_shape_chains: bool = False,
    strip_initializers: bool = False,
) -> None:
    """
    Export the image-embedding injection graph (source line 1773
    ``masked_scatter``), with the source-aligned IO contract:

        inputs_embeds : (B, S, H_text)         float
        image_mask    : (B, S, H_text)         bool   ← from ImageMaskBuildBlock
        image_embeds  : (N_image_tokens, H_text)   float

    The exported graph is the source's ``masked_scatter`` semantics
    verbatim. Because PyTorch's ONNX exporter lowers ``masked_scatter``
    to ``NonZero -> Transpose -> ScatterND``, the file carries a few
    ``unk__N`` placeholders (the runtime count of True positions in
    ``image_mask`` is a data-dependent dimension that ONNX static shape
    inference cannot resolve). This matches the source's intrinsic
    semantics — the same kind of inherent unknown that
    ``vision_cu_seqlens_*.onnx`` carries from
    ``torch.repeat_interleave``.

    The dummy ``image_mask`` is constructed to set the first
    ``image_token_count`` rows fully True (a row-uniform layout exactly
    matching what ``ImageMaskBuildBlock`` produces for the
    representative scenario), so the trace records the correct branch.
    """
    text_cfg = _text_config(model)
    text_hidden = int(text_cfg.hidden_size)
    dtype = _model_float_dtype(model)

    total_positions = batch_size * seq_len
    if image_token_count > total_positions:
        raise ValueError(
            f"image_token_count={image_token_count} exceeds total positions "
            f"{batch_size}*{seq_len}={total_positions}"
        )

    # Source-aligned dummy ``image_mask``: row-uniform (every selected
    # ``(b, s)`` row is fully True across the hidden dim), exactly as
    # ``ImageMaskBuildBlock``'s ``Equal -> Unsqueeze -> Expand`` produces.
    image_mask = torch.zeros(batch_size, seq_len, text_hidden, dtype=torch.bool)
    image_mask[:, :image_token_count, :] = True

    sample_inputs = (
        torch.randn(batch_size, seq_len, text_hidden, dtype=dtype),
        image_mask,
        torch.randn(image_token_count, text_hidden, dtype=dtype),
    )
    save_path = os.path.join(out_dir, f"mm_inject_{_seq_tag(seq_len)}.onnx")
    block = MMInjectBlock()
    _onnx_export(
        block,
        sample_inputs,
        save_path,
        input_names=["inputs_embeds", "image_mask", "image_embeds"],
        output_names=["inputs_embeds_out"],
        dynamic_axes={},
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
        fold_pure_shape_chains=fold_pure_shape_chains,
    )


# ---------------------------------------------------------------------------
# 3. mrope_position_ids_prefill (statically-unrolled get_rope_index)
# ---------------------------------------------------------------------------

def export_mrope_position_ids_prefill(
    model: Qwen3_5MoeModelLike,
    out_dir: str,
    batch_size: int,
    opset: int,
    simplify: bool,
    seq_len: int,
    *,
    text_pre_len: int,
    image_token_count: int,
    image_grid_t: int = 1,
    image_grid_h: int,
    image_grid_w: int,
    static_grid: bool = False,
    fold_pure_shape_chains: bool = False,
    strip_initializers: bool = False,
) -> None:
    vcfg = _vision_config(model)
    spatial_merge_size = int(vcfg.spatial_merge_size)

    block = MRoPEPositionIdsPrefillBlock(
        batch_size=batch_size,
        seq_len=seq_len,
        text_pre_len=text_pre_len,
        spatial_merge_size=spatial_merge_size,
        static_grid=(image_grid_t, image_grid_h, image_grid_w) if static_grid else None,
    )
    expected_image = (image_grid_h // spatial_merge_size) * (
        image_grid_w // spatial_merge_size
    )
    if expected_image != image_token_count:
        raise ValueError(
            "image_token_count does not match (H // merge) * (W // merge): "
            f"expected {expected_image}, got {image_token_count}",
        )
    if text_pre_len + image_token_count > seq_len:
        raise ValueError(
            "text_pre_len + image_token_count exceeds seq_len: "
            f"{text_pre_len} + {image_token_count} > {seq_len}",
        )

    # Source-aligned dummy inputs: arbitrary input_ids, an mm_token_type_ids
    # that matches the [text_pre | image | text_post] layout, and the
    # single-image grid_thw.
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.int64)
    mm_token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.int32)
    mm_token_type_ids[:, text_pre_len : text_pre_len + image_token_count] = 1
    image_grid_thw = torch.tensor(
        [[image_grid_t, image_grid_h, image_grid_w]], dtype=torch.int64,
    )

    sample_inputs = (input_ids, mm_token_type_ids, image_grid_thw)
    save_path = os.path.join(
        out_dir,
        f"mrope_position_ids_prefill_{_seq_tag(seq_len)}.onnx",
    )
    _onnx_export(
        block,
        sample_inputs,
        save_path,
        input_names=["input_ids", "mm_token_type_ids", "image_grid_thw"],
        output_names=["position_ids", "mrope_position_deltas"],
        dynamic_axes={},
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
        fold_pure_shape_chains=fold_pure_shape_chains,
    )


# ---------------------------------------------------------------------------
# 4. mrope_position_ids_decode (compute_3d_position_ids elif branch)
# ---------------------------------------------------------------------------

def export_mrope_position_ids_decode(
    model: Qwen3_5MoeModelLike,
    out_dir: str,
    batch_size: int,
    opset: int,
    simplify: bool,
    *,
    decode_context_len: int,
    fold_pure_shape_chains: bool = False,
    strip_initializers: bool = False,
) -> None:
    block = MRoPEPositionIdsDecodeBlock(batch_size=batch_size)

    # Source line 1722-1725: attention_mask spans full ctx + 1 (the new
    # decode token). int64 matches the source ``.long()`` cast at line
    # 1723.
    attn_len = decode_context_len + 1
    attention_mask = torch.ones(batch_size, attn_len, dtype=torch.int64)
    rope_deltas = torch.zeros(batch_size, 1, dtype=torch.int64)

    sample_inputs = (attention_mask, rope_deltas)
    save_path = os.path.join(
        out_dir,
        f"mrope_position_ids_decode_ctx{_seq_tag(decode_context_len)}.onnx",
    )
    _onnx_export(
        block,
        sample_inputs,
        save_path,
        input_names=["attention_mask", "rope_deltas"],
        output_names=["position_ids"],
        dynamic_axes={},
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
        fold_pure_shape_chains=fold_pure_shape_chains,
    )
