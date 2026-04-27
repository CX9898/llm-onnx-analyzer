"""
ONNX export wrappers for the Qwen3.5-MoE vision tower and multimodal injection.

These mirror the text-side wrappers in ``qwen_export_shared.py`` /
``qwen_merged_block_export.py`` but target the four extra graphs introduced
by the multimodal variant:

  - ``vision_patch_embed_<seq>.onnx``
        Conv3d patchification.
  - ``vision_block_<idx>_repr_<seq>.onnx``
        A single representative ViT block (norm1 -> attn -> + residual ->
        norm2 -> mlp -> + residual). Single-segment attention path; see
        ``VisionBlockRepr`` for why.
  - ``vision_patch_merger_<seq>.onnx``
        Spatial concat + LayerNorm + 2-layer MLP, mapping vision hidden to
        text hidden so the result can be injected into the text sequence.
  - ``mm_inject_<seq>.onnx``
        ``inputs_embeds.masked_scatter(image_mask, image_embeds)`` —
        write merged image features into the text token slots.

What we deliberately do *not* export
------------------------------------
The vision tower has two pieces of *Python control flow* that drive shapes
and indices but contain no learnable weights:

  - ``Qwen3_5MoeVisionModel.fast_pos_embed_interpolate(grid_thw)`` —
        bilinear interpolation over the 48x48 ``pos_embed`` lookup table.
  - ``Qwen3_5MoeVisionModel.rot_pos_emb(grid_thw)`` —
        2D (height, width) rotary index construction.

Both walk over ``grid_thw`` with Python loops and ``.tolist()`` calls. They
produce inputs to the exported graphs (``cos`` / ``sin``) but cannot be
faithfully ONNX-exported themselves. The inference engine is expected to
compute them on CPU once per multimodal request and feed the results in.

Similarly, the M-RoPE index (``position_ids`` of shape ``[3, B, S]``) is
produced by ``Qwen3_5MoeModel.get_rope_index``, which contains
``itertools.groupby`` over Python lists — also not exportable. The text-side
``RotaryEmbeddingBlockMoE`` already accepts a 3D ``position_ids`` input and
applies interleaved M-RoPE correctly; only the index *producer* lives
outside the ONNX graphs.

Static seq_lens used during export
----------------------------------
The vision graphs are exported with a *static* ``seq_len`` chosen as the
"representative" number of vision patch tokens — controlled by
``--vision_token_seq_len`` (default 1024). Real inference can run the same
graphs at any seq_len that matches the upstream shape semantics. The static
choice keeps shape propagation deterministic and stats reproducible.
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
    _visual_model,
)
from qwen_onnx_blocks_vision import (  # noqa: E402
    MMInjectBlock,
    VisionBlockRepr,
    VisionPatchEmbedBlock,
    VisionPatchMergerBlock,
)


def _vision_dtype(model: Qwen3_5MoeModelLike) -> torch.dtype:
    """Return the dtype of the vision tower's parameters."""
    visual = _visual_model(model)
    return visual.patch_embed.proj.weight.dtype


# ---------------------------------------------------------------------------
# 1. vision_patch_embed
# ---------------------------------------------------------------------------

def export_vision_patch_embed(
    model: Qwen3_5MoeModelLike,
    out_dir: str,
    opset: int,
    simplify: bool,
    vision_token_seq_len: int,
    *,
    fold_pure_shape_chains: bool = False,
    strip_initializers: bool = False,
) -> None:
    visual = _visual_model(model)
    vcfg = _vision_config(model)
    block = VisionPatchEmbedBlock(visual.patch_embed)

    in_channels = int(vcfg.in_channels)
    t_patch = int(vcfg.temporal_patch_size)
    p = int(vcfg.patch_size)
    flat_dim = in_channels * t_patch * p * p
    dtype = _vision_dtype(model)

    sample_inputs = (
        torch.randn(vision_token_seq_len, flat_dim, dtype=dtype),
    )
    save_path = os.path.join(
        out_dir,
        f"vision_patch_embed_{_seq_tag(vision_token_seq_len)}.onnx",
    )
    _onnx_export(
        block,
        sample_inputs,
        save_path,
        input_names=["pixel_values_flat"],
        output_names=["patch_embeds"],
        dynamic_axes={},
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
        fold_pure_shape_chains=fold_pure_shape_chains,
    )


# ---------------------------------------------------------------------------
# 2. vision_block_<idx>_repr (representative ViT block)
# ---------------------------------------------------------------------------

def export_vision_block_repr(
    model: Qwen3_5MoeModelLike,
    layer_idx: int,
    out_dir: str,
    opset: int,
    simplify: bool,
    vision_token_seq_len: int,
    *,
    fold_pure_shape_chains: bool = False,
    strip_initializers: bool = False,
) -> None:
    visual = _visual_model(model)
    vcfg = _vision_config(model)

    if not (0 <= layer_idx < len(visual.blocks)):
        raise IndexError(
            f"vision block layer_idx={layer_idx} out of range [0, {len(visual.blocks)})"
        )
    block = VisionBlockRepr(visual.blocks[layer_idx])

    hidden = int(vcfg.hidden_size)
    head_dim = hidden // int(vcfg.num_heads)
    dtype = _vision_dtype(model)

    sample_inputs = (
        torch.randn(vision_token_seq_len, hidden, dtype=dtype),
        torch.randn(vision_token_seq_len, head_dim, dtype=dtype),
        torch.randn(vision_token_seq_len, head_dim, dtype=dtype),
    )
    save_path = os.path.join(
        out_dir,
        f"vision_block_{layer_idx:02d}_repr_{_seq_tag(vision_token_seq_len)}.onnx",
    )
    _onnx_export(
        block,
        sample_inputs,
        save_path,
        input_names=["hidden_states", "cos", "sin"],
        output_names=["hidden_states_out"],
        dynamic_axes={},
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
        fold_pure_shape_chains=fold_pure_shape_chains,
    )


# ---------------------------------------------------------------------------
# 3. vision_patch_merger
# ---------------------------------------------------------------------------

def export_vision_patch_merger(
    model: Qwen3_5MoeModelLike,
    out_dir: str,
    opset: int,
    simplify: bool,
    vision_token_seq_len: int,
    *,
    fold_pure_shape_chains: bool = False,
    strip_initializers: bool = False,
) -> None:
    visual = _visual_model(model)
    vcfg = _vision_config(model)
    block = VisionPatchMergerBlock(visual.merger)

    hidden = int(vcfg.hidden_size)
    spatial_merge = int(vcfg.spatial_merge_size)
    if vision_token_seq_len % (spatial_merge * spatial_merge) != 0:
        raise ValueError(
            f"vision_token_seq_len={vision_token_seq_len} is not divisible by "
            f"spatial_merge_size**2={spatial_merge**2}; the patch merger requires "
            f"complete spatial groups."
        )
    dtype = _vision_dtype(model)

    sample_inputs = (
        torch.randn(vision_token_seq_len, hidden, dtype=dtype),
    )
    save_path = os.path.join(
        out_dir,
        f"vision_patch_merger_{_seq_tag(vision_token_seq_len)}.onnx",
    )
    _onnx_export(
        block,
        sample_inputs,
        save_path,
        input_names=["vision_features"],
        output_names=["image_embeds"],
        dynamic_axes={},
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
        fold_pure_shape_chains=fold_pure_shape_chains,
    )


# ---------------------------------------------------------------------------
# 4. mm_inject (masked_scatter image_embeds into inputs_embeds)
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
    Export the image-embedding injection graph.

    The dummy inputs are constructed so that exactly ``image_token_count``
    positions across ``(batch_size, seq_len, H_text)`` are flagged for
    replacement, matching the row count of ``image_embeds``. Real inference
    can vary the number of image tokens per request; the static dummy is
    only to drive shape inference.
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

    # Build a (B, S) bool mask with exactly ``image_token_count`` True entries
    # at the front of the flattened sequence; broadcast to the embedding shape.
    flat_mask = torch.zeros(total_positions, dtype=torch.bool)
    flat_mask[:image_token_count] = True
    mask_2d = flat_mask.view(batch_size, seq_len)
    image_mask = mask_2d.unsqueeze(-1).expand(batch_size, seq_len, text_hidden).contiguous()

    sample_inputs = (
        torch.randn(batch_size, seq_len, text_hidden, dtype=dtype),
        image_mask,
        torch.randn(image_token_count, text_hidden, dtype=dtype),
    )
    save_path = os.path.join(out_dir, f"mm_inject_{_seq_tag(seq_len)}.onnx")
    _onnx_export(
        MMInjectBlock(),
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
