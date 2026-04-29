"""
ONNX export wrappers for the Qwen3.5-MoE vision tower.

These mirror the text-side wrappers in ``qwen_export_shared.py`` /
``qwen_merged_block_export.py`` but target every learnable / structural
piece of ``Qwen3_5MoeVisionModel.forward`` (``modeling_qwen3_5_moe.py``
lines 1228-1275):

  - ``vision_patch_embed_<seq>.onnx``                Conv3d patchification.
  - ``vision_pos_embed_interp_<seq>.onnx``           Bilinear interpolation
                                                    of the learnable
                                                    ``pos_embed`` table
                                                    against ``grid_thw``,
                                                    plus the residual add
                                                    onto the patch tokens
                                                    (source lines 1241-1242).
  - ``vision_rot_pos_emb_<seq>.onnx``                2D height/width rotary
                                                    ``cos / sin`` tables
                                                    (source lines 1244-1250).
  - ``vision_cu_seqlens_<seq>.onnx``                 Variable-length packing
                                                    prefix-sum tensor
                                                    (source lines 1252-1260).
  - ``vision_block_<idx>_repr_<seq>.onnx``           A single representative
                                                    ViT block (norm1 ->
                                                    cu_seqlens-aware attn ->
                                                    + residual -> norm2 ->
                                                    mlp -> + residual).
  - ``vision_patch_merger_<seq>.onnx``               Spatial concat +
                                                    LayerNorm + 2-layer MLP,
                                                    mapping vision hidden
                                                    to text hidden.

Why every step is exported separately
-------------------------------------
Per the project README, rule 1 ("source-aligned") and rule 4 ("continuous
data flow / no analytical islands") forbid leaving any source-side tensor
work outside ONNX. The graphs above stitch end-to-end:

    pixel_values
        |  vision_patch_embed
        v
    hidden_states_pre  ------+
                              | vision_pos_embed_interp
        grid_thw  ------------+
                              v
    hidden_states_post  -----+
                             |
        grid_thw  -----------+--> vision_rot_pos_emb  -> cos, sin
                             |
        grid_thw  -----------+--> vision_cu_seqlens   -> cu_seqlens
                             |
                             v
    [hidden, cos, sin, cu_seqlens] ---> vision_block_<idx>_repr (x depth)
                                              |
                                              v
                                  vision_patch_merger -> image_embeds

Static seq_lens used during export
----------------------------------
The vision graphs are exported with a *static* ``seq_len`` chosen as the
"representative" number of vision patch tokens — controlled by
``--vision_token_seq_len`` (default 1024). Real inference can run the same
graphs at any seq_len that matches the upstream shape semantics. The
static choice keeps shape propagation deterministic and stats reproducible.
"""

from __future__ import annotations

import math
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
    _vision_config,
    _visual_model,
)
from qwen_onnx_blocks_vision import (  # noqa: E402
    VisionBlockRepr,
    VisionCuSeqlensBlock,
    VisionPatchEmbedBlock,
    VisionPatchMergerBlock,
    VisionPosEmbedInterpBlock,
    VisionRotPosEmbBlock,
)


def _vision_dtype(model: Qwen3_5MoeModelLike) -> torch.dtype:
    """Return the dtype of the vision tower's parameters."""
    visual = _visual_model(model)
    return visual.patch_embed.proj.weight.dtype


def _representative_grid_thw(
    vision_token_seq_len: int,
    *,
    grid_t: int = 1,
) -> tuple[int, int, int]:
    """
    Choose a static ``(T, H, W)`` such that ``T * H * W == vision_token_seq_len``
    and ``H == W`` when possible. Mirrors the project's existing
    convention that ``vision_token_seq_len = 1024`` represents a 32x32
    single-image grid.
    """
    if vision_token_seq_len <= 0:
        raise ValueError(f"vision_token_seq_len must be > 0, got {vision_token_seq_len}")
    spatial = vision_token_seq_len // max(grid_t, 1)
    side = int(round(math.sqrt(spatial)))
    if side * side != spatial:
        raise ValueError(
            f"vision_token_seq_len={vision_token_seq_len} (with grid_t={grid_t}) "
            "is not a perfect square; pick a value with H == W for the "
            "representative export."
        )
    return grid_t, side, side


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
# 2. vision_pos_embed_interp (learnable pos_embed bilinear lookup + add)
# ---------------------------------------------------------------------------

def export_vision_pos_embed_interp(
    model: Qwen3_5MoeModelLike,
    out_dir: str,
    opset: int,
    simplify: bool,
    vision_token_seq_len: int,
    *,
    grid_t: int = 1,
    static_grid: bool = False,
    fold_pure_shape_chains: bool = False,
    strip_initializers: bool = False,
) -> None:
    visual = _visual_model(model)
    vcfg = _vision_config(model)
    t, h, w = _representative_grid_thw(vision_token_seq_len, grid_t=grid_t)

    block = VisionPosEmbedInterpBlock(
        visual,
        static_grid=(t, h, w) if static_grid else None,
    )
    hidden = int(vcfg.hidden_size)
    dtype = _vision_dtype(model)

    sample_inputs = (
        torch.randn(vision_token_seq_len, hidden, dtype=dtype),
        torch.tensor([[t, h, w]], dtype=torch.int64),
    )
    save_path = os.path.join(
        out_dir,
        f"vision_pos_embed_interp_{_seq_tag(vision_token_seq_len)}.onnx",
    )
    _onnx_export(
        block,
        sample_inputs,
        save_path,
        input_names=["hidden_states_pre", "grid_thw"],
        output_names=["hidden_states_post"],
        dynamic_axes={},
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
        fold_pure_shape_chains=fold_pure_shape_chains,
    )


# ---------------------------------------------------------------------------
# 3. vision_rot_pos_emb (2D height/width rotary cos/sin tables)
# ---------------------------------------------------------------------------

def export_vision_rot_pos_emb(
    model: Qwen3_5MoeModelLike,
    out_dir: str,
    opset: int,
    simplify: bool,
    vision_token_seq_len: int,
    *,
    grid_t: int = 1,
    static_grid: bool = False,
    fold_pure_shape_chains: bool = False,
    strip_initializers: bool = False,
) -> None:
    visual = _visual_model(model)
    t, h, w = _representative_grid_thw(vision_token_seq_len, grid_t=grid_t)
    output_dtype = _vision_dtype(model)

    block = VisionRotPosEmbBlock(
        visual,
        output_dtype=output_dtype,
        static_grid=(t, h, w) if static_grid else None,
    )

    # Source-aligned input: grid_thw matches the upstream signature.
    sample_inputs = (
        torch.tensor([[t, h, w]], dtype=torch.int64),
    )
    save_path = os.path.join(
        out_dir,
        f"vision_rot_pos_emb_{_seq_tag(vision_token_seq_len)}.onnx",
    )
    _onnx_export(
        block,
        sample_inputs,
        save_path,
        input_names=["grid_thw"],
        output_names=["cos", "sin"],
        dynamic_axes={},
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
        fold_pure_shape_chains=fold_pure_shape_chains,
    )


# ---------------------------------------------------------------------------
# 4. vision_cu_seqlens (variable-length packing prefix-sum tensor)
# ---------------------------------------------------------------------------

def export_vision_cu_seqlens(
    model: Qwen3_5MoeModelLike,
    out_dir: str,
    opset: int,
    simplify: bool,
    vision_token_seq_len: int,
    *,
    grid_t: int = 1,
    fold_pure_shape_chains: bool = False,
    strip_initializers: bool = False,
) -> None:
    t, h, w = _representative_grid_thw(vision_token_seq_len, grid_t=grid_t)
    block = VisionCuSeqlensBlock()
    sample_inputs = (
        torch.tensor([[t, h, w]], dtype=torch.int64),
    )
    save_path = os.path.join(
        out_dir,
        f"vision_cu_seqlens_{_seq_tag(vision_token_seq_len)}.onnx",
    )
    _onnx_export(
        block,
        sample_inputs,
        save_path,
        input_names=["grid_thw"],
        output_names=["cu_seqlens"],
        dynamic_axes={},
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
        fold_pure_shape_chains=fold_pure_shape_chains,
    )


# ---------------------------------------------------------------------------
# 5. vision_block_<idx>_repr (representative ViT block, cu_seqlens-aware)
# ---------------------------------------------------------------------------

def export_vision_block_repr(
    model: Qwen3_5MoeModelLike,
    layer_idx: int,
    out_dir: str,
    opset: int,
    simplify: bool,
    vision_token_seq_len: int,
    *,
    num_segments: int = 1,
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

    if num_segments < 1:
        raise ValueError(f"num_segments must be >= 1, got {num_segments}")
    # Representative cu_seqlens for a single-segment scenario:
    # ``[0, vision_token_seq_len]``. Multi-segment scenarios use the same
    # graph topology with a longer cu_seqlens at runtime. Dtype is
    # ``int64`` to match the source-semantics-faithful output of
    # ``vision_cu_seqlens_*.onnx`` — the source's
    # ``dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32``
    # (modeling_qwen3_5_moe.py line 1258, HF PR #34852 workaround) picks
    # ``grid_thw.dtype = int64`` under both the legacy ``torch.jit.trace``
    # and the new dynamo ``torch.onnx.export`` paths. The downstream
    # ``_VisionCuSeqlensSegmentAttention.forward`` opens with a
    # ``cu_seqlens.long()`` cast, so the int64 graph contract is identical
    # in semantic to the eager-mode int32 contract.
    cu_seqlens_dummy = torch.tensor(
        [0, vision_token_seq_len], dtype=torch.int64,
    )

    sample_inputs = (
        torch.randn(vision_token_seq_len, hidden, dtype=dtype),
        torch.randn(vision_token_seq_len, head_dim, dtype=dtype),
        torch.randn(vision_token_seq_len, head_dim, dtype=dtype),
        cu_seqlens_dummy,
    )
    save_path = os.path.join(
        out_dir,
        f"vision_block_{layer_idx:02d}_repr_{_seq_tag(vision_token_seq_len)}.onnx",
    )
    _onnx_export(
        block,
        sample_inputs,
        save_path,
        input_names=["hidden_states", "cos", "sin", "cu_seqlens"],
        output_names=["hidden_states_out"],
        dynamic_axes={},
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
        fold_pure_shape_chains=fold_pure_shape_chains,
    )


# ---------------------------------------------------------------------------
# 6. vision_patch_merger
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
