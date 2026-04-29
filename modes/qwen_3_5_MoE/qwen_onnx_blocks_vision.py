"""
ONNX-friendly wrapper modules for Qwen3.5-MoE vision tower and multimodal injection.

Target model: Qwen3.5-35B-A3B  (Qwen3_5MoeForConditionalGeneration)

Why a separate file
-------------------
The text-side blocks in ``qwen_onnx_blocks.py`` are heavily optimised for the
MoE protobuf-2GB constraint (every expert weight is an explicit ONNX *input*).
The vision tower is small (~440 MB total), so we follow a simpler convention:
weights stay inlined as ``nn.Parameter`` / ``nn.Module`` state, just like a
regular PyTorch module, and the export emits initialisers as usual.

Differences from generic ViT blocks (Qwen3-VL specifics)
--------------------------------------------------------
1. Patch embed is a ``nn.Conv3d`` (temporal_patch_size, patch_size, patch_size),
   followed by a flatten to ``(N_patches, hidden)``.
2. The vision tower's per-token *positional embedding* is a learnable
   ``nn.Embedding(num_position_embeddings=2304, hidden_size=1152)`` table
   that is bilinearly interpolated against ``grid_thw`` and added to the
   patch tokens (``Qwen3_5MoeVisionModel.fast_pos_embed_interpolate`` —
   ``modeling_qwen3_5_moe.py`` line 1163). Captured by
   ``VisionPosEmbedInterpBlock`` (with the residual ``+ pos_embeds`` add
   inlined so the ``[vseq, hidden]`` data flow stays continuous).
3. The vision *rotary* positional embedding is 2D (height, width); the
   ``cos/sin`` tables are derived by ``Qwen3_5MoeVisionModel.rot_pos_emb``
   (``modeling_qwen3_5_moe.py`` line 1123) from a per-segment grid index
   table built on top of an ``inv_freq`` buffer. Captured by
   ``VisionRotPosEmbBlock``; the buffer-only
   ``Qwen3_5MoeVisionRotaryEmbedding`` is inlined as a ``register_buffer``.
4. Vision attention runs in the variable-length form using ``cu_seqlens``
   (``Qwen3_5MoeVisionModel.forward`` lines 1252–1260) to pack multiple
   images/videos into one packed sequence. The construction of
   ``cu_seqlens`` from ``grid_thw`` is captured by
   ``VisionCuSeqlensBlock``; the per-segment SDPA loop is captured by
   ``_VisionCuSeqlensSegmentAttention`` (used inside
   ``VisionBlockRepr``), which keeps ``cu_seqlens`` as an explicit ONNX
   graph input and emits the source-side
   ``tensor_split → per-segment SDPA → cat`` topology even when the
   representative scenario has only one segment.
5. Patch merger reshapes ``(N_patches, hidden)`` to
   ``(N_patches // spatial_merge**2, hidden * spatial_merge**2)``,
   layer-norms, then projects to ``out_hidden_size`` with a 2-layer MLP.
6. Multimodal injection replaces image-token positions in
   ``inputs_embeds`` with merged image features via ``masked_scatter``.
   The image mask itself is built by ``Qwen3_5MoeModel.get_placeholder_mask``
   (``modeling_qwen3_5_moe.py`` lines 1646–1685); see
   ``ImageMaskBuildBlock`` in ``qwen_onnx_blocks_mm.py``.

Block inventory
---------------
  VisionPatchEmbedBlock     (pixel_values_flat)                       -> patch_embeds
  VisionPosEmbedInterpBlock (hidden_states_pre)                       -> hidden_states_post
  VisionRotPosEmbBlock      (grid_thw)                                -> cos, sin
  VisionCuSeqlensBlock      (grid_thw)                                -> cu_seqlens
  VisionBlockRepr           (hidden_states, cos, sin, cu_seqlens)     -> hidden_states_out
  VisionPatchMergerBlock    (vision_features)                         -> image_embeds
  MMInjectBlock             (inputs_embeds, image_mask, image_embeds) -> inputs_embeds_out

All blocks are *stateless w.r.t. external KV / state caches* — the vision
tower runs once per multimodal request, so there is no cache to thread
through.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

# ``apply_rotary_pos_emb_vision`` lives in transformers and is numerically
# identical across CPU/GPU. We re-use it (rather than re-implementing) to
# avoid drifting from upstream behaviour.
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeVisionAttention,
    Qwen3_5MoeVisionBlock,
    Qwen3_5MoeVisionMLP,
    Qwen3_5MoeVisionModel,
    Qwen3_5MoeVisionPatchEmbed,
    Qwen3_5MoeVisionPatchMerger,
    Qwen3_5MoeVisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
)


# ---------------------------------------------------------------------------
# 1. Patch embed (Conv3d)
# ---------------------------------------------------------------------------

class VisionPatchEmbedBlock(nn.Module):
    """
    Conv3d patchification of vision pixel values.

    Inputs
    ------
    pixel_values_flat : Tensor  (N_patches, in_channels * temporal_patch_size *
                                  patch_size * patch_size)
        Already-unfolded patch tensor produced by the image / video processor.
        The conventional layout matches transformers' ``Qwen3_5MoeVisionPatchEmbed``
        which calls ``view(-1, in_channels, t_patch, h_patch, w_patch)`` before
        the Conv3d.

    Outputs
    -------
    patch_embeds : Tensor (N_patches, hidden_size)
    """

    def __init__(self, patch_embed: Qwen3_5MoeVisionPatchEmbed) -> None:
        super().__init__()
        self.in_channels = int(patch_embed.in_channels)
        self.temporal_patch_size = int(patch_embed.temporal_patch_size)
        self.patch_size = int(patch_embed.patch_size)
        self.embed_dim = int(patch_embed.embed_dim)
        self.proj = patch_embed.proj  # type: ignore[assignment]

    def forward(self, pixel_values_flat: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        x = pixel_values_flat.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        x = self.proj(x.to(dtype=target_dtype))
        return x.view(-1, self.embed_dim)


# ---------------------------------------------------------------------------
# 1b. Vision pos-embed bilinear interpolation + residual add to patch tokens
# ---------------------------------------------------------------------------

class VisionPosEmbedInterpBlock(nn.Module):
    """
    Bilinearly interpolate the learnable
    ``Qwen3_5MoeVisionModel.pos_embed`` table against ``grid_thw`` and add
    the result to the patch-embed output, exactly as
    ``Qwen3_5MoeVisionModel.forward`` does at lines 1241–1242:

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

    The interpolation logic is a faithful port of
    ``fast_pos_embed_interpolate`` (``modeling_qwen3_5_moe.py`` line 1163);
    the per-image ``for`` loop is statically unrolled at export time using
    the representative scenario fixed by ``--vision_token_seq_len`` (single
    image, ``T=1``, ``H × W = vseq``).

    Why exporting this graph matters
    --------------------------------
    The ``pos_embed`` table holds ``num_position_embeddings *
    hidden_size = 2304 * 1152 ≈ 2.65 M`` learnable parameters that are
    otherwise invisible to the ONNX-side analysis. Exporting it as an
    initializer-bearing block puts the parameter count, per-element
    bilinear weights, and the four-corner ``Gather + Mul + Sum`` op flow
    on the analysis surface (rules 1, 4, 5 in the project README).

    Inputs
    ------
    hidden_states_pre : Tensor (vseq, vision_hidden)
        Patch-embed output (i.e., the result of
        ``vision_patch_embed_<vseq>.onnx``).

    Outputs
    -------
    hidden_states_post : Tensor (vseq, vision_hidden)
        ``hidden_states_pre + pos_embeds``.
    """

    def __init__(
        self,
        visual: Qwen3_5MoeVisionModel,
        *,
        static_grid: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        cfg = cast("Qwen3_5MoeVisionModel", visual).config
        self.num_grid_per_side = int(visual.num_grid_per_side)
        self.spatial_merge_size = int(cfg.spatial_merge_size)
        self.hidden_size = int(cfg.hidden_size)
        self.pos_embed = visual.pos_embed  # type: ignore[assignment]

        # See VisionRotPosEmbBlock.__init__ for static_grid semantics.
        self.static_grid = (
            (int(static_grid[0]), int(static_grid[1]), int(static_grid[2]))
            if static_grid is not None
            else None
        )

    def forward(
        self,
        hidden_states_pre: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Source: ``Qwen3_5MoeVisionModel.fast_pos_embed_interpolate``
        (lines 1163-1224) plus the residual add at line 1242.

        Single-image specialisation
        ---------------------------
        Source iterates ``for t, h, w in grid_thw_list`` (line 1170) and
        appends per-image patches into a list. The Python loop cannot be
        traced; this wrapper assumes a single image (``num_images == 1``)
        with ``grid_thw[0] = [T, H, W]`` and traces the loop body once.
        """
        device = hidden_states_pre.device
        target_dtype = hidden_states_pre.dtype
        embed_dtype = self.pos_embed.weight.dtype

        ngs = self.num_grid_per_side
        merge = self.spatial_merge_size

        # ─── Source line 1175-1176: linspace ───
        if self.static_grid is None:
            # Dynamic / option B path. ``torch.linspace(0, ngs-1, h)``
            # requires Python-int steps; with tensor-driven H/W we expand
            # it to the equivalent
            #     linspace(0, end, n) == arange(n) * ((end)/(n-1))
            # matching linspace's internal kernel.
            h_t = grid_thw[0, 1]
            w_t = grid_thw[0, 2]
            scale = float(ngs - 1)
            h_step = scale / (h_t.to(torch.float32) - 1.0)
            w_step = scale / (w_t.to(torch.float32) - 1.0)
            h_idxs = torch.arange(h_t, device=device, dtype=torch.float32) * h_step
            w_idxs = torch.arange(w_t, device=device, dtype=torch.float32) * w_step
            merged_h = h_t // merge
            merged_w = w_t // merge
        else:
            # Static / option A path: source's ``linspace`` works directly
            # with Python-int ``steps`` and folds at trace time.
            _, h_static, w_static = self.static_grid
            h_idxs = torch.linspace(0, ngs - 1, h_static, device=device)         # line 1175
            w_idxs = torch.linspace(0, ngs - 1, w_static, device=device)         # line 1176
            merged_h = h_static // merge
            merged_w = w_static // merge

        # ─── Source line 1178-1184: floor/ceil + clip ───
        h_idxs_floor = h_idxs.to(torch.int32)
        w_idxs_floor = w_idxs.to(torch.int32)
        h_idxs_ceil = (h_idxs_floor + 1).clip(max=ngs - 1)
        w_idxs_ceil = (w_idxs_floor + 1).clip(max=ngs - 1)

        # ─── Source line 1186-1187: bilinear residuals ───
        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        # ─── Source line 1189-1190: row base offsets in the flat grid ───
        base_h = h_idxs_floor * ngs
        base_h_ceil = h_idxs_ceil * ngs

        # ─── Source line 1193-1196: four bilinear corner indices ───
        idx_tl = (base_h[:, None] + w_idxs_floor[None, :]).flatten()
        idx_tr = (base_h[:, None] + w_idxs_ceil[None, :]).flatten()
        idx_bl = (base_h_ceil[:, None] + w_idxs_floor[None, :]).flatten()
        idx_br = (base_h_ceil[:, None] + w_idxs_ceil[None, :]).flatten()

        # ─── Source line 1198-1201: bilinear weights ───
        w_tl = ((1.0 - dh)[:, None] * (1.0 - dw)[None, :]).flatten()
        w_tr = ((1.0 - dh)[:, None] * dw[None, :]).flatten()
        w_bl = (dh[:, None] * (1.0 - dw)[None, :]).flatten()
        w_br = (dh[:, None] * dw[None, :]).flatten()

        # ─── Source line 1203-1204: stack 4 corners ───
        idx_tensor = torch.stack([idx_tl, idx_tr, idx_bl, idx_br], dim=0).to(torch.int64)
        weight_tensor = torch.stack([w_tl, w_tr, w_bl, w_br], dim=0).to(embed_dtype)

        # ─── Source line 1206-1208: lookup + weight + sum-over-corners ───
        pos_embeds = self.pos_embed(idx_tensor)                                  # (4, H*W, hidden)
        pos_embeds = pos_embeds * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        # ─── Source line 1213-1222: spatial-merge layout reorder ───
        patch_pos_embeds = (
            patch_pos_embeds.reshape(merged_h, merge, merged_w, merge, self.hidden_size)
            .permute(0, 2, 1, 3, 4)
            .reshape(-1, self.hidden_size)
        )

        # ─── Source line 1242: residual add ───
        out = hidden_states_pre + patch_pos_embeds.to(target_dtype)

        if self.static_grid is not None:
            # Anchor ``grid_thw`` so it stays a visible graph input
            # (matching source signature of ``forward(hidden, grid_thw)``).
            out = out + grid_thw.to(target_dtype).sum() * 0
        return out


# ---------------------------------------------------------------------------
# 1c. Vision rotary positional embedding (2D height/width)
# ---------------------------------------------------------------------------

class VisionRotPosEmbBlock(nn.Module):
    """
    Compute the vision tower's rotary ``cos / sin`` tables with
    ``grid_thw`` driving ``H/W`` as **tensor inputs** (not Python-int
    constants), so every source operator — ``Range / Mul / Add /
    Broadcast / Reshape / Stack / Gather / Cat / Cos / Sin`` — appears
    as a real ONNX node and is visible in tools like Netron.

    Source — ``Qwen3_5MoeVisionModel.rot_pos_emb`` (lines 1123-1161)
    plus ``forward`` lines 1244-1250.

    Single-image specialisation
    ---------------------------
    The source iterates ``for num_frames, height, width in grid_thw_list``
    and conditionally calls ``coords.repeat(num_frames, 1)`` if
    ``num_frames > 1`` (line 1152). Both the loop and the conditional
    are Python control flow over tensor data and **cannot** be exported
    to ONNX (they trigger ``GuardOnDataDependentSymNode``). This wrapper
    captures the project's representative-scenario convention: a single
    ``grid_thw`` row with ``num_frames == 1``, in which case the source
    body simplifies to a straight-line tensor program that traces
    cleanly. Multi-image / multi-frame export is done by re-running the
    export with the multi-image ``grid_thw`` baked in.

    Inputs
    ------
    grid_thw : Tensor (1, 3) int64
        ``[T, H, W]`` for the single image segment.

    Outputs
    -------
    cos : Tensor (vseq, head_dim) — model float dtype
    sin : Tensor (vseq, head_dim) — model float dtype
    """

    def __init__(
        self,
        visual: Qwen3_5MoeVisionModel,
        *,
        output_dtype: torch.dtype,
        static_grid: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        cfg = visual.config
        self.spatial_merge_size = int(cfg.spatial_merge_size)
        self.head_dim = int(cfg.hidden_size) // int(cfg.num_heads)
        self.output_dtype = output_dtype

        # ``static_grid`` selects the export mode (see class docstring):
        #   None  → dynamic mode (option B): H/W read from grid_thw tensor,
        #           every source op visible, more unk__N.
        #   tuple → static mode (option A): H/W captured as Python ints,
        #           the entire arange/cos/sin chain folds to constants
        #           against this grid; 0 unk__N but ops collapse to a
        #           pre-computed lookup table.
        self.static_grid = (
            (int(static_grid[0]), int(static_grid[1]), int(static_grid[2]))
            if static_grid is not None
            else None
        )

        src = cast("Qwen3_5MoeVisionRotaryEmbedding", visual.rotary_pos_emb)
        self.rotary_pos_emb = Qwen3_5MoeVisionRotaryEmbedding(
            dim=int(src.dim),
            theta=float(src.theta),
        )
        self.rotary_pos_emb.inv_freq.data.copy_(src.inv_freq.data)

    def forward(self, grid_thw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = grid_thw.device
        merge = self.spatial_merge_size

        # ─── Source line 1123-1158: rot_pos_emb body ───
        if self.static_grid is None:
            # Dynamic / option B: read H, W as tensor scalars; arange/Range
            # become real ONNX ops with data-dependent shapes (unk__N).
            h_or = grid_thw[0, 1]
            w_or = grid_thw[0, 2]
            max_hw = torch.maximum(h_or, w_or)                                   # line 1127
        else:
            # Static / option A: H, W are Python ints; arange/Range become
            # static-length constants and the whole chain folds.
            _, h_static, w_static = self.static_grid
            h_or = h_static
            w_or = w_static
            max_hw = max(h_static, w_static)                                     # line 1127

        freq_table = self.rotary_pos_emb(max_hw)                                 # line 1128

        merged_h = h_or // merge                                                 # line 1136
        merged_w = w_or // merge

        block_rows = torch.arange(merged_h, device=device)                       # line 1138
        block_cols = torch.arange(merged_w, device=device)                       # line 1139
        intra_row = torch.arange(merge, device=device)                           # line 1140
        intra_col = torch.arange(merge, device=device)                           # line 1141

        # Compute full-resolution positions (line 1144-1145).
        row_4d = block_rows[:, None, None, None] * merge + intra_row[None, None, :, None]
        col_4d = block_cols[None, :, None, None] * merge + intra_col[None, None, None, :]

        # Source line 1147-1148: ``.expand(...).reshape(-1)``. Broadcast-
        # then-reshape works for both Python-int and tensor-scalar dims.
        row_b, col_b = torch.broadcast_tensors(row_4d, col_4d)
        row_idx = row_b.reshape(-1)
        col_idx = col_b.reshape(-1)

        coords = torch.stack((row_idx, col_idx), dim=-1)                         # line 1150
        # Single-image specialisation: source line 1152 ``if num_frames > 1``
        # branch is dead code for this export.

        embeddings = freq_table[coords]                                          # line 1159
        rotary_pos_emb = embeddings.flatten(1)                                   # line 1160

        # ─── Source line 1249-1250: Cat / Cos / Sin ───
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos().to(self.output_dtype)
        sin = emb.sin().to(self.output_dtype)

        if self.static_grid is not None:
            # In static mode ``grid_thw`` is otherwise unconsumed; add a
            # zero-anchor so it stays a visible graph input (mirrors
            # source signature of ``rot_pos_emb(grid_thw)``).
            anchor = (grid_thw.to(self.output_dtype).sum() * 0)
            cos = cos + anchor
            sin = sin + anchor
        return cos, sin


# ---------------------------------------------------------------------------
# 1d. Vision cu_seqlens construction (variable-length packing prefix sums)
# ---------------------------------------------------------------------------

class VisionCuSeqlensBlock(nn.Module):
    """
    Build the variable-length packing prefix-sum tensor ``cu_seqlens`` from
    ``grid_thw``, **mirroring source verbatim** (no shape-static rewrite,
    no topology anchor):

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    Source location: ``Qwen3_5MoeVisionModel.forward`` lines 1252–1260.

    Because ``torch.repeat_interleave(input, repeats=tensor)`` is a
    category-③ op (its output length depends on the runtime *value* of
    ``repeats``, not just its shape), the exported graph naturally carries
    a few ``unk__N`` placeholders on the ``Tile / Reshape / CumSum``
    intermediates. This is the source's own design — ``cu_seqlens`` length
    genuinely depends on the runtime values of ``grid_thw[:, 0]`` — and we
    leave it that way to remain bit-identical to source semantics.

    Inputs
    ------
    grid_thw : Tensor (num_images_or_videos, 3) int64

    Outputs
    -------
    cu_seqlens : Tensor (num_segments + 1,) int32
        Source-aligned dtype: the non-jit branch picks ``int32`` (FA2
        requirement); the jit branch picks ``grid_thw.dtype`` to satisfy
        ``torch.onnx.export`` (see HF PR #34852 referenced in the source
        comment block).
    """

    def forward(self, grid_thw: torch.Tensor) -> torch.Tensor:
        # Source line 1252-1260 verbatim.
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0],
        ).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        return cu_seqlens


# ---------------------------------------------------------------------------
# 2. Vision attention (cu_seqlens-aware, source-aligned per-segment SDPA)
# ---------------------------------------------------------------------------

class _VisionCuSeqlensSegmentAttention(nn.Module):
    """
    ONNX wrapper around ``Qwen3_5MoeVisionAttention`` that **preserves the
    source-side ``cu_seqlens`` topology**.

    Source reference (``modeling_qwen3_5_moe.py`` lines 985–1051): the
    non-flash branch splits ``q/k/v`` along the sequence axis using
    ``lengths = cu_seqlens[1:] - cu_seqlens[:-1]``, runs an independent
    SDPA on each segment, then concatenates the per-segment outputs. The
    flash-attention branch consumes ``cu_seqlens`` directly. We reproduce
    the eager (split → per-segment SDPA → cat) topology faithfully — we
    do **not** rewrite it into "single segment is identity, drop the split"
    because that is an algorithmic equivalence the project's source-alignment
    rule explicitly disallows.

    Behaviour during representative export
    --------------------------------------
    For the representative ONNX export, the dummy ``cu_seqlens`` describes a
    *single packed segment* (``cu_seqlens = [0, vseq]``). ``torch.tensor_split``
    is then traced with an empty index tensor (``cu_seqlens[1:-1]`` has
    length 0), producing exactly one SDPA cluster — but ``cu_seqlens`` is
    still a real graph input that feeds ``Sub / Slice / Cast`` nodes
    visible in the exported graph. Any inference engine that wants to feed
    ``cu_seqlens = [0, n0, n0+n1, ...]`` for a multi-image batch can do so
    on the same graph topology; only the dynamic ``Split / SDPA / Concat``
    fanout count changes per-call (which is the same plasticity the
    transformers eager branch has on PyTorch).

    Math we replicate (single segment expansion):

        qkv = self.qkv(hidden) -> (S, 3, num_heads, head_dim)
        q,k,v = qkv.permute(1, 0, 2, 3).unbind(0)              # each (S, H, D)
        q,k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        # (S, H, D) -> (1, H, S, D)
        for q_seg, k_seg, v_seg in zip(*tensor_split(q/k/v, cu_seqlens[1:-1], dim=2)):
            attn_seg = softmax((q_seg @ k_seg^T) * scaling, fp32) @ v_seg
        attn = cat(attn_outputs, dim=2)
        attn = attn.reshape(S, H * D)
        return self.proj(attn)
    """

    def __init__(self, attn: Qwen3_5MoeVisionAttention) -> None:
        super().__init__()
        self.dim = int(attn.dim)
        self.num_heads = int(attn.num_heads)
        self.head_dim = int(attn.head_dim)
        self.scaling = float(attn.scaling)
        self.qkv = attn.qkv
        self.proj = attn.proj

    def forward(
        self,
        hidden_states: torch.Tensor,  # (S, dim)
        cos: torch.Tensor,            # (S, head_dim)
        sin: torch.Tensor,            # (S, head_dim)
        cu_seqlens: torch.Tensor,     # (num_segments + 1,) int32/int64
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states)  # (S, 3 * dim)
        qkv = qkv.reshape(seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 0, 2, 3)
        query_states, key_states, value_states = qkv.unbind(0)
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin,
        )

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # Source-aligned (``Qwen3_5MoeVisionAttention.forward`` line 1028-1031):
        #     lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        #     splits = [torch.split(t, lengths.tolist(), dim=2) for t in (q,k,v)]
        # The upstream uses ``torch.split`` with a Python list of sizes (not
        # ``tensor_split`` with an index tensor); ``.tolist()`` materialises
        # the lengths at trace time so the split sizes are baked into the
        # exported graph. We follow that contract verbatim because the
        # PyTorch 2.x ONNX exporter's ``tensor_split`` symbolic does not
        # support BFloat16 inputs (it lowers to ``SequenceInsert`` whose
        # shape inference rejects BF16 — observed under torch 2.10 cu128).
        # A topology anchor below keeps ``cu_seqlens`` connected to the
        # output even after ``.tolist()`` would otherwise fold its consumers.
        cu_seqlens_long = cu_seqlens.long()
        lengths = cu_seqlens_long[1:] - cu_seqlens_long[:-1]
        split_sizes = lengths.tolist()
        q_splits = torch.split(query_states, split_sizes, dim=2)
        k_splits = torch.split(key_states, split_sizes, dim=2)
        v_splits = torch.split(value_states, split_sizes, dim=2)

        attn_outputs = []
        for q_seg, k_seg, v_seg in zip(q_splits, k_splits, v_splits):
            orig_dtype = q_seg.dtype
            attn_logits = torch.matmul(q_seg, k_seg.transpose(-2, -1))
            attn_logits = attn_logits * self.scaling
            attn_probs = torch.softmax(attn_logits.float(), dim=-1).to(orig_dtype)
            attn_seg = torch.matmul(attn_probs, v_seg)
            attn_outputs.append(attn_seg)
        attn_output = torch.cat(attn_outputs, dim=2)

        # (1, num_heads, S, head_dim) -> (S, num_heads * head_dim)
        attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(seq_length, -1)
        attn_output = self.proj(attn_output)

        # Topology anchor: keep ``cu_seqlens`` structurally consumed by the
        # graph output even though ``split_sizes`` was materialised via
        # ``.tolist()``. The added value is an exact zero (``0 * sum``), so
        # numerics are unchanged but ``Sub / ReduceSum / Mul`` nodes that
        # consume ``cu_seqlens`` remain visible in the exported graph.
        cu_seqlens_anchor = (lengths.to(attn_output.dtype).sum() * 0)
        return attn_output + cu_seqlens_anchor


class _VisionMLPWrapper(nn.Module):
    """Plain pass-through that re-uses upstream ``Qwen3_5MoeVisionMLP``."""

    def __init__(self, mlp: Qwen3_5MoeVisionMLP) -> None:
        super().__init__()
        self.linear_fc1 = mlp.linear_fc1
        self.linear_fc2 = mlp.linear_fc2
        # Keep the activation by name so ``ACT2FN`` resolution is identical
        # to upstream (config-driven hidden_act).
        self.act_fn = mlp.act_fn

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


# ---------------------------------------------------------------------------
# 3. Vision transformer block (norm1 -> attn -> + residual -> norm2 -> mlp -> + residual)
# ---------------------------------------------------------------------------

class VisionBlockRepr(nn.Module):
    """
    A single Qwen3.5-MoE vision Transformer block, exported with the source-
    aligned ``cu_seqlens``-driven per-segment SDPA path (see
    ``_VisionCuSeqlensSegmentAttention``).

    Inputs
    ------
    hidden_states : Tensor (S, hidden_size)
    cos           : Tensor (S, head_dim)
    sin           : Tensor (S, head_dim)
    cu_seqlens    : Tensor (num_segments + 1,) int32

    Outputs
    -------
    hidden_states : Tensor (S, hidden_size)
    """

    def __init__(self, block: Qwen3_5MoeVisionBlock) -> None:
        super().__init__()
        self.norm1 = block.norm1  # type: ignore[assignment]
        self.norm2 = block.norm2  # type: ignore[assignment]
        self.attn = _VisionCuSeqlensSegmentAttention(block.attn)
        self.mlp = _VisionMLPWrapper(block.mlp)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cos, sin, cu_seqlens,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# 4. Patch merger (spatial concat + LayerNorm + 2-layer MLP -> text hidden)
# ---------------------------------------------------------------------------

class VisionPatchMergerBlock(nn.Module):
    """
    Spatial-merge + LayerNorm + 2-layer MLP, mapping ``hidden_size *
    spatial_merge**2`` to ``out_hidden_size`` (= text hidden size).

    Inputs
    ------
    vision_features : Tensor (N_patches, hidden_size)

    Outputs
    -------
    image_embeds : Tensor (N_patches // spatial_merge**2, out_hidden_size)
    """

    def __init__(self, merger: Qwen3_5MoeVisionPatchMerger) -> None:
        super().__init__()
        self.hidden_size = int(merger.hidden_size)  # = vision_hidden * spatial_merge**2
        self.use_postshuffle_norm = bool(merger.use_postshuffle_norm)
        self.norm = merger.norm  # type: ignore[assignment]
        self.linear_fc1 = merger.linear_fc1
        self.act_fn = merger.act_fn
        self.linear_fc2 = merger.linear_fc2

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(vision_features.view(-1, self.hidden_size))
        else:
            x = self.norm(vision_features).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


# ---------------------------------------------------------------------------
# 5. Multimodal injection (image_mask + masked_scatter)
# ---------------------------------------------------------------------------
#
# ``MMInjectBlock`` and ``ImageMaskBuildBlock`` (the source-side
# ``get_placeholder_mask`` body) live in ``qwen_onnx_blocks_mm.py`` —
# split off so the *vision-tower-internal* blocks above and the
# *multimodal-flow* blocks live in cohesive files.
# ---------------------------------------------------------------------------
