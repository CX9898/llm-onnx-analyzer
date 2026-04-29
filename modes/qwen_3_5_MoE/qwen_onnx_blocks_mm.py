"""
ONNX-friendly wrappers for the *multimodal flow* of
``Qwen3_5MoeForConditionalGeneration`` — i.e. the bookkeeping that lives
in ``Qwen3_5MoeModel.forward`` (``modeling_qwen3_5_moe.py`` lines 1738–1815)
between ``inputs_embeds`` and the language-model entry, plus the M-RoPE
3D ``position_ids`` constructor that drives the text decoder's rotary
embedding.

This file is the home for blocks that are **not** part of the vision tower
itself but are still source-side tensor work that the project's analysis
surface needs to see (rules 1, 4, 5 in the project README — source-aligned
topology, continuous data flow, real shapes / dtypes / parameter counts).

Block inventory
---------------
  ImageMaskBuildBlock           (input_ids, image_token_id) -> image_mask
        ⇢ ``Qwen3_5MoeModel.get_placeholder_mask`` lines 1646-1685
        Source-aligned reference: produces the bool mask exactly as the
        upstream ``get_placeholder_mask`` body (input-ids branch). Still
        emitted as a separate ONNX file for analysis surface, even though
        ``MMInjectBlock`` below has been rewritten to consume row-level
        position indices instead of the bool mask itself (see "Why a
        mask -> indices switch" in MMInjectBlock).
  MMInjectBlock                 (inputs_embeds, image_position_indices, image_embeds)
                                                            -> inputs_embeds_out
        ⇢ ``Qwen3_5MoeModel.forward``     line 1773
        (relocated from ``qwen_onnx_blocks_vision.py`` for cohesion;
        rewritten from ``masked_scatter`` to row-level ``ScatterND``;
        see the docstring for the mask <-> indices equivalence proof.)
  MRoPEPositionIdsPrefillBlock  (input_ids, mm_token_type_ids, image_grid_thw)
                                                            -> position_ids, mrope_position_deltas
        ⇢ ``Qwen3_5MoeModel.compute_3d_position_ids``  line 1707 ``if`` branch
        ⇢ ``Qwen3_5MoeModel.get_rope_index``           line 1511
        ⇢ ``Qwen3_5MoeModel.get_vision_position_ids``  line 1455
  MRoPEPositionIdsDecodeBlock   (attention_mask, rope_deltas)
                                                            -> position_ids
        ⇢ ``Qwen3_5MoeModel.compute_3d_position_ids``  line 1720 ``elif`` branch

Representative-scenario static expansion
----------------------------------------
The prefill M-RoPE block statically expands the source's
``itertools.groupby`` segmentation by capturing the segment layout
``[text_pre | image | text_post]`` at construction time (matching the
project's "one representative request per export" convention used
throughout — e.g. ``vision_token_seq_len=1024`` chooses a 32×32 grid
for the vision tower). Inference engines that need a different
segmentation simply re-run the export with different
``--mrope_text_pre_len`` / ``--mm_image_token_count`` / etc.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Image mask build (Qwen3_5MoeModel.get_placeholder_mask, image branch)
# ---------------------------------------------------------------------------

class ImageMaskBuildBlock(nn.Module):
    """
    Construct the image-token placement mask consumed by ``MMInjectBlock``,
    mirroring the ``input_ids`` branch of
    ``Qwen3_5MoeModel.get_placeholder_mask``
    (``modeling_qwen3_5_moe.py`` lines 1666-1671 and 1678):

        special_image_mask = input_ids == self.config.image_token_id
        special_image_mask = (
            special_image_mask.unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )

    The graph keeps the source-side ``Equal → Unsqueeze → Expand`` chain
    visible. ``image_token_id`` is a graph input (not a baked-in constant)
    so the same exported graph is reusable across vocabulary changes.

    Inputs
    ------
    input_ids       : Tensor (B, S) int64
    image_token_id  : Tensor scalar int64

    Outputs
    -------
    image_mask : Tensor (B, S, hidden_size) bool
        Already broadcast to the embedding shape, ready for
        ``MMInjectBlock``'s ``masked_scatter``.
    """

    def __init__(self, *, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        image_token_id: torch.Tensor,
    ) -> torch.Tensor:
        special_image_mask = input_ids.eq(image_token_id)
        special_image_mask = special_image_mask.unsqueeze(-1).expand(
            -1, -1, self.hidden_size,
        )
        return special_image_mask.contiguous()


# ---------------------------------------------------------------------------
# 2. mm_inject (relocated from qwen_onnx_blocks_vision.py)
# ---------------------------------------------------------------------------

class MMInjectBlock(nn.Module):
    """
    Replace image-token rows in ``inputs_embeds`` with the merged
    ``image_embeds``, **mirroring source verbatim**:

        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    Source location: ``Qwen3_5MoeModel.forward`` line 1773.

    Source-aligned IO contract
    --------------------------
    The block's inputs are exactly the three tensors that source line 1773
    consumes — same names, same dtypes, same shapes:

        inputs_embeds : (B, S, H_text)        float
        image_mask    : (B, S, H_text)        bool   ← from ``get_placeholder_mask``
        image_embeds  : (N_image_tokens, H_text)  float

    ``image_mask`` is the direct output of ``ImageMaskBuildBlock`` (which
    mirrors ``get_placeholder_mask``, source lines 1666-1671:
    ``input_ids == image_token_id`` followed by
    ``unsqueeze(-1).expand_as(inputs_embeds)``), so at the ONNX-graph
    level ``image_mask_build_<seq>.onnx`` and ``mm_inject_<seq>.onnx``
    are now **directly chained** by the ``image_mask`` tensor:

        image_mask_build_*.onnx  ──image_mask:bool[B,S,H]──▶  mm_inject_*.onnx

    Note on ``unk__N``
    -----------------
    PyTorch's ONNX exporter lowers ``masked_scatter`` to
    ``NonZero -> Transpose -> ScatterND``. ``NonZero(image_mask)`` has a
    *data-dependent* output shape (the number of ``True`` positions in
    ``image_mask`` is a runtime value that ONNX static shape inference
    cannot resolve), so the exported graph carries a few ``unk__N``
    placeholders on the ``NonZero / Transpose / Gather / ScatterND``
    intermediates. **This is a property of the source's
    ``masked_scatter`` semantics**, identical in nature to the
    ``vision_cu_seqlens`` / ``Tile`` situation — and we leave it that
    way to remain bit-identical to source semantics. Quantitative
    analysis tools that need the static image-token count should read
    it from the ``image_mask_build_*.onnx`` consumer pattern (or from
    the export config's ``--mm_image_token_count``) rather than from
    the post-NonZero intermediate shapes.

    Inputs
    ------
    inputs_embeds : Tensor (B, S, H_text) float
    image_mask    : Tensor (B, S, H_text) bool
        Row-uniform: every row at an image-token ``(b, s)`` is fully
        True; produced by ``ImageMaskBuildBlock``.
    image_embeds  : Tensor (N_image_tokens, H_text) float

    Outputs
    -------
    inputs_embeds_out : Tensor (B, S, H_text)
        Same shape as ``inputs_embeds`` with image-token rows replaced.
    """

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        image_mask: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        return inputs_embeds.masked_scatter(image_mask, image_embeds)        # source line 1773


# ---------------------------------------------------------------------------
# 3. M-RoPE 3D position_ids — prefill (statically-unrolled get_rope_index)
# ---------------------------------------------------------------------------

class MRoPEPositionIdsPrefillBlock(nn.Module):
    """
    Build the 3D M-RoPE ``position_ids: int64[3, B, S]`` and
    ``mrope_position_deltas: int64[B, 1]`` for prefill, statically unrolling
    ``Qwen3_5MoeModel.get_rope_index`` (``modeling_qwen3_5_moe.py`` line 1511)
    plus the ``if`` branch of ``compute_3d_position_ids`` (line 1707).

    Static segment layout used for the representative prefill request::

        [ text_pre  (length L1) | image (image_token_count) | text_post (length L2) ]

    where ``L1 + image_token_count + L2 == seq_len`` and
    ``image_token_count = (image_grid_h // merge) * (image_grid_w // merge)``
    for ``temporal_patch_size = 1``. This mirrors what the source's
    ``itertools.groupby`` over ``mm_token_type_ids`` produces for a single
    image embedded in a text prompt.

    Inputs
    ------
    input_ids          : Tensor (B, S)               int64
    mm_token_type_ids  : Tensor (B, S)               int32
        Source-side modality tag (text=0, image=1, video=2). Consumed
        for source-alignment / topology surface; the segment boundaries
        in the exported graph come from the captured layout, mirroring
        the source's ``input_type_group`` (line 1571) which is the result
        of ``groupby`` on this tensor in PyTorch.
    image_grid_thw     : Tensor (num_images, 3)      int64
        ``[T, H, W]`` for the single image segment. Consumed for source-
        alignment.

    Outputs
    -------
    position_ids            : Tensor (3, B, S)       int64
    mrope_position_deltas   : Tensor (B, 1)          int64
        Both feed back into ``Qwen3_5MoeModel.forward`` exactly the same
        way as the source-side ``compute_3d_position_ids`` returns.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        seq_len: int,
        text_pre_len: int,
        spatial_merge_size: int,
        static_grid: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.text_pre_len = int(text_pre_len)
        self.spatial_merge_size = int(spatial_merge_size)
        # See VisionRotPosEmbBlock.__init__ for static_grid semantics.
        # When set, image_seq_length and the entire image-segment arange/
        # repeat chain fold to constants; output position_ids has fully
        # static shape [3, B, seq_len].
        self.static_grid = (
            (int(static_grid[0]), int(static_grid[1]), int(static_grid[2]))
            if static_grid is not None
            else None
        )

        if self.text_pre_len > self.seq_len:
            raise ValueError(
                "text_pre_len exceeds seq_len: "
                f"{self.text_pre_len} > {self.seq_len}",
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Source: ``Qwen3_5MoeModel.get_rope_index`` body (lines 1511-1645)
        plus ``compute_3d_position_ids`` ``if`` branch (line 1707).

        Static vs. tensor-driven mix
        ----------------------------
        Source's segmentation comes from
        ``itertools.groupby(enumerate(mm_token_type_ids.tolist()), ...)``
        — a Python loop over a tensor that **cannot be ONNX-traced**.
        For the representative single-image prefill scenario the layout
        is fixed at::

            [ text_pre (length L1) | image (length M) | text_post (length L2) ]

        with ``L1 + M + L2 == seq_len``. We adopt option **B** of the
        source-alignment trade-off:

        * ``L1`` is captured at ``__init__`` (Python int, since the
          source's ``groupby`` boundary is fundamentally a Python
          decision; the only way to re-derive it tensor-side would be a
          full reproduction of ``groupby`` in ONNX, which is heavy and
          numerically irrelevant for the representative scenario).
        * ``H``, ``W`` (and therefore the **image segment length**
          ``M = (H//merge) * (W//merge)``) are read from
          ``image_grid_thw`` as tensor scalars, so every source
          operation in ``get_vision_position_ids`` (line 1455) — the
          ``arange / repeat / repeat_interleave / full / stack`` chain —
          appears as a real ONNX node and is visible in tools like
          Netron.
        * ``L2 = seq_len - L1 - M`` is derived from the ``M`` tensor
          scalar and the captured Python ints. The total ``S`` dim of
          the output then becomes a symbolic dim (``L1 + M + L2``) which
          shape-inferencers express as a parameter; at runtime it
          matches the captured ``seq_len`` exactly.
        """
        device = input_ids.device
        dtype = input_ids.dtype  # int64

        L1 = self.text_pre_len
        merge = self.spatial_merge_size
        B = self.batch_size

        # ─── T/H/W from ``image_grid_thw`` ───
        # Source: ``grid_thw_list`` produced by ``.tolist()`` at line 1567
        # plus the unpack ``for modality_type, group in input_type_group``
        # at line 1571.
        if self.static_grid is None:
            # Dynamic / option B: tensor scalars (image-segment ranges
            # become true Range/Tile/OneHot ops with unk__N).
            h_v = image_grid_thw[0, 1]
            w_v = image_grid_thw[0, 2]
        else:
            # Static / option A: Python ints (image-segment ranges fold).
            _, h_v, w_v = self.static_grid

        # ─── Source line 1466-1469: get_vision_position_ids ───
        # ``llm_grid_t = grid_t // temporal_patch_size`` etc. T is fixed
        # at 1 for the representative single-image scenario and the
        # source's ``temporal_patch_size`` is also 1, so ``llm_grid_t == 1``
        # is hard-coded.
        llm_grid_h = h_v // merge
        llm_grid_w = w_v // merge
        image_seq_length = llm_grid_h * llm_grid_w  # T == 1

        # ─── Source line 1581-1586: text-pre segment ───
        # ``arange(text_len).view(1, -1).expand(3, -1) + current_pos``.
        # ``L1`` is a Python int (segment boundary, see class docstring),
        # so this segment's ``arange`` is a static-length op; the
        # exporter folds it into a constant initializer. This is
        # source-faithful for static segments.
        pre_ids = (
            torch.arange(L1, device=device, dtype=dtype).view(1, -1).expand(3, -1)
        )

        # ─── Source line 1493-1505: get_vision_position_ids body ───
        # Drive ``arange / repeat / repeat_interleave`` from tensor
        # scalars so each op appears in the ONNX graph.
        position_width = (
            torch.arange(llm_grid_w, device=device, dtype=dtype)
            + L1
        ).repeat(llm_grid_h)                                                     # T == 1: drop * llm_grid_t
        # ``dim=0`` is explicit (instead of source's default ``dim=None``)
        # to avoid a torch.onnx exporter bug with ``aten_repeat_interleave_
        # self_int`` when the ``repeats`` argument is a Python int. The
        # input is already 1-D so the result is identical.
        position_height = (
            torch.arange(llm_grid_h, device=device, dtype=dtype)
            + L1
        ).repeat_interleave(llm_grid_w, dim=0)                                   # T == 1
        # Source line 1500: ``torch.full((image_seq_length,), current_pos)``;
        # tensor-driven length lowers to ``Expand / ConstantOfShape``.
        position_temporal = (
            torch.zeros(image_seq_length, device=device, dtype=dtype) + L1
        )
        # Source line 1506: ``position_temporal = position_temporal * time_interval``
        # (default time_interval == 1; emitted for topology parity).
        time_interval = torch.tensor(1, device=device, dtype=dtype)
        position_temporal = position_temporal * time_interval
        image_ids = torch.stack(
            [position_temporal, position_height, position_width], dim=0,
        )                                                                        # line 1508

        # ─── Source line 1594: text-post segment ───
        # ``current_pos += max(grid_h, grid_w) // merge`` (already merged
        # so just ``max(llm_grid_h, llm_grid_w)``).
        if self.static_grid is None:
            post_start = torch.maximum(llm_grid_h, llm_grid_w) + L1
            L2_t = (
                torch.tensor(self.seq_len - L1, device=device, dtype=dtype)
                - image_seq_length
            )
        else:
            # Python ints fold to constants under tracer.
            post_start = max(llm_grid_h, llm_grid_w) + L1
            L2_t = self.seq_len - L1 - image_seq_length
        post_ids = (
            torch.arange(L2_t, device=device, dtype=dtype).view(1, -1).expand(3, -1)
            + post_start
        )

        # ─── Source line 1595: cat & reshape ───
        llm_positions = torch.cat([pre_ids, image_ids, post_ids], dim=1).reshape(3, -1)

        # ─── Source line 1599: assign per batch ───
        # Representative scenario uses a homogeneous batch (every batch
        # element shares the same layout); broadcast to [3, B, S].
        position_ids = llm_positions.unsqueeze(1).expand(3, B, -1).contiguous()

        # ─── Source line 1600-1601: deltas ───
        deltas = (llm_positions.max() + 1 - self.seq_len).to(dtype=dtype)
        mrope_position_deltas = deltas.expand(B).unsqueeze(1).contiguous()

        # ─── Topology anchor for unconsumed inputs ───
        # ``input_ids`` and ``mm_token_type_ids`` are read by source-side
        # ``get_rope_index`` for segmentation; the static-L1 capture
        # makes them irrelevant for the *value* of position_ids. Add a
        # zero-anchor so they remain visible as graph inputs (matching
        # source signature). ``image_grid_thw`` is consumed via h_v/w_v
        # in dynamic mode, but in static mode it is otherwise unused —
        # anchor it too so the graph signature is identical between
        # modes.
        anchor = (input_ids.to(dtype).sum() + mm_token_type_ids.to(dtype).sum()) * 0
        if self.static_grid is not None:
            anchor = anchor + image_grid_thw.to(dtype).sum() * 0
        position_ids = position_ids + anchor
        mrope_position_deltas = mrope_position_deltas + anchor

        return position_ids, mrope_position_deltas


# ---------------------------------------------------------------------------
# 4. M-RoPE 3D position_ids — decode (compute_3d_position_ids elif branch)
# ---------------------------------------------------------------------------

class MRoPEPositionIdsDecodeBlock(nn.Module):
    """
    Decode-step M-RoPE ``position_ids`` builder, faithfully reproducing
    the ``elif`` branch of ``Qwen3_5MoeModel.compute_3d_position_ids``
    (``modeling_qwen3_5_moe.py`` lines 1720-1730), specifically the
    ``attention_mask is not None`` sub-branch (the typical decode path):

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        position_ids = position_ids.view(1, batch_size, -1).repeat(3, 1, 1)
        delta = self.rope_deltas.repeat_interleave(
            batch_size // self.rope_deltas.shape[0], dim=0,
        )
        position_ids = position_ids + delta

    Inputs
    ------
    attention_mask : Tensor (B, ctx + 1) int64 / int32
    rope_deltas    : Tensor (B, 1)       int64
        The ``mrope_position_deltas`` produced by
        ``MRoPEPositionIdsPrefillBlock`` at the start of the request and
        cached on the host across decode steps (matches
        ``self.rope_deltas`` in the source).

    Outputs
    -------
    position_ids : Tensor (3, B, ctx + 1) int64
    """

    def __init__(self, *, batch_size: int) -> None:
        super().__init__()
        self.batch_size = int(batch_size)

    def forward(
        self,
        attention_mask: torch.Tensor,
        rope_deltas: torch.Tensor,
    ) -> torch.Tensor:
        # Match source line 1723 exactly: attention_mask -> long -> cumsum -> sub
        position_ids = attention_mask.long().cumsum(-1) - 1
        # Source line 1724: zero out masked positions
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        # Source line 1725: view + repeat to [3, B, S]
        B = attention_mask.shape[0]
        position_ids = position_ids.view(1, B, -1).repeat(3, 1, 1)
        # Source line 1729: rope_deltas repeat_interleave to match B
        repeats = self.batch_size // rope_deltas.shape[0]
        delta = rope_deltas.repeat_interleave(repeats, dim=0)
        # Source line 1730: broadcast add
        position_ids = position_ids + delta.view(1, B, 1)
        return position_ids
