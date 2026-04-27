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
2. Vision attention runs in the variable-length form using ``cu_seqlens`` to
   pack multiple images/videos into one sequence. For ONNX export we collapse
   to the *single-segment* case (one image / one video at a time). This
   matches what the inference engine computes inside one image's tokens
   without changing the math.
3. Rotary positional embedding is 2D (height, width), pre-computed by the
   inference engine from ``grid_thw`` and fed in as ``cos / sin`` tensors.
   The Python control flow that derives them from ``grid_thw`` is *not*
   exported (Python-side data, not ONNX).
4. Patch merger reshapes ``(N_patches, hidden)`` to
   ``(N_patches // spatial_merge**2, hidden * spatial_merge**2)``,
   layer-norms, then projects to ``out_hidden_size`` with a 2-layer MLP.
5. Multimodal injection replaces image-token positions in
   ``inputs_embeds`` with merged image features via ``masked_scatter``.

Block inventory
---------------
  VisionPatchEmbedBlock  (pixel_values_flat)                  -> patch_embeds
  VisionBlockRepr        (hidden_states, cos, sin)            -> hidden_states
  VisionPatchMergerBlock (vision_features)                    -> image_embeds
  MMInjectBlock          (inputs_embeds, image_mask, image_embeds)
                                                              -> inputs_embeds

All blocks are *stateless w.r.t. external KV / state caches* — the vision
tower runs once per multimodal request, so there is no cache to thread
through.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ``apply_rotary_pos_emb_vision`` lives in transformers and is numerically
# identical across CPU/GPU. We re-use it (rather than re-implementing) to
# avoid drifting from upstream behaviour.
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeVisionAttention,
    Qwen3_5MoeVisionBlock,
    Qwen3_5MoeVisionMLP,
    Qwen3_5MoeVisionPatchEmbed,
    Qwen3_5MoeVisionPatchMerger,
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
# 2. Vision attention (single-segment, ONNX-friendly)
# ---------------------------------------------------------------------------

class _VisionSingleSegmentAttention(nn.Module):
    """
    ONNX-friendly variant of ``Qwen3_5MoeVisionAttention`` that assumes a
    *single packed segment* (e.g. one image or one video at a time). The math
    is identical to the per-segment branch of the upstream module — we just
    bypass the ``cu_seqlens``-based split because for a single segment the
    split is the identity, and ``ALL_ATTENTION_FUNCTIONS`` would otherwise
    introduce backend-specific control flow that does not lower cleanly to
    ONNX standard ops.

    The math we replicate:

        qkv = self.qkv(hidden) -> (S, 3, num_heads, head_dim)
        q,k,v = qkv.permute(1, 0, 2, 3).unbind(0)              # each (S, H, D)
        q,k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        # per-segment SDPA over a single (S,S) block, no causal mask
        attn = softmax((q @ k^T) * scaling) @ v
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
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states)  # (S, 3 * dim)
        qkv = qkv.reshape(seq_length, 3, self.num_heads, self.head_dim)
        # Layout to match upstream: (3, S, num_heads, head_dim)
        qkv = qkv.permute(1, 0, 2, 3)
        query_states, key_states, value_states = qkv.unbind(0)
        # Apply 2D rotary embedding (height, width) — identical to upstream
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin,
        )

        # (S, num_heads, head_dim) -> (1, num_heads, S, head_dim)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # Standard SDPA over a single (S, S) block, no causal mask.
        # Cast to float32 for numerically-stable softmax then back to model
        # dtype, matching the precision policy used throughout the codebase.
        orig_dtype = query_states.dtype
        attn_logits = torch.matmul(query_states, key_states.transpose(-2, -1))
        attn_logits = attn_logits * self.scaling
        attn_probs = torch.softmax(attn_logits.float(), dim=-1).to(orig_dtype)
        attn_output = torch.matmul(attn_probs, value_states)

        # (1, num_heads, S, head_dim) -> (S, num_heads * head_dim)
        attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(seq_length, -1)
        return self.proj(attn_output)


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
    A single Qwen3.5-MoE vision Transformer block, exported with a
    *single-segment* attention path (see ``_VisionSingleSegmentAttention``).

    Inputs
    ------
    hidden_states : Tensor (S, hidden_size)
    cos           : Tensor (S, head_dim)
    sin           : Tensor (S, head_dim)

    Outputs
    -------
    hidden_states : Tensor (S, hidden_size)
    """

    def __init__(self, block: Qwen3_5MoeVisionBlock) -> None:
        super().__init__()
        self.norm1 = block.norm1  # type: ignore[assignment]
        self.norm2 = block.norm2  # type: ignore[assignment]
        self.attn = _VisionSingleSegmentAttention(block.attn)
        self.mlp = _VisionMLPWrapper(block.mlp)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cos, sin)
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
# 5. Multimodal injection (image embeds -> inputs_embeds at image-token slots)
# ---------------------------------------------------------------------------

class MMInjectBlock(nn.Module):
    """
    Replace image-token positions in ``inputs_embeds`` with merged
    ``image_embeds``. Mirrors the line in
    ``Qwen3_5MoeModel.forward``::

        image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    For ONNX export we accept the ``image_mask`` directly (already broadcast
    to the embedding shape) instead of building it from ``input_ids`` so that
    the graph does not depend on a magic token-id constant — exactly one
    boolean tensor is the "where to inject" specification.

    Inputs
    ------
    inputs_embeds : Tensor (B, S, H_text)
    image_mask    : Bool   (B, S, H_text)   — True where image tokens live
    image_embeds  : Tensor (N_image_tokens, H_text)

    Outputs
    -------
    inputs_embeds_out : Tensor (B, S, H_text)
        Same shape as ``inputs_embeds`` with image-token slots replaced.
    """

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        image_mask: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        # ``masked_scatter`` is supported by the PyTorch ONNX exporter via
        # ScatterND lowering. The boolean mask determines linearised target
        # positions; ``image_embeds`` is read in flat (row-major) order to
        # fill them. This requires
        #     image_mask.sum() == image_embeds.numel()
        # which is guaranteed by the upstream tokenisation pipeline.
        return inputs_embeds.masked_scatter(image_mask, image_embeds.to(inputs_embeds.dtype))
