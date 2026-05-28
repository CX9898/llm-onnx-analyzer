"""ONNX export blocks aligned with diffusers Z-Image source modules."""

from __future__ import annotations

import torch
import torch.nn as nn

from z_image_onnx_rope import AttentionStaticShape, install_export_processor, split_adaln_modulation


class TimestepEmbedBlock(nn.Module):
    """
    ``t_embedder(t * t_scale).type_as(x)`` — matches ``ZImageTransformer2DModel.forward``.
    """

    def __init__(self, embedder: nn.Module, t_scale: float, output_dtype: torch.dtype) -> None:
        super().__init__()
        self.embedder = embedder
        self.t_scale = float(t_scale)
        self.output_dtype = output_dtype

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.embedder(t * self.t_scale).to(self.output_dtype)


class CapEmbedBlock(nn.Module):
    def __init__(self, cap_embedder: nn.Sequential) -> None:
        super().__init__()
        self.cap_embedder = cap_embedder

    def forward(self, cap_feats: torch.Tensor) -> torch.Tensor:
        return self.cap_embedder(cap_feats)


class XPatchEmbedBlock(nn.Module):
    def __init__(self, x_embedder: nn.Linear) -> None:
        super().__init__()
        self.x_embedder = x_embedder

    def forward(self, patch_feats: torch.Tensor) -> torch.Tensor:
        return self.x_embedder(patch_feats)


class PatchifyAndEmbedBasicBlock(nn.Module):
    """
    ``ZImageTransformer2DModel.patchify_and_embed`` (basic mode, transformer_z_image.py:588).

    Inputs match pipeline boundary: latent ``[B,C,F,H,W]`` (P1), cap ``[B,S,cap_feat_dim]`` (TE hidden).
    """

    def __init__(self, transformer: nn.Module, patch_size: int, f_patch_size: int) -> None:
        super().__init__()
        self.transformer = transformer
        self.patch_size = patch_size
        self.f_patch_size = f_patch_size

    def forward(
        self,
        latent: torch.Tensor,
        cap_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        images = [latent[i] for i in range(latent.shape[0])]
        caps = [cap_feats[i] for i in range(cap_feats.shape[0])]
        x, cap, _, x_pos, cap_pos, x_pad, cap_pad = self.transformer.patchify_and_embed(
            images, caps, self.patch_size, self.f_patch_size
        )
        return (
            torch.stack(x, dim=0),
            torch.stack(cap, dim=0),
            torch.stack(x_pos, dim=0).to(torch.int32),
            torch.stack(cap_pos, dim=0).to(torch.int32),
            torch.stack(x_pad, dim=0),
            torch.stack(cap_pad, dim=0),
        )


class PrepareSequenceBlock(nn.Module):
    """
    Export-friendly ``_prepare_sequence`` for fixed-length representative scenes.

    Mirrors transformer_z_image.py:778-797 when all items share ``max_seqlen``:
    pad_token ``Where`` → RoPE cos/sin → no batch padding → attn_mask all-ones.
    """

    def __init__(
        self,
        transformer: nn.Module,
        *,
        pad_token: nn.Parameter,
        batch_size: int,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.pad_token = pad_token
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rope = RopeEmbedBlock(transformer.rope_embedder)

    def forward(
        self,
        feats: torch.Tensor,
        pos_ids: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = pad_mask.unsqueeze(-1)
        feats = torch.where(mask, self.pad_token, feats)
        rope_cos, rope_sin = self.rope(pos_ids)
        attn_mask = torch.ones(
            self.batch_size, self.seq_len, dtype=torch.bool, device=feats.device
        )
        return feats, rope_cos, rope_sin, attn_mask


class XEmbedPrepareBlock(nn.Module):
    """T3: ``all_x_embedder`` + ``_prepare_sequence`` (x branch)."""

    def __init__(
        self,
        transformer: nn.Module,
        x_embedder: nn.Linear,
        *,
        batch_size: int,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.x_embedder = XPatchEmbedBlock(x_embedder)
        self.prepare = PrepareSequenceBlock(
            transformer,
            pad_token=transformer.x_pad_token,
            batch_size=batch_size,
            seq_len=seq_len,
        )

    def forward(
        self,
        x_patch_feats: torch.Tensor,
        x_pos_ids: torch.Tensor,
        x_pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_emb = self.x_embedder(x_patch_feats)
        return self.prepare(x_emb, x_pos_ids, x_pad_mask)


class XBranchBlock(nn.Module):
    """T3+T4: ``x_embed_prepare`` + ``noise_refiner`` ×n (layer[0] repr unrolled in-graph)."""

    def __init__(
        self,
        transformer: nn.Module,
        x_embedder: nn.Linear,
        *,
        batch_size: int,
        seq_len: int,
        n_refiner_layers: int,
    ) -> None:
        super().__init__()
        self.embed = XEmbedPrepareBlock(
            transformer,
            x_embedder,
            batch_size=batch_size,
            seq_len=seq_len,
        )
        self.refiner = ZImageTransformerBlockExportBlock(
            transformer.noise_refiner[0],
            AttentionStaticShape.from_transformer(transformer, batch_size, seq_len),
        )
        self.n_refiner_layers = n_refiner_layers

    def forward(
        self,
        x_patch_feats: torch.Tensor,
        x_pos_ids: torch.Tensor,
        x_pad_mask: torch.Tensor,
        adaln_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_tokens, rope_cos, rope_sin, attn_mask = self.embed(x_patch_feats, x_pos_ids, x_pad_mask)
        for _ in range(self.n_refiner_layers):
            x_tokens = self.refiner(x_tokens, attn_mask, rope_cos, rope_sin, adaln_input)
        return x_tokens, rope_cos, rope_sin, attn_mask


class CapBranchBlock(nn.Module):
    """T5+T6: ``cap_embed_prepare`` + ``context_refiner`` ×n (layer[0] repr unrolled in-graph)."""

    def __init__(
        self,
        transformer: nn.Module,
        cap_embedder: nn.Sequential,
        *,
        batch_size: int,
        seq_len: int,
        n_refiner_layers: int,
    ) -> None:
        super().__init__()
        self.embed = CapEmbedPrepareBlock(
            transformer,
            cap_embedder,
            batch_size=batch_size,
            seq_len=seq_len,
        )
        self.refiner = ZImageTransformerBlockExportBlock(
            transformer.context_refiner[0],
            AttentionStaticShape.from_transformer(transformer, batch_size, seq_len),
        )
        self.n_refiner_layers = n_refiner_layers

    def forward(
        self,
        cap_feats_padded: torch.Tensor,
        cap_pos_ids: torch.Tensor,
        cap_pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cap_tokens, rope_cos, rope_sin, attn_mask = self.embed(
            cap_feats_padded, cap_pos_ids, cap_pad_mask
        )
        for _ in range(self.n_refiner_layers):
            cap_tokens = self.refiner(cap_tokens, attn_mask, rope_cos, rope_sin)
        return cap_tokens, rope_cos, rope_sin, attn_mask


class CapEmbedPrepareBlock(nn.Module):
    """T5: ``cap_embedder`` + ``_prepare_sequence`` (cap branch)."""

    def __init__(
        self,
        transformer: nn.Module,
        cap_embedder: nn.Sequential,
        *,
        batch_size: int,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.cap_embedder = CapEmbedBlock(cap_embedder)
        self.prepare = PrepareSequenceBlock(
            transformer,
            pad_token=transformer.cap_pad_token,
            batch_size=batch_size,
            seq_len=seq_len,
        )

    def forward(
        self,
        cap_feats_padded: torch.Tensor,
        cap_pos_ids: torch.Tensor,
        cap_pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cap_emb = self.cap_embedder(cap_feats_padded)
        return self.prepare(cap_emb, cap_pos_ids, cap_pad_mask)


class FinalOutputBlock(nn.Module):
    """T9: ``final_layer`` → image token slice → static ``unpatchify`` + pipeline cast."""

    def __init__(
        self,
        final_layer: nn.Module,
        *,
        batch_size: int,
        unified_seq_len: int,
        image_seq_len: int,
        patch_dim: int,
        out_channels: int,
        f_size: int,
        h_size: int,
        w_size: int,
        patch_size: int,
        f_patch_size: int,
    ) -> None:
        super().__init__()
        self.final = FinalLayerBlock(final_layer)
        self.slice = SliceImageTokensBlock(
            batch_size=batch_size,
            unified_seq_len=unified_seq_len,
            image_seq_len=image_seq_len,
            patch_dim=patch_dim,
        )
        self.unpatch = UnpatchifyStaticBlock(
            batch_size=batch_size,
            image_seq_len=image_seq_len,
            out_channels=out_channels,
            f_size=f_size,
            h_size=h_size,
            w_size=w_size,
            patch_size=patch_size,
            f_patch_size=f_patch_size,
        )

    def forward(
        self, hidden_states: torch.Tensor, adaln_input: torch.Tensor
    ) -> torch.Tensor:
        patch_out = self.final(hidden_states, adaln_input)
        image_patch = self.slice(patch_out)
        return self.unpatch(image_patch)


class RopeEmbedBlock(nn.Module):
    """
    ``RopeEmbedder`` with cos/sin outputs; supports ``pos_ids`` ``[B, S, 3]`` or ``[S, 3]``.
    """

    def __init__(self, rope_embedder) -> None:
        super().__init__()
        freqs_cis_list = rope_embedder.precompute_freqs_cis(
            rope_embedder.axes_dims,
            rope_embedder.axes_lens,
            rope_embedder.theta,
        )
        self.n_axes = len(rope_embedder.axes_dims)
        for i, freqs_cis in enumerate(freqs_cis_list):
            self.register_buffer(f"cos_axis_{i}", freqs_cis.real.contiguous(), persistent=False)
            self.register_buffer(f"sin_axis_{i}", freqs_cis.imag.contiguous(), persistent=False)

    def _embed_single(self, pos_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos_parts: list[torch.Tensor] = []
        sin_parts: list[torch.Tensor] = []
        for i in range(self.n_axes):
            index = pos_ids[:, i].long()
            cos_parts.append(self.get_buffer(f"cos_axis_{i}")[index])
            sin_parts.append(self.get_buffer(f"sin_axis_{i}")[index])
        return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)

    def forward(self, pos_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pos_ids.ndim == 2:
            cos, sin = self._embed_single(pos_ids)
            return cos.unsqueeze(0), sin.unsqueeze(0)
        cos_list, sin_list = [], []
        for b in range(pos_ids.shape[0]):
            c, s = self._embed_single(pos_ids[b])
            cos_list.append(c)
            sin_list.append(s)
        return torch.stack(cos_list, dim=0), torch.stack(sin_list, dim=0)


class SequenceConcatBasicBlock(nn.Module):
    """Basic mode ``_build_unified_sequence``: concat [x, cap] tokens and RoPE."""

    def forward(
        self,
        x_tokens: torch.Tensor,
        cap_tokens: torch.Tensor,
        x_rope_cos: torch.Tensor,
        x_rope_sin: torch.Tensor,
        cap_rope_cos: torch.Tensor,
        cap_rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unified = torch.cat([x_tokens, cap_tokens], dim=1)
        rope_cos = torch.cat([x_rope_cos, cap_rope_cos], dim=1)
        rope_sin = torch.cat([x_rope_sin, cap_rope_sin], dim=1)
        return unified, rope_cos, rope_sin


class SliceImageTokensBlock(nn.Module):
    def __init__(
        self,
        batch_size: int,
        unified_seq_len: int,
        image_seq_len: int,
        patch_dim: int,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.unified_seq_len = unified_seq_len
        self.image_seq_len = image_seq_len
        self.patch_dim = patch_dim
        self._cap_tail = unified_seq_len - image_seq_len

    def forward(self, patch_output: torch.Tensor) -> torch.Tensor:
        x = patch_output.reshape(self.batch_size, self.unified_seq_len, self.patch_dim)
        image, _ = x.split([self.image_seq_len, self._cap_tail], dim=1)
        return image.reshape(self.batch_size, self.image_seq_len, self.patch_dim)


class UnpatchifyStaticBlock(nn.Module):
    """
    Static ``unpatchify`` for fixed scene geometry (no list / ``Gather`` / dynamic ``Slice``).

    Output ``noise_pred`` ``[B, C, H, W] float32`` for scheduler (``pipeline_z_image.py`` L556-562).
    """

    def __init__(
        self,
        batch_size: int,
        image_seq_len: int,
        out_channels: int,
        f_size: int,
        h_size: int,
        w_size: int,
        patch_size: int,
        f_patch_size: int,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.image_seq_len = image_seq_len
        self.out_channels = out_channels
        self.f_size = f_size
        self.h_size = h_size
        self.w_size = w_size
        self.f_tokens = f_size // f_patch_size
        self.h_tokens = h_size // patch_size
        self.w_tokens = w_size // patch_size
        self.pf = f_patch_size
        self.ph = patch_size
        self.pw = patch_size

    def forward(self, image_patch_output: torch.Tensor) -> torch.Tensor:
        x = image_patch_output.reshape(
            self.batch_size,
            self.f_tokens,
            self.h_tokens,
            self.w_tokens,
            self.pf,
            self.ph,
            self.pw,
            self.out_channels,
        )
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
            self.batch_size, self.out_channels, self.f_size, self.h_size, self.w_size
        )
        return (-x.float()).squeeze(2)


class ZImageTransformerBlockExportBlock(nn.Module):
    """``ZImageTransformerBlock.forward`` with export attention processor."""

    def __init__(self, block: nn.Module, static: AttentionStaticShape) -> None:
        super().__init__()
        self.block = block
        self.static = static
        install_export_processor(block.attention, static)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        adaln_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        block = self.block
        x = hidden_states

        if block.modulation:
            assert adaln_input is not None
            mod = block.adaLN_modulation(adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = split_adaln_modulation(mod, self.static)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            attn_out = block.attention(
                block.attention_norm1(x) * scale_msa,
                attention_mask=attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            x = x + gate_msa * block.attention_norm2(attn_out)
            x = x + gate_mlp * block.ffn_norm2(block.feed_forward(block.ffn_norm1(x) * scale_mlp))
        else:
            attn_out = block.attention(
                block.attention_norm1(x),
                attention_mask=attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            x = x + block.attention_norm2(attn_out)
            x = x + block.ffn_norm2(block.feed_forward(block.ffn_norm1(x)))
        return x


class FinalLayerBlock(nn.Module):
    def __init__(self, final_layer: nn.Module) -> None:
        super().__init__()
        self.final_layer = final_layer

    def forward(self, hidden_states: torch.Tensor, adaln_input: torch.Tensor) -> torch.Tensor:
        return self.final_layer(hidden_states, c=adaln_input)


class VAEDecodeBlock(nn.Module):
    """``pipeline_z_image``: ``latents(f32).to(vae.dtype)`` → scale → decode."""

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae
        self.scaling_factor = float(vae.config.scaling_factor)
        shift = getattr(vae.config, "shift_factor", 0.0)
        self.shift_factor = float(shift) if shift is not None else 0.0

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        x = latents.to(dtype=self.vae.dtype)
        x = x / self.scaling_factor + self.shift_factor
        return self.vae.decode(x, return_dict=False)[0]


class TextEncodePaddedBlock(nn.Module):
    """
    ``text_encoder(...).hidden_states[-2]`` in ``text_encoder`` weight dtype (bf16).
    """

    def __init__(self, text_encoder: nn.Module) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.output_dtype = next(text_encoder.parameters()).dtype

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return out.hidden_states[-2].to(self.output_dtype)
