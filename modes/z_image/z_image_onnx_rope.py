"""ONNX-exportable RoPE + attention helpers for Z-Image blocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AttentionStaticShape:
    """Fixed tensor geometry for a single exported attention subgraph."""

    batch_size: int
    seq_len: int
    hidden_size: int
    num_heads: int
    head_dim: int

    @classmethod
    def from_transformer(cls, transformer, batch_size: int, seq_len: int) -> AttentionStaticShape:
        hidden_size = int(transformer.dim)
        num_heads = int(transformer.n_heads)
        head_dim = hidden_size // num_heads
        return cls(batch_size, seq_len, hidden_size, num_heads, head_dim)


def freqs_cis_to_cos_sin(freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert complex ``freqs_cis`` to cos/sin without complex ops in the graph."""
    if freqs_cis.is_complex():
        cos = freqs_cis.real
        sin = freqs_cis.imag
    else:
        raise TypeError("freqs_cis must be complex for conversion helper")
    return cos, sin


def apply_rotary_emb_onnx(
    x_in: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    static: AttentionStaticShape,
) -> torch.Tensor:
    """
    Real-valued RoPE matching ``ZSingleStreamAttnProcessor`` inner ``apply_rotary_emb``.

    Uses literal reshape sizes so ONNX shape inference stays fully static.
    """
    dtype = x_in.dtype
    b, s, h, d = static.batch_size, static.seq_len, static.num_heads, static.head_dim
    x = x_in.float().reshape(b, s, h, d // 2, 2)
    x0 = x[..., 0]
    x1 = x[..., 1]
    cos = cos.unsqueeze(2).float()
    sin = sin.unsqueeze(2).float()
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    out = torch.stack([y0, y1], dim=-1).reshape(b, s, h, d)
    return out.to(dtype)


def scaled_dot_product_attention_static(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    *,
    head_dim: int,
) -> torch.Tensor:
    """Explicit attention (MatMul + Softmax) for static ONNX export."""
    scale = head_dim**-0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    if attn_mask is not None:
        scores = scores + attn_mask
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, value)


class ZSingleStreamAttnProcessorExport:
    """Export surface for ``ZSingleStreamAttnProcessor`` (standard ONNX ops only)."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, static: AttentionStaticShape) -> None:
        self.static = static

    def __call__(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del encoder_hidden_states

        static = self.static
        b, s, h, d = static.batch_size, static.seq_len, static.num_heads, static.head_dim

        query = attn.to_q(hidden_states).reshape(b, s, h, d)
        key = attn.to_k(hidden_states).reshape(b, s, h, d)
        value = attn.to_v(hidden_states).reshape(b, s, h, d)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if rope_cos is not None and rope_sin is not None:
            cos, sin = rope_cos, rope_sin
        elif freqs_cis is not None:
            cos, sin = freqs_cis_to_cos_sin(freqs_cis)
        else:
            cos = sin = None

        if cos is not None and sin is not None:
            query = apply_rotary_emb_onnx(query, cos, sin, static)
            key = apply_rotary_emb_onnx(key, cos, sin, static)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        if attention_mask is not None and attention_mask.dtype == torch.bool:
            mask = torch.zeros_like(attention_mask, dtype=dtype)
            mask = mask.masked_fill(~attention_mask, float("-inf"))
        else:
            mask = attention_mask

        hidden_states = scaled_dot_product_attention_static(q, k, v, mask, head_dim=d)
        hidden_states = hidden_states.transpose(1, 2).reshape(b, s, h * d).to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        return output


def install_export_processor(attention_module: nn.Module, static: AttentionStaticShape) -> None:
    attention_module.set_processor(ZSingleStreamAttnProcessorExport(static))


def split_adaln_modulation(mod: torch.Tensor, static: AttentionStaticShape) -> tuple[torch.Tensor, ...]:
    """Split AdaLN ``[B, 4*H]`` into four ``[B, 1, H]`` tensors without ``chunk``/dynamic ``Slice``."""
    b, h = static.batch_size, static.hidden_size
    mod4 = mod.reshape(b, 4, h)
    return tuple(mod4[:, i, :].reshape(b, 1, h) for i in range(4))
