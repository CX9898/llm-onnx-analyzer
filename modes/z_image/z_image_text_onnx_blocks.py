"""ONNX-friendly Qwen3 text encoder blocks for Z-Image ``text_encode`` export."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TextEncodeStaticShape:
    batch_size: int
    seq_len: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int

    @classmethod
    def from_model(cls, model: nn.Module, batch_size: int, seq_len: int) -> TextEncodeStaticShape:
        cfg = model.config
        hidden_size = int(cfg.hidden_size)
        num_heads = int(cfg.num_attention_heads)
        num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_heads))
        head_dim = int(getattr(cfg, "head_dim", hidden_size // num_heads))
        return cls(batch_size, seq_len, hidden_size, num_heads, num_kv_heads, head_dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb_onnx(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = query * cos + _rotate_half(query) * sin
    k_embed = key * cos + _rotate_half(key) * sin
    return q_embed, k_embed


def repeat_kv_onnx(hidden_states: torch.Tensor, n_rep: int, static: TextEncodeStaticShape) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    b, kv, s, d = static.batch_size, static.num_kv_heads, static.seq_len, static.head_dim
    expanded = static.num_heads
    chunks = [hidden_states for _ in range(n_rep)]
    return torch.cat(chunks, dim=1).reshape(b, expanded, s, d)


def make_causal_attention_mask(
    batch_size: int,
    seq_len: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    rows = torch.arange(seq_len, device=device)
    cols = torch.arange(seq_len, device=device)
    mask = torch.where(
        cols[None, :] > rows[:, None],
        torch.tensor(float("-inf"), device=device, dtype=dtype),
        torch.zeros((), device=device, dtype=dtype),
    )
    return mask.view(1, 1, seq_len, seq_len).expand(batch_size, -1, -1, -1)


class TextRMSNormBlock(nn.Module):
    def __init__(self, norm: nn.Module) -> None:
        super().__init__()
        self.weight = norm.weight
        self.eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class TextRotaryEmbeddingBlock(nn.Module):
    def __init__(self, rotary: nn.Module, static: TextEncodeStaticShape, output_dtype: torch.dtype) -> None:
        super().__init__()
        self.register_buffer("inv_freq", rotary.inv_freq.detach().float(), persistent=False)
        self.attention_scaling = float(getattr(rotary, "attention_scaling", 1.0))
        self.static = static
        self.output_dtype = output_dtype

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        static = self.static
        b = static.batch_size
        inv = self.inv_freq.view(1, -1, 1).expand(b, -1, 1)
        pos = position_ids[:, None, :].float()
        freqs = (inv @ pos).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=self.output_dtype), sin.to(dtype=self.output_dtype)


class TextSelfAttentionBlock(nn.Module):
    def __init__(self, attn: nn.Module, static: TextEncodeStaticShape) -> None:
        super().__init__()
        self.static = static
        self.num_kv_groups = static.num_heads // static.num_kv_heads
        self.scaling = float(attn.scaling)
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.o_proj = attn.o_proj
        self.q_norm = TextRMSNormBlock(attn.q_norm)
        self.k_norm = TextRMSNormBlock(attn.k_norm)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        static = self.static
        b, s, h, nh, kv, d = (
            static.batch_size,
            static.seq_len,
            static.hidden_size,
            static.num_heads,
            static.num_kv_heads,
            static.head_dim,
        )

        q = self.q_norm(self.q_proj(hidden_states).view(b, s, nh, d)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(b, s, kv, d)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, s, kv, d).transpose(1, 2)

        q, k = apply_rotary_pos_emb_onnx(q, k, cos, sin)

        k = repeat_kv_onnx(k, self.num_kv_groups, static)
        v = repeat_kv_onnx(v, self.num_kv_groups, static)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(b, s, nh * d)
        return self.o_proj(attn_output)


class TextDecoderLayerExportBlock(nn.Module):
    def __init__(self, layer: nn.Module, static: TextEncodeStaticShape) -> None:
        super().__init__()
        self.input_layernorm = TextRMSNormBlock(layer.input_layernorm)
        self.self_attn = TextSelfAttentionBlock(layer.self_attn, static)
        self.post_attention_layernorm = TextRMSNormBlock(layer.post_attention_layernorm)
        self.mlp = layer.mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self.self_attn(x, cos, sin, attention_mask)
        hidden_states = residual + x

        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        x = self.mlp(x)
        return residual + x


class TextEmbedPrepareBlock(nn.Module):
    """Embedding + RoPE + causal/padding attention mask (text_encode front segment)."""

    def __init__(self, text_encoder: nn.Module, static: TextEncodeStaticShape) -> None:
        super().__init__()
        self.static = static
        self.output_dtype = next(text_encoder.parameters()).dtype
        self.embed_tokens = text_encoder.embed_tokens
        self.rotary = TextRotaryEmbeddingBlock(text_encoder.rotary_emb, static, self.output_dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        static = self.static
        b, s = static.batch_size, static.seq_len
        device = input_ids.device
        dtype = self.output_dtype

        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(s, device=device, dtype=torch.long).unsqueeze(0).expand(b, -1)
        cos, sin = self.rotary(position_ids)

        attn_mask = make_causal_attention_mask(b, s, dtype=dtype, device=device)
        pad = (1 - attention_mask.to(dtype)).view(b, 1, 1, s) * torch.tensor(-1e4, device=device, dtype=dtype)
        attn_mask = attn_mask + pad
        return hidden_states, cos, sin, attn_mask


class TextEncodeExportBlock(nn.Module):
    """
    Qwen3Model forward aligned with ``pipeline_z_image`` text path:
    ``model(..., output_hidden_states=True).hidden_states[-2]``.

    Uses eager attention + real-valued RoPE (no hub kernels / SDPA / COMPLEX128).
    """

    def __init__(self, text_encoder: nn.Module, static: TextEncodeStaticShape) -> None:
        super().__init__()
        self.static = static
        self.output_dtype = next(text_encoder.parameters()).dtype
        self.front = TextEmbedPrepareBlock(text_encoder, static)
        self.layers = nn.ModuleList(
            TextDecoderLayerExportBlock(layer, static) for layer in text_encoder.layers
        )
        self.norm = TextRMSNormBlock(text_encoder.norm)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states, cos, sin, attn_mask = self.front(input_ids, attention_mask)

        for idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, cos, sin, attn_mask)
            if idx == len(self.layers) - 2:
                return hidden_states.to(self.output_dtype)

        return self.norm(hidden_states).to(self.output_dtype)
