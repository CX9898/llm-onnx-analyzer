"""
ONNX-friendly wrapper modules for Qwen3.5-MoE transformer blocks.

Target model: Qwen3.5-35B-A3B  (Qwen3_5MoeForCausalLM / Qwen3_5MoeTextModel)

Architecture differences from Dense Qwen3
------------------------------------------
1. Layer types  — alternating pattern (default: every 4th layer is full_attention,
   the other 3 are linear_attention / GatedDeltaNet):
       layer_types[i] = "full_attention" if (i+1) % 4 == 0 else "linear_attention"

2. Full-attention — three changes vs Dense Qwen3:
   a. q_proj outputs num_heads * head_dim * 2 (second half is an attention gate).
   b. Partial RoPE: only the first (head_dim * partial_rotary_factor) dims are
      rotated (partial_rotary_factor = 0.25 by default, giving 64/256 = 64 dims).
   c. Gated output: attn_out *= sigmoid(gate)  before o_proj.

3. FFN → SparseMoeBlock (all layers, both full and linear attention):
   - Qwen3_5MoeTopKRouter : top_k = 8 out of num_experts = 256
   - Qwen3_5MoeExperts    : 256 experts; weights stored as 3-D tensors
   - shared_expert        : always-active MLP
   - shared_expert_gate   : sigmoid-gated scaling of shared output

4. RMSNorm formula: output = ((x / rms(x)) * (1 + weight)).to(orig_dtype)
   This differs from standard RMSNorm (weight * x, where weight is centred at 1).

5. Linear-attention (GatedDeltaNet):
   Recurrent state-space model with two export routes:
   - `GatedDeltaNetBlock`        : decode-only, seq_len = 1
   - `GatedDeltaNetPrefillBlock` : chunk-prefill, seq_len > 1
   Both expose the same state tensors (`conv_state`, `recurrent_state`), but
   the prefill block consumes a whole `[B, S, H]` sequence in one ONNX graph.

Block inventory
---------------
  EmbeddingBlock           (input_ids, embedding_weight) → hidden_states
  LMHeadBlock              (hidden_states, lm_head_weight) → logits
  RotaryEmbeddingBlockMoE  (position_ids)                  → cos, sin  [float32]
      Same structure as RotaryEmbeddingBlock but respects partial_rotary_factor
      via a pre-sliced inv_freq.

  MoENormBlock             (hidden_states)                 → output
      Uses (1 + weight) * normalised_x formula.

  MoeSelfAttentionBlock    (hidden_states, cos, sin, attention_mask,
                            past_key, past_value)
                           → attn_output, new_key, new_value
      Input hidden_states already layer-normed; output is before residual add.
      Handles gated q_proj and partial RoPE internally.

  GatedDeltaNetBlock       (hidden_states, conv_state, recurrent_state)
                           → output, new_conv_state, new_recurrent_state
      Decode mode only (seq_len = 1).  Input already layer-normed;
      output is before residual add.

  GatedDeltaNetPrefillBlock(hidden_states, conv_state, recurrent_state)
                           → output, new_conv_state, new_recurrent_state
      Chunk-prefill mode. Input already layer-normed; output is before
      residual add.

  MoeSparseMoeBlock        (hidden_states,
                            experts_gate_up, experts_down,
                            shared_gate_proj_w, shared_up_proj_w, shared_down_proj_w,
                            shared_expert_gate_w)
                           → ffn_output
      All weight matrices are *explicit ONNX inputs* (stateless).
      This keeps every ONNX file small regardless of num_experts, avoiding
      the protobuf 2 GB serialisation limit.
      Input already layer-normed; output is before residual add.

Residual connections
--------------------
All sub-blocks exclude residual adds.  The inference engine is responsible for:
    hidden = hidden + attn_output   (or gated_delta_net_output)
    hidden = hidden + ffn_output
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch._C as _C
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx import symbolic_helper as onnx_symbolic_helper


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _PureMoERMSNorm(nn.Module):
    """
    Pure-PyTorch RMSNorm matching Qwen3_5MoeRMSNorm's (1 + weight) formula.

    Standard formula  : output = (x / rms(x)) * weight
    Qwen3_5Moe formula: output = (x / rms(x)) * (1 + weight)
    (weight is initialised to 0, so (1+weight) starts at 1.)
    """

    def __init__(self, weight: torch.Tensor, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.clone().detach())
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        normed = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * (1.0 + self.weight.float())).to(orig_dtype)


def _wrap_moe_norm(norm_module) -> _PureMoERMSNorm:
    # Qwen3_5MoeRMSNorm uses .eps; Qwen3RMSNorm uses .variance_epsilon
    eps = getattr(norm_module, "eps",
                  getattr(norm_module, "variance_epsilon", 1e-6))
    return _PureMoERMSNorm(norm_module.weight.detach(), eps)


def _rotate_half_moe(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _shape_as_list(value) -> list[int | None] | None:
    try:
        sizes = onnx_symbolic_helper._get_tensor_sizes(value)
    except Exception:
        return None
    if sizes is None:
        return None
    return list(sizes)


def _tensor_type_like(value, sizes: list[int | None] | None = None, dtype: torch.dtype | None = None):
    value_type = value.type()
    if value_type is None or not isinstance(value_type, _C.TensorType):
        return None
    if dtype is not None:
        try:
            value_type = value_type.with_dtype(dtype)
        except Exception:
            pass
    if sizes is None:
        return value_type
    try:
        return value_type.with_sizes(sizes)
    except Exception:
        return value_type


def _set_value_type(value, value_type) -> None:
    if value_type is not None:
        value.setType(value_type)


def _chunk_gated_delta_rule_onnx(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    fuse_layout_prep_for_onnx_export: bool = False,
    fuse_mask_decay_for_onnx_export: bool = False,
    fuse_triangular_for_onnx_export: bool = False,
    fuse_chunk_step_for_onnx_export: bool = False,
    fuse_chunk_scan_for_onnx_export: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Pure-PyTorch chunk-prefill DeltaNet recurrence without in-place updates.

    This mirrors the HuggingFace torch fallback path used for Qwen3.5-MoE
    prefill, but rewrites the stateful chunk math in a trace-friendly way so
    it can be exported as one ONNX graph.
    """

    initial_dtype = query.dtype
    sequence_length = query.shape[1]
    query, key, value, k_beta, v_beta, g = _delta_net_chunk_layout_prep(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=chunk_size,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        fuse_for_onnx_export=fuse_layout_prep_for_onnx_export,
    )

    batch_size, num_heads, num_chunks, chunk_size, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    total_sequence_length = num_chunks * chunk_size
    g_cumsum, decay_mask, lower_keep_mask, upper_mask = _delta_net_mask_decay(
        g,
        fuse_for_onnx_export=fuse_mask_decay_for_onnx_export,
    )

    attn_base = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(lower_keep_mask, 0.0)
    attn = _delta_net_triangular_solve(
        attn_base,
        fuse_for_onnx_export=fuse_triangular_for_onnx_export,
    )

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g_cumsum.exp().unsqueeze(-1))

    if initial_state is None:
        last_recurrent_state = value.new_zeros(batch_size, num_heads, k_head_dim, v_head_dim)
    else:
        last_recurrent_state = initial_state.to(value)

    core_attn_out, last_recurrent_state = _delta_net_chunk_scan(
        query,
        key,
        value,
        decay_mask,
        g_cumsum,
        k_cumdecay,
        last_recurrent_state,
        upper_mask,
        fuse_for_onnx_export=fuse_chunk_scan_for_onnx_export,
        fuse_chunk_step_for_onnx_export=fuse_chunk_step_for_onnx_export,
    )
    core_attn_out = core_attn_out.reshape(batch_size, num_heads, total_sequence_length, v_head_dim)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)

    if not output_final_state:
        last_recurrent_state = None
    return core_attn_out, last_recurrent_state


def _delta_net_triangular_solve_expanded(attn_base: torch.Tensor) -> torch.Tensor:
    """
    Expanded lower-triangular solve used by the runnable PyTorch path.

    Given a strictly lower-triangular `attn_base = -A`, this returns
    `(I - attn_base)^-1 = (I + A)^-1`.
    """

    chunk_size = attn_base.shape[-1]
    solved_rows: list[torch.Tensor] = []
    for row_idx in range(chunk_size):
        if row_idx == 0:
            solved_row = torch.zeros_like(attn_base[..., 0, :])
        else:
            row = attn_base[..., row_idx, :row_idx]
            prev = torch.stack([solved_rows[j][..., :row_idx] for j in range(row_idx)], dim=-2)
            prefix = row + torch.matmul(row.unsqueeze(-2), prev).squeeze(-2)
            suffix = torch.zeros_like(attn_base[..., row_idx, row_idx:])
            solved_row = torch.cat([prefix, suffix], dim=-1)
        solved_rows.append(solved_row)

    attn = torch.stack(solved_rows, dim=-2)
    eye = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device).view(1, 1, 1, chunk_size, chunk_size)
    return attn + eye


class _DeltaNetTriangularSolveFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn_base: torch.Tensor) -> torch.Tensor:
        return _delta_net_triangular_solve_expanded(attn_base)

    @staticmethod
    def symbolic(g, attn_base):
        out = g.op("qwen_onnx::DeltaNetTriangularSolve", attn_base)
        attn_base_sizes = _shape_as_list(attn_base)
        _set_value_type(out, _tensor_type_like(attn_base, attn_base_sizes))
        return out


def _delta_net_triangular_solve(
    attn_base: torch.Tensor,
    *,
    fuse_for_onnx_export: bool,
) -> torch.Tensor:
    if fuse_for_onnx_export and torch.onnx.is_in_onnx_export():
        return cast(torch.Tensor, _DeltaNetTriangularSolveFn.apply(attn_base))
    return _delta_net_triangular_solve_expanded(attn_base)


def _delta_net_mask_decay_expanded(
    g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    chunk_size = int(g.shape[-1])
    g_cumsum = g.cumsum(dim=-1)
    decay_mask = ((g_cumsum.unsqueeze(-1) - g_cumsum.unsqueeze(-2)).tril().exp().float()).tril()
    lower_keep_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=g.device),
        diagonal=0,
    )
    upper_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=g.device),
        diagonal=1,
    )
    return g_cumsum, decay_mask, lower_keep_mask, upper_mask


class _DeltaNetMaskDecayFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _delta_net_mask_decay_expanded(g)

    @staticmethod
    def symbolic(g_graph, g):
        g_cumsum, decay_mask, lower_keep_mask, upper_mask = g_graph.op(
            "qwen_onnx::DeltaNetMaskDecay",
            g,
            outputs=4,
        )
        g_sizes = _shape_as_list(g)
        _set_value_type(g_cumsum, _tensor_type_like(g, g_sizes))
        if g_sizes is not None and len(g_sizes) == 4:
            decay_sizes = [g_sizes[0], g_sizes[1], g_sizes[2], g_sizes[3], g_sizes[3]]
            _set_value_type(decay_mask, _tensor_type_like(g, decay_sizes))
            mask_sizes = [g_sizes[3], g_sizes[3]]
            _set_value_type(lower_keep_mask, _tensor_type_like(g, mask_sizes, dtype=torch.bool))
            _set_value_type(upper_mask, _tensor_type_like(g, mask_sizes, dtype=torch.bool))
        return g_cumsum, decay_mask, lower_keep_mask, upper_mask


def _delta_net_mask_decay(
    g: torch.Tensor,
    *,
    fuse_for_onnx_export: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if fuse_for_onnx_export and torch.onnx.is_in_onnx_export():
        return cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            _DeltaNetMaskDecayFn.apply(g),
        )
    return _delta_net_mask_decay_expanded(g)


def _delta_net_chunk_step_expanded(
    q_i: torch.Tensor,
    k_i: torch.Tensor,
    v_i: torch.Tensor,
    decay_i: torch.Tensor,
    g_i: torch.Tensor,
    g_last: torch.Tensor,
    k_cumdecay_i: torch.Tensor,
    last_recurrent_state: torch.Tensor,
    upper_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    attn_qk = (q_i @ k_i.transpose(-1, -2) * decay_i).masked_fill(upper_mask, 0.0)
    v_prime = k_cumdecay_i @ last_recurrent_state
    v_new = v_i - v_prime
    attn_inter = (q_i * g_i.exp().unsqueeze(-1)) @ last_recurrent_state
    chunk_output = attn_inter + attn_qk @ v_new

    g_diff = (g_last.unsqueeze(-1) - g_i).exp().unsqueeze(-1)
    kgv = (k_i * g_diff).transpose(-1, -2) @ v_new
    new_state = last_recurrent_state * g_last.exp().unsqueeze(-1).unsqueeze(-1) + kgv
    return chunk_output, new_state


class _DeltaNetChunkStepFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_i: torch.Tensor,
        k_i: torch.Tensor,
        v_i: torch.Tensor,
        decay_i: torch.Tensor,
        g_i: torch.Tensor,
        g_last: torch.Tensor,
        k_cumdecay_i: torch.Tensor,
        last_recurrent_state: torch.Tensor,
        upper_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _delta_net_chunk_step_expanded(
            q_i,
            k_i,
            v_i,
            decay_i,
            g_i,
            g_last,
            k_cumdecay_i,
            last_recurrent_state,
            upper_mask,
        )

    @staticmethod
    def symbolic(
        g,
        q_i,
        k_i,
        v_i,
        decay_i,
        g_i,
        g_last,
        k_cumdecay_i,
        last_recurrent_state,
        upper_mask,
    ):
        chunk_output, new_state = g.op(
            "qwen_onnx::DeltaNetChunkStep",
            q_i,
            k_i,
            v_i,
            decay_i,
            g_i,
            g_last,
            k_cumdecay_i,
            last_recurrent_state,
            upper_mask,
            outputs=2,
        )
        _set_value_type(chunk_output, _tensor_type_like(v_i))
        _set_value_type(new_state, _tensor_type_like(last_recurrent_state))
        return chunk_output, new_state


def _delta_net_chunk_step(
    q_i: torch.Tensor,
    k_i: torch.Tensor,
    v_i: torch.Tensor,
    decay_i: torch.Tensor,
    g_i: torch.Tensor,
    g_last: torch.Tensor,
    k_cumdecay_i: torch.Tensor,
    last_recurrent_state: torch.Tensor,
    upper_mask: torch.Tensor,
    *,
    fuse_for_onnx_export: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if fuse_for_onnx_export and torch.onnx.is_in_onnx_export():
        return cast(
            tuple[torch.Tensor, torch.Tensor],
            _DeltaNetChunkStepFn.apply(
                q_i,
                k_i,
                v_i,
                decay_i,
                g_i,
                g_last,
                k_cumdecay_i,
                last_recurrent_state,
                upper_mask,
            ),
        )
    return _delta_net_chunk_step_expanded(
        q_i,
        k_i,
        v_i,
        decay_i,
        g_i,
        g_last,
        k_cumdecay_i,
        last_recurrent_state,
        upper_mask,
    )


def _delta_net_chunk_scan_expanded(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay_mask: torch.Tensor,
    g_cumsum: torch.Tensor,
    k_cumdecay: torch.Tensor,
    last_recurrent_state: torch.Tensor,
    upper_mask: torch.Tensor,
    *,
    fuse_chunk_step_for_onnx_export: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_chunks = int(query.shape[2])
    core_attn_chunks: list[torch.Tensor] = []
    current_state = last_recurrent_state
    for chunk_idx in range(num_chunks):
        q_i = query[:, :, chunk_idx]
        k_i = key[:, :, chunk_idx]
        v_i = value[:, :, chunk_idx]
        decay_i = decay_mask[:, :, chunk_idx]
        g_i = g_cumsum[:, :, chunk_idx]
        g_last = g_i[:, :, -1]

        chunk_output, current_state = _delta_net_chunk_step(
            q_i,
            k_i,
            v_i,
            decay_i,
            g_i,
            g_last,
            k_cumdecay[:, :, chunk_idx],
            current_state,
            upper_mask,
            fuse_for_onnx_export=fuse_chunk_step_for_onnx_export,
        )
        core_attn_chunks.append(chunk_output)
    return torch.stack(core_attn_chunks, dim=2), current_state


class _DeltaNetChunkScanFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        decay_mask: torch.Tensor,
        g_cumsum: torch.Tensor,
        k_cumdecay: torch.Tensor,
        last_recurrent_state: torch.Tensor,
        upper_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _delta_net_chunk_scan_expanded(
            query,
            key,
            value,
            decay_mask,
            g_cumsum,
            k_cumdecay,
            last_recurrent_state,
            upper_mask,
            fuse_chunk_step_for_onnx_export=False,
        )

    @staticmethod
    def symbolic(
        g,
        query,
        key,
        value,
        decay_mask,
        g_cumsum,
        k_cumdecay,
        last_recurrent_state,
        upper_mask,
    ):
        core_attn_chunks, new_state = g.op(
            "qwen_onnx::DeltaNetChunkScan",
            query,
            key,
            value,
            decay_mask,
            g_cumsum,
            k_cumdecay,
            last_recurrent_state,
            upper_mask,
            outputs=2,
        )
        _set_value_type(core_attn_chunks, _tensor_type_like(value))
        _set_value_type(new_state, _tensor_type_like(last_recurrent_state))
        return core_attn_chunks, new_state


def _delta_net_chunk_scan(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay_mask: torch.Tensor,
    g_cumsum: torch.Tensor,
    k_cumdecay: torch.Tensor,
    last_recurrent_state: torch.Tensor,
    upper_mask: torch.Tensor,
    *,
    fuse_for_onnx_export: bool,
    fuse_chunk_step_for_onnx_export: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if fuse_for_onnx_export and torch.onnx.is_in_onnx_export():
        return cast(
            tuple[torch.Tensor, torch.Tensor],
            _DeltaNetChunkScanFn.apply(
                query,
                key,
                value,
                decay_mask,
                g_cumsum,
                k_cumdecay,
                last_recurrent_state,
                upper_mask,
            ),
        )

    return _delta_net_chunk_scan_expanded(
        query,
        key,
        value,
        decay_mask,
        g_cumsum,
        k_cumdecay,
        last_recurrent_state,
        upper_mask,
        fuse_chunk_step_for_onnx_export=fuse_chunk_step_for_onnx_export,
    )


def _optional_bias(bias: torch.Tensor, has_bias: bool) -> torch.Tensor | None:
    return bias if has_bias else None


def _delta_net_input_proj_pack_expanded(
    hidden_states: torch.Tensor,
    z_weight: torch.Tensor,
    z_bias: torch.Tensor,
    b_weight: torch.Tensor,
    b_bias: torch.Tensor,
    a_weight: torch.Tensor,
    a_bias: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    *,
    z_has_bias: bool,
    b_has_bias: bool,
    a_has_bias: bool,
    qkv_has_bias: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z = F.linear(hidden_states, z_weight, _optional_bias(z_bias, z_has_bias))
    b = F.linear(hidden_states, b_weight, _optional_bias(b_bias, b_has_bias))
    a = F.linear(hidden_states, a_weight, _optional_bias(a_bias, a_has_bias))
    qkv = F.linear(hidden_states, qkv_weight, _optional_bias(qkv_bias, qkv_has_bias)).transpose(1, 2)
    return z, b, a, qkv


class _DeltaNetInputProjPackFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        z_weight: torch.Tensor,
        z_bias: torch.Tensor,
        b_weight: torch.Tensor,
        b_bias: torch.Tensor,
        a_weight: torch.Tensor,
        a_bias: torch.Tensor,
        qkv_weight: torch.Tensor,
        qkv_bias: torch.Tensor,
        z_has_bias: bool,
        b_has_bias: bool,
        a_has_bias: bool,
        qkv_has_bias: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _delta_net_input_proj_pack_expanded(
            hidden_states,
            z_weight,
            z_bias,
            b_weight,
            b_bias,
            a_weight,
            a_bias,
            qkv_weight,
            qkv_bias,
            z_has_bias=z_has_bias,
            b_has_bias=b_has_bias,
            a_has_bias=a_has_bias,
            qkv_has_bias=qkv_has_bias,
        )

    @staticmethod
    def symbolic(
        g,
        hidden_states,
        z_weight,
        z_bias,
        b_weight,
        b_bias,
        a_weight,
        a_bias,
        qkv_weight,
        qkv_bias,
        z_has_bias,
        b_has_bias,
        a_has_bias,
        qkv_has_bias,
    ):
        z, b, a, qkv = g.op(
            "qwen_onnx::DeltaNetInputProjPack",
            hidden_states,
            z_weight,
            z_bias,
            b_weight,
            b_bias,
            a_weight,
            a_bias,
            qkv_weight,
            qkv_bias,
            z_has_bias_i=int(z_has_bias),
            b_has_bias_i=int(b_has_bias),
            a_has_bias_i=int(a_has_bias),
            qkv_has_bias_i=int(qkv_has_bias),
            outputs=4,
        )
        hs_sizes = _shape_as_list(hidden_states)
        z_w_sizes = _shape_as_list(z_weight)
        b_w_sizes = _shape_as_list(b_weight)
        a_w_sizes = _shape_as_list(a_weight)
        qkv_w_sizes = _shape_as_list(qkv_weight)
        if hs_sizes is not None and len(hs_sizes) == 3:
            if z_w_sizes is not None and len(z_w_sizes) >= 1:
                _set_value_type(z, _tensor_type_like(hidden_states, [hs_sizes[0], hs_sizes[1], z_w_sizes[0]]))
            if b_w_sizes is not None and len(b_w_sizes) >= 1:
                _set_value_type(b, _tensor_type_like(hidden_states, [hs_sizes[0], hs_sizes[1], b_w_sizes[0]]))
            if a_w_sizes is not None and len(a_w_sizes) >= 1:
                _set_value_type(a, _tensor_type_like(hidden_states, [hs_sizes[0], hs_sizes[1], a_w_sizes[0]]))
            if qkv_w_sizes is not None and len(qkv_w_sizes) >= 1:
                _set_value_type(qkv, _tensor_type_like(hidden_states, [hs_sizes[0], qkv_w_sizes[0], hs_sizes[1]]))
        return z, b, a, qkv


def _delta_net_input_proj_pack(
    hidden_states: torch.Tensor,
    z_weight: torch.Tensor,
    z_bias: torch.Tensor,
    b_weight: torch.Tensor,
    b_bias: torch.Tensor,
    a_weight: torch.Tensor,
    a_bias: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    *,
    z_has_bias: bool,
    b_has_bias: bool,
    a_has_bias: bool,
    qkv_has_bias: bool,
    fuse_for_onnx_export: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if fuse_for_onnx_export and torch.onnx.is_in_onnx_export():
        return cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            _DeltaNetInputProjPackFn.apply(
                hidden_states,
                z_weight,
                z_bias,
                b_weight,
                b_bias,
                a_weight,
                a_bias,
                qkv_weight,
                qkv_bias,
                z_has_bias,
                b_has_bias,
                a_has_bias,
                qkv_has_bias,
            ),
        )
    return _delta_net_input_proj_pack_expanded(
        hidden_states,
        z_weight,
        z_bias,
        b_weight,
        b_bias,
        a_weight,
        a_bias,
        qkv_weight,
        qkv_bias,
        z_has_bias=z_has_bias,
        b_has_bias=b_has_bias,
        a_has_bias=a_has_bias,
        qkv_has_bias=qkv_has_bias,
    )


def _delta_net_causal_conv_prefill_expanded(
    qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    *,
    conv_has_bias: bool,
    conv_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    extended = torch.cat([conv_state, qkv], dim=-1)
    conv_out = F.conv1d(
        extended,
        conv_weight.unsqueeze(1),
        bias=_optional_bias(conv_bias, conv_has_bias),
        groups=conv_dim,
    )
    mixed_qkv = F.silu(conv_out[:, :, 1:]).transpose(1, 2).contiguous()
    new_conv_state = extended[:, :, -conv_weight.shape[-1]:]
    return mixed_qkv, new_conv_state


class _DeltaNetCausalConvPrefillFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        conv_has_bias: bool,
        conv_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _delta_net_causal_conv_prefill_expanded(
            qkv,
            conv_state,
            conv_weight,
            conv_bias,
            conv_has_bias=conv_has_bias,
            conv_dim=conv_dim,
        )

    @staticmethod
    def symbolic(g, qkv, conv_state, conv_weight, conv_bias, conv_has_bias, conv_dim):
        mixed_qkv, new_conv_state = g.op(
            "qwen_onnx::DeltaNetCausalConvPrefill",
            qkv,
            conv_state,
            conv_weight,
            conv_bias,
            conv_has_bias_i=int(conv_has_bias),
            conv_dim_i=int(conv_dim),
            outputs=2,
        )
        qkv_sizes = _shape_as_list(qkv)
        conv_state_sizes = _shape_as_list(conv_state)
        if qkv_sizes is not None and len(qkv_sizes) == 3:
            _set_value_type(mixed_qkv, _tensor_type_like(qkv, [qkv_sizes[0], qkv_sizes[2], qkv_sizes[1]]))
        _set_value_type(new_conv_state, _tensor_type_like(conv_state))
        return mixed_qkv, new_conv_state


def _delta_net_causal_conv_prefill(
    qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    *,
    conv_has_bias: bool,
    conv_dim: int,
    fuse_for_onnx_export: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if fuse_for_onnx_export and torch.onnx.is_in_onnx_export():
        return cast(
            tuple[torch.Tensor, torch.Tensor],
            _DeltaNetCausalConvPrefillFn.apply(
                qkv,
                conv_state,
                conv_weight,
                conv_bias,
                conv_has_bias,
                conv_dim,
            ),
        )
    return _delta_net_causal_conv_prefill_expanded(
        qkv,
        conv_state,
        conv_weight,
        conv_bias,
        conv_has_bias=conv_has_bias,
        conv_dim=conv_dim,
    )


def _delta_net_qkv_layout_gate_prep_expanded(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    head_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_len, _ = mixed_qkv.shape
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, num_k_heads, head_k_dim)
    key = key.reshape(batch_size, seq_len, num_k_heads, head_k_dim)
    value = value.reshape(batch_size, seq_len, num_v_heads, head_v_dim)
    if head_ratio > 1:
        query = query.repeat_interleave(head_ratio, dim=2)
        key = key.repeat_interleave(head_ratio, dim=2)
    beta = b.sigmoid()
    A = A_log.float().exp()
    g = -A * F.softplus(a.float() + dt_bias.float())
    return query, key, value, beta, g


class _DeltaNetQkvLayoutGatePrepFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
        head_ratio: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _delta_net_qkv_layout_gate_prep_expanded(
            mixed_qkv,
            b,
            a,
            A_log,
            dt_bias,
            num_k_heads=num_k_heads,
            head_k_dim=head_k_dim,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            head_ratio=head_ratio,
        )

    @staticmethod
    def symbolic(
        g,
        mixed_qkv,
        b,
        a,
        A_log,
        dt_bias,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
        head_ratio,
    ):
        query, key, value, beta, g_out = g.op(
            "qwen_onnx::DeltaNetQkvLayoutGatePrep",
            mixed_qkv,
            b,
            a,
            A_log,
            dt_bias,
            num_k_heads_i=int(num_k_heads),
            head_k_dim_i=int(head_k_dim),
            num_v_heads_i=int(num_v_heads),
            head_v_dim_i=int(head_v_dim),
            head_ratio_i=int(head_ratio),
            outputs=5,
        )
        mixed_sizes = _shape_as_list(mixed_qkv)
        if mixed_sizes is not None and len(mixed_sizes) == 3:
            query_key_sizes = [mixed_sizes[0], mixed_sizes[1], int(num_v_heads), int(head_k_dim)]
            value_sizes = [mixed_sizes[0], mixed_sizes[1], int(num_v_heads), int(head_v_dim)]
            beta_g_sizes = [mixed_sizes[0], mixed_sizes[1], int(num_v_heads)]
            _set_value_type(query, _tensor_type_like(mixed_qkv, query_key_sizes))
            _set_value_type(key, _tensor_type_like(mixed_qkv, query_key_sizes))
            _set_value_type(value, _tensor_type_like(mixed_qkv, value_sizes))
            _set_value_type(beta, _tensor_type_like(mixed_qkv, beta_g_sizes))
            _set_value_type(g_out, _tensor_type_like(mixed_qkv, beta_g_sizes))
        return query, key, value, beta, g_out


def _delta_net_qkv_layout_gate_prep(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    head_ratio: int,
    fuse_for_onnx_export: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if fuse_for_onnx_export and torch.onnx.is_in_onnx_export():
        return cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            _DeltaNetQkvLayoutGatePrepFn.apply(
                mixed_qkv,
                b,
                a,
                A_log,
                dt_bias,
                num_k_heads,
                head_k_dim,
                num_v_heads,
                head_v_dim,
                head_ratio,
            ),
        )
    return _delta_net_qkv_layout_gate_prep_expanded(
        mixed_qkv,
        b,
        a,
        A_log,
        dt_bias,
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
        num_v_heads=num_v_heads,
        head_v_dim=head_v_dim,
        head_ratio=head_ratio,
    )


def _delta_net_chunk_layout_prep_expanded(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)

    query = query.transpose(1, 2).contiguous().to(torch.float32)
    key = key.transpose(1, 2).contiguous().to(torch.float32)
    value = value.transpose(1, 2).contiguous().to(torch.float32)
    beta = beta.transpose(1, 2).contiguous().to(torch.float32)
    g = g.transpose(1, 2).contiguous().to(torch.float32)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

    if pad_size > 0:
        query_pad = query.new_zeros((batch_size, num_heads, pad_size, k_head_dim))
        key_pad = key.new_zeros((batch_size, num_heads, pad_size, k_head_dim))
        value_pad = value.new_zeros((batch_size, num_heads, pad_size, v_head_dim))
        scalar_pad = beta.new_zeros((batch_size, num_heads, pad_size))

        query = torch.cat((query, query_pad), dim=2)
        key = torch.cat((key, key_pad), dim=2)
        value = torch.cat((value, value_pad), dim=2)
        beta = torch.cat((beta, scalar_pad), dim=2)
        g = torch.cat((g, scalar_pad), dim=2)

    total_sequence_length = sequence_length + pad_size
    num_chunks = total_sequence_length // chunk_size
    scale = k_head_dim ** -0.5
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query_chunks = query.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
    key_chunks = key.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
    value_chunks = value.reshape(batch_size, num_heads, num_chunks, chunk_size, v_head_dim)
    k_beta_chunks = k_beta.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
    v_beta_chunks = v_beta.reshape(batch_size, num_heads, num_chunks, chunk_size, v_head_dim)
    g_chunks = g.reshape(batch_size, num_heads, num_chunks, chunk_size)
    return query_chunks, key_chunks, value_chunks, k_beta_chunks, v_beta_chunks, g_chunks


class _DeltaNetChunkLayoutPrepFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int,
        use_qk_l2norm_in_kernel: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _delta_net_chunk_layout_prep_expanded(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=chunk_size,
            use_qk_l2norm_in_kernel=bool(use_qk_l2norm_in_kernel),
        )

    @staticmethod
    def symbolic(
        g_graph,
        query,
        key,
        value,
        g,
        beta,
        chunk_size,
        use_qk_l2norm_in_kernel,
    ):
        query_chunks, key_chunks, value_chunks, k_beta_chunks, v_beta_chunks, g_chunks = g_graph.op(
            "qwen_onnx::DeltaNetChunkLayoutPrep",
            query,
            key,
            value,
            g,
            beta,
            chunk_size_i=int(chunk_size),
            use_qk_l2norm_in_kernel_i=int(use_qk_l2norm_in_kernel),
            outputs=6,
        )
        query_sizes = _shape_as_list(query)
        value_sizes = _shape_as_list(value)
        g_sizes = _shape_as_list(g)
        num_chunks = None
        if query_sizes is not None and len(query_sizes) == 4 and isinstance(query_sizes[1], int):
            seq_len = int(query_sizes[1])
            pad_size = (int(chunk_size) - seq_len % int(chunk_size)) % int(chunk_size)
            num_chunks = (seq_len + pad_size) // int(chunk_size)
        if query_sizes is not None and len(query_sizes) == 4:
            chunk_query_sizes = [query_sizes[0], query_sizes[2], num_chunks, int(chunk_size), query_sizes[3]]
            _set_value_type(query_chunks, _tensor_type_like(query, chunk_query_sizes))
            _set_value_type(key_chunks, _tensor_type_like(key, chunk_query_sizes))
            _set_value_type(k_beta_chunks, _tensor_type_like(key, chunk_query_sizes))
        if value_sizes is not None and len(value_sizes) == 4:
            chunk_value_sizes = [value_sizes[0], value_sizes[2], num_chunks, int(chunk_size), value_sizes[3]]
            _set_value_type(value_chunks, _tensor_type_like(value, chunk_value_sizes))
            _set_value_type(v_beta_chunks, _tensor_type_like(value, chunk_value_sizes))
        if g_sizes is not None and len(g_sizes) == 3:
            chunk_g_sizes = [g_sizes[0], g_sizes[2], num_chunks, int(chunk_size)]
            _set_value_type(g_chunks, _tensor_type_like(g, chunk_g_sizes))
        return query_chunks, key_chunks, value_chunks, k_beta_chunks, v_beta_chunks, g_chunks


def _delta_net_chunk_layout_prep(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int,
    use_qk_l2norm_in_kernel: bool,
    fuse_for_onnx_export: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if fuse_for_onnx_export and torch.onnx.is_in_onnx_export():
        return cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            _DeltaNetChunkLayoutPrepFn.apply(
                query,
                key,
                value,
                g,
                beta,
                chunk_size,
                use_qk_l2norm_in_kernel,
            ),
        )
    return _delta_net_chunk_layout_prep_expanded(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=chunk_size,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


class _ChunkGatedDeltaRuleFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, new_state = _chunk_gated_delta_rule_onnx(
            query,
            key,
            value,
            g=g,
            beta=beta,
            chunk_size=chunk_size,
            initial_state=recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            # Keep the trace-time execution fully expanded. The symbolic() below
            # replaces the whole block with one custom node; nesting more custom
            # autograd ONNX functions inside this forward path can trigger JIT
            # tuple-lowering failures during export.
            fuse_layout_prep_for_onnx_export=False,
            fuse_mask_decay_for_onnx_export=False,
            fuse_triangular_for_onnx_export=False,
            fuse_chunk_step_for_onnx_export=False,
            fuse_chunk_scan_for_onnx_export=False,
        )
        assert new_state is not None
        return out, new_state

    @staticmethod
    def symbolic(g_graph, query, key, value, g, beta, recurrent_state, chunk_size):
        out, new_state = g_graph.op(
            "qwen_onnx::ChunkGatedDeltaRule",
            query,
            key,
            value,
            g,
            beta,
            recurrent_state,
            chunk_size_i=int(chunk_size),
            outputs=2,
        )
        query_sizes = _shape_as_list(query)
        value_sizes = _shape_as_list(value)
        if query_sizes is not None and len(query_sizes) == 4:
            out_sizes = [
                query_sizes[0],
                query_sizes[1],
                query_sizes[2],
                value_sizes[-1] if value_sizes is not None and len(value_sizes) >= 1 else None,
            ]
            _set_value_type(out, _tensor_type_like(value, out_sizes))
        _set_value_type(new_state, _tensor_type_like(recurrent_state))
        return out, new_state


def _chunk_gated_delta_rule_structured(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    *,
    chunk_size: int,
    fuse_for_onnx_export: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if fuse_for_onnx_export and torch.onnx.is_in_onnx_export():
        return cast(
            tuple[torch.Tensor, torch.Tensor],
            _ChunkGatedDeltaRuleFn.apply(
                query,
                key,
                value,
                g,
                beta,
                recurrent_state,
                chunk_size,
            ),
        )
    out, new_state = _chunk_gated_delta_rule_onnx(
        query,
        key,
        value,
        g=g,
        beta=beta,
        chunk_size=chunk_size,
        initial_state=recurrent_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        fuse_layout_prep_for_onnx_export=fuse_for_onnx_export,
        fuse_mask_decay_for_onnx_export=False,
        fuse_triangular_for_onnx_export=True,
        fuse_chunk_step_for_onnx_export=True,
        fuse_chunk_scan_for_onnx_export=False,
    )
    assert new_state is not None
    return out, new_state


def _recurrent_gated_delta_rule_expanded(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    *,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, _ = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = recurrent_state.to(value)

    for token_idx in range(sequence_length):
        q_t = query[:, :, token_idx]
        k_t = key[:, :, token_idx]
        v_t = value[:, :, token_idx]
        g_t = g[:, :, token_idx].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, token_idx].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, token_idx] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class _RecurrentGatedDeltaRuleFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _recurrent_gated_delta_rule_expanded(
            query,
            key,
            value,
            g,
            beta,
            recurrent_state,
            use_qk_l2norm_in_kernel=True,
        )

    @staticmethod
    def symbolic(g_graph, query, key, value, g, beta, recurrent_state):
        out, new_state = g_graph.op(
            "qwen_onnx::RecurrentGatedDeltaRule",
            query,
            key,
            value,
            g,
            beta,
            recurrent_state,
            outputs=2,
        )
        query_sizes = _shape_as_list(query)
        value_sizes = _shape_as_list(value)
        if query_sizes is not None and len(query_sizes) == 4:
            out_sizes = [
                query_sizes[0],
                query_sizes[1],
                query_sizes[2],
                value_sizes[-1] if value_sizes is not None and len(value_sizes) >= 1 else None,
            ]
            _set_value_type(out, _tensor_type_like(value, out_sizes))
        _set_value_type(new_state, _tensor_type_like(recurrent_state))
        return out, new_state


def _recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    *,
    fuse_for_onnx_export: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if fuse_for_onnx_export and torch.onnx.is_in_onnx_export():
        return cast(
            tuple[torch.Tensor, torch.Tensor],
            _RecurrentGatedDeltaRuleFn.apply(
                query,
                key,
                value,
                g,
                beta,
                recurrent_state,
            ),
        )
    return _recurrent_gated_delta_rule_expanded(
        query,
        key,
        value,
        g,
        beta,
        recurrent_state,
        use_qk_l2norm_in_kernel=True,
    )


def _delta_net_gated_norm_out_proj_expanded(
    core_out: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
    *,
    norm_eps: float,
    num_v_heads: int,
    head_v_dim: int,
    out_proj_has_bias: bool,
) -> torch.Tensor:
    batch_size, seq_len = core_out.shape[0], core_out.shape[1]
    core_flat = core_out.reshape(-1, head_v_dim)
    z_flat = z.reshape(batch_size, seq_len, num_v_heads, head_v_dim).reshape(-1, head_v_dim)
    orig_dtype = core_flat.dtype
    x_f = core_flat.float()
    normed = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + norm_eps)
    normed = norm_weight * normed.to(orig_dtype)
    normed = (normed * F.silu(z_flat.float())).to(orig_dtype)
    normed = normed.reshape(batch_size, seq_len, -1)
    return F.linear(normed, out_proj_weight, _optional_bias(out_proj_bias, out_proj_has_bias))


class _DeltaNetGatedNormOutProjFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        core_out: torch.Tensor,
        z: torch.Tensor,
        norm_weight: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
        norm_eps: float,
        num_v_heads: int,
        head_v_dim: int,
        out_proj_has_bias: bool,
    ) -> torch.Tensor:
        return _delta_net_gated_norm_out_proj_expanded(
            core_out,
            z,
            norm_weight,
            out_proj_weight,
            out_proj_bias,
            norm_eps=norm_eps,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            out_proj_has_bias=out_proj_has_bias,
        )

    @staticmethod
    def symbolic(
        g,
        core_out,
        z,
        norm_weight,
        out_proj_weight,
        out_proj_bias,
        norm_eps,
        num_v_heads,
        head_v_dim,
        out_proj_has_bias,
    ):
        out = g.op(
            "qwen_onnx::DeltaNetGatedNormOutProj",
            core_out,
            z,
            norm_weight,
            out_proj_weight,
            out_proj_bias,
            norm_eps_f=float(norm_eps),
            num_v_heads_i=int(num_v_heads),
            head_v_dim_i=int(head_v_dim),
            out_proj_has_bias_i=int(out_proj_has_bias),
        )
        core_sizes = _shape_as_list(core_out)
        out_w_sizes = _shape_as_list(out_proj_weight)
        if core_sizes is not None and len(core_sizes) >= 2 and out_w_sizes is not None and len(out_w_sizes) >= 1:
            _set_value_type(out, _tensor_type_like(core_out, [core_sizes[0], core_sizes[1], out_w_sizes[0]]))
        return out


def _delta_net_gated_norm_out_proj(
    core_out: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
    *,
    norm_eps: float,
    num_v_heads: int,
    head_v_dim: int,
    out_proj_has_bias: bool,
    fuse_for_onnx_export: bool,
) -> torch.Tensor:
    if fuse_for_onnx_export and torch.onnx.is_in_onnx_export():
        return cast(
            torch.Tensor,
            _DeltaNetGatedNormOutProjFn.apply(
                core_out,
                z,
                norm_weight,
                out_proj_weight,
                out_proj_bias,
                norm_eps,
                num_v_heads,
                head_v_dim,
                out_proj_has_bias,
            ),
        )
    return _delta_net_gated_norm_out_proj_expanded(
        core_out,
        z,
        norm_weight,
        out_proj_weight,
        out_proj_bias,
        norm_eps=norm_eps,
        num_v_heads=num_v_heads,
        head_v_dim=head_v_dim,
        out_proj_has_bias=out_proj_has_bias,
    )


# ---------------------------------------------------------------------------
# 1. Stateless embedding / lm-head blocks
# ---------------------------------------------------------------------------

class EmbeddingBlock(nn.Module):
    """
    Token embedding look-up table (stateless; weight is an explicit input).

    Inputs
    ------
    input_ids        : LongTensor  (B, S)
    embedding_weight : FloatTensor (vocab_size, hidden_size)

    Outputs
    -------
    hidden_states : FloatTensor (B, S, hidden_size)
    """

    def forward(
        self,
        input_ids: torch.LongTensor,
        embedding_weight: torch.Tensor,
    ) -> torch.Tensor:
        return torch.nn.functional.embedding(input_ids, embedding_weight)


class LMHeadBlock(nn.Module):
    """
    Language model head (stateless; weight is an explicit input).

    Inputs
    ------
    hidden_states  : FloatTensor (B, S, hidden_size)
    lm_head_weight : FloatTensor (vocab_size, hidden_size)

    Outputs
    -------
    logits : FloatTensor (B, S, vocab_size)
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        return torch.nn.functional.linear(hidden_states, lm_head_weight)


# ---------------------------------------------------------------------------
# 2. Rotary embedding block (partial RoPE aware)
# ---------------------------------------------------------------------------

class RotaryEmbeddingBlockMoE(nn.Module):
    """
    Computes RoPE cos/sin tables for Qwen3.5-MoE.

    Qwen3.5-MoE uses partial_rotary_factor = 0.25, meaning only
    head_dim * 0.25 = 64 dimensions are rotated (out of 256).
    This block outputs cos/sin of shape (B, S, partial_dim), not full head_dim.
    The SelfAttention block applies them to the first partial_dim dims of q/k
    and concatenates the un-rotated remainder.

    Inputs
    ------
    position_ids : int64 (B, S)

    Outputs
    -------
    cos : model float dtype (B, S, partial_dim)
    sin : model float dtype (B, S, partial_dim)
    """

    def __init__(
        self,
        inv_freq: torch.Tensor,
        attention_scaling: float,
        *,
        output_dtype: torch.dtype,
        mrope_section: list[int] | tuple[int, ...] | None = None,
        mrope_interleaved: bool = False,
        export_batch_size: int | None = None,
        export_num_grids: int | None = None,
    ) -> None:
        super().__init__()
        # inv_freq already covers only the partial dims
        self.register_buffer("inv_freq", inv_freq.clone().detach().float())
        self.attention_scaling = float(attention_scaling)
        self.output_dtype = output_dtype
        self.mrope_interleaved = bool(mrope_interleaved)
        self.export_batch_size = int(export_batch_size) if export_batch_size is not None else None
        self.export_num_grids = int(export_num_grids) if export_num_grids is not None else None
        section = tuple(int(x) for x in (mrope_section or ()))
        if section and len(section) != 3:
            raise ValueError(f"mrope_section must have length 3, got {section!r}")
        self.mrope_section = section
        rotary_half_dim = int(inv_freq.numel())
        source_map = [0] * rotary_half_dim
        if self.mrope_interleaved and self.mrope_section:
            for source_idx, offset in ((1, 1), (2, 2)):
                length = min(int(self.mrope_section[source_idx]) * 3, rotary_half_dim)
                for dim_idx in range(offset, length, 3):
                    source_map[dim_idx] = source_idx
        self._mrope_source_map = tuple(source_map)

    def _apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        if not self.mrope_interleaved or not self.mrope_section:
            return freqs[0]
        pieces = [
            freqs[source_idx, :, :, dim_idx: dim_idx + 1]
            for dim_idx, source_idx in enumerate(self._mrope_source_map)
        ]
        return torch.cat(pieces, dim=-1)

    def forward(self, position_ids: torch.LongTensor) -> tuple:
        if position_ids.ndim == 2:
            num_grids = self.export_num_grids if self.export_num_grids is not None else 3
            batch_size = self.export_batch_size if self.export_batch_size is not None else position_ids.shape[0]
            # Prefer static repeat during export. The ONNX lowering for expand(-1)
            # keeps symbolic helper shapes alive, which in turn leaves rotary
            # gather outputs annotated as float32[unk__*, unk__*, 32] even when
            # this export path is fully static (e.g. decode B=1, S=1).
            if self.export_batch_size is not None:
                grid_position_ids = torch.cat([position_ids.unsqueeze(0) for _ in range(num_grids)], dim=0)
            else:
                grid_position_ids = position_ids[None, ...].expand(num_grids, batch_size, -1)
        else:
            grid_position_ids = position_ids
            num_grids = self.export_num_grids if self.export_num_grids is not None else grid_position_ids.shape[0]
            batch_size = self.export_batch_size if self.export_batch_size is not None else grid_position_ids.shape[1]
        inv = self.inv_freq.view(1, 1, -1, 1)
        if self.export_num_grids is not None and self.export_batch_size is not None:
            inv = torch.cat([inv for _ in range(num_grids)], dim=0)
            inv = torch.cat([inv for _ in range(batch_size)], dim=1)
        else:
            inv = inv.expand(num_grids, batch_size, -1, 1)
        pos = grid_position_ids[:, :, None, :].float()
        freqs = (inv @ pos).transpose(2, 3)
        if num_grids == 3:
            freqs = self._apply_interleaved_mrope(freqs)
        else:
            freqs = freqs[0]
        emb = torch.cat((freqs, freqs), dim=-1)                 # (B, S, partial_dim)
        scale = self.attention_scaling
        cos_out = emb.cos() * scale
        sin_out = emb.sin() * scale
        return cos_out.to(dtype=self.output_dtype), sin_out.to(dtype=self.output_dtype)


# ---------------------------------------------------------------------------
# 3. MoE RMSNorm block  (reused for both layer norms and the final norm)
# ---------------------------------------------------------------------------

class MoENormBlock(nn.Module):
    """
    Single Qwen3_5MoeRMSNorm with (1 + weight) formula.

    Inputs  : hidden_states (B, S, H)
    Outputs : output        (B, S, H)
    """

    def __init__(self, norm_module) -> None:
        super().__init__()
        self.norm = _wrap_moe_norm(norm_module)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


# ---------------------------------------------------------------------------
# 4. Full-attention self-attention sub-block  (no layernorm, no residual)
# ---------------------------------------------------------------------------

class MoeSelfAttentionBlock(nn.Module):
    """
    Pure self-attention sub-block for Qwen3.5-MoE full-attention layers.

    Differences from Dense Qwen3 SelfAttentionBlock:
      - q_proj size is 2× (second half is a per-head gate).
      - Partial RoPE: only the first partial_dim dims of q/k are rotated.
      - Attention output is gated: attn_out *= sigmoid(gate_flat).

    The input ``hidden_states`` must already be layer-normed.
    The returned ``attn_output`` must be added to the pre-norm residual.

    Inputs
    ------
    hidden_states  : Tensor  (B, S,  H)   — already layer-normed
    cos            : Tensor  (B, S,  partial_dim)
    sin            : Tensor  (B, S,  partial_dim)
    attention_mask : Tensor  (B, 1,  S, T)  additive mask (0 / -inf)
    past_key       : Tensor  (B, KV, P, D)
    past_value     : Tensor  (B, KV, P, D)

    Outputs
    -------
    attn_output : Tensor  (B, S,  H)   — before residual add
    new_key     : Tensor  (B, KV, T, D)   T = P + S
    new_value   : Tensor  (B, KV, T, D)
    """

    def __init__(
        self,
        layer,
        *,
        export_batch_size: int | None = None,
        export_seq_len: int | None = None,
        export_total_len: int | None = None,
        export_partial_dim: int | None = None,
    ) -> None:
        super().__init__()
        attn = layer.self_attn
        self.head_dim      = int(attn.head_dim)
        self.num_heads     = int(attn.q_proj.out_features // (self.head_dim * 2))
        self.num_kv_heads  = int(attn.k_proj.out_features // self.head_dim)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling       = float(attn.scaling)
        self.export_batch_size = int(export_batch_size) if export_batch_size is not None else None
        self.export_seq_len = int(export_seq_len) if export_seq_len is not None else None
        self.export_total_len = int(export_total_len) if export_total_len is not None else None
        self.export_partial_dim = int(export_partial_dim) if export_partial_dim is not None else None

        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.o_proj = attn.o_proj

        def _norm_eps(n):
            return getattr(n, "eps", getattr(n, "variance_epsilon", 1e-6))

        self.q_norm = _PureMoERMSNorm(attn.q_norm.weight.detach(), _norm_eps(attn.q_norm))
        self.k_norm = _PureMoERMSNorm(attn.k_norm.weight.detach(), _norm_eps(attn.k_norm))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
    ) -> tuple:
        B = self.export_batch_size if self.export_batch_size is not None else hidden_states.shape[0]
        S = self.export_seq_len if self.export_seq_len is not None else hidden_states.shape[1]
        D = self.head_dim
        merged_head_dim = self.num_heads * D

        # q_proj emits [q | gate] interleaved per-head; reshape and split
        q_raw = self.q_proj(hidden_states).view(B, S, self.num_heads, D * 2)
        q_proj_out, gate = q_raw[..., :D], q_raw[..., D:]   # each (B, S, NH, D)
        gate = gate.reshape(B, S, merged_head_dim)            # (B, S, NH*D)

        q = self.q_norm(q_proj_out).transpose(1, 2)           # (B, NH,  S, D)
        k = self.k_norm(
            self.k_proj(hidden_states).view(B, S, self.num_kv_heads, D)
        ).transpose(1, 2)                                      # (B, KV,  S, D)
        v = self.v_proj(hidden_states).view(
            B, S, self.num_kv_heads, D
        ).transpose(1, 2)                                      # (B, KV,  S, D)

        # Partial RoPE: rotate only the first partial_dim channels of q and k
        partial_dim = self.export_partial_dim if self.export_partial_dim is not None else cos.shape[-1]
        cos_ = cos.unsqueeze(1)   # (B, 1, S, partial_dim)
        sin_ = sin.unsqueeze(1)

        q_rot  = q[..., :partial_dim]
        q_pass = q[..., partial_dim:]
        k_rot  = k[..., :partial_dim]
        k_pass = k[..., partial_dim:]

        q_rot = q_rot * cos_ + _rotate_half_moe(q_rot) * sin_
        k_rot = k_rot * cos_ + _rotate_half_moe(k_rot) * sin_

        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)

        # KV cache
        k = torch.cat([past_key,   k], dim=2)
        v = torch.cat([past_value, v], dim=2)
        new_key, new_value = k, v

        KV = self.num_kv_heads
        T = self.export_total_len if self.export_total_len is not None else k.shape[2]
        expanded_heads = KV * self.num_kv_groups
        if self.export_batch_size is not None and self.export_seq_len is not None and self.export_total_len is not None:
            # Prefer explicit static head replication during export. The ONNX
            # lowering for expand() preserves symbolic helper shapes here and
            # leaves intermediate KV-cache tensors with unk__* dimensions even
            # though B/S/T are fixed for the canonical exports.
            k_exp = torch.cat([k for _ in range(self.num_kv_groups)], dim=1)
            v_exp = torch.cat([v for _ in range(self.num_kv_groups)], dim=1)
        else:
            k_exp = k[:, :, None, :, :].expand(B, KV, self.num_kv_groups, T, D)
            k_exp = k_exp.reshape(B, expanded_heads, T, D)
            v_exp = v[:, :, None, :, :].expand(B, KV, self.num_kv_groups, T, D)
            v_exp = v_exp.reshape(B, expanded_heads, T, D)

        attn_w = torch.matmul(q, k_exp.transpose(2, 3)) * self.scaling
        attn_w = attn_w + attention_mask
        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_out = torch.matmul(attn_w, v_exp)   # (B, NH, S, D)

        # Gate: sigmoid applied to the gate projection, then multiply
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, merged_head_dim)  # (B, S, NH*D)
        attn_out = attn_out * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_out)
        return attn_output, new_key, new_value


# ---------------------------------------------------------------------------
# 5. GatedDeltaNet block  (linear-attention, decode mode, seq_len = 1)
# ---------------------------------------------------------------------------

class GatedDeltaNetBlock(nn.Module):
    """
    GatedDeltaNet linear-attention sub-block — *decode mode only* (seq_len = 1).

    The model's chunk-based prefill implementation uses in-place mutations that
    cannot be TorchScript-traced.  This wrapper always uses the pure recurrent
    formula, which is ONNX-safe.  Run this ONNX file token-by-token for prefill.

    The input ``hidden_states`` must already be layer-normed.
    The returned ``output`` must be added to the pre-norm residual.

    Inputs
    ------
    hidden_states    : Tensor  (B, 1, H)  — seq_len MUST be 1
    conv_state       : Tensor  (B, conv_dim, kernel_size)
    recurrent_state  : Tensor  (B, num_v_heads, k_head_dim, v_head_dim)

    Outputs
    -------
    output           : Tensor  (B, 1, H)  — before residual add
    new_conv_state   : Tensor  (B, conv_dim, kernel_size)
    new_recurrent_state : Tensor  (B, num_v_heads, k_head_dim, v_head_dim)
    """

    def __init__(
        self,
        layer,
        *,
        fuse_recurrent_gated_delta_rule_for_onnx_export: bool = False,
        standardize_decode_conv_for_onnx_export: bool = False,
    ) -> None:
        super().__init__()
        la = layer.linear_attn
        self.fuse_recurrent_gated_delta_rule_for_onnx_export = bool(
            fuse_recurrent_gated_delta_rule_for_onnx_export
        )
        self.standardize_decode_conv_for_onnx_export = bool(
            standardize_decode_conv_for_onnx_export
        )

        self.num_v_heads  = int(la.num_v_heads)
        self.num_k_heads  = int(la.num_k_heads)
        self.head_k_dim   = int(la.head_k_dim)
        self.head_v_dim   = int(la.head_v_dim)
        self.key_dim      = int(la.key_dim)
        self.value_dim    = int(la.value_dim)
        self.conv_dim     = int(la.conv_dim)       # key_dim*2 + value_dim
        self.kernel_size  = int(la.conv_kernel_size)
        self.head_ratio   = self.num_v_heads // self.num_k_heads

        self.in_proj_qkv  = la.in_proj_qkv
        self.in_proj_z    = la.in_proj_z
        self.in_proj_b    = la.in_proj_b
        self.in_proj_a    = la.in_proj_a
        self.out_proj     = la.out_proj

        # Convolution weights  (depthwise: each channel has its own filter)
        # shape: (conv_dim, 1, kernel_size) → squeeze to (conv_dim, kernel_size)
        self.register_buffer(
            "conv_weight",
            la.conv1d.weight.detach().squeeze(1),   # (conv_dim, kernel_size)
        )
        if la.conv1d.bias is not None:
            self.register_buffer("conv_bias", la.conv1d.bias.detach())
        else:
            self.conv_bias = None

        # Recurrent decay parameters
        self.register_buffer("A_log",   la.A_log.detach())
        self.register_buffer("dt_bias", la.dt_bias.detach())

        # Gated RMSNorm (applied to core attention output)
        self.norm_weight = nn.Parameter(la.norm.weight.detach())
        self.norm_eps    = getattr(la.norm, "eps",
                                   getattr(la.norm, "variance_epsilon", 1e-6))

    # ------------------------------------------------------------------
    def _gated_rms_norm(
        self,
        x: torch.Tensor,       # (T, head_v_dim)
        gate: torch.Tensor,    # (T, head_v_dim)
    ) -> torch.Tensor:
        """Pure-PyTorch Qwen3_5MoeRMSNormGated."""
        orig = x.dtype
        x_f = x.float()
        normed = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        normed = self.norm_weight * normed.to(orig)
        return (normed * F.silu(gate.float())).to(orig)

    def _decode_causal_conv_update(
        self,
        qkv: torch.Tensor,         # (B, conv_dim, 1)
        conv_state: torch.Tensor,  # (B, conv_dim, kernel_size)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cat = torch.cat([conv_state, qkv], dim=-1)       # (B, conv_dim, kernel+1)
        new_conv_state = cat[:, :, 1:]                   # shift by 1

        if self.standardize_decode_conv_for_onnx_export and torch.onnx.is_in_onnx_export():
            conv_out = F.conv1d(
                cat,
                self.conv_weight.unsqueeze(1),
                bias=self.conv_bias,
                groups=self.conv_dim,
            )
            mixed_qkv = F.silu(conv_out[:, :, -1:]).transpose(1, 2).contiguous()
            return mixed_qkv, new_conv_state

        # Keep the eager path trace-friendly and close to the source update logic.
        win = cat[:, :, -self.kernel_size:]                                # (B, conv_dim, kernel_size)
        mixed_qkv = (win * self.conv_weight.unsqueeze(0)).sum(dim=-1)      # (B, conv_dim)
        if self.conv_bias is not None:
            mixed_qkv = mixed_qkv + self.conv_bias
        mixed_qkv = F.silu(mixed_qkv).unsqueeze(1)                         # (B, 1, conv_dim)
        return mixed_qkv, new_conv_state

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,   # (B, 1, H)
        conv_state: torch.Tensor,      # (B, conv_dim, kernel_size)
        recurrent_state: torch.Tensor, # (B, num_v_heads, k_head_dim, v_head_dim)
    ) -> tuple:
        B, _, _ = hidden_states.shape

        # ── Input projections ──────────────────────────────────────────
        z   = self.in_proj_z(hidden_states)           # (B, 1, value_dim)
        b   = self.in_proj_b(hidden_states)           # (B, 1, num_v_heads)
        a   = self.in_proj_a(hidden_states)           # (B, 1, num_v_heads)

        # qkv needs a conv1d step; move to (B, conv_dim, 1) for the conv
        qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)   # (B, conv_dim, 1)
        mixed_qkv, new_conv_state = self._decode_causal_conv_update(qkv, conv_state)

        # ── Split Q / K / V ────────────────────────────────────────────
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )  # each (B, 1, *)

        query = query.reshape(B, 1, self.num_k_heads, self.head_k_dim)
        key   = key.reshape  (B, 1, self.num_k_heads, self.head_k_dim)
        value = value.reshape(B, 1, self.num_v_heads, self.head_v_dim)

        # GQA expansion when num_v_heads > num_k_heads
        if self.head_ratio > 1:
            query = query.repeat_interleave(self.head_ratio, dim=2)
            key   = key.repeat_interleave(self.head_ratio, dim=2)

        # ── Recurrent parameters ────────────────────────────────────────
        beta  = b.sigmoid()                                 # (B, 1, num_v_heads)
        # g = -A.exp() * softplus(a + dt_bias)
        A     = self.A_log.float().exp()                    # (num_v_heads,)
        a_sq  = a.float()                                   # (B, 1, num_v_heads)
        g     = -A * F.softplus(a_sq + self.dt_bias.float())   # (B, num_v_heads)
        core_out, new_recurrent_state = _recurrent_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            recurrent_state,
            fuse_for_onnx_export=self.fuse_recurrent_gated_delta_rule_for_onnx_export,
        )

        # ── Gated RMSNorm ──────────────────────────────────────────────
        core_flat = core_out.reshape(-1, self.head_v_dim)   # (B*NH_v, v_dim)
        z_flat    = z.reshape(B, self.num_v_heads, self.head_v_dim).reshape(-1, self.head_v_dim)
        normed    = self._gated_rms_norm(core_flat, z_flat)
        normed    = normed.reshape(B, 1, -1)                # (B, 1, value_dim)

        output = self.out_proj(normed)                      # (B, 1, H)
        return output, new_conv_state, new_recurrent_state


class GatedDeltaNetPrefillBlock(GatedDeltaNetBlock):
    """
    GatedDeltaNet linear-attention sub-block — chunk-prefill mode.

    This wrapper consumes a whole prefill sequence and returns the full output
    sequence plus the final conv / recurrent states. Its recurrence math follows
    the chunk-prefill path rather than the single-step decode path.

    Inputs
    ------
    hidden_states    : Tensor  (B, S, H)  — seq_len can be > 1
    conv_state       : Tensor  (B, conv_dim, kernel_size)
    recurrent_state  : Tensor  (B, num_v_heads, k_head_dim, v_head_dim)

    Outputs
    -------
    output              : Tensor  (B, S, H)  — before residual add
    new_conv_state      : Tensor  (B, conv_dim, kernel_size)
    new_recurrent_state : Tensor  (B, num_v_heads, k_head_dim, v_head_dim)
    """

    def __init__(
        self,
        layer,
        chunk_size: int = 64,
        fuse_mask_decay_for_onnx_export: bool = False,
        fuse_triangular_for_onnx_export: bool = False,
        fuse_chunk_step_for_onnx_export: bool = False,
        fuse_chunk_scan_for_onnx_export: bool = False,
    ) -> None:
        super().__init__(layer)
        self.chunk_size = int(chunk_size)
        self.fuse_mask_decay_for_onnx_export = bool(fuse_mask_decay_for_onnx_export)
        self.fuse_triangular_for_onnx_export = bool(fuse_triangular_for_onnx_export)
        self.fuse_chunk_step_for_onnx_export = bool(fuse_chunk_step_for_onnx_export)
        self.fuse_chunk_scan_for_onnx_export = bool(fuse_chunk_scan_for_onnx_export)

    def _causal_conv_prefill(
        self,
        qkv: torch.Tensor,         # (B, conv_dim, S)
        conv_state: torch.Tensor,  # (B, conv_dim, kernel_size)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the same causal depthwise conv as decode mode, but over a full
        sequence using the provided initial conv state.
        """

        extended = torch.cat([conv_state, qkv], dim=-1)  # (B, conv_dim, kernel_size + S)
        conv_out = F.conv1d(
            extended,
            self.conv_weight.unsqueeze(1),
            bias=self.conv_bias,
            groups=self.conv_dim,
        )
        mixed_qkv = F.silu(conv_out[:, :, 1:]).transpose(1, 2).contiguous()
        new_conv_state = extended[:, :, -self.kernel_size:]
        return mixed_qkv, new_conv_state

    def forward(
        self,
        hidden_states: torch.Tensor,   # (B, S, H)
        conv_state: torch.Tensor,      # (B, conv_dim, kernel_size)
        recurrent_state: torch.Tensor, # (B, num_v_heads, k_head_dim, v_head_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, _ = hidden_states.shape

        z = self.in_proj_z(hidden_states)                 # (B, S, value_dim)
        b = self.in_proj_b(hidden_states)                 # (B, S, num_v_heads)
        a = self.in_proj_a(hidden_states)                 # (B, S, num_v_heads)

        qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # (B, conv_dim, S)
        mixed_qkv, new_conv_state = self._causal_conv_prefill(qkv, conv_state)

        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        query = query.reshape(B, S, self.num_k_heads, self.head_k_dim)
        key = key.reshape(B, S, self.num_k_heads, self.head_k_dim)
        value = value.reshape(B, S, self.num_v_heads, self.head_v_dim)

        if self.head_ratio > 1:
            query = query.repeat_interleave(self.head_ratio, dim=2)
            key = key.repeat_interleave(self.head_ratio, dim=2)

        beta = b.sigmoid()
        A = self.A_log.float().exp()
        g = -A * F.softplus(a.float() + self.dt_bias.float())
        core_out, new_recurrent_state = _chunk_gated_delta_rule_onnx(
            query,
            key,
            value,
            g=g,
            beta=beta,
            chunk_size=self.chunk_size,
            initial_state=recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            fuse_mask_decay_for_onnx_export=self.fuse_mask_decay_for_onnx_export,
            fuse_triangular_for_onnx_export=self.fuse_triangular_for_onnx_export,
            fuse_chunk_step_for_onnx_export=self.fuse_chunk_step_for_onnx_export,
            fuse_chunk_scan_for_onnx_export=self.fuse_chunk_scan_for_onnx_export,
        )
        assert new_recurrent_state is not None

        core_flat = core_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(B, S, self.num_v_heads, self.head_v_dim).reshape(-1, self.head_v_dim)
        normed = self._gated_rms_norm(core_flat, z_flat)
        normed = normed.reshape(B, S, -1)

        output = self.out_proj(normed)
        return output, new_conv_state, new_recurrent_state


class _DeltaNetStageBase(nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        la = cast(Any, layer.linear_attn)

        self.num_v_heads = int(la.num_v_heads)
        self.num_k_heads = int(la.num_k_heads)
        self.head_k_dim = int(la.head_k_dim)
        self.head_v_dim = int(la.head_v_dim)
        self.key_dim = int(la.key_dim)
        self.value_dim = int(la.value_dim)
        self.conv_dim = int(la.conv_dim)
        self.kernel_size = int(la.conv_kernel_size)
        self.head_ratio = self.num_v_heads // self.num_k_heads

        self.in_proj_qkv = cast(nn.Linear, la.in_proj_qkv)
        self.in_proj_z = cast(nn.Linear, la.in_proj_z)
        self.in_proj_b = cast(nn.Linear, la.in_proj_b)
        self.in_proj_a = cast(nn.Linear, la.in_proj_a)
        self.out_proj = cast(nn.Linear, la.out_proj)

        self.conv_weight: torch.Tensor
        self.register_buffer("conv_weight", la.conv1d.weight.detach().squeeze(1))
        if la.conv1d.bias is not None:
            self.conv_bias: torch.Tensor | None
            self.register_buffer("conv_bias", la.conv1d.bias.detach())
        else:
            self.conv_bias = None

        self.A_log: torch.Tensor
        self.dt_bias: torch.Tensor
        self.register_buffer("A_log", la.A_log.detach())
        self.register_buffer("dt_bias", la.dt_bias.detach())

        self.norm_weight = nn.Parameter(la.norm.weight.detach())
        self.norm_eps = getattr(la.norm, "eps", getattr(la.norm, "variance_epsilon", 1e-6))


class DeltaNetInputProjPackBlock(_DeltaNetStageBase):
    def __init__(self, layer, *, fuse_for_onnx_export: bool = False) -> None:
        super().__init__(layer)
        self.fuse_for_onnx_export = bool(fuse_for_onnx_export)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_bias = self.in_proj_z.bias if self.in_proj_z.bias is not None else self.in_proj_z.weight.new_empty(0)
        b_bias = self.in_proj_b.bias if self.in_proj_b.bias is not None else self.in_proj_b.weight.new_empty(0)
        a_bias = self.in_proj_a.bias if self.in_proj_a.bias is not None else self.in_proj_a.weight.new_empty(0)
        qkv_bias = (
            self.in_proj_qkv.bias if self.in_proj_qkv.bias is not None else self.in_proj_qkv.weight.new_empty(0)
        )
        return _delta_net_input_proj_pack(
            hidden_states,
            self.in_proj_z.weight,
            z_bias,
            self.in_proj_b.weight,
            b_bias,
            self.in_proj_a.weight,
            a_bias,
            self.in_proj_qkv.weight,
            qkv_bias,
            z_has_bias=self.in_proj_z.bias is not None,
            b_has_bias=self.in_proj_b.bias is not None,
            a_has_bias=self.in_proj_a.bias is not None,
            qkv_has_bias=self.in_proj_qkv.bias is not None,
            fuse_for_onnx_export=self.fuse_for_onnx_export,
        )


class DeltaNetCausalConvPrefillBlock(_DeltaNetStageBase):
    def __init__(self, layer, *, fuse_for_onnx_export: bool = False) -> None:
        super().__init__(layer)
        self.fuse_for_onnx_export = bool(fuse_for_onnx_export)

    def forward(
        self,
        qkv: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conv_bias = self.conv_bias if self.conv_bias is not None else self.conv_weight.new_empty(0)
        return _delta_net_causal_conv_prefill(
            qkv,
            conv_state,
            self.conv_weight,
            conv_bias,
            conv_has_bias=self.conv_bias is not None,
            conv_dim=self.conv_dim,
            fuse_for_onnx_export=self.fuse_for_onnx_export,
        )


class DeltaNetQkvLayoutGatePrepBlock(_DeltaNetStageBase):
    def __init__(self, layer, *, fuse_for_onnx_export: bool = False) -> None:
        super().__init__(layer)
        self.fuse_for_onnx_export = bool(fuse_for_onnx_export)

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _delta_net_qkv_layout_gate_prep(
            mixed_qkv,
            b,
            a,
            self.A_log,
            self.dt_bias,
            num_k_heads=self.num_k_heads,
            head_k_dim=self.head_k_dim,
            num_v_heads=self.num_v_heads,
            head_v_dim=self.head_v_dim,
            head_ratio=self.head_ratio,
            fuse_for_onnx_export=self.fuse_for_onnx_export,
        )


class DeltaNetChunkLayoutPrepBlock(nn.Module):
    def __init__(
        self,
        chunk_size: int = 64,
        *,
        use_qk_l2norm_in_kernel: bool = True,
        fuse_for_onnx_export: bool = False,
    ) -> None:
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.use_qk_l2norm_in_kernel = bool(use_qk_l2norm_in_kernel)
        self.fuse_for_onnx_export = bool(fuse_for_onnx_export)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _delta_net_chunk_layout_prep(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=self.chunk_size,
            use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
            fuse_for_onnx_export=self.fuse_for_onnx_export,
        )


class ChunkGatedDeltaRuleBlock(nn.Module):
    def __init__(
        self,
        chunk_size: int = 64,
        *,
        fuse_for_onnx_export: bool = False,
    ) -> None:
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.fuse_for_onnx_export = bool(fuse_for_onnx_export)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _chunk_gated_delta_rule_structured(
            query,
            key,
            value,
            g,
            beta,
            recurrent_state,
            chunk_size=self.chunk_size,
            fuse_for_onnx_export=self.fuse_for_onnx_export,
        )


class RecurrentGatedDeltaRuleBlock(nn.Module):
    def __init__(self, *, fuse_for_onnx_export: bool = False) -> None:
        super().__init__()
        self.fuse_for_onnx_export = bool(fuse_for_onnx_export)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _recurrent_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            recurrent_state,
            fuse_for_onnx_export=self.fuse_for_onnx_export,
        )


class DeltaNetMaskDecayBlock(nn.Module):
    def forward(
        self,
        g_chunks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _delta_net_mask_decay_expanded(g_chunks)


class DeltaNetTriangularSolveBlock(nn.Module):
    def forward(
        self,
        attn_base: torch.Tensor,
    ) -> torch.Tensor:
        return _delta_net_triangular_solve_expanded(attn_base)


class DeltaNetChunkStepBlock(nn.Module):
    def forward(
        self,
        q_i: torch.Tensor,
        k_i: torch.Tensor,
        v_i: torch.Tensor,
        decay_i: torch.Tensor,
        g_i: torch.Tensor,
        g_last: torch.Tensor,
        k_cumdecay_i: torch.Tensor,
        last_recurrent_state: torch.Tensor,
        upper_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _delta_net_chunk_step_expanded(
            q_i,
            k_i,
            v_i,
            decay_i,
            g_i,
            g_last,
            k_cumdecay_i,
            last_recurrent_state,
            upper_mask,
        )


class DeltaNetChunkScanBlock(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        decay_mask: torch.Tensor,
        g_cumsum: torch.Tensor,
        k_cumdecay: torch.Tensor,
        last_recurrent_state: torch.Tensor,
        upper_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _delta_net_chunk_scan(
            query,
            key,
            value,
            decay_mask,
            g_cumsum,
            k_cumdecay,
            last_recurrent_state,
            upper_mask,
            fuse_for_onnx_export=False,
            fuse_chunk_step_for_onnx_export=True,
        )


class DeltaNetGatedNormOutProjBlock(_DeltaNetStageBase):
    def __init__(self, layer, *, fuse_for_onnx_export: bool = False) -> None:
        super().__init__(layer)
        self.fuse_for_onnx_export = bool(fuse_for_onnx_export)

    def forward(
        self,
        core_out: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        out_proj_bias = self.out_proj.bias if self.out_proj.bias is not None else self.out_proj.weight.new_empty(0)
        return _delta_net_gated_norm_out_proj(
            core_out,
            z,
            self.norm_weight,
            self.out_proj.weight,
            out_proj_bias,
            norm_eps=self.norm_eps,
            num_v_heads=self.num_v_heads,
            head_v_dim=self.head_v_dim,
            out_proj_has_bias=self.out_proj.bias is not None,
            fuse_for_onnx_export=self.fuse_for_onnx_export,
        )


class StructuredGatedDeltaNetPrefillBlock(_DeltaNetStageBase):
    """
    Analysis-oriented DeltaNet prefill export with a clean top-level graph.

    The graph keeps the chunk-recurrence stages as custom nodes so the main
    ONNX stays readable, while the simpler front/back compute stages are
    expanded directly into the parent graph.
    """

    def __init__(
        self,
        layer,
        chunk_size: int = 64,
        *,
        fuse_stage_ops_for_onnx_export: bool = True,
        fuse_chunk_gated_delta_rule_for_onnx_export: bool = False,
    ) -> None:
        super().__init__(layer)
        self.chunk_size = int(chunk_size)
        self.fuse_stage_ops_for_onnx_export = bool(fuse_stage_ops_for_onnx_export)
        self.fuse_chunk_gated_delta_rule_for_onnx_export = bool(
            fuse_chunk_gated_delta_rule_for_onnx_export
        )
        if self.fuse_stage_ops_for_onnx_export and self.fuse_chunk_gated_delta_rule_for_onnx_export:
            raise ValueError(
                "fuse_stage_ops_for_onnx_export and "
                "fuse_chunk_gated_delta_rule_for_onnx_export are mutually exclusive."
            )
        self.input_proj_pack = DeltaNetInputProjPackBlock(
            layer,
            fuse_for_onnx_export=False,
        )
        self.causal_conv_prefill = DeltaNetCausalConvPrefillBlock(
            layer,
            fuse_for_onnx_export=False,
        )
        self.qkv_layout_gate_prep = DeltaNetQkvLayoutGatePrepBlock(
            layer,
            fuse_for_onnx_export=False,
        )
        self.gated_norm_out_proj = DeltaNetGatedNormOutProjBlock(
            layer,
            fuse_for_onnx_export=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, b, a, qkv = self.input_proj_pack(hidden_states)
        mixed_qkv, new_conv_state = self.causal_conv_prefill(qkv, conv_state)
        query, key, value, beta, g = self.qkv_layout_gate_prep(mixed_qkv, b, a)
        if self.fuse_chunk_gated_delta_rule_for_onnx_export:
            core_out, new_recurrent_state = _chunk_gated_delta_rule_structured(
                query,
                key,
                value,
                g,
                beta,
                recurrent_state,
                chunk_size=self.chunk_size,
                fuse_for_onnx_export=True,
            )
        else:
            core_out, new_recurrent_state = _chunk_gated_delta_rule_onnx(
                query,
                key,
                value,
                g=g,
                beta=beta,
                chunk_size=self.chunk_size,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                fuse_layout_prep_for_onnx_export=self.fuse_stage_ops_for_onnx_export,
                fuse_mask_decay_for_onnx_export=False,
                fuse_triangular_for_onnx_export=self.fuse_stage_ops_for_onnx_export,
                fuse_chunk_step_for_onnx_export=False,
                fuse_chunk_scan_for_onnx_export=self.fuse_stage_ops_for_onnx_export,
            )
        assert new_recurrent_state is not None
        output = self.gated_norm_out_proj(core_out, z)
        return output, new_conv_state, new_recurrent_state


# ---------------------------------------------------------------------------
# 6. SparseMoeBlock  (all layers; stateless — weights are explicit ONNX inputs)
# ---------------------------------------------------------------------------

class MoeSparseMoeBlock(nn.Module):
    """
    ONNX-friendly SparseMoeBlock.

    All expert and shared-expert weight matrices are passed as explicit
    ONNX *inputs* rather than baked-in constants.  This avoids the protobuf
    2 GB serialisation limit for large expert counts.

    Inputs
    ------
    hidden_states       : Tensor  (B, S, H)  — already layer-normed
    experts_gate_up     : Tensor  (E, 2*I, H)   E = model num_experts
    experts_down        : Tensor  (E, H,   I)
    shared_gate_proj_w  : Tensor  (I_s, H)
    shared_up_proj_w    : Tensor  (I_s, H)
    shared_down_proj_w  : Tensor  (H,   I_s)
    shared_expert_gate_w: Tensor  (1,   H)

    Outputs
    -------
    ffn_output : Tensor  (B, S, H)  — before residual add
    """

    def __init__(
        self,
        layer,
        *,
        export_batch_size: int | None = None,
        export_seq_len: int | None = None,
    ) -> None:
        super().__init__()
        moe = layer.mlp
        self.num_experts = int(moe.gate.num_experts)
        self.top_k = int(moe.gate.top_k)

        self.register_buffer(
            "router_weight",
            moe.gate.weight.detach(),
        )
        self.act_fn_name = moe.experts.act_fn.__class__.__name__
        self.export_batch_size = int(export_batch_size) if export_batch_size is not None else None
        self.export_seq_len = int(export_seq_len) if export_seq_len is not None else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        experts_gate_up: torch.Tensor,
        experts_down: torch.Tensor,
        shared_gate_proj_w: torch.Tensor,
        shared_up_proj_w: torch.Tensor,
        shared_down_proj_w: torch.Tensor,
        shared_expert_gate_w: torch.Tensor,
    ) -> torch.Tensor:
        B = self.export_batch_size if self.export_batch_size is not None else hidden_states.shape[0]
        S = self.export_seq_len if self.export_seq_len is not None else hidden_states.shape[1]
        H = hidden_states.shape[2]
        x = hidden_states.reshape(B * S, H)   # (T, H)

        # ── Router ─────────────────────────────────────────────────────
        # router_weight matches the real model expert count exactly.
        router_logits = F.linear(x, self.router_weight)               # (T, E)
        router_probs  = F.softmax(router_logits.float(), dim=-1).to(x.dtype)
        topk_w, topk_idx = router_probs.topk(self.top_k, dim=-1)      # (T, top_k)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

        # One-hot routing matrix over the real expert count.
        one_hot = F.one_hot(topk_idx, self.num_experts).to(x.dtype)  # (T, top_k, E)
        routing = (one_hot * topk_w.unsqueeze(-1)).sum(dim=1)               # (T, E)

        # ── Batched expert forward via einsum ──────────────────────────
        # experts_gate_up : (E, 2*I, H) — weight referenced ONCE as a graph input,
        #   never sliced into per-expert constants.  This avoids the TorchScript
        #   constant-folding that embeds 256×8 MB = 2 GB into the ONNX proto.
        #
        # F.linear(x, experts_gate_up[e]) ≡ x @ experts_gate_up[e].T
        # In batch form: all_gu[t,e,:] = Σ_h  x[t,h] * experts_gate_up[e,:,h]
        #   → einsum 'th,eih->tei'
        all_gu   = torch.einsum("th,eih->tei", x, experts_gate_up)    # (T, E, 2*I)
        gate_v, up_v = all_gu.chunk(2, dim=-1)                        # each (T, E, I)
        activated = F.silu(gate_v) * up_v                             # (T, E, I)

        # experts_down : (E, H, I)
        # F.linear(act, experts_down[e]) ≡ act @ experts_down[e].T
        # In batch: all_out[t,e,h] = Σ_i  activated[t,e,i] * experts_down[e,h,i]
        #   → einsum 'tei,ehi->teh'
        all_out  = torch.einsum("tei,ehi->teh", activated, experts_down)  # (T, E, H)

        # Weighted sum over experts
        output   = torch.einsum("te,teh->th", routing, all_out)        # (T, H)

        # ── Shared expert ──────────────────────────────────────────────
        shared_g  = F.linear(x, shared_gate_proj_w)
        shared_u  = F.linear(x, shared_up_proj_w)
        shared_out = F.linear(F.silu(shared_g) * shared_u, shared_down_proj_w)

        shared_w = torch.sigmoid(F.linear(x, shared_expert_gate_w))   # (T, 1)
        output   = output + shared_w * shared_out

        return output.reshape(B, S, H)
