#!/usr/bin/env python3
"""
Export merged ONNX blocks for representative Qwen3.5-MoE layers.

This script sits between the current very fine-grained sub-block export and a
future full-layer export. It exports one complete representative merged set:

- embedding_<seq>.onnx
- layer_00_linear_attn_block.onnx
- layer_03_full_attn_block_<seq>.onnx
- layer_00_moe_block_<seq>.onnx
- layer_03_moe_block_<seq>.onnx
- norm_<seq>.onnx
- lm_head_<seq>.onnx

The merged layer blocks add back the layer-local residual adds that were
missing from the fine-grained exports, so they are easier to connect into a
layer-level data flow.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from safetensors import safe_open

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qwen_export_shared import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeModelLike,
    _infer_checkpoint_torch_dtype,
    _layer_type,
    _model_float_dtype,
    _moe_weight_from_model,
    _onnx_export,
    _seq_tag,
    _text_config,
    _text_model,
    _torch_dtype_to_name,
    _write_structured_manifest,
)
from qwen_onnx_blocks import (
    ChunkGatedDeltaRuleBlock,
    DeltaNetChunkStepBlock,
    GatedDeltaNetBlock,
    GatedDeltaNetPrefillBlock,
    MoENormBlock,
    MoeSelfAttentionBlock,
    MoeSparseMoeBlock,
    RecurrentGatedDeltaRuleBlock,
    RotaryEmbeddingBlockMoE,
    StructuredGatedDeltaNetPrefillBlock,
)

_TORCH_TO_SAFETENSORS_DTYPE = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float64: "F64",
}


def _resolve_checkpoint_tensor_name(weight_map: dict[str, str], tensor_name: str) -> str | None:
    candidates = [tensor_name]
    if tensor_name.startswith("model."):
        candidates.append(f"model.language_model.{tensor_name[len('model.'):]}")
    for candidate in candidates:
        if candidate in weight_map:
            return candidate
    return None


def _module_attr_by_name(root: nn.Module, qualified_name: str) -> tuple[object, str]:
    current: object = root
    parts = qualified_name.split(".")
    for part in parts[:-1]:
        if isinstance(current, (nn.ModuleList, nn.Sequential)) and part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current, parts[-1]


def _restore_checkpoint_tensor_dtypes(model: Qwen3_5MoeModelLike, model_path: str) -> list[str]:
    index_path = Path(model_path) / "model.safetensors.index.json"
    if not index_path.exists():
        return []

    with index_path.open("r", encoding="utf-8") as handle:
        weight_map: dict[str, str] = json.load(handle)["weight_map"]

    param_map = dict(model.named_parameters())
    mismatch_names_by_file: dict[str, list[tuple[str, str]]] = {}
    for tensor_name, param in param_map.items():
        if not param.is_floating_point():
            continue
        checkpoint_name = _resolve_checkpoint_tensor_name(weight_map, tensor_name)
        if checkpoint_name is None:
            continue
        mismatch_names_by_file.setdefault(weight_map[checkpoint_name], []).append((tensor_name, checkpoint_name))

    restored: list[str] = []
    with torch.no_grad():
        for tensor_file, entries in mismatch_names_by_file.items():
            with safe_open(str(Path(model_path) / tensor_file), framework="pt", device="cpu") as reader:
                for tensor_name, checkpoint_name in entries:
                    param = param_map[tensor_name]
                    old_dtype = param.dtype
                    expected_dtype = _TORCH_TO_SAFETENSORS_DTYPE.get(old_dtype)
                    source_tensor = reader.get_tensor(checkpoint_name)
                    actual_dtype = _TORCH_TO_SAFETENSORS_DTYPE.get(source_tensor.dtype)
                    if tuple(source_tensor.shape) != tuple(param.shape):
                        continue
                    if expected_dtype == actual_dtype:
                        continue
                    owner, attr_name = _module_attr_by_name(model, tensor_name)
                    target = getattr(owner, attr_name)
                    target.data = source_tensor.to(device=target.device)
                    restored.append(f"{tensor_name}: {old_dtype} -> {source_tensor.dtype}")

    return restored


class LinearAttentionMergedBlock(nn.Module):
    """
    input_norm -> padding mask -> delta_net -> residual add
    """

    def __init__(
        self,
        model: Qwen3_5MoeModelLike,
        layer_idx: int,
        *,
        prefill: bool = False,
        chunk_size: int = 64,
        structured_prefill: bool = False,
        fuse_chunk_gated_delta_rule_for_onnx_export: bool = False,
        fuse_recurrent_gated_delta_rule_for_onnx_export: bool = False,
        standardize_decode_conv_for_onnx_export: bool = False,
    ) -> None:
        super().__init__()
        layer = _text_model(model).layers[layer_idx]
        la = cast(Any, layer.linear_attn)

        self.input_norm = MoENormBlock(layer.input_layernorm)
        self.prefill = bool(prefill)
        if structured_prefill:
            self.token_mixer = StructuredGatedDeltaNetPrefillBlock(
                layer,
                chunk_size=chunk_size,
                fuse_stage_ops_for_onnx_export=not fuse_chunk_gated_delta_rule_for_onnx_export,
                fuse_chunk_gated_delta_rule_for_onnx_export=fuse_chunk_gated_delta_rule_for_onnx_export,
            )
        elif self.prefill:
            self.token_mixer = GatedDeltaNetPrefillBlock(layer, chunk_size=chunk_size)
        else:
            self.token_mixer = GatedDeltaNetBlock(
                layer,
                fuse_recurrent_gated_delta_rule_for_onnx_export=fuse_recurrent_gated_delta_rule_for_onnx_export,
                standardize_decode_conv_for_onnx_export=standardize_decode_conv_for_onnx_export,
            )
        self.conv_dim = int(la.conv_dim)
        self.kernel_size = int(la.conv_kernel_size)
        self.num_v_heads = int(la.num_v_heads)
        self.head_k_dim = int(la.head_k_dim)
        self.head_v_dim = int(la.head_v_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        x = self.input_norm(hidden_states)
        x = x * padding_mask.to(x.dtype).unsqueeze(-1)
        x, new_conv_state, new_recurrent_state = self.token_mixer(x, conv_state, recurrent_state)
        hidden_states = residual + x
        return hidden_states, new_conv_state, new_recurrent_state


class FullAttentionMergedBlock(nn.Module):
    """
    input_norm -> rotary -> self_attn -> residual add
    """

    def __init__(
        self,
        model: Qwen3_5MoeModelLike,
        layer_idx: int,
        *,
        export_batch_size: int | None = None,
        export_seq_len: int | None = None,
        export_past_seq_len: int | None = None,
    ) -> None:
        super().__init__()
        text_model = _text_model(model)
        layer = text_model.layers[layer_idx]
        cfg = cast(Any, _text_config(model))
        rotary = cast(Any, text_model.rotary_emb)
        rope_parameters = getattr(cfg, "rope_parameters", {}) or {}
        float_dtype = _model_float_dtype(model)

        partial_factor = float(rope_parameters.get("partial_rotary_factor", 1.0))
        head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
        partial_dim = int(head_dim * partial_factor)
        inv_freq_partial = rotary.inv_freq[: partial_dim // 2].detach()

        self.input_norm = MoENormBlock(layer.input_layernorm)
        self.rotary = RotaryEmbeddingBlockMoE(
            inv_freq_partial,
            rotary.attention_scaling,
            output_dtype=float_dtype,
            mrope_section=rope_parameters.get("mrope_section"),
            mrope_interleaved=bool(rope_parameters.get("mrope_interleaved", False)),
            export_batch_size=export_batch_size,
            export_num_grids=3,
        )
        self.token_mixer = MoeSelfAttentionBlock(
            layer,
            export_batch_size=export_batch_size,
            export_seq_len=export_seq_len,
            export_total_len=(
                export_seq_len + export_past_seq_len
                if export_seq_len is not None and export_past_seq_len is not None
                else None
            ),
            export_partial_dim=partial_dim,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        x = self.input_norm(hidden_states)
        cos, sin = self.rotary(position_ids)
        x, new_key, new_value = self.token_mixer(x, cos, sin, attention_mask, past_key, past_value)
        hidden_states = residual + x
        return hidden_states, new_key, new_value


class MergedMoeBlock(nn.Module):
    """
    post_norm -> moe_ffn -> residual add
    """

    def __init__(
        self,
        model: Qwen3_5MoeModelLike,
        layer_idx: int,
        *,
        export_batch_size: int | None = None,
        export_seq_len: int | None = None,
    ) -> None:
        super().__init__()
        layer = _text_model(model).layers[layer_idx]
        self.post_norm = MoENormBlock(layer.post_attention_layernorm)
        self.moe = MoeSparseMoeBlock(
            layer,
            export_batch_size=export_batch_size,
            export_seq_len=export_seq_len,
        )

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
        residual = hidden_states
        x = self.post_norm(hidden_states)
        x = self.moe(
            x,
            experts_gate_up,
            experts_down,
            shared_gate_proj_w,
            shared_up_proj_w,
            shared_down_proj_w,
            shared_expert_gate_w,
        )
        return residual + x


def _export_prefill_linear_attn_stage_subgraphs(
    *,
    out_dir: str,
    prefix: str,
    batch_size: int,
    seq_len: int,
    chunk_size: int,
    nh_v: int,
    kd: int,
    vd: int,
    float_dtype: torch.dtype,
    opset: int,
    simplify: bool,
    fold_pure_shape_chains: bool,
) -> dict[str, str]:
    chunk_rule_prefix = f"{prefix}_ChunkGatedDeltaRule"
    recurrent_state_dtype = torch.float32
    step_compute_dtype = torch.float32

    stage_paths: dict[str, str] = {}
    step_path = os.path.join(
        out_dir,
        f"{chunk_rule_prefix}_DeltaNetChunkStep_chunk{chunk_size}.onnx",
    )
    _onnx_export(
        DeltaNetChunkStepBlock(),
        (
            torch.randn(batch_size, nh_v, chunk_size, kd, dtype=step_compute_dtype),
            torch.randn(batch_size, nh_v, chunk_size, kd, dtype=step_compute_dtype),
            torch.randn(batch_size, nh_v, chunk_size, vd, dtype=step_compute_dtype),
            torch.randn(batch_size, nh_v, chunk_size, chunk_size, dtype=step_compute_dtype),
            torch.randn(batch_size, nh_v, chunk_size, dtype=step_compute_dtype),
            torch.randn(batch_size, nh_v, dtype=step_compute_dtype),
            torch.randn(batch_size, nh_v, chunk_size, kd, dtype=step_compute_dtype),
            torch.zeros(batch_size, nh_v, kd, vd, dtype=recurrent_state_dtype),
            torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=1),
        ),
        step_path,
        input_names=[
            "q_i",
            "k_i",
            "v_i",
            "decay_i",
            "g_i",
            "g_last",
            "k_cumdecay_i",
            "recurrent_state",
            "upper_mask",
        ],
        output_names=["core_out", "new_recurrent_state"],
        dynamic_axes={},
        opset=opset,
        simplify=simplify,
        fold_pure_shape_chains=fold_pure_shape_chains,
    )
    stage_paths["delta_net_chunk_step"] = step_path
    return stage_paths


def _load_model(model_path: str, *, variant: str = "text") -> Qwen3_5MoeModelLike:
    """Load a Qwen3.5-MoE checkpoint as either the text-only or VL top-level class.

    Parameters
    ----------
    model_path : str
        Path to the HuggingFace-style checkpoint folder.
    variant : {"text", "vl"}
        - ``"text"``: load as ``Qwen3_5MoeForCausalLM``. Even when the on-disk
          checkpoint is a multimodal one, ``transformers`` transparently remaps
          ``model.language_model.*`` to ``model.*`` and ignores unused vision
          / MTP weights, so the text backbone loads with the correct values.
        - ``"vl"``: load as ``Qwen3_5MoeForConditionalGeneration``. Both the
          vision tower (``model.visual``) and the text backbone
          (``model.language_model``) are populated from the same checkpoint.
    """
    if variant not in ("text", "vl"):
        raise ValueError(f"variant must be 'text' or 'vl', got {variant!r}")

    torch_dtype = _infer_checkpoint_torch_dtype(model_path)
    resolved_dtype_name = _torch_dtype_to_name(torch_dtype)
    cls = Qwen3_5MoeForConditionalGeneration if variant == "vl" else Qwen3_5MoeForCausalLM
    print(
        f"Loading model from: {model_path!r} "
        f"(dtype={resolved_dtype_name}, variant={variant}, class={cls.__name__})"
    )
    model = cast(Any, cls).from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    restored = _restore_checkpoint_tensor_dtypes(model, model_path)
    if restored:
        print("Restored checkpoint storage dtypes for mixed-precision tensors:")
        for item in restored:
            print(f"  - {item}")
    model.eval()
    return model


def export_linear_attn_block(
    model: Qwen3_5MoeModelLike,
    layer_idx: int,
    out_dir: str,
    batch_size: int,
    opset: int,
    simplify: bool,
    seq_len: int,
    *,
    prefill: bool = False,
    chunk_size: int = 64,
    structured_prefill: bool = False,
    fuse_chunk_gated_delta_rule_for_onnx_export: bool = False,
    fuse_recurrent_gated_delta_rule_for_onnx_export: bool = False,
    standardize_decode_conv_for_onnx_export: bool = False,
    fold_pure_shape_chains: bool = False,
    strip_initializers: bool = False,
) -> None:
    cfg = cast(Any, _text_config(model))
    layer = _text_model(model).layers[layer_idx]
    la = cast(Any, layer.linear_attn)
    float_dtype = _model_float_dtype(model)
    h = cfg.hidden_size
    nh_v = int(la.num_v_heads)
    kd = int(la.head_k_dim)
    vd = int(la.head_v_dim)
    cd = int(la.conv_dim)
    ks = int(la.conv_kernel_size)
    recurrent_state_dtype = torch.float32

    block = LinearAttentionMergedBlock(
        model,
        layer_idx,
        prefill=prefill,
        chunk_size=chunk_size,
        structured_prefill=structured_prefill,
        fuse_chunk_gated_delta_rule_for_onnx_export=fuse_chunk_gated_delta_rule_for_onnx_export,
        fuse_recurrent_gated_delta_rule_for_onnx_export=fuse_recurrent_gated_delta_rule_for_onnx_export,
        standardize_decode_conv_for_onnx_export=standardize_decode_conv_for_onnx_export,
    )
    token_seq_len = seq_len if prefill else 1
    sample_inputs = (
        torch.randn(batch_size, token_seq_len, h, dtype=float_dtype),
        torch.zeros(batch_size, cd, ks, dtype=float_dtype),
        torch.zeros(batch_size, nh_v, kd, vd, dtype=recurrent_state_dtype),
        torch.ones(batch_size, token_seq_len, dtype=torch.bool),
    )
    tag = _seq_tag(seq_len)
    prefix = f"layer_{layer_idx:02d}_linear_attn_block"
    file_name = f"{prefix}_{tag}.onnx" if prefill else f"{prefix}.onnx"
    custom_opsets = None
    if fuse_chunk_gated_delta_rule_for_onnx_export or fuse_recurrent_gated_delta_rule_for_onnx_export:
        custom_opsets = {"qwen_onnx": 1}
    _onnx_export(
        block,
        sample_inputs,
        os.path.join(out_dir, file_name),
        input_names=["hidden_states", "conv_state", "recurrent_state", "padding_mask"],
        output_names=["hidden_states", "new_conv_state", "new_recurrent_state"],
        dynamic_axes={},
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
        custom_opsets=custom_opsets,
        fold_pure_shape_chains=fold_pure_shape_chains,
    )

    top_path = os.path.join(out_dir, file_name)

    if prefill and fuse_chunk_gated_delta_rule_for_onnx_export:
        stage_paths = _export_prefill_linear_attn_stage_subgraphs(
            out_dir=out_dir,
            prefix=prefix,
            batch_size=batch_size,
            seq_len=seq_len,
            chunk_size=chunk_size,
            nh_v=nh_v,
            kd=kd,
            vd=vd,
            float_dtype=float_dtype,
            opset=opset,
            simplify=simplify,
            fold_pure_shape_chains=fold_pure_shape_chains,
        )
        chunk_rule_path = os.path.join(
            out_dir,
            f"{prefix}_ChunkGatedDeltaRule_chunk{chunk_size}_{tag}.onnx",
        )
        _onnx_export(
            ChunkGatedDeltaRuleBlock(
                chunk_size=chunk_size,
                fuse_for_onnx_export=False,
            ),
            (
                torch.randn(batch_size, seq_len, nh_v, kd, dtype=float_dtype),
                torch.randn(batch_size, seq_len, nh_v, kd, dtype=float_dtype),
                torch.randn(batch_size, seq_len, nh_v, vd, dtype=float_dtype),
                torch.randn(batch_size, seq_len, nh_v, dtype=float_dtype),
                torch.randn(batch_size, seq_len, nh_v, dtype=float_dtype),
                torch.zeros(batch_size, nh_v, kd, vd, dtype=recurrent_state_dtype),
            ),
            chunk_rule_path,
            input_names=["query", "key", "value", "g", "beta", "recurrent_state"],
            output_names=["core_out", "new_recurrent_state"],
            dynamic_axes={},
            opset=opset,
            simplify=simplify,
            strip_initializers=strip_initializers,
            custom_opsets={"qwen_onnx": 1},
            fold_pure_shape_chains=fold_pure_shape_chains,
        )
        manifest_graph_defs: dict[str, dict[str, object | None]] = {
            "linear_attn_block_prefill_structured": {
                "path": top_path,
                "node_type": None,
            },
            "chunk_gated_delta_rule": {
                "path": chunk_rule_path,
                "node_type": "qwen_onnx::ChunkGatedDeltaRule",
                "inputs": ["query", "key", "value", "g", "beta", "recurrent_state"],
                "outputs": ["core_out", "new_recurrent_state"],
            },
            "delta_net_chunk_step": {
                "path": stage_paths["delta_net_chunk_step"],
                "node_type": "qwen_onnx::DeltaNetChunkStep",
                "inputs": [
                    "q_i",
                    "k_i",
                    "v_i",
                    "decay_i",
                    "g_i",
                    "g_last",
                    "k_cumdecay_i",
                    "recurrent_state",
                    "upper_mask",
                ],
                "outputs": ["core_out", "new_recurrent_state"],
            },
        }
        _write_structured_manifest(
            cast(Any, manifest_graph_defs),
            root_key="linear_attn_block_prefill_structured",
            save_path=os.path.join(out_dir, f"{prefix}_{tag}.json"),
        )

    if (not prefill) and fuse_recurrent_gated_delta_rule_for_onnx_export:
        recurrent_rule_path = os.path.join(
            out_dir,
            f"{prefix}_RecurrentGatedDeltaRule.onnx",
        )
        _onnx_export(
            RecurrentGatedDeltaRuleBlock(fuse_for_onnx_export=False),
            (
                torch.randn(batch_size, 1, nh_v, kd, dtype=float_dtype),
                torch.randn(batch_size, 1, nh_v, kd, dtype=float_dtype),
                torch.randn(batch_size, 1, nh_v, vd, dtype=float_dtype),
                torch.randn(batch_size, 1, nh_v, dtype=float_dtype),
                torch.randn(batch_size, 1, nh_v, dtype=float_dtype),
                torch.zeros(batch_size, nh_v, kd, vd, dtype=recurrent_state_dtype),
            ),
            recurrent_rule_path,
            input_names=["query", "key", "value", "g", "beta", "recurrent_state"],
            output_names=["core_out", "new_recurrent_state"],
            dynamic_axes={},
            opset=opset,
            simplify=simplify,
            strip_initializers=strip_initializers,
            fold_pure_shape_chains=fold_pure_shape_chains,
        )
        _write_structured_manifest(
            cast(Any, {
                "linear_attn_block_decode_structured": {
                    "path": top_path,
                    "node_type": None,
                },
                "recurrent_gated_delta_rule": {
                    "path": recurrent_rule_path,
                    "node_type": "qwen_onnx::RecurrentGatedDeltaRule",
                    "inputs": ["query", "key", "value", "g", "beta", "recurrent_state"],
                    "outputs": ["core_out", "new_recurrent_state"],
                },
            }),
            root_key="linear_attn_block_decode_structured",
            save_path=os.path.join(out_dir, f"{prefix}.json"),
        )


def export_linear_attn_prefill_bundle(
    model: Qwen3_5MoeModelLike,
    layer_idx: int,
    out_dir: str,
    batch_size: int,
    opset: int,
    simplify: bool,
    seq_len: int,
    *,
    chunk_size: int = 64,
    fold_pure_shape_chains: bool = False,
    strip_initializers: bool = False,
) -> None:
    """Canonical prefill export: structured top graph + chunk-rule subgraphs."""
    export_linear_attn_block(
        model,
        layer_idx,
        out_dir,
        batch_size,
        opset,
        simplify,
        seq_len,
        prefill=True,
        chunk_size=chunk_size,
        structured_prefill=True,
        fuse_chunk_gated_delta_rule_for_onnx_export=True,
        fold_pure_shape_chains=fold_pure_shape_chains,
        strip_initializers=strip_initializers,
    )


def export_linear_attn_decode_bundle(
    model: Qwen3_5MoeModelLike,
    layer_idx: int,
    out_dir: str,
    batch_size: int,
    opset: int,
    simplify: bool,
    *,
    fold_pure_shape_chains: bool = False,
    strip_initializers: bool = False,
) -> None:
    """Canonical decode export: recurrent-rule top graph + structured manifest."""
    export_linear_attn_block(
        model,
        layer_idx,
        out_dir,
        batch_size,
        opset,
        simplify,
        1,
        prefill=False,
        fuse_recurrent_gated_delta_rule_for_onnx_export=True,
        standardize_decode_conv_for_onnx_export=True,
        fold_pure_shape_chains=fold_pure_shape_chains,
        strip_initializers=strip_initializers,
    )


def export_full_attn_block(
    model: Qwen3_5MoeModelLike,
    layer_idx: int,
    out_dir: str,
    batch_size: int,
    opset: int,
    simplify: bool,
    seq_len: int,
    *,
    past_seq_len: int = 0,
    fold_pure_shape_chains: bool = False,
    strip_initializers: bool = False,
) -> None:
    cfg = _text_config(model)
    float_dtype = _model_float_dtype(model)
    h = cfg.hidden_size
    kv = cfg.num_key_value_heads
    d = getattr(cfg, "head_dim", h // cfg.num_attention_heads)
    b, s, p = batch_size, seq_len, past_seq_len
    total_len = p + s
    causal_mask = torch.zeros((b, 1, s, total_len), dtype=float_dtype)
    for token_idx in range(s):
        causal_mask[:, :, token_idx, p + token_idx + 1 :] = float("-inf")
    position_ids = torch.arange(p, p + s, dtype=torch.long).unsqueeze(0).expand(b, -1).contiguous()
    if p == 0:
        file_name = f"layer_{layer_idx:02d}_full_attn_block_{_seq_tag(seq_len)}.onnx"
    else:
        file_name = (
            f"layer_{layer_idx:02d}_full_attn_block_decode_ctx{_seq_tag(p)}.onnx"
            if s == 1
            else f"layer_{layer_idx:02d}_full_attn_block_{_seq_tag(seq_len)}_ctx{_seq_tag(p)}.onnx"
        )

    block = FullAttentionMergedBlock(
        model,
        layer_idx,
        export_batch_size=b,
        export_seq_len=s,
        export_past_seq_len=p,
    )
    sample_inputs = (
        torch.randn(b, s, h, dtype=float_dtype),
        position_ids,
        causal_mask,
        torch.zeros(b, kv, p, d, dtype=float_dtype),
        torch.zeros(b, kv, p, d, dtype=float_dtype),
    )
    _onnx_export(
        block,
        sample_inputs,
        os.path.join(out_dir, file_name),
        input_names=["hidden_states", "position_ids", "attention_mask", "past_key", "past_value"],
        output_names=["hidden_states", "new_key", "new_value"],
        dynamic_axes={},
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
        fold_pure_shape_chains=fold_pure_shape_chains,
    )


def export_moe_block(
    model: Qwen3_5MoeModelLike,
    layer_idx: int,
    out_dir: str,
    batch_size: int,
    opset: int,
    simplify: bool,
    seq_len: int,
    fold_pure_shape_chains: bool = False,
    strip_initializers: bool = False,
) -> None:
    cfg = _text_config(model)
    float_dtype = _model_float_dtype(model)
    h = cfg.hidden_size

    block = MergedMoeBlock(
        model,
        layer_idx,
        export_batch_size=batch_size,
        export_seq_len=seq_len,
    )
    moe_weights = _moe_weight_from_model(model, layer_idx)
    sample_inputs = (torch.randn(batch_size, seq_len, h, dtype=float_dtype),) + moe_weights
    _onnx_export(
        block,
        sample_inputs,
        os.path.join(out_dir, f"layer_{layer_idx:02d}_moe_block_{_seq_tag(seq_len)}.onnx"),
        input_names=[
            "hidden_states",
            "experts_gate_up",
            "experts_down",
            "shared_gate_proj_w",
            "shared_up_proj_w",
            "shared_down_proj_w",
            "shared_expert_gate_w",
        ],
        output_names=["hidden_states"],
        dynamic_axes={},
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
        fold_pure_shape_chains=fold_pure_shape_chains,
    )


if __name__ == "__main__":
    raise SystemExit(
        "This module is an implementation detail. Use export_qwen_onnx_main.py as the canonical CLI."
    )
