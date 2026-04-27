from __future__ import annotations

import os
import sys
from pathlib import Path

import onnx
import torch
import torch.nn as nn

_EXPORT_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_EXPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPORT_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from export_common.checkpoint_metadata import (  # noqa: E402
    infer_checkpoint_torch_dtype as _common_infer_checkpoint_torch_dtype,
    model_float_dtype as _common_model_float_dtype,
    torch_dtype_to_name as _common_torch_dtype_to_name,
)
from export_common.export_pipeline import (  # noqa: E402
    layer_tag as _common_layer_tag,
    onnx_export as _common_onnx_export,
    seq_tag as _common_seq_tag,
    shape_enrich_onnx_file as _common_shape_enrich_onnx_file,
    strip_initializers_to_inputs as _common_strip_initializers_to_inputs,
)
from export_common.manifest_utils import write_structured_manifest as _common_write_structured_manifest  # noqa: E402
from qwen_onnx_blocks import EmbeddingBlock, LMHeadBlock, MoENormBlock  # noqa: E402
from qwen_shape_propagation import _static_shape_propagation  # noqa: E402
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
)

# A loaded Qwen3.5-MoE model may be either of the two top-level classes below.
# All shared helpers/exports use the structural ``Qwen3_5MoeModelLike`` alias so
# that text-only and vision-language paths share one code surface.
Qwen3_5MoeModelLike = Qwen3_5MoeForCausalLM | Qwen3_5MoeForConditionalGeneration


# ---------------------------------------------------------------------------
# Variant accessors
# ---------------------------------------------------------------------------
#
# Topology of the two supported top-level classes::
#
#   Qwen3_5MoeForCausalLM
#     .model            : Qwen3_5MoeTextModel       <- text backbone
#     .lm_head          : nn.Linear
#     .config           : Qwen3_5MoeTextConfig      <- flat fields
#
#   Qwen3_5MoeForConditionalGeneration
#     .model            : Qwen3_5MoeModel
#       .visual         : Qwen3_5MoeVisionModel     <- vision tower
#       .language_model : Qwen3_5MoeTextModel       <- text backbone
#     .lm_head          : nn.Linear
#     .config           : Qwen3_5MoeConfig
#       .text_config    : Qwen3_5MoeTextConfig      <- nested
#       .vision_config  : Qwen3_5MoeVisionConfig    <- nested
#
# These helpers abstract the difference so that downstream export code never
# needs to know which top-level class it is dealing with.


def _text_model(model):
    """Return the text backbone (a ``Qwen3_5MoeTextModel`` instance)."""
    inner = model.model
    return inner.language_model if hasattr(inner, "language_model") else inner


def _text_config(model):
    """Return the text config (always flat field access from the result)."""
    cfg = model.config
    return cfg.text_config if hasattr(cfg, "text_config") else cfg


def _visual_model(model):
    """Return the vision tower; raises if the loaded model is text-only."""
    inner = model.model
    if not hasattr(inner, "visual"):
        raise AttributeError(
            "Loaded model has no .model.visual; expected a multimodal entry "
            "(Qwen3_5MoeForConditionalGeneration). Use --variant vl when loading."
        )
    return inner.visual


def _vision_config(model):
    """Return the vision config; raises if the loaded model is text-only."""
    cfg = model.config
    if not hasattr(cfg, "vision_config"):
        raise AttributeError(
            "model.config has no .vision_config; expected a multimodal model. "
            "Use --variant vl when loading."
        )
    return cfg.vision_config


def _seq_tag(seq_len: int) -> str:
    return _common_seq_tag(seq_len)


def _layer_type(model: Qwen3_5MoeModelLike, idx: int) -> str:
    return _text_model(model).layers[idx].layer_type


def _torch_dtype_to_name(dtype: torch.dtype) -> str:
    return _common_torch_dtype_to_name(dtype)


def _infer_checkpoint_torch_dtype(model_path: str) -> torch.dtype:
    return _common_infer_checkpoint_torch_dtype(model_path)


def _model_float_dtype(model: nn.Module) -> torch.dtype:
    return _common_model_float_dtype(model)


def _layer_tag(layer_indices: list[int]) -> str:
    return _common_layer_tag(layer_indices)


def _strip_initializers_to_inputs(save_path: str) -> None:
    _common_strip_initializers_to_inputs(save_path)


def _onnx_export(
    module: nn.Module,
    dummy_inputs: tuple,
    save_path: str,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict,
    opset: int = 17,
    simplify: bool = True,
    strip_initializers: bool = False,
    custom_opsets: dict[str, int] | None = None,
    fold_pure_shape_chains: bool = False,
    shape_enrich_after_fold: bool = True,
    collect_onnx_stats: bool = True,
) -> None:
    _common_onnx_export(
        module,
        dummy_inputs,
        save_path,
        input_names,
        output_names,
        dynamic_axes,
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
        custom_opsets=custom_opsets,
        fold_pure_shape_chains=fold_pure_shape_chains,
        shape_enrich_after_fold=shape_enrich_after_fold,
        collect_onnx_stats=collect_onnx_stats,
        static_shape_propagator=_static_shape_propagation,
    )


def _shape_enrich_onnx_file(save_path: str) -> None:
    _common_shape_enrich_onnx_file(
        save_path,
        static_shape_propagator=_static_shape_propagation,
    )


def _write_structured_manifest(
    graph_defs: dict[str, dict[str, str | None]],
    *,
    root_key: str,
    save_path: str,
) -> None:
    _common_write_structured_manifest(graph_defs, root_key=root_key, save_path=save_path)


def _norm_dummy(model, batch_size, seq_len):
    bsz, slen, hidden = batch_size, seq_len, _text_config(model).hidden_size
    return (torch.randn(bsz, slen, hidden, dtype=_model_float_dtype(model)),)


def _moe_weight_dummies(model):
    cfg = _text_config(model)
    num_experts = cfg.num_experts
    intermediate = cfg.moe_intermediate_size
    shared_intermediate = cfg.shared_expert_intermediate_size
    hidden = cfg.hidden_size
    dtype = _model_float_dtype(model)
    return (
        torch.randn(num_experts, 2 * intermediate, hidden, dtype=dtype),
        torch.randn(num_experts, hidden, intermediate, dtype=dtype),
        torch.randn(shared_intermediate, hidden, dtype=dtype),
        torch.randn(shared_intermediate, hidden, dtype=dtype),
        torch.randn(hidden, shared_intermediate, dtype=dtype),
        torch.randn(1, hidden, dtype=dtype),
    )


def _moe_weight_from_model(model, layer_idx):
    moe = _text_model(model).layers[layer_idx].mlp
    experts = moe.experts
    shared = moe.shared_expert
    return (
        experts.gate_up_proj.detach(),
        experts.down_proj.detach(),
        shared.gate_proj.weight.detach(),
        shared.up_proj.weight.detach(),
        shared.down_proj.weight.detach(),
        moe.shared_expert_gate.weight.detach(),
    )


def export_embedding(
    model,
    out_dir,
    batch_size,
    opset,
    simplify,
    seq_len,
    *,
    strip_initializers: bool = False,
    fold_pure_shape_chains: bool = False,
):
    block = EmbeddingBlock()
    bsz, slen = batch_size, seq_len
    text_cfg = _text_config(model)
    vocab, hidden = text_cfg.vocab_size, text_cfg.hidden_size
    sample_inputs = (
        torch.randint(0, vocab, (bsz, slen)),
        torch.randn(vocab, hidden, dtype=_model_float_dtype(model)),
    )
    _onnx_export(
        block,
        sample_inputs,
        os.path.join(out_dir, f"embedding_{_seq_tag(seq_len)}.onnx"),
        input_names=["input_ids", "embedding_weight"],
        output_names=["hidden_states"],
        dynamic_axes={},
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
        fold_pure_shape_chains=fold_pure_shape_chains,
    )


def export_norm(
    model,
    out_dir,
    batch_size,
    opset,
    simplify,
    seq_len,
    *,
    strip_initializers: bool = False,
    fold_pure_shape_chains: bool = False,
):
    block = MoENormBlock(_text_model(model).norm)
    sample_inputs = _norm_dummy(model, batch_size, seq_len)
    _onnx_export(
        block,
        sample_inputs,
        os.path.join(out_dir, f"norm_{_seq_tag(seq_len)}.onnx"),
        input_names=["hidden_states"],
        output_names=["output"],
        dynamic_axes={},
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
        fold_pure_shape_chains=fold_pure_shape_chains,
    )


def export_lm_head(
    model,
    out_dir,
    batch_size,
    opset,
    simplify,
    seq_len,
    *,
    strip_initializers: bool = False,
    fold_pure_shape_chains: bool = False,
):
    block = LMHeadBlock()
    bsz, slen = batch_size, seq_len
    text_cfg = _text_config(model)
    hidden, vocab = text_cfg.hidden_size, text_cfg.vocab_size
    dtype = _model_float_dtype(model)
    sample_inputs = (
        torch.randn(bsz, slen, hidden, dtype=dtype),
        torch.randn(vocab, hidden, dtype=dtype),
    )
    _onnx_export(
        block,
        sample_inputs,
        os.path.join(out_dir, f"lm_head_{_seq_tag(seq_len)}.onnx"),
        input_names=["hidden_states", "lm_head_weight"],
        output_names=["logits"],
        dynamic_axes={},
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
        fold_pure_shape_chains=fold_pure_shape_chains,
    )



