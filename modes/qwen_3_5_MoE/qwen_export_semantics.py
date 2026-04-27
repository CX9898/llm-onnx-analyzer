from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from export_common.checkpoint_metadata import tensorproto_from_safetensors_dtype
from onnx import TensorProto

@dataclass(frozen=True)
class DtypeAuditFinding:
    category: str
    export_symbol: str | None
    source_symbol: str | None
    export_snippet: str | None
    source_snippet: str | None
    note: str


def _contains(text: str, needle: str | tuple[str, ...]) -> bool:
    if isinstance(needle, tuple):
        return all(part in text for part in needle)
    return needle in text


def build_dtype_audit_report(
    export_wrapper_text: str,
    source_text: str,
) -> list[DtypeAuditFinding]:
    findings: list[DtypeAuditFinding] = []

    checks = [
        (
            "_chunk_gated_delta_rule_onnx",
            "torch_chunk_gated_delta_rule",
            (
                "query = query.transpose(1, 2).contiguous().to(torch.float32)",
                "key = key.transpose(1, 2).contiguous().to(torch.float32)",
                "value = value.transpose(1, 2).contiguous().to(torch.float32)",
            ),
            "x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)",
            "Chunked gated delta rule promotes q/k/v/beta/g to fp32 in both export wrapper and Transformers source.",
        ),
        (
            "_recurrent_gated_delta_rule_onnx",
            "torch_recurrent_gated_delta_rule",
            (
                "query, key, value, beta, g = [",
                "x.transpose(1, 2).contiguous().to(torch.float32)",
            ),
            "x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)",
            "Recurrent gated delta rule promotes q/k/v/beta/g to fp32 in both export wrapper and Transformers source.",
        ),
        (
            "MoeSelfAttentionBlock.forward",
            "eager_attention_forward",
            "F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)",
            "softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)",
            "Attention softmax is explicitly evaluated in fp32, then cast back to query dtype.",
        ),
        (
            None,
            "Qwen3_5MoeRMSNormGated.forward",
            None,
            "hidden_states.to(torch.float32)",
            "Gated RMSNorm in Transformers source explicitly promotes hidden/gate path to fp32.",
        ),
    ]

    for export_symbol, source_symbol, export_snippet, source_snippet, note in checks:
        export_hit = _contains(export_wrapper_text, export_snippet) if export_symbol is not None else False
        source_hit = _contains(source_text, source_snippet)
        if export_hit and source_hit:
            category = "source_explicit_fp32"
        elif source_hit:
            category = "source_only_fp32"
        elif export_hit:
            category = "export_only_hardcoded_dtype"
        else:
            category = "missing"
        findings.append(
            DtypeAuditFinding(
                category=category,
                export_symbol=export_symbol,
                source_symbol=source_symbol,
                export_snippet=str(export_snippet) if export_hit else None,
                source_snippet=str(source_snippet) if source_hit else None,
                note=note,
            )
        )

    return findings


def render_dtype_audit_markdown(findings: list[DtypeAuditFinding]) -> str:
    lines = [
        "# Qwen3.5-MoE dtype semantic audit",
        "",
        "| Category | Export symbol | Source symbol | Note |",
        "| --- | --- | --- | --- |",
    ]
    for item in findings:
        lines.append(
            f"| `{item.category}` | `{item.export_symbol or '-'}` | `{item.source_symbol or '-'}` | {item.note} |"
        )
    return "\n".join(lines) + "\n"


def qwen_custom_output_specs(
    op_type: str,
    input_shapes: list[list[int] | None],
    input_elem_types: list[int | None],
) -> list[tuple[list[int] | None, int | None]]:
    if op_type == "DeltaNetTriangularSolve":
        return [(input_shapes[0], input_elem_types[0])]
    if op_type == "DeltaNetChunkStep":
        value_shape = input_shapes[2] if len(input_shapes) > 2 else None
        value_type = input_elem_types[2] if len(input_elem_types) > 2 else None
        state_shape = input_shapes[7] if len(input_shapes) > 7 else None
        state_type = input_elem_types[7] if len(input_elem_types) > 7 else None
        return [(value_shape, value_type), (state_shape, state_type)]
    if op_type == "ChunkGatedDeltaRule":
        query_shape = input_shapes[0] if len(input_shapes) > 0 else None
        value_shape = input_shapes[2] if len(input_shapes) > 2 else None
        state_shape = input_shapes[5] if len(input_shapes) > 5 else None
        value_type = input_elem_types[2] if len(input_elem_types) > 2 else None
        state_type = input_elem_types[5] if len(input_elem_types) > 5 else None
        out_shape = None
        if query_shape is not None and value_shape is not None and len(query_shape) >= 3 and len(value_shape) >= 4:
            out_shape = [query_shape[0], query_shape[1], query_shape[2], value_shape[3]]
        return [(out_shape, value_type), (state_shape, state_type)]
    if op_type == "RecurrentGatedDeltaRule":
        query_shape = input_shapes[0] if len(input_shapes) > 0 else None
        value_shape = input_shapes[2] if len(input_shapes) > 2 else None
        state_shape = input_shapes[5] if len(input_shapes) > 5 else None
        value_type = input_elem_types[2] if len(input_elem_types) > 2 else None
        state_type = input_elem_types[5] if len(input_elem_types) > 5 else None
        out_shape = None
        if query_shape is not None and value_shape is not None and len(query_shape) >= 3 and len(value_shape) >= 4:
            out_shape = [query_shape[0], query_shape[1], query_shape[2], value_shape[3]]
        return [(out_shape, value_type), (state_shape, state_type)]
    return []
