#!/usr/bin/env python3
"""
Compare two ONNX models and highlight structural vs shape/value differences.

This is especially useful for Qwen3.5-MoE exports where two graphs may look
nearly identical in a visualizer, but differ in MoE expert count, TopK, or
input tensor shapes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto, numpy_helper


def _shape_from_value_info(value_info) -> list[int | str] | None:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None

    shape: list[int | str] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            shape.append(dim.dim_param)
        else:
            shape.append("?")
    return shape


def _tensor_type_name(elem_type: int | None) -> str:
    if elem_type is None:
        return "unknown"
    try:
        return TensorProto.DataType.Name(elem_type)
    except ValueError:
        return str(elem_type)


def _shape_str(shape: list[int | str] | None) -> str:
    if shape is None:
        return "?"
    return "[" + ", ".join(str(x) for x in shape) + "]"


def _to_python_value(tensor: onnx.TensorProto) -> Any:
    array = numpy_helper.to_array(tensor)
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _collect_value_info(model: onnx.ModelProto) -> dict[str, dict[str, Any]]:
    info: dict[str, dict[str, Any]] = {}

    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        tensor_type = vi.type.tensor_type
        elem_type = tensor_type.elem_type if tensor_type.HasField("elem_type") else None
        info[vi.name] = {
            "shape": _shape_from_value_info(vi),
            "elem_type": elem_type,
        }

    for init in model.graph.initializer:
        info[init.name] = {
            "shape": [int(d) for d in init.dims],
            "elem_type": int(init.data_type),
        }

    return info


def _collect_constants(model: onnx.ModelProto) -> dict[str, Any]:
    constants: dict[str, Any] = {}

    for init in model.graph.initializer:
        constants[init.name] = _to_python_value(init)

    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attr in node.attribute:
            if attr.name == "value":
                constants[node.output[0]] = _to_python_value(attr.t)
                break

    return constants


def _extract_topk_values(model: onnx.ModelProto, constants: dict[str, Any]) -> list[int | None]:
    values: list[int | None] = []
    for node in model.graph.node:
        if node.op_type != "TopK":
            continue
        if len(node.input) < 2:
            values.append(None)
            continue
        raw = constants.get(node.input[1])
        if isinstance(raw, list) and raw:
            raw = raw[0]
        if raw is None:
            values.append(None)
            continue
        try:
            values.append(int(raw))
        except (TypeError, ValueError):
            values.append(None)
    return values


def _model_summary(model_path: Path) -> dict[str, Any]:
    model = onnx.load(str(model_path))
    tensor_info = _collect_value_info(model)
    constants = _collect_constants(model)

    graph_inputs = []
    initializer_names = {init.name for init in model.graph.initializer}
    for vi in model.graph.input:
        if vi.name in initializer_names:
            continue
        meta = tensor_info.get(vi.name, {})
        graph_inputs.append(
            {
                "name": vi.name,
                "shape": meta.get("shape"),
                "elem_type": _tensor_type_name(meta.get("elem_type")),
            }
        )

    graph_outputs = []
    for vi in model.graph.output:
        meta = tensor_info.get(vi.name, {})
        graph_outputs.append(
            {
                "name": vi.name,
                "shape": meta.get("shape"),
                "elem_type": _tensor_type_name(meta.get("elem_type")),
            }
        )

    op_counts = Counter(node.op_type for node in model.graph.node)
    op_sequence = [node.op_type for node in model.graph.node]

    experts_gate_up = next((item for item in graph_inputs if item["name"] == "experts_gate_up"), None)
    experts_down = next((item for item in graph_inputs if item["name"] == "experts_down"), None)

    expert_count = None
    for candidate in (experts_gate_up, experts_down):
        if not candidate or not candidate["shape"]:
            continue
        first_dim = candidate["shape"][0]
        if isinstance(first_dim, int):
            expert_count = first_dim
            break

    topk_values = _extract_topk_values(model, constants)

    return {
        "path": str(model_path),
        "file_size_bytes": model_path.stat().st_size,
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "op_counts": dict(op_counts),
        "op_sequence": op_sequence,
        "inputs": graph_inputs,
        "outputs": graph_outputs,
        "experts_gate_up_shape": experts_gate_up["shape"] if experts_gate_up else None,
        "experts_down_shape": experts_down["shape"] if experts_down else None,
        "expert_count": expert_count,
        "topk_values": topk_values,
    }


def _compare_named_tensors(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    left_map = {item["name"]: item for item in left}
    right_map = {item["name"]: item for item in right}
    names = sorted(set(left_map) | set(right_map))

    diffs = []
    for name in names:
        l_item = left_map.get(name)
        r_item = right_map.get(name)
        if l_item == r_item:
            continue
        diffs.append(
            {
                "name": name,
                "left": l_item,
                "right": r_item,
            }
        )
    return diffs


def _top_op_diffs(left_ops: dict[str, int], right_ops: dict[str, int]) -> list[dict[str, Any]]:
    names = sorted(set(left_ops) | set(right_ops))
    diffs = []
    for name in names:
        l_val = left_ops.get(name, 0)
        r_val = right_ops.get(name, 0)
        if l_val != r_val:
            diffs.append({"op_type": name, "left": l_val, "right": r_val})
    return diffs


def _print_summary(tag: str, summary: dict[str, Any]) -> None:
    print(f"\n[{tag}] {summary['path']}")
    print(f"  file_size_bytes : {summary['file_size_bytes']}")
    print(f"  node_count      : {summary['node_count']}")
    print(f"  initializer_cnt : {summary['initializer_count']}")
    print(f"  expert_count    : {summary['expert_count']}")
    print(f"  topk_values     : {summary['topk_values']}")
    print(f"  experts_gate_up : {_shape_str(summary['experts_gate_up_shape'])}")
    print(f"  experts_down    : {_shape_str(summary['experts_down_shape'])}")

    print("  inputs:")
    for item in summary["inputs"]:
        print(f"    - {item['name']}: {_shape_str(item['shape'])} {item['elem_type']}")

    print("  outputs:")
    for item in summary["outputs"]:
        print(f"    - {item['name']}: {_shape_str(item['shape'])} {item['elem_type']}")

    op_str = "  ".join(f"{name}×{count}" for name, count in Counter(summary["op_sequence"]).most_common())
    print(f"  ops            : {op_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="First ONNX file")
    parser.add_argument("right", type=Path, help="Second ONNX file")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON report path")
    args = parser.parse_args()

    left = _model_summary(args.left)
    right = _model_summary(args.right)

    input_diffs = _compare_named_tensors(left["inputs"], right["inputs"])
    output_diffs = _compare_named_tensors(left["outputs"], right["outputs"])
    op_count_diffs = _top_op_diffs(left["op_counts"], right["op_counts"])
    same_op_sequence = left["op_sequence"] == right["op_sequence"]

    report = {
        "left": left,
        "right": right,
        "comparison": {
            "same_node_count": left["node_count"] == right["node_count"],
            "same_initializer_count": left["initializer_count"] == right["initializer_count"],
            "same_op_sequence": same_op_sequence,
            "input_diffs": input_diffs,
            "output_diffs": output_diffs,
            "op_count_diffs": op_count_diffs,
            "expert_count_diff": {
                "left": left["expert_count"],
                "right": right["expert_count"],
            },
            "topk_diff": {
                "left": left["topk_values"],
                "right": right["topk_values"],
            },
        },
    }

    _print_summary("LEFT", left)
    _print_summary("RIGHT", right)

    print("\n[COMPARISON]")
    print(f"  same_node_count      : {report['comparison']['same_node_count']}")
    print(f"  same_initializer_cnt : {report['comparison']['same_initializer_count']}")
    print(f"  same_op_sequence     : {report['comparison']['same_op_sequence']}")
    print(
        "  expert_count         : "
        f"{left['expert_count']} vs {right['expert_count']}"
    )
    print(
        "  topk_values          : "
        f"{left['topk_values']} vs {right['topk_values']}"
    )

    if input_diffs:
        print("  input_diffs:")
        for diff in input_diffs:
            print(
                f"    - {diff['name']}: "
                f"{_shape_str(diff['left']['shape']) if diff['left'] else 'missing'} vs "
                f"{_shape_str(diff['right']['shape']) if diff['right'] else 'missing'}"
            )
    else:
        print("  input_diffs          : none")

    if output_diffs:
        print("  output_diffs:")
        for diff in output_diffs:
            print(
                f"    - {diff['name']}: "
                f"{_shape_str(diff['left']['shape']) if diff['left'] else 'missing'} vs "
                f"{_shape_str(diff['right']['shape']) if diff['right'] else 'missing'}"
            )
    else:
        print("  output_diffs         : none")

    if op_count_diffs:
        print("  op_count_diffs:")
        for diff in op_count_diffs:
            print(f"    - {diff['op_type']}: {diff['left']} vs {diff['right']}")
    else:
        print("  op_count_diffs       : none")

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON report written to {args.json_out}")


if __name__ == "__main__":
    main()
