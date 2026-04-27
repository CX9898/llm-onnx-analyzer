"""
Shared ONNX graph/statistics helpers reused by export scripts.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


_2GB = 2 * (1 << 30)
_records: list[dict] = []


def reset_records() -> None:
    _records.clear()


def record_export(
    save_path: str,
    file_size: int,
    shape_ok: bool,
    onnxsim_status: str,
    param_count: Optional[int],
    op_counter: Optional[Counter],
    graph_optimizations: Optional[list[str]] = None,
) -> None:
    _records.append({
        "filename": os.path.basename(save_path),
        "path": save_path,
        "size_mb": round(file_size / (1 << 20), 3),
        "shape_inference": "ok" if shape_ok else "skipped",
        "onnxsim": onnxsim_status,
        "params_initializer": param_count,
        "ops": dict(op_counter) if op_counter is not None else None,
        "graph_optimizations": list(graph_optimizations or []),
    })


def save_stats_json(output_dir: str) -> str:
    total_params = sum(
        r["params_initializer"] for r in _records
        if r["params_initializer"] is not None
    )
    all_ops: Counter = Counter()
    for record in _records:
        if record["ops"]:
            all_ops.update(record["ops"])

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": output_dir,
        "total_files": len(_records),
        "total_params_initializer": total_params,
        "all_ops": dict(all_ops.most_common()),
        "files": _records,
    }

    out_path = os.path.join(output_dir, "onnx_stats.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return out_path


def print_simplification_report() -> None:
    if not _records:
        return

    shape_ok = [r for r in _records if r["shape_inference"] == "ok"]
    sim_ok = [r for r in _records if r["onnxsim"] in ("ok", "ok_skip_cf")]
    total = len(_records)

    print(f"\n  Simplification status  ({len(shape_ok)}/{total} shape_inference OK"
          f", {len(sim_ok)}/{total} onnxsim OK)")
    print(f"  {'File':<52} {'shape_inf':>10}  {'onnxsim':<25}")
    print("  " + "-" * 92)
    for record in _records:
        shape_state = "✓ ok" if record["shape_inference"] == "ok" else "✗ " + record["shape_inference"]
        sim_state = "✓ ok" if record["onnxsim"] in ("ok", "ok_skip_cf") else "• " + record["onnxsim"]
        print(f"  {record['filename']:<52} {shape_state:>10}  {sim_state:<25}")

    not_simplified = [
        record["filename"] for record in _records
        if record["shape_inference"] != "ok" or record["onnxsim"] not in ("ok", "ok_skip_cf", "check_failed")
    ]
    if not_simplified:
        print("\n  ⚠  Files without full simplification:")
        for filename in not_simplified:
            record = next(r for r in _records if r["filename"] == filename)
            print(f"     {filename}  → shape_inference={record['shape_inference']}"
                  f"  onnxsim={record['onnxsim']}")


def onnx_stats(save_path: str) -> tuple[Optional[int], Optional[Counter]]:
    try:
        file_size = os.path.getsize(save_path)
    except OSError:
        return None, None

    if file_size > _2GB:
        return None, None

    try:
        model_proto = onnx.load(save_path)
    except Exception:
        return None, None

    total = 0
    for init in model_proto.graph.initializer:
        n = 1
        for dim in init.dims:
            n *= dim
        total += n

    op_counter = Counter(node.op_type for node in model_proto.graph.node)
    return total, op_counter


def _human(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f} B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f} M"
    if n >= 1_000:
        return f"{n / 1e3:.1f} K"
    return str(n)


def print_onnx_stats(save_path: str, file_size_bytes: int, indent: str = "    ") -> None:
    if file_size_bytes > _2GB:
        print(f"{indent}stats  : skipped (file > 2 GB)")
        return

    param_count, op_counter = onnx_stats(save_path)
    if param_count is None:
        print(f"{indent}stats  : could not load ONNX file for analysis")
        return

    if param_count > 0:
        print(f"{indent}params : {param_count:,}  (~{_human(param_count)})  [initializers]")
    else:
        print(f"{indent}params : 0  (weights passed as explicit inputs — stateless block)")

    if op_counter:
        ops_str = "  ".join(f"{op}×{cnt}" for op, cnt in op_counter.most_common())
        print(f"{indent}ops    : {ops_str}")
    else:
        print(f"{indent}ops    : (empty graph)")


def fold_pure_shape_chains_in_file(save_path: str) -> dict[str, int]:
    model = onnx.load(save_path)
    graph = model.graph

    shapes: dict[str, list[int]] = {}
    values: dict[str, np.ndarray] = {}
    elem_types: dict[str, int] = {}

    def _dims_from_vi(vi) -> list[int] | None:
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            return None
        dims: list[int] = []
        for dim in tt.shape.dim:
            if not dim.HasField("dim_value"):
                return None
            dims.append(int(dim.dim_value))
        return dims

    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        tt = vi.type.tensor_type
        if tt.elem_type:
            elem_types[vi.name] = int(tt.elem_type)
        dims = _dims_from_vi(vi)
        if dims is not None:
            shapes[vi.name] = dims

    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        values[init.name] = np.asarray(arr)
        elem_types[init.name] = int(init.data_type)
        shapes[init.name] = list(arr.shape)

    def _get_attr(node, name: str):
        for attr in node.attribute:
            if attr.name == name:
                return attr
        return None

    def _get_attr_int(node, name: str, default: int) -> int:
        attr = _get_attr(node, name)
        return int(attr.i) if attr is not None else default

    def _get_attr_ints(node, name: str) -> list[int] | None:
        attr = _get_attr(node, name)
        if attr is None:
            return None
        return [int(x) for x in attr.ints]

    def _record(name: str, value: np.ndarray | None, elem_type: int | None = None) -> None:
        if value is None:
            return
        arr = np.asarray(value)
        values[name] = arr
        shapes[name] = list(arr.shape)
        if elem_type is not None:
            elem_types[name] = int(elem_type)

    progress = True
    while progress:
        progress = False
        for node in graph.node:
            op = node.op_type

            if op == "Constant":
                attr = _get_attr(node, "value")
                if attr is not None and node.output[0] not in values:
                    arr = numpy_helper.to_array(attr.t)
                    _record(node.output[0], arr, int(attr.t.data_type))
                    progress = True
                continue

            if op in {"Identity", "Cast"} and node.input and node.input[0] in values:
                arr = np.asarray(values[node.input[0]])
                out_type = elem_types.get(node.input[0], TensorProto.FLOAT)
                if op == "Cast":
                    attr = _get_attr(node, "to")
                    out_type = int(attr.i) if attr is not None else out_type
                    dtype_map = {
                        TensorProto.INT64: np.int64,
                        TensorProto.INT32: np.int32,
                        TensorProto.FLOAT: np.float32,
                        TensorProto.FLOAT16: np.float16,
                        TensorProto.BOOL: np.bool_,
                    }
                    arr = arr.astype(dtype_map.get(out_type, arr.dtype), copy=False)
                if node.output[0] not in values:
                    _record(node.output[0], arr, out_type)
                    progress = True
                continue

            if op == "Shape":
                in_shape = shapes.get(node.input[0])
                if in_shape is not None and node.output[0] not in values:
                    _record(node.output[0], np.asarray(in_shape, dtype=np.int64), TensorProto.INT64)
                    progress = True
                continue

            if op == "Slice" and len(node.input) >= 3 and all(name in values for name in node.input[:3]):
                data = np.asarray(values[node.input[0]])
                starts = np.asarray(values[node.input[1]]).reshape(-1)
                ends = np.asarray(values[node.input[2]]).reshape(-1)
                axes = (
                    np.asarray(values[node.input[3]]).reshape(-1)
                    if len(node.input) > 3 and node.input[3] in values
                    else np.arange(len(starts), dtype=np.int64)
                )
                steps = (
                    np.asarray(values[node.input[4]]).reshape(-1)
                    if len(node.input) > 4 and node.input[4] in values
                    else np.ones(len(starts), dtype=np.int64)
                )
                slices = [slice(None)] * data.ndim
                for start, end, axis, step in zip(starts.tolist(), ends.tolist(), axes.tolist(), steps.tolist()):
                    slices[int(axis)] = slice(int(start), int(end), int(step))
                arr = data[tuple(slices)]
                if node.output[0] not in values:
                    _record(node.output[0], arr, elem_types.get(node.input[0], TensorProto.INT64))
                    progress = True
                continue

            if op == "Concat" and all(name in values for name in node.input):
                axis = _get_attr_int(node, "axis", 0)
                arr = np.concatenate([np.asarray(values[name]) for name in node.input], axis=axis)
                if node.output[0] not in values:
                    _record(node.output[0], arr, elem_types.get(node.input[0], TensorProto.INT64))
                    progress = True
                continue

            if op == "Unsqueeze" and len(node.input) >= 2 and node.input[0] in values and node.input[1] in values:
                arr = np.asarray(values[node.input[0]])
                axes = sorted(int(x) for x in np.asarray(values[node.input[1]]).reshape(-1).tolist())
                for axis in axes:
                    arr = np.expand_dims(arr, axis)
                if node.output[0] not in values:
                    _record(node.output[0], arr, elem_types.get(node.input[0], TensorProto.INT64))
                    progress = True
                continue

            if op == "Squeeze" and node.input and node.input[0] in values:
                arr = np.asarray(values[node.input[0]])
                axes = (
                    [int(x) for x in np.asarray(values[node.input[1]]).reshape(-1).tolist()]
                    if len(node.input) > 1 and node.input[1] in values
                    else _get_attr_ints(node, "axes")
                )
                arr = np.squeeze(arr, axis=tuple(axes) if axes is not None else None)
                if node.output[0] not in values:
                    _record(node.output[0], arr, elem_types.get(node.input[0], TensorProto.INT64))
                    progress = True
                continue

            if op == "Gather" and len(node.input) >= 2 and node.input[0] in values and node.input[1] in values:
                axis = _get_attr_int(node, "axis", 0)
                arr = np.take(np.asarray(values[node.input[0]]), np.asarray(values[node.input[1]]), axis=axis)
                if node.output[0] not in values:
                    _record(node.output[0], arr, elem_types.get(node.input[0], TensorProto.INT64))
                    progress = True
                continue

            if op == "Reshape" and len(node.input) >= 2 and node.input[0] in values and node.input[1] in values:
                data = np.asarray(values[node.input[0]])
                target = [int(x) for x in np.asarray(values[node.input[1]]).reshape(-1).tolist()]
                arr = np.reshape(data, target)
                if node.output[0] not in values:
                    _record(node.output[0], arr, elem_types.get(node.input[0], TensorProto.INT64))
                    progress = True
                continue

            if op == "Tile" and len(node.input) >= 2 and node.input[0] in values and node.input[1] in values:
                arr = np.tile(np.asarray(values[node.input[0]]), np.asarray(values[node.input[1]]))
                if node.output[0] not in values:
                    _record(node.output[0], arr, elem_types.get(node.input[0], TensorProto.INT64))
                    progress = True
                continue

            if op == "Expand" and len(node.input) >= 2 and node.input[0] in values and node.input[1] in values:
                arr = np.broadcast_to(np.asarray(values[node.input[0]]), tuple(int(x) for x in np.asarray(values[node.input[1]])))
                if node.output[0] not in values:
                    _record(node.output[0], arr, elem_types.get(node.input[0], TensorProto.INT64))
                    progress = True
                continue

            if op in {"Add", "Sub", "Mul", "Div", "Mod"} and len(node.input) >= 2 and node.input[0] in values and node.input[1] in values:
                lhs = np.asarray(values[node.input[0]])
                rhs = np.asarray(values[node.input[1]])
                if op == "Add":
                    arr = lhs + rhs
                elif op == "Sub":
                    arr = lhs - rhs
                elif op == "Mul":
                    arr = lhs * rhs
                elif op == "Div":
                    arr = lhs / rhs
                else:
                    arr = np.mod(lhs, rhs)
                if node.output[0] not in values:
                    _record(node.output[0], arr, elem_types.get(node.input[0], TensorProto.INT64))
                    progress = True
                continue

    target_input_positions = {
        "Reshape": {1},
        "Expand": {1},
        "Tile": {1},
        "Unsqueeze": {1},
        "Squeeze": {1},
        "Slice": {1, 2, 3, 4},
        "Gather": {1},
        "ConstantOfShape": {0},
    }

    added_initializers = 0
    replaced_inputs = 0
    initializer_names = {init.name for init in graph.initializer}
    for node in graph.node:
        positions = target_input_positions.get(node.op_type)
        if not positions:
            continue
        for idx in positions:
            if idx >= len(node.input):
                continue
            in_name = node.input[idx]
            if not in_name or in_name in initializer_names or in_name not in values:
                continue
            arr = np.asarray(values[in_name])
            if arr.size > 256:
                continue
            new_name = f"{in_name}__folded"
            if all(init.name != new_name for init in graph.initializer):
                graph.initializer.append(numpy_helper.from_array(arr, new_name))
                initializer_names.add(new_name)
                added_initializers += 1
            node.input[idx] = new_name
            replaced_inputs += 1

    live_tensors = {output.name for output in graph.output}
    for node in graph.node:
        live_tensors.update(inp for inp in node.input if inp)

    removed_nodes = 0
    removed = True
    helper_ops = {
        "Constant",
        "Shape",
        "Slice",
        "Concat",
        "Unsqueeze",
        "Squeeze",
        "Reshape",
        "Tile",
        "Expand",
        "Gather",
        "Cast",
        "Identity",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Mod",
    }
    while removed:
        removed = False
        for node in list(graph.node):
            if node.op_type not in helper_ops:
                continue
            if any(out in live_tensors for out in node.output):
                continue
            graph.node.remove(node)
            removed_nodes += 1
            removed = True
        if removed:
            live_tensors = {output.name for output in graph.output}
            for node in graph.node:
                live_tensors.update(inp for inp in node.input if inp)

    if added_initializers or replaced_inputs or removed_nodes:
        onnx.save(model, save_path)

    return {
        "replaced_inputs": replaced_inputs,
        "added_initializers": added_initializers,
        "removed_nodes": removed_nodes,
    }

