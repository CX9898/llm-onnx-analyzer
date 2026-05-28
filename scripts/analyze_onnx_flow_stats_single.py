#!/usr/bin/env python3
"""
Generate per-node flow statistics for an ONNX graph.

Outputs a TSV file in graph/topological order with columns:
    Name, Type, Forward_MACs, FPercent, Memory, MPercent,
    Params, PPercent, InShape, OutShape

Conventions
-----------
* Forward_MACs:
    Estimated multiply-accumulate count for compute-heavy ops only.
    Currently implemented for MatMul, Gemm, Conv, and common Einsum forms.
    Other ops default to 0.
* Memory:
    Sum of output activation tensor sizes in bytes for this node.
* Params:
    Sum of initializer elements directly consumed by this node.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import onnx
from onnx import TensorProto, shape_inference


def _dtype_nbytes(elem_type: int) -> int:
    mapping: dict[int, int] = {
        TensorProto.FLOAT: 4,
        TensorProto.UINT8: 1,
        TensorProto.INT8: 1,
        TensorProto.UINT16: 2,
        TensorProto.INT16: 2,
        TensorProto.INT32: 4,
        TensorProto.INT64: 8,
        TensorProto.STRING: 0,
        TensorProto.BOOL: 1,
        TensorProto.FLOAT16: 2,
        TensorProto.DOUBLE: 8,
        TensorProto.UINT32: 4,
        TensorProto.UINT64: 8,
        TensorProto.COMPLEX64: 8,
        TensorProto.COMPLEX128: 16,
        TensorProto.BFLOAT16: 2,
    }
    return mapping.get(elem_type, 0)


def _prod(shape: Iterable[int]) -> int:
    out = 1
    for x in shape:
        out *= int(x)
    return out


def _shape_to_str(shape: list[int | str] | None) -> str:
    if shape is None:
        return "?"
    return "[" + ", ".join(str(x) for x in shape) + "]"


def _shape_numel(shape: list[int | str] | None) -> int | None:
    if shape is None:
        return None
    numel = 1
    for d in shape:
        if not isinstance(d, int):
            return None
        numel *= d
    return numel


def _concretize_shape(shape: list[int | str] | None, batch_size: int | None) -> list[int | str] | None:
    if shape is None or batch_size is None:
        return shape
    out: list[int | str] = []
    for i, d in enumerate(shape):
        if isinstance(d, int):
            out.append(d)
        elif i == 0:
            out.append(batch_size)
        else:
            out.append(d)
    return out


def _infer_batch_size(model: onnx.ModelProto, tensor_info: dict[str, dict]) -> int | None:
    for vi in model.graph.input:
        meta = tensor_info.get(vi.name)
        if not meta:
            continue
        shape = meta.get("shape")
        if shape and len(shape) >= 1 and isinstance(shape[0], int) and shape[0] > 0:
            return int(shape[0])
    return None


_ACTIVATION_INPUT_NAMES = frozenset(
    {
        "hidden_states",
        "attn_mask",
        "rope_cos",
        "rope_sin",
        "adaln_input",
        "latent",
        "cap_feats",
        "cap_feats_padded",
        "timestep",
        "input_ids",
        "attention_mask",
        "x_patch_feats",
        "x_embedded",
        "x_pos_ids",
        "x_pad_mask",
        "cap_embedded",
        "cap_pos_ids",
        "cap_pad_mask",
        "x_tokens",
        "cap_tokens",
        "x_rope_cos",
        "x_rope_sin",
        "cap_rope_cos",
        "cap_rope_sin",
        "patch_output",
        "image_patch_output",
        "latents",
    }
)


def _is_activation_graph_input(name: str) -> bool:
    if name in _ACTIVATION_INPUT_NAMES:
        return True
    if name.endswith("_mask") or name.endswith("_ids"):
        return True
    if "rope" in name and ("cos" in name or "sin" in name):
        return True
    if ".weight" in name or ".bias" in name:
        return False
    return False


def _refresh_tensor_meta(entry: dict) -> None:
    shape = entry.get("shape")
    elem_type = entry.get("elem_type") or 0
    entry["numel"] = _shape_numel(shape)
    entry["nbytes"] = (entry["numel"] or 0) * _dtype_nbytes(elem_type)


def _matmul_output_shape(a_shape, b_shape) -> list[int | str] | None:
    if a_shape is None or b_shape is None or len(a_shape) < 2 or len(b_shape) < 2:
        return None
    if not all(isinstance(x, int) for x in a_shape + b_shape):
        return None
    batch_a = a_shape[:-2]
    batch_b = b_shape[:-2]
    m, k = int(a_shape[-2]), int(a_shape[-1])
    k2, n = int(b_shape[-2]), int(b_shape[-1])
    if k != k2:
        return None
    if batch_a and batch_b and batch_a != batch_b:
        if len(batch_a) == len(batch_b) == 1:
            batch = [batch_a[0] if isinstance(batch_a[0], int) else batch_b[0]]
        else:
            batch = batch_a if len(batch_a) >= len(batch_b) else batch_b
    else:
        batch = batch_a or batch_b
    return list(batch) + [m, n]


def _propagate_shapes(model: onnx.ModelProto, tensor_info: dict[str, dict]) -> None:
    """Best-effort shape propagation for stripped graphs (symbolic batch -> concrete)."""
    batch_size = _infer_batch_size(model, tensor_info)
    for meta in tensor_info.values():
        meta["shape"] = _concretize_shape(meta.get("shape"), batch_size)
        _refresh_tensor_meta(meta)

    for _ in range(8):
        changed = False
        for node in model.graph.node:
            input_shapes = [tensor_info.get(n, {}).get("shape") for n in node.input if n]
            output_names = [n for n in node.output if n]
            if not output_names:
                continue

            new_shape = None
            if node.op_type in {"Add", "Mul", "Sub", "Div", "Where"} and input_shapes:
                known = [s for s in input_shapes if s and all(isinstance(x, int) for x in s)]
                if known:
                    new_shape = list(known[0])
            elif node.op_type == "MatMul" and len(input_shapes) >= 2:
                new_shape = _matmul_output_shape(input_shapes[0], input_shapes[1])
            elif node.op_type == "Gemm" and len(input_shapes) >= 2:
                b_shape = input_shapes[1]
                trans_b = _get_attr_int(node, "transB", 0)
                if b_shape is not None and trans_b and len(b_shape) == 2 and all(
                    isinstance(x, int) for x in b_shape
                ):
                    b_shape = [int(b_shape[1]), int(b_shape[0])]
                new_shape = _matmul_output_shape(input_shapes[0], b_shape)
            elif node.op_type == "Softmax" and input_shapes and input_shapes[0]:
                new_shape = list(input_shapes[0])
            elif node.op_type in {"LayerNormalization", "InstanceNormalization", "Sigmoid", "Tanh", "Cast"}:
                if input_shapes and input_shapes[0]:
                    new_shape = list(input_shapes[0])
            elif node.op_type == "Concat":
                axis = _get_attr_int(node, "axis", 0)
                known = [s for s in input_shapes if s and all(isinstance(x, int) for x in s)]
                if len(known) >= 2:
                    new_shape = list(known[0])
                    total = sum(int(s[axis]) for s in known)
                    new_shape[axis] = total

            if new_shape is None:
                continue
            for out_name in output_names:
                meta = tensor_info.get(out_name)
                if meta is None:
                    meta = {"shape": None, "elem_type": None, "is_initializer": False}
                    tensor_info[out_name] = meta
                if meta.get("shape") != new_shape:
                    meta["shape"] = new_shape
                    _refresh_tensor_meta(meta)
                    changed = True
        if not changed:
            break


def _external_weight_params(model: onnx.ModelProto, tensor_info: dict[str, dict]) -> int:
    init_names = {init.name for init in model.graph.initializer}
    total = 0
    for vi in model.graph.input:
        if vi.name in init_names or _is_activation_graph_input(vi.name):
            continue
        meta = tensor_info.get(vi.name)
        if meta and meta.get("numel"):
            total += int(meta["numel"])
    return total


def _get_attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def _get_attr_ints(node: onnx.NodeProto, name: str) -> list[int] | None:
    for attr in node.attribute:
        if attr.name == name:
            return list(attr.ints)
    return None


def _get_attr_str(node: onnx.NodeProto, name: str) -> str | None:
    for attr in node.attribute:
        if attr.name == name:
            return attr.s.decode("utf-8")
    return None


def _extract_shape_from_value_info(value_info) -> tuple[list[int | str] | None, int | None]:
    t = value_info.type.tensor_type
    elem_type = t.elem_type if t.HasField("elem_type") else None
    if not t.HasField("shape"):
        return None, elem_type
    shape: list[int | str] = []
    for dim in t.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            shape.append(dim.dim_param)
        else:
            shape.append("?")
    return shape, elem_type


def _build_tensor_info(model: onnx.ModelProto):
    info: dict[str, dict] = {}

    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        shape, elem_type = _extract_shape_from_value_info(vi)
        info[vi.name] = {
            "shape": shape,
            "elem_type": elem_type,
            "is_initializer": False,
            "numel": _shape_numel(shape),
            "nbytes": (_shape_numel(shape) or 0) * _dtype_nbytes(elem_type or 0),
        }

    for init in model.graph.initializer:
        shape = [int(d) for d in init.dims]
        elem_type = init.data_type
        info[init.name] = {
            "shape": shape,
            "elem_type": elem_type,
            "is_initializer": True,
            "numel": _prod(shape),
            "nbytes": _prod(shape) * _dtype_nbytes(elem_type),
        }
    return info


def _format_io_shapes(names: list[str], tensor_info: dict[str, dict]) -> str:
    parts = []
    for name in names:
        if not name:
            continue
        meta = tensor_info.get(name)
        parts.append(f"{name}:{_shape_to_str(meta['shape']) if meta else '?'}")
    return " | ".join(parts)


def _estimate_matmul_macs(a_shape, b_shape) -> int:
    if a_shape is None or b_shape is None or len(a_shape) < 2 or len(b_shape) < 2:
        return 0
    if not all(isinstance(x, int) for x in a_shape + b_shape):
        return 0
    m = int(a_shape[-2])
    k = int(a_shape[-1])
    n = int(b_shape[-1])
    batch_shape = a_shape[:-2]
    batch = _prod(batch_shape) if batch_shape else 1
    return batch * m * n * k


def _estimate_conv_macs(node: onnx.NodeProto, inp_shape, weight_shape, out_shape) -> int:
    if inp_shape is None or weight_shape is None or out_shape is None:
        return 0
    if not all(isinstance(x, int) for x in inp_shape + weight_shape + out_shape):
        return 0
    groups = _get_attr_int(node, "group", 1)
    out_numel = _prod(out_shape)
    cin_per_group = int(weight_shape[1])
    kernel = _prod(weight_shape[2:])
    return out_numel * cin_per_group * kernel


def _estimate_einsum_macs(node: onnx.NodeProto, input_shapes: list[list[int | str] | None]) -> int:
    eq = _get_attr_str(node, "equation")
    if not eq or len(input_shapes) < 2:
        return 0
    if any(s is None or not all(isinstance(x, int) for x in s) for s in input_shapes):
        return 0

    s0_shape = input_shapes[0]
    s1_shape = input_shapes[1]
    if s0_shape is None or s1_shape is None:
        return 0

    s0 = [int(x) for x in s0_shape]
    s1 = [int(x) for x in s1_shape]

    if eq == "th,eih->tei":
        t, h = s0
        e, i, h2 = s1
        return t * e * i * h if h == h2 else 0
    if eq == "tei,ehi->teh":
        t, e, i = s0
        e2, h, i2 = s1
        return t * e * h * i if e == e2 and i == i2 else 0
    if eq == "te,teh->th":
        t, e = s0
        t2, e2, h = s1
        return t * h * e if t == t2 and e == e2 else 0
    return 0


def _estimate_node_macs(node: onnx.NodeProto, tensor_info: dict[str, dict]) -> int:
    input_shapes = [tensor_info.get(name, {}).get("shape") for name in node.input if name]
    output_shapes = [tensor_info.get(name, {}).get("shape") for name in node.output if name]

    if node.op_type == "MatMul" and len(input_shapes) >= 2:
        return _estimate_matmul_macs(input_shapes[0], input_shapes[1])

    if node.op_type == "Gemm" and len(input_shapes) >= 2:
        a_shape = input_shapes[0]
        b_shape = input_shapes[1]
        trans_b = _get_attr_int(node, "transB", 0)
        if b_shape is not None and trans_b and len(b_shape) == 2 and all(isinstance(x, int) for x in b_shape):
            b_shape = [int(b_shape[1]), int(b_shape[0])]
        return _estimate_matmul_macs(a_shape, b_shape)

    if node.op_type == "Conv" and len(input_shapes) >= 2 and output_shapes:
        return _estimate_conv_macs(node, input_shapes[0], input_shapes[1], output_shapes[0])

    if node.op_type == "Einsum":
        return _estimate_einsum_macs(node, input_shapes)

    return 0


def analyze_model(model_path: Path):
    model = onnx.load(str(model_path))
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass
    tensor_info = _build_tensor_info(model)
    _propagate_shapes(model, tensor_info)

    rows = []
    for idx, node in enumerate(model.graph.node):
        name = node.name or f"{node.op_type}_{idx:04d}"
        params = 0
        for inp in node.input:
            meta = tensor_info.get(inp)
            if meta and meta.get("is_initializer"):
                params += int(meta.get("numel", 0))

        memory = 0
        for out in node.output:
            meta = tensor_info.get(out)
            if meta:
                memory += int(meta.get("nbytes", 0))

        macs = _estimate_node_macs(node, tensor_info)
        rows.append({
            "Name": name,
            "Type": node.op_type,
            "Forward_MACs": int(macs),
            "Memory": int(memory),
            "Params": int(params),
            "InShape": _format_io_shapes(list(node.input), tensor_info),
            "OutShape": _format_io_shapes(list(node.output), tensor_info),
        })

    total_macs = sum(r["Forward_MACs"] for r in rows)
    total_memory = sum(r["Memory"] for r in rows)
    total_params_init = sum(r["Params"] for r in rows)
    total_params_external = _external_weight_params(model, tensor_info)
    total_params = total_params_init + total_params_external

    for r in rows:
        r["FPercent"] = 100.0 * r["Forward_MACs"] / total_macs if total_macs else 0.0
        r["MPercent"] = 100.0 * r["Memory"] / total_memory if total_memory else 0.0
        r["PPercent"] = 100.0 * r["Params"] / total_params if total_params else 0.0

    return {
        "rows": rows,
        "summary": {
            "model_path": str(model_path),
            "node_count": len(rows),
            "total_forward_macs": total_macs,
            "total_output_memory_bytes": total_memory,
            "total_params_initializer": total_params_init,
            "total_params_external": total_params_external,
            "total_params": total_params,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_path", help="Path to ONNX file")
    ap.add_argument(
        "--out_tsv",
        default=None,
        help="Output TSV path (default: <input>.flow_stats.tsv next to the input ONNX)",
    )
    ap.add_argument(
        "--out_json",
        default=None,
        help="Output summary JSON path (default: <input>.flow_stats.summary.json next to the input ONNX)",
    )
    args = ap.parse_args()

    model_path = Path(args.model_path).resolve()
    default_tsv = model_path.with_name(model_path.stem + ".flow_stats.tsv")
    default_json = model_path.with_name(model_path.stem + ".flow_stats.summary.json")
    out_tsv = Path(args.out_tsv).resolve() if args.out_tsv else default_tsv
    out_json = Path(args.out_json).resolve() if args.out_json else default_json
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    result = analyze_model(model_path)

    fieldnames = [
        "Name", "Type", "Forward_MACs", "FPercent",
        "Memory", "MPercent", "Params", "PPercent",
        "InShape", "OutShape",
    ]
    with open(out_tsv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({
                "Name": row["Name"],
                "Type": row["Type"],
                "Forward_MACs": row["Forward_MACs"],
                "FPercent": f"{row['FPercent']:.6f}",
                "Memory": row["Memory"],
                "MPercent": f"{row['MPercent']:.6f}",
                "Params": row["Params"],
                "PPercent": f"{row['PPercent']:.6f}",
                "InShape": row["InShape"],
                "OutShape": row["OutShape"],
            })

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, indent=2, ensure_ascii=False)

    print(f"TSV saved  : {out_tsv}")
    print(f"JSON saved : {out_json}")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
