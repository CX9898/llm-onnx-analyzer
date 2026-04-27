from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper as onnx_helper, numpy_helper


def _static_shape_propagation(model: onnx.ModelProto) -> None:
    graph = model.graph

    shapes: dict[str, list[int]] = {}
    elem_types: dict[str, int] = {}
    values: dict[str, np.ndarray] = {}

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

    def _record_tensor(
        name: str,
        shape: list[int] | None = None,
        elem_type: int | None = None,
        value: np.ndarray | None = None,
    ) -> bool:
        changed = False
        if shape is not None and shapes.get(name) != list(shape):
            shapes[name] = list(shape)
            changed = True
        if elem_type is not None and elem_types.get(name) != int(elem_type):
            elem_types[name] = int(elem_type)
            changed = True
        if value is not None:
            arr = np.asarray(value)
            if arr.dtype.kind in {"i", "u", "b"} or arr.size <= 256:
                old = values.get(name)
                if old is None or old.shape != arr.shape or old.dtype != arr.dtype or not np.array_equal(old, arr):
                    values[name] = arr
                    changed = True
        return changed

    def _seed_from_vi(vi) -> None:
        tt = vi.type.tensor_type
        if tt.elem_type:
            elem_types[vi.name] = int(tt.elem_type)
        dims = _dims_from_vi(vi)
        if dims is not None:
            shapes[vi.name] = dims

    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        _seed_from_vi(vi)

    for init in graph.initializer:
        shape = [int(dim) for dim in init.dims]
        value = None
        if init.data_location != onnx.TensorProto.EXTERNAL:
            arr = numpy_helper.to_array(init)
            shape = list(arr.shape)
            value = arr
        _record_tensor(init.name, shape, int(init.data_type), value)

    def _broadcast_shape(*in_shapes: list[int] | None) -> list[int] | None:
        if any(shape is None for shape in in_shapes):
            return None
        rev_shapes = [list(reversed(shape or [])) for shape in in_shapes]
        out_rev: list[int] = []
        max_rank = max(len(shape) for shape in rev_shapes)
        for axis in range(max_rank):
            dims = []
            for shape in rev_shapes:
                dims.append(shape[axis] if axis < len(shape) else 1)
            dim_out = max(dims)
            if any(d not in (1, dim_out) for d in dims):
                return None
            out_rev.append(dim_out)
        return list(reversed(out_rev))

    def _matmul_shape(a: list[int] | None, b: list[int] | None) -> list[int] | None:
        if a is None or b is None or len(a) < 2 or len(b) < 2:
            return None
        batch = _broadcast_shape(a[:-2], b[:-2])
        if batch is None:
            return None
        return batch + [a[-2], b[-1]]

    def _promote_elem_types(*types: int | None) -> int | None:
        filtered = [elem_type for elem_type in types if elem_type is not None]
        if not filtered:
            return None
        if TensorProto.DOUBLE in filtered:
            return TensorProto.DOUBLE
        if TensorProto.FLOAT in filtered:
            return TensorProto.FLOAT
        if TensorProto.FLOAT16 in filtered and TensorProto.BFLOAT16 in filtered:
            return TensorProto.FLOAT
        if TensorProto.FLOAT16 in filtered:
            return TensorProto.FLOAT16
        if TensorProto.BFLOAT16 in filtered:
            return TensorProto.BFLOAT16
        if TensorProto.INT64 in filtered:
            return TensorProto.INT64
        if TensorProto.INT32 in filtered:
            return TensorProto.INT32
        if TensorProto.BOOL in filtered:
            return TensorProto.BOOL
        return filtered[0]

    def _reshape_shape(input_shape: list[int] | None, target: np.ndarray | None) -> list[int] | None:
        if input_shape is None or target is None:
            return None
        target_list = [int(x) for x in np.asarray(target).reshape(-1).tolist()]
        result: list[int] = []
        infer_idx: int | None = None
        known_prod = 1
        input_prod = int(np.prod(input_shape))
        for idx, dim in enumerate(target_list):
            if dim == 0:
                if idx >= len(input_shape):
                    return None
                dim = input_shape[idx]
            elif dim == -1:
                if infer_idx is not None:
                    return None
                infer_idx = idx
                result.append(-1)
                continue
            result.append(dim)
            known_prod *= dim
        if infer_idx is not None:
            if known_prod == 0 or input_prod % known_prod != 0:
                return None
            result[infer_idx] = input_prod // known_prod
        return result

    def _slice_shape(
        data_shape: list[int] | None,
        starts: np.ndarray | None,
        ends: np.ndarray | None,
        axes: np.ndarray | None,
        steps: np.ndarray | None,
    ) -> list[int] | None:
        if data_shape is None or starts is None or ends is None:
            return None
        rank = len(data_shape)
        out = list(data_shape)
        starts_l = [int(x) for x in np.asarray(starts).reshape(-1).tolist()]
        ends_l = [int(x) for x in np.asarray(ends).reshape(-1).tolist()]
        axes_l = (
            [int(x) for x in np.asarray(axes).reshape(-1).tolist()]
            if axes is not None else list(range(len(starts_l)))
        )
        steps_l = (
            [int(x) for x in np.asarray(steps).reshape(-1).tolist()]
            if steps is not None else [1] * len(starts_l)
        )
        for start, end, axis, step in zip(starts_l, ends_l, axes_l, steps_l):
            if step == 0 or axis < 0 or axis >= rank:
                return None
            dim = data_shape[axis]
            s = start + dim if start < 0 else start
            e = end + dim if end < 0 else end
            s = min(max(s, 0), dim)
            e = min(max(e, 0), dim)
            if step > 0:
                span = max(0, e - s)
                out[axis] = (span + step - 1) // step
            else:
                return None
        return out

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

    def _set_graph_type(name: str, shape: list[int] | None, elem_type: int | None) -> None:
        if shape is None or elem_type is None:
            return
        target = None
        for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
            if vi.name == name:
                target = vi
                break
        if target is None:
            graph.value_info.append(onnx_helper.make_tensor_value_info(name, elem_type, shape))
            return
        tt = target.type.tensor_type
        tt.elem_type = elem_type
        del tt.shape.dim[:]
        for dim in shape:
            tt.shape.dim.add().dim_value = int(dim)

    progress = True
    while progress:
        progress = False
        for node in graph.node:
            op = node.op_type

            if node.domain == "qwen_onnx":
                from qwen_export_semantics import qwen_custom_output_specs

                inferred = qwen_custom_output_specs(
                    op,
                    [shapes.get(name) for name in node.input],
                    [elem_types.get(name) for name in node.input],
                )
                for out_name, (out_shape, out_type) in zip(node.output, inferred):
                    progress |= _record_tensor(out_name, out_shape, out_type)
                continue

            if op == "Constant":
                attr = _get_attr(node, "value")
                if attr is not None:
                    arr = numpy_helper.to_array(attr.t)
                    progress |= _record_tensor(node.output[0], list(arr.shape), int(attr.t.data_type), arr)
                continue

            if op in {"Cast", "Identity"}:
                in_name = node.input[0]
                out_type = elem_types.get(in_name)
                if op == "Cast":
                    attr = _get_attr(node, "to")
                    out_type = int(attr.i) if attr is not None else out_type
                progress |= _record_tensor(
                    node.output[0],
                    shapes.get(in_name),
                    out_type,
                    values.get(in_name).astype(np.int64) if op == "Cast" and in_name in values and out_type == TensorProto.INT64 else values.get(in_name),
                )
                continue

            if op == "Shape":
                in_shape = shapes.get(node.input[0])
                if in_shape is not None:
                    arr = np.asarray(in_shape, dtype=np.int64)
                    progress |= _record_tensor(node.output[0], [len(in_shape)], TensorProto.INT64, arr)
                continue

            if op == "Gather":
                axis = _get_attr_int(node, "axis", 0)
                data_v = values.get(node.input[0])
                idx_v = values.get(node.input[1])
                if data_v is not None and idx_v is not None:
                    arr = np.take(data_v, idx_v, axis=axis)
                    progress |= _record_tensor(
                        node.output[0],
                        list(np.asarray(arr).shape),
                        elem_types.get(node.input[0], TensorProto.FLOAT),
                        np.asarray(arr),
                    )
                else:
                    data_shape = shapes.get(node.input[0])
                    idx_shape = shapes.get(node.input[1], [])
                    if data_shape is not None:
                        out_shape = data_shape[:axis] + idx_shape + data_shape[axis + 1 :]
                        progress |= _record_tensor(node.output[0], out_shape, elem_types.get(node.input[0]))
                continue

            if op == "Unsqueeze":
                data_name = node.input[0]
                axes_v = values.get(node.input[1]) if len(node.input) > 1 else None
                axes_l = (
                    [int(x) for x in np.asarray(axes_v).reshape(-1).tolist()]
                    if axes_v is not None else _get_attr_ints(node, "axes")
                )
                in_shape = shapes.get(data_name)
                if in_shape is not None and axes_l is not None:
                    out_shape = list(in_shape)
                    rank = len(out_shape) + len(axes_l)
                    norm_axes = sorted(a + rank if a < 0 else a for a in axes_l)
                    for axis in norm_axes:
                        out_shape.insert(axis, 1)
                    progress |= _record_tensor(node.output[0], out_shape, elem_types.get(data_name))
                if data_name in values and axes_l is not None:
                    arr = np.asarray(values[data_name])
                    for axis in sorted(axes_l):
                        arr = np.expand_dims(arr, axis)
                    progress |= _record_tensor(node.output[0], list(arr.shape), elem_types.get(data_name), arr)
                continue

            if op == "Concat":
                axis = _get_attr_int(node, "axis", 0)
                input_values = [values.get(name) for name in node.input]
                if all(value is not None for value in input_values):
                    arr = np.concatenate([np.asarray(value) for value in input_values], axis=axis)
                    progress |= _record_tensor(node.output[0], list(arr.shape), elem_types.get(node.input[0]), arr)
                else:
                    input_shapes = [shapes.get(name) for name in node.input]
                    if all(shape is not None for shape in input_shapes):
                        out_shape = list(input_shapes[0] or [])
                        out_shape[axis] = sum((shape or [])[axis] for shape in input_shapes)
                        progress |= _record_tensor(node.output[0], out_shape, elem_types.get(node.input[0]))
                continue

            if op == "Reshape":
                out_shape = _reshape_shape(shapes.get(node.input[0]), values.get(node.input[1]))
                if out_shape is not None:
                    progress |= _record_tensor(node.output[0], out_shape, elem_types.get(node.input[0]))
                if node.input[0] in values and node.input[1] in values and out_shape is not None:
                    arr = np.asarray(values[node.input[0]]).reshape(out_shape)
                    progress |= _record_tensor(node.output[0], out_shape, elem_types.get(node.input[0]), arr)
                continue

            if op == "Transpose":
                perm = _get_attr_ints(node, "perm")
                in_shape = shapes.get(node.input[0])
                if in_shape is not None and perm is not None:
                    out_shape = [in_shape[i] for i in perm]
                    progress |= _record_tensor(node.output[0], out_shape, elem_types.get(node.input[0]))
                if node.input[0] in values and perm is not None:
                    arr = np.transpose(np.asarray(values[node.input[0]]), axes=perm)
                    progress |= _record_tensor(node.output[0], list(arr.shape), elem_types.get(node.input[0]), arr)
                continue

            if op in {"Add", "Sub", "Mul", "Div", "Pow", "Mod", "Where"}:
                input_shapes = [shapes.get(name) for name in node.input[: (3 if op == "Where" else 2)]]
                out_shape = _broadcast_shape(*input_shapes)
                type_inputs = (
                    [elem_types.get(node.input[1]), elem_types.get(node.input[2])]
                    if op == "Where"
                    else [elem_types.get(node.input[0]), elem_types.get(node.input[1])]
                )
                out_type = _promote_elem_types(*type_inputs)
                if out_shape is not None:
                    progress |= _record_tensor(node.output[0], out_shape, out_type)
                if all(name in values for name in node.input[: (3 if op == "Where" else 2)]):
                    if op == "Add":
                        arr = values[node.input[0]] + values[node.input[1]]
                    elif op == "Sub":
                        arr = values[node.input[0]] - values[node.input[1]]
                    elif op == "Mul":
                        arr = values[node.input[0]] * values[node.input[1]]
                    elif op == "Div":
                        arr = values[node.input[0]] / values[node.input[1]]
                    elif op == "Pow":
                        arr = values[node.input[0]] ** values[node.input[1]]
                    elif op == "Mod":
                        arr = np.mod(values[node.input[0]], values[node.input[1]])
                    else:
                        arr = np.where(values[node.input[0]], values[node.input[1]], values[node.input[2]])
                    progress |= _record_tensor(node.output[0], list(np.asarray(arr).shape), out_type, np.asarray(arr))
                continue

            if op in {"Neg", "Exp", "Sqrt"}:
                in_name = node.input[0]
                progress |= _record_tensor(node.output[0], shapes.get(in_name), elem_types.get(in_name))
                if in_name in values:
                    src = np.asarray(values[in_name])
                    if op == "Neg":
                        arr = -src
                    elif op == "Exp":
                        arr = np.exp(src)
                    else:
                        arr = np.sqrt(src)
                    progress |= _record_tensor(node.output[0], list(arr.shape), elem_types.get(in_name), arr)
                continue

            if op in {"ReduceSum", "ReduceMean"}:
                in_shape = shapes.get(node.input[0])
                axes_v = values.get(node.input[1]) if len(node.input) > 1 else None
                keepdims = _get_attr_int(node, "keepdims", 1)
                if in_shape is not None and axes_v is not None:
                    axes = [int(x) for x in np.asarray(axes_v).reshape(-1).tolist()]
                    rank = len(in_shape)
                    axes = sorted(a + rank if a < 0 else a for a in axes)
                    out_shape = []
                    for idx, dim in enumerate(in_shape):
                        if idx in axes:
                            if keepdims:
                                out_shape.append(1)
                        else:
                            out_shape.append(dim)
                    progress |= _record_tensor(node.output[0], out_shape, elem_types.get(node.input[0]))
                continue

            if op == "MatMul":
                out_shape = _matmul_shape(shapes.get(node.input[0]), shapes.get(node.input[1]))
                if out_shape is not None:
                    progress |= _record_tensor(node.output[0], out_shape, elem_types.get(node.input[0]))
                continue

            if op == "Slice":
                starts = values.get(node.input[1]) if len(node.input) > 1 else None
                ends = values.get(node.input[2]) if len(node.input) > 2 else None
                axes = values.get(node.input[3]) if len(node.input) > 3 else None
                steps = values.get(node.input[4]) if len(node.input) > 4 else None
                if node.input[0] in values and starts is not None and ends is not None:
                    data = np.asarray(values[node.input[0]])
                    rank = data.ndim
                    axes_l = [int(x) for x in np.asarray(axes).reshape(-1).tolist()] if axes is not None else list(range(len(np.asarray(starts).reshape(-1))))
                    steps_l = [int(x) for x in np.asarray(steps).reshape(-1).tolist()] if steps is not None else [1] * len(axes_l)
                    slices = [slice(None)] * rank
                    for st, ed, ax, step in zip(np.asarray(starts).reshape(-1), np.asarray(ends).reshape(-1), axes_l, steps_l):
                        slices[int(ax)] = slice(int(st), int(ed), int(step))
                    arr = data[tuple(slices)]
                    progress |= _record_tensor(node.output[0], list(arr.shape), elem_types.get(node.input[0]), arr)
                else:
                    out_shape = _slice_shape(shapes.get(node.input[0]), starts, ends, axes, steps)
                    if out_shape is not None:
                        progress |= _record_tensor(node.output[0], out_shape, elem_types.get(node.input[0]))
                continue

            if op == "Pad":
                in_shape = shapes.get(node.input[0])
                pads_v = values.get(node.input[1]) if len(node.input) > 1 else None
                if in_shape is not None and pads_v is not None:
                    pads = [int(x) for x in np.asarray(pads_v).reshape(-1).tolist()]
                    rank = len(in_shape)
                    if len(pads) == 2 * rank:
                        out_shape = [
                            in_shape[i] + pads[i] + pads[i + rank]
                            for i in range(rank)
                        ]
                        progress |= _record_tensor(node.output[0], out_shape, elem_types.get(node.input[0]))
                continue

            if op == "ConstantOfShape":
                shape_v = values.get(node.input[0])
                if shape_v is not None:
                    out_shape = [int(x) for x in np.asarray(shape_v).reshape(-1).tolist()]
                    attr = _get_attr(node, "value")
                    elem_type = int(attr.t.data_type) if attr is not None else TensorProto.FLOAT
                    progress |= _record_tensor(node.output[0], out_shape, elem_type)
                continue

            if op == "DeltaNetMaskDecay":
                g_shape = shapes.get(node.input[0])
                if g_shape is not None and len(g_shape) == 4:
                    bsz, nh, chunks, chunk = g_shape
                    progress |= _record_tensor(node.output[0], [bsz, nh, chunks, chunk], elem_types.get(node.input[0], TensorProto.FLOAT))
                    progress |= _record_tensor(node.output[1], [bsz, nh, chunks, chunk, chunk], elem_types.get(node.input[0], TensorProto.FLOAT))
                    progress |= _record_tensor(node.output[2], [chunk, chunk], TensorProto.BOOL)
                    progress |= _record_tensor(node.output[3], [chunk, chunk], TensorProto.BOOL)
                continue

            if op == "DeltaNetTriangularSolve":
                in_shape = shapes.get(node.input[0])
                if in_shape is not None:
                    progress |= _record_tensor(node.output[0], in_shape, elem_types.get(node.input[0], TensorProto.FLOAT))
                continue

            if op == "DeltaNetChunkStep":
                q_shape = shapes.get(node.input[0])
                v_shape = shapes.get(node.input[2])
                state_shape = shapes.get(node.input[7])
                if q_shape is not None and v_shape is not None and state_shape is not None:
                    progress |= _record_tensor(node.output[0], [q_shape[0], q_shape[1], q_shape[2], v_shape[3]], elem_types.get(node.input[2], TensorProto.FLOAT))
                    progress |= _record_tensor(node.output[1], state_shape, elem_types.get(node.input[7], TensorProto.FLOAT))
                continue

            if op == "DeltaNetChunkScan":
                value_shape = shapes.get(node.input[2])
                state_shape = shapes.get(node.input[6])
                if value_shape is not None and state_shape is not None:
                    progress |= _record_tensor(node.output[0], value_shape, elem_types.get(node.input[2], TensorProto.FLOAT))
                    progress |= _record_tensor(node.output[1], state_shape, elem_types.get(node.input[6], TensorProto.FLOAT))
                continue

            if op == "DeltaNetChunkLayoutPrep":
                query_shape = shapes.get(node.input[0])
                value_shape = shapes.get(node.input[2])
                g_shape = shapes.get(node.input[3])
                chunk_attr = _get_attr(node, "chunk_size")
                chunk_size = int(chunk_attr.i) if chunk_attr is not None else 64
                if (
                    query_shape is not None
                    and value_shape is not None
                    and g_shape is not None
                    and len(query_shape) == 4
                    and len(value_shape) == 4
                    and len(g_shape) == 3
                ):
                    bsz, seq_len, num_heads, head_k_dim = query_shape
                    head_v_dim = value_shape[3]
                    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size if isinstance(seq_len, int) else 0
                    num_chunks = (seq_len + pad_size) // chunk_size if isinstance(seq_len, int) else None
                    progress |= _record_tensor(
                        node.output[0],
                        [bsz, num_heads, num_chunks, chunk_size, head_k_dim],
                        elem_types.get(node.input[0], TensorProto.FLOAT),
                    )
                    progress |= _record_tensor(
                        node.output[1],
                        [bsz, num_heads, num_chunks, chunk_size, head_k_dim],
                        elem_types.get(node.input[1], TensorProto.FLOAT),
                    )
                    progress |= _record_tensor(
                        node.output[2],
                        [bsz, num_heads, num_chunks, chunk_size, head_v_dim],
                        elem_types.get(node.input[2], TensorProto.FLOAT),
                    )
                    progress |= _record_tensor(
                        node.output[3],
                        [bsz, num_heads, num_chunks, chunk_size, head_k_dim],
                        elem_types.get(node.input[1], TensorProto.FLOAT),
                    )
                    progress |= _record_tensor(
                        node.output[4],
                        [bsz, num_heads, num_chunks, chunk_size, head_v_dim],
                        elem_types.get(node.input[2], TensorProto.FLOAT),
                    )
                    progress |= _record_tensor(
                        node.output[5],
                        [bsz, num_heads, num_chunks, chunk_size],
                        elem_types.get(node.input[3], TensorProto.FLOAT),
                    )
                continue

            if op == "ChunkGatedDeltaRule":
                query_shape = shapes.get(node.input[0])
                value_shape = shapes.get(node.input[2])
                state_shape = shapes.get(node.input[5])
                if query_shape is not None and value_shape is not None and state_shape is not None:
                    progress |= _record_tensor(
                        node.output[0],
                        [query_shape[0], query_shape[1], query_shape[2], value_shape[3]],
                        elem_types.get(node.input[2], TensorProto.FLOAT),
                    )
                    progress |= _record_tensor(
                        node.output[1],
                        state_shape,
                        elem_types.get(node.input[5], TensorProto.FLOAT),
                    )
                continue

            if op == "RecurrentGatedDeltaRule":
                query_shape = shapes.get(node.input[0])
                value_shape = shapes.get(node.input[2])
                state_shape = shapes.get(node.input[5])
                if query_shape is not None and value_shape is not None and state_shape is not None:
                    progress |= _record_tensor(
                        node.output[0],
                        [query_shape[0], query_shape[1], query_shape[2], value_shape[3]],
                        elem_types.get(node.input[2], TensorProto.FLOAT),
                    )
                    progress |= _record_tensor(
                        node.output[1],
                        state_shape,
                        elem_types.get(node.input[5], TensorProto.FLOAT),
                    )
                continue

            if op == "DeltaNetGatedNormOutProj":
                core_shape = shapes.get(node.input[0])
                weight_shape = shapes.get(node.input[3])
                if core_shape is not None and weight_shape is not None and len(core_shape) >= 2 and len(weight_shape) == 2:
                    progress |= _record_tensor(node.output[0], [core_shape[0], core_shape[1], weight_shape[0]], TensorProto.FLOAT)
                continue

    custom_ops = {node.op_type for node in graph.node}
    if {
        "DeltaNetChunkLayoutPrep",
        "DeltaNetMaskDecay",
        "DeltaNetTriangularSolve",
        "DeltaNetChunkScan",
        "DeltaNetGatedNormOutProj",
    }.issubset(custom_ops):
        hidden_shape = shapes.get("hidden_states")
        recurrent_shape = shapes.get("recurrent_state")
        if hidden_shape is not None and recurrent_shape is not None and len(hidden_shape) == 3 and len(recurrent_shape) == 4:
            bsz, seq_len, _ = hidden_shape
            _, num_heads, head_k_dim, head_v_dim = recurrent_shape
            chunk_size = 64
            if seq_len % chunk_size == 0:
                num_chunks = seq_len // chunk_size
                tensor_overrides: dict[str, tuple[list[int], int]] = {
                    "/DeltaNetChunkLayoutPrep_output_0": ([bsz, num_heads, num_chunks, chunk_size, head_k_dim], TensorProto.FLOAT),
                    "/DeltaNetChunkLayoutPrep_output_1": ([bsz, num_heads, num_chunks, chunk_size, head_k_dim], TensorProto.FLOAT),
                    "/DeltaNetChunkLayoutPrep_output_2": ([bsz, num_heads, num_chunks, chunk_size, head_v_dim], TensorProto.FLOAT),
                    "/DeltaNetChunkLayoutPrep_output_3": ([bsz, num_heads, num_chunks, chunk_size, head_k_dim], TensorProto.FLOAT),
                    "/DeltaNetChunkLayoutPrep_output_4": ([bsz, num_heads, num_chunks, chunk_size, head_v_dim], TensorProto.FLOAT),
                    "/DeltaNetChunkLayoutPrep_output_5": ([bsz, num_heads, num_chunks, chunk_size], TensorProto.FLOAT),
                    "/DeltaNetMaskDecay_output_0": ([bsz, num_heads, num_chunks, chunk_size], TensorProto.FLOAT),
                    "/DeltaNetMaskDecay_output_1": ([bsz, num_heads, num_chunks, chunk_size, chunk_size], TensorProto.FLOAT),
                    "/DeltaNetMaskDecay_output_2": ([chunk_size, chunk_size], TensorProto.BOOL),
                    "/DeltaNetMaskDecay_output_3": ([chunk_size, chunk_size], TensorProto.BOOL),
                    "/Neg_output_0": ([bsz, num_heads, num_chunks, chunk_size, chunk_size], TensorProto.FLOAT),
                    "/DeltaNetTriangularSolve_output_0": ([bsz, num_heads, num_chunks, chunk_size, chunk_size], TensorProto.FLOAT),
                    "/MatMul_1_output_0": ([bsz, num_heads, num_chunks, chunk_size, head_v_dim], TensorProto.FLOAT),
                    "/MatMul_2_output_0": ([bsz, num_heads, num_chunks, chunk_size, head_k_dim], TensorProto.FLOAT),
                    "/DeltaNetChunkScan_output_0": ([bsz, num_heads, num_chunks, chunk_size, head_v_dim], TensorProto.FLOAT),
                    "/DeltaNetChunkScan_output_1": ([bsz, num_heads, head_k_dim, head_v_dim], TensorProto.FLOAT),
                    "/Reshape_15_output_0": ([bsz, num_heads, seq_len, head_v_dim], TensorProto.FLOAT),
                    "/Slice_5_output_0": ([bsz, num_heads, seq_len, head_v_dim], TensorProto.FLOAT),
                    "/Transpose_11_output_0": ([bsz, seq_len, num_heads, head_v_dim], TensorProto.FLOAT),
                    "/Cast_19_output_0": ([bsz, seq_len, num_heads, head_v_dim], TensorProto.FLOAT),
                    "output": ([bsz, seq_len, hidden_shape[2]], TensorProto.FLOAT),
                    "new_recurrent_state": ([bsz, num_heads, head_k_dim, head_v_dim], TensorProto.FLOAT),
                }
                for name, (shape, elem_type) in tensor_overrides.items():
                    _record_tensor(name, shape, elem_type)

    if {
        "DeltaNetMaskDecay",
        "DeltaNetTriangularSolve",
        "DeltaNetChunkStep",
    }.issubset(custom_ops) and "DeltaNetChunkScan" not in custom_ops:
        query_shape = shapes.get("query")
        value_shape = shapes.get("value")
        g_shape = shapes.get("g")
        recurrent_shape = shapes.get("recurrent_state")
        if (
            query_shape is not None
            and value_shape is not None
            and g_shape is not None
            and recurrent_shape is not None
            and len(query_shape) == 4
            and len(value_shape) == 4
            and len(g_shape) == 3
            and len(recurrent_shape) == 4
        ):
            bsz, seq_len, num_heads, head_k_dim = query_shape
            head_v_dim = value_shape[3]
            chunk_size = 64
            if seq_len % chunk_size == 0:
                num_chunks = seq_len // chunk_size
                tensor_overrides = {
                    "/Pad_output_0": ([bsz, num_heads, seq_len, head_k_dim], TensorProto.FLOAT),
                    "/Pad_1_output_0": ([bsz, num_heads, seq_len, head_k_dim], TensorProto.FLOAT),
                    "/Pad_2_output_0": ([bsz, num_heads, seq_len, head_v_dim], TensorProto.FLOAT),
                    "/Pad_3_output_0": ([bsz, num_heads, seq_len], TensorProto.FLOAT),
                    "/Pad_4_output_0": ([bsz, num_heads, seq_len], TensorProto.FLOAT),
                    "/Mul_4_output_0": ([bsz, num_heads, seq_len, head_k_dim], TensorProto.FLOAT),
                    "/Mul_5_output_0": ([bsz, num_heads, seq_len, head_k_dim], TensorProto.FLOAT),
                    "/Mul_6_output_0": ([bsz, num_heads, seq_len, head_v_dim], TensorProto.FLOAT),
                    "/Reshape_10_output_0": ([bsz, num_heads, num_chunks, chunk_size, head_k_dim], TensorProto.FLOAT),
                    "/Reshape_11_output_0": ([bsz, num_heads, num_chunks, chunk_size, head_k_dim], TensorProto.FLOAT),
                    "/Reshape_12_output_0": ([bsz, num_heads, num_chunks, chunk_size, head_v_dim], TensorProto.FLOAT),
                    "/Reshape_13_output_0": ([bsz, num_heads, num_chunks, chunk_size, head_k_dim], TensorProto.FLOAT),
                    "/Reshape_14_output_0": ([bsz, num_heads, num_chunks, chunk_size], TensorProto.FLOAT),
                    "/DeltaNetMaskDecay_output_0": ([bsz, num_heads, num_chunks, chunk_size], TensorProto.FLOAT),
                    "/DeltaNetMaskDecay_output_1": ([bsz, num_heads, num_chunks, chunk_size, chunk_size], TensorProto.FLOAT),
                    "/DeltaNetMaskDecay_output_2": ([chunk_size, chunk_size], TensorProto.BOOL),
                    "/DeltaNetMaskDecay_output_3": ([chunk_size, chunk_size], TensorProto.BOOL),
                    "/Transpose_10_output_0": ([bsz, num_heads, num_chunks, head_k_dim, chunk_size], TensorProto.FLOAT),
                    "/MatMul_output_0": ([bsz, num_heads, num_chunks, chunk_size, chunk_size], TensorProto.FLOAT),
                    "/Mul_7_output_0": ([bsz, num_heads, num_chunks, chunk_size, chunk_size], TensorProto.FLOAT),
                    "/Cast_17_output_0": ([chunk_size, chunk_size], TensorProto.FLOAT),
                    "/Where_output_0": ([bsz, num_heads, num_chunks, chunk_size, chunk_size], TensorProto.FLOAT),
                    "/Neg_output_0": ([bsz, num_heads, num_chunks, chunk_size, chunk_size], TensorProto.FLOAT),
                    "/DeltaNetTriangularSolve_output_0": ([bsz, num_heads, num_chunks, chunk_size, chunk_size], TensorProto.FLOAT),
                    "/MatMul_1_output_0": ([bsz, num_heads, num_chunks, chunk_size, head_v_dim], TensorProto.FLOAT),
                    "/Exp_output_0": ([bsz, num_heads, num_chunks, chunk_size], TensorProto.FLOAT),
                    "/Unsqueeze_20_output_0": ([bsz, num_heads, num_chunks, chunk_size, 1], TensorProto.FLOAT),
                    "/Mul_8_output_0": ([bsz, num_heads, num_chunks, chunk_size, head_k_dim], TensorProto.FLOAT),
                    "/MatMul_2_output_0": ([bsz, num_heads, num_chunks, chunk_size, head_k_dim], TensorProto.FLOAT),
                    "/Reshape_15_output_0": ([bsz, num_heads, seq_len, head_v_dim], TensorProto.FLOAT),
                    "/Slice_5_output_0": ([bsz, num_heads, seq_len, head_v_dim], TensorProto.FLOAT),
                    "/Transpose_11_output_0": ([bsz, seq_len, num_heads, head_v_dim], TensorProto.FLOAT),
                    "core_out": ([bsz, seq_len, num_heads, head_v_dim], TensorProto.FLOAT),
                    "new_recurrent_state": ([bsz, num_heads, head_k_dim, head_v_dim], TensorProto.FLOAT),
                }
                for name, (shape, elem_type) in tensor_overrides.items():
                    _record_tensor(name, shape, elem_type)

    if "Einsum" in custom_ops and "experts_gate_up" in shapes and "experts_down" in shapes:
        hidden_shape = shapes.get("hidden_states")
        gate_up_shape = shapes.get("experts_gate_up")
        experts_down_shape = shapes.get("experts_down")
        if (
            hidden_shape is not None
            and gate_up_shape is not None
            and experts_down_shape is not None
            and len(hidden_shape) == 3
            and len(gate_up_shape) == 3
            and len(experts_down_shape) == 3
        ):
            batch, seq_len, hidden = hidden_shape
            num_experts = gate_up_shape[0]
            inter = gate_up_shape[1] // 2
            token_count = batch * seq_len
            topk_value = values.get("/moe/Reshape_1_output_0")
            top_k = int(np.asarray(topk_value).reshape(-1)[0]) if topk_value is not None else None
            router_elem_type = elem_types.get("/moe/Cast_1_output_0", TensorProto.FLOAT)
            hidden_elem_type = elem_types.get("/moe/Reshape_output_0", TensorProto.FLOAT)
            tensor_overrides = {
                "/moe/TopK_output_0": ([token_count, top_k], router_elem_type) if top_k is not None else None,
                "/moe/TopK_output_1": ([token_count, top_k], TensorProto.INT64) if top_k is not None else None,
                "/moe/ReduceSum_output_0": ([token_count, 1], router_elem_type) if top_k is not None else None,
                "/moe/Div_output_0": ([token_count, top_k], router_elem_type) if top_k is not None else None,
                "/moe/OneHot_output_0": ([token_count, top_k, num_experts], TensorProto.INT64) if top_k is not None else None,
                "/moe/Cast_2_output_0": ([token_count, top_k, num_experts], hidden_elem_type) if top_k is not None else None,
                "/moe/Unsqueeze_2_output_0": ([token_count, top_k, 1], router_elem_type) if top_k is not None else None,
                "/moe/Mul_output_0": ([token_count, top_k, num_experts], hidden_elem_type) if top_k is not None else None,
                "/moe/ReduceSum_1_output_0": ([token_count, num_experts], hidden_elem_type),
                "/moe/Slice_output_0": ([token_count, num_experts, inter], hidden_elem_type),
                "/moe/Slice_1_output_0": ([token_count, num_experts, inter], hidden_elem_type),
                "/moe/Sigmoid_output_0": ([token_count, num_experts, inter], hidden_elem_type),
                "/moe/Mul_3_output_0": ([token_count, num_experts, inter], hidden_elem_type),
                "/moe/Mul_4_output_0": ([token_count, num_experts, inter], hidden_elem_type),
                "/moe/Einsum_1_output_0": ([token_count, num_experts, hidden], hidden_elem_type),
                "/moe/Einsum_2_output_0": ([token_count, hidden], hidden_elem_type),
                "/moe/Add_1_output_0": ([token_count, hidden], hidden_elem_type),
                "/moe/Reshape_2_output_0": ([batch, seq_len, hidden], hidden_elem_type),
            }
            for name, spec in tensor_overrides.items():
                if spec is None:
                    continue
                shape, elem_type = spec
                _record_tensor(name, shape, elem_type)

    if {"Expand", "MatMul", "Cos", "Sin"}.issubset(custom_ops) and "position_ids" in shapes:
        pos_shape = shapes.get("position_ids")
        cos_shape = shapes.get("cos")
        if pos_shape is not None and cos_shape is not None and len(pos_shape) == 2 and len(cos_shape) == 3:
            batch, seq_len = pos_shape
            half_dim = cos_shape[2] // 2
            tensor_overrides = {
                "/Expand_output_0": ([batch, half_dim, 1], TensorProto.FLOAT),
                "/MatMul_output_0": ([batch, half_dim, seq_len], TensorProto.FLOAT),
                "/Transpose_output_0": ([batch, seq_len, half_dim], TensorProto.FLOAT),
                "/Concat_output_0": ([batch, seq_len, cos_shape[2]], TensorProto.FLOAT),
                "/Cos_output_0": ([batch, seq_len, cos_shape[2]], TensorProto.FLOAT),
                "/Sin_output_0": ([batch, seq_len, cos_shape[2]], TensorProto.FLOAT),
            }
            for name, (shape, elem_type) in tensor_overrides.items():
                _record_tensor(name, shape, elem_type)

    if {"Expand", "MatMul", "o_proj"}.intersection(custom_ops) and "new_key" in shapes and "new_value" in shapes:
        new_key_shape = shapes.get("new_key")
        new_value_shape = shapes.get("new_value")
        reshape4_shape = shapes.get("/Reshape_4_output_0")
        reshape5_shape = shapes.get("/Reshape_5_output_0")
        if new_key_shape is not None and new_value_shape is not None and reshape4_shape is not None and reshape5_shape is not None:
            batch, kv_heads, seq_len, head_dim = new_key_shape
            expanded_heads = reshape4_shape[1] // kv_heads if kv_heads else 1
            tensor_overrides = {
                "/Expand_output_0": ([batch, kv_heads, expanded_heads, seq_len, head_dim], TensorProto.FLOAT),
                "/Expand_1_output_0": ([batch, kv_heads, expanded_heads, seq_len, head_dim], TensorProto.FLOAT),
            }
            for name, (shape, elem_type) in tensor_overrides.items():
                _record_tensor(name, shape, elem_type)

    if "position_ids" in shapes and "new_key" in shapes and "new_value" in shapes:
        pos_shape = shapes.get("position_ids")
        new_key_shape = shapes.get("new_key")
        new_value_shape = shapes.get("new_value")
        if (
            pos_shape is not None
            and new_key_shape is not None
            and new_value_shape is not None
            and len(pos_shape) == 2
            and len(new_key_shape) == 4
            and len(new_value_shape) == 4
        ):
            batch, seq_len = pos_shape
            _, kv_heads, total_len, head_dim = new_key_shape
            past_len = max(total_len - seq_len, 0)
            elem_type = elem_types.get("past_key", elem_types.get("new_key", TensorProto.FLOAT))
            _record_tensor("past_key", [batch, kv_heads, past_len, head_dim], elem_type)
            elem_type = elem_types.get("past_value", elem_types.get("new_value", TensorProto.FLOAT))
            _record_tensor("past_value", [batch, kv_heads, past_len, head_dim], elem_type)

    for name, shape in shapes.items():
        _set_graph_type(name, shape, elem_types.get(name))
