from __future__ import annotations

import argparse
import os
from collections.abc import Callable

import onnx
import torch
import torch.nn as nn

try:
    import onnxsim
    _ONNXSIM_AVAILABLE = True
except Exception:
    onnxsim = None
    _ONNXSIM_AVAILABLE = False

from export_common.onnx_graph_utils import (
    fold_pure_shape_chains_in_file,
    onnx_stats,
    print_onnx_stats,
    record_export,
)
from export_common.shape_inference_core import enrich_onnx_file


StaticShapePropagator = Callable[[onnx.ModelProto], None]


def _has_bfloat16_initializer_cast(model: onnx.ModelProto) -> bool:
    initializer_types = {init.name: int(init.data_type) for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Cast" or not node.input:
            continue
        if initializer_types.get(node.input[0]) == onnx.TensorProto.BFLOAT16:
            return True
    return False


def _restore_static_input_shapes(
    model: onnx.ModelProto,
    input_shapes: dict[str, list[int]],
) -> None:
    """
    Re-apply the dummy-input static shapes onto ``model.graph.input`` after
    ``onnxsim.simplify``. ``onnxsim`` may downgrade input ``value_info``
    annotations to ``unk__N`` when it folds a defensive Reshape sitting
    between an input and a downstream consumer, even though the graph's
    runtime contract still requires the original static shape (matched by
    every consumer node, e.g. ``ScatterND``).

    Operates only on inputs whose names appear in ``input_shapes`` and
    whose existing rank already matches the requested rank, so dynamic
    inputs that legitimately need ``dim_param`` placeholders are not
    rewritten.
    """
    for inp in model.graph.input:
        if inp.name not in input_shapes:
            continue
        target = input_shapes[inp.name]
        shape_proto = inp.type.tensor_type.shape
        if len(shape_proto.dim) != len(target):
            continue
        # Only restore when at least one dim is currently a placeholder
        # (don't churn proto bytes when the static shape is already there).
        needs_restore = any(
            (not d.HasField("dim_value")) or d.dim_param.startswith("unk")
            for d in shape_proto.dim
        )
        if not needs_restore:
            continue
        shape_proto.ClearField("dim")
        for size in target:
            new_dim = shape_proto.dim.add()
            new_dim.dim_value = int(size)


def parse_bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def seq_tag(seq_len: int) -> str:
    return f"{seq_len // 1024}k" if seq_len % 1024 == 0 else str(seq_len)


def layer_tag(layer_indices: list[int]) -> str:
    return "_".join(f"{idx:02d}" for idx in layer_indices)


def strip_initializers_to_inputs(save_path: str) -> None:
    model_proto = onnx.load(save_path)
    graph = model_proto.graph
    existing_inputs = {vi.name for vi in graph.input}
    retained_initializers = []

    for init in list(graph.initializer):
        if init.name.endswith("__folded"):
            retained_initializers.append(init)
            continue
        if init.name not in existing_inputs:
            graph.input.append(
                onnx.helper.make_tensor_value_info(init.name, init.data_type, list(init.dims))
            )
            existing_inputs.add(init.name)

    del graph.initializer[:]
    graph.initializer.extend(retained_initializers)
    onnx.save(model_proto, save_path)


# Per-tensor threshold: larger initializers go to a sidecar ``*.onnx.data`` file.
# Graph topology keeps weights as initializer → MatMul/Gemm inputs (Netron-friendly).
EXTERNAL_DATA_TENSOR_THRESHOLD_BYTES = 1024


def externalize_large_initializers(
    save_path: str,
    *,
    size_threshold: int = EXTERNAL_DATA_TENSOR_THRESHOLD_BYTES,
) -> bool:
    """Move bulky initializer bytes to ``<stem>.onnx.data``; keep initializer nodes in-graph."""
    model = onnx.load(save_path)
    inline_bytes = sum(len(init.raw_data or b"") for init in model.graph.initializer)
    if inline_bytes <= size_threshold:
        return False

    try:
        from onnx.external_data_helper import convert_model_to_external_data
    except ImportError:
        print("    external_data: onnx.external_data_helper unavailable — keeping inline initializers")
        return False

    location = os.path.basename(save_path) + ".data"
    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=location,
        size_threshold=size_threshold,
        convert_attribute=False,
    )
    onnx.save(model, save_path)
    data_path = os.path.join(os.path.dirname(save_path) or ".", location)
    data_mb = os.path.getsize(data_path) / (1 << 20) if os.path.isfile(data_path) else 0.0
    print(f"    external_data: sidecar {location} ({data_mb:.1f} MB), initializers remain in-graph")
    return True


def _external_data_paths(model: onnx.ModelProto, base_dir: str) -> set[str]:
    """Collect on-disk paths referenced by external-data tensors in *model*."""
    paths: set[str] = set()

    def add_tensor(tensor: onnx.TensorProto) -> None:
        if tensor.data_location != onnx.TensorProto.EXTERNAL and not tensor.external_data:
            return
        location = next((e.value for e in tensor.external_data if e.key == "location"), None)
        if location:
            paths.add(os.path.normpath(os.path.join(base_dir, location)))

    for init in model.graph.initializer:
        add_tensor(init)
    for init in model.graph.sparse_initializer:
        add_tensor(init)
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                add_tensor(attr.t)
            elif attr.type == onnx.AttributeProto.TENSORS:
                for tensor in attr.tensors:
                    add_tensor(tensor)
    return paths


def finalize_analysis_initializers(save_path: str) -> int:
    """
    Drop weight bytes for analysis-only exports.

    Keeps ``graph.initializer`` entries (shape/dtype + MatMul/Gemm edges) but
    removes inline ``raw_data``, external-data refs, and any external sidecar files.
    """
    base_dir = os.path.dirname(os.path.abspath(save_path)) or "."
    sidecar = save_path + ".data"
    model = onnx.load(save_path, load_external_data=True)
    ext_paths = _external_data_paths(model, base_dir)
    ext_paths.add(os.path.abspath(sidecar))

    dropped = 0
    for init in model.graph.initializer:
        has_bytes = bool(init.raw_data) or init.external_data or init.data_location == onnx.TensorProto.EXTERNAL
        if not has_bytes:
            continue
        init.ClearField("raw_data")
        init.data_location = onnx.TensorProto.DEFAULT
        while init.external_data:
            init.external_data.pop()
        dropped += 1

    onnx.save(model, save_path)
    removed_files = 0
    for path in ext_paths:
        if os.path.isfile(path):
            os.remove(path)
            removed_files += 1
    if dropped:
        print(
            f"    analysis_weights: dropped bytes for {dropped} initializers "
            f"(topology + shape kept; removed {removed_files} external file(s))"
        )
    return dropped


def _snapshot_dir_files(base_dir: str) -> set[str]:
    if not os.path.isdir(base_dir):
        return set()
    return {
        os.path.abspath(os.path.join(base_dir, name))
        for name in os.listdir(base_dir)
        if os.path.isfile(os.path.join(base_dir, name))
    }


def _remove_export_artifact_files(save_path: str, before: set[str]) -> int:
    """Delete intermediate external-weight files left by torch/onnxsim in *base_dir*."""
    base_dir = os.path.dirname(os.path.abspath(save_path)) or "."
    keep = {os.path.abspath(save_path)}
    removed = 0
    for path in _snapshot_dir_files(base_dir) - before - keep:
        os.remove(path)
        removed += 1
    if removed:
        print(f"    analysis_weights: removed {removed} intermediate artifact file(s)")
    return removed


def shape_enrich_onnx_file(
    save_path: str,
    *,
    static_shape_propagator: StaticShapePropagator | None = None,
) -> None:
    enrich_onnx_file(save_path, static_shape_propagator=static_shape_propagator)


def onnx_export(
    module: nn.Module,
    dummy_inputs: tuple,
    save_path: str,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict,
    *,
    opset: int = 17,
    simplify: bool = True,
    strip_initializers: bool = False,
    custom_opsets: dict[str, int] | None = None,
    fold_pure_shape_chains: bool = False,
    shape_enrich_after_fold: bool = True,
    collect_onnx_stats: bool = True,
    static_shape_propagator: StaticShapePropagator | None = None,
) -> None:
    base_dir = os.path.dirname(os.path.abspath(save_path)) or "."
    os.makedirs(base_dir, exist_ok=True)
    dir_before = _snapshot_dir_files(base_dir)
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            dummy_inputs,
            save_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            # Keep source-level dtype transitions visible in ONNX instead of
            # folding casts on checkpoint weights into new initializers.
            do_constant_folding=False,
            dynamo=False,
            custom_opsets=custom_opsets,
        )

    max_size = 2 * (1 << 30)
    file_size = os.path.getsize(save_path)
    shape_ok = False
    sim_status = "not_run"
    graph_optimizations: list[str] = []
    should_shape_enrich = simplify or fold_pure_shape_chains or strip_initializers

    if should_shape_enrich:
        try:
            shape_enrich_onnx_file(save_path, static_shape_propagator=static_shape_propagator)
            print("    shape_inference: OK" + (" (custom ops + static propagation)" if custom_opsets else ""))
            shape_ok = True
        except Exception as exc:
            print(f"    shape_inference: skipped ({exc})")

        if fold_pure_shape_chains:
            try:
                fold_stats = fold_pure_shape_chains_in_file(save_path)
                if any(fold_stats.values()):
                    graph_optimizations.append("fold_pure_shape_chains")
                    print(
                        "    shape_fold: "
                        f"replaced_inputs={fold_stats['replaced_inputs']}  "
                        f"added_initializers={fold_stats['added_initializers']}  "
                        f"removed_nodes={fold_stats['removed_nodes']}"
                    )
                    if shape_enrich_after_fold:
                        shape_enrich_onnx_file(save_path, static_shape_propagator=static_shape_propagator)
                        shape_ok = True
                    else:
                        print("    shape_inference(after fold): skipped by config")
                else:
                    print("    shape_fold: no eligible pure shape chains found")
            except Exception as exc:
                print(f"    shape_fold: skipped ({exc})")

    if simplify:
        if custom_opsets:
            print("    onnxsim: skipped (custom ops present)")
        elif file_size > max_size:
            sim_status = "skipped_2gb"
            print(
                f"    onnxsim: skipped (file {file_size/(1<<20):.0f} MB > 2 GB limit — "
                "shape_inference already applied)"
            )
        elif _ONNXSIM_AVAILABLE and onnxsim is not None:
            model_proto = onnx.load(save_path)
            graph_input_names = {inp.name for inp in model_proto.graph.input}
            input_shapes = {
                name: list(t.shape)
                for name, t in zip(input_names, dummy_inputs)
                if isinstance(t, torch.Tensor) and name in graph_input_names
            }
            skip_constant_folding = _has_bfloat16_initializer_cast(model_proto)
            try:
                simplified, ok = onnxsim.simplify(
                    model_proto,
                    test_input_shapes=input_shapes,
                    skip_constant_folding=skip_constant_folding,
                )
                if ok:
                    # ``onnxsim.simplify`` occasionally downgrades the
                    # graph-input ``value_info`` shape annotations when
                    # it folds a defensive Reshape sitting between an
                    # input and a downstream consumer (notably ScatterND,
                    # see ``MMInjectBlock``). The graph topology is
                    # correct — only the input's *declared* shape gets
                    # turned into ``unk__N``. Since ``dummy_inputs`` is
                    # authoritative for the static representative
                    # scenario (and ``dynamic_axes`` is honoured by the
                    # tracer separately, well before this pass), we
                    # restore the static shape annotations here so the
                    # downstream MAC / memory / shape analysis tools see
                    # full static IO contracts.
                    _restore_static_input_shapes(simplified, input_shapes)
                    onnx.save(simplified, save_path)
                    sim_status = "ok_skip_cf" if skip_constant_folding else "ok"
                    if skip_constant_folding:
                        print("    onnxsim: simplified OK (constant folding skipped for BF16 initializer casts)")
                    else:
                        print("    onnxsim: simplified OK")
                else:
                    sim_status = "check_failed"
                    print("    onnxsim: check failed — keeping shape-inferred model")
            except Exception as exc:
                sim_status = f"error({exc})"
                print(f"    onnxsim: skipped ({exc})")
        else:
            sim_status = "not_installed"
            print("    onnxsim: not installed (run `pip install onnx-simplifier`)")

    if strip_initializers:
        strip_initializers_to_inputs(save_path)
        print("    strip_initializers: converted all initializers to graph inputs (discouraged for visualization)")
        try:
            shape_enrich_onnx_file(save_path, static_shape_propagator=static_shape_propagator)
            print("    shape_inference(after strip): OK")
            shape_ok = True
        except Exception as exc:
            print(f"    shape_inference(after strip): skipped ({exc})")
    else:
        externalize_large_initializers(save_path)
        finalize_analysis_initializers(save_path)
        _remove_export_artifact_files(save_path, dir_before)

    file_size = os.path.getsize(save_path)
    print(f"    → {save_path}  ({file_size / (1 << 20):.1f} MB)")
    param_count = None
    op_counter = None
    if collect_onnx_stats:
        param_count, op_counter = onnx_stats(save_path)
        print_onnx_stats(save_path, file_size)
    else:
        print("    stats  : skipped by config")
    record_export(save_path, file_size, shape_ok, sim_status, param_count, op_counter, graph_optimizations)

