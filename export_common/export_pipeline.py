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
        print("    strip_initializers: converted all initializers to graph inputs")
        try:
            shape_enrich_onnx_file(save_path, static_shape_propagator=static_shape_propagator)
            print("    shape_inference(after strip): OK")
            shape_ok = True
        except Exception as exc:
            print(f"    shape_inference(after strip): skipped ({exc})")

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

