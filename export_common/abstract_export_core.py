from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import onnx
from onnx import helper as onnx_helper

from export_common.checkpoint_metadata import ParameterBinding, WeightMetadataReader


@dataclass(frozen=True)
class TensorSpec:
    name: str
    elem_type: int
    shape: list[int]
    role: str


@dataclass(frozen=True)
class ExportScene:
    batch_size: int
    seq_len: int
    decode_context_len: int
    phase: str


@dataclass
class ModuleSemantics:
    runtime_inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    parameter_bindings: list[ParameterBinding]


class TypeShapeEnv:
    def __init__(self) -> None:
        self._items: dict[str, TensorSpec] = {}

    def register(self, spec: TensorSpec) -> None:
        self._items[spec.name] = spec

    def extend(self, specs: Iterable[TensorSpec]) -> None:
        for spec in specs:
            self.register(spec)

    def values(self) -> list[TensorSpec]:
        return list(self._items.values())


def tensor_dims(vi: onnx.ValueInfoProto) -> list[int]:
    dims: list[int] = []
    for dim in vi.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        else:
            dims.append(0)
    return dims


def make_value_info(spec: TensorSpec) -> onnx.ValueInfoProto:
    return onnx_helper.make_tensor_value_info(spec.name, spec.elem_type, spec.shape)


def replace_initializer(graph: onnx.GraphProto, tensor: onnx.TensorProto) -> None:
    for idx, init in enumerate(graph.initializer):
        if init.name == tensor.name:
            graph.initializer[idx].CopyFrom(tensor)
            return
    graph.initializer.append(tensor)


def clear_external_parameter_inputs(graph: onnx.GraphProto, runtime_input_names: set[str]) -> None:
    kept = [item for item in graph.input if item.name in runtime_input_names]
    del graph.input[:]
    graph.input.extend(kept)


def set_outputs(graph: onnx.GraphProto, outputs: list[TensorSpec]) -> None:
    del graph.output[:]
    for spec in outputs:
        graph.output.append(make_value_info(spec))


def set_runtime_inputs(graph: onnx.GraphProto, inputs: list[TensorSpec]) -> None:
    del graph.input[:]
    for spec in inputs:
        graph.input.append(make_value_info(spec))


def clear_value_info(graph: onnx.GraphProto) -> None:
    del graph.value_info[:]


BuildModuleSemantics = Callable[[onnx.ModelProto, str, ExportScene, WeightMetadataReader], ModuleSemantics]
ShapeEnricher = Callable[[str], None]


def rewrite_template_to_abstract_model(
    template_path: Path,
    output_path: Path,
    reader: WeightMetadataReader,
    scene: ExportScene,
    *,
    build_module_semantics: BuildModuleSemantics,
    shape_enricher: ShapeEnricher | None = None,
) -> None:
    model = onnx.load(template_path)
    semantics = build_module_semantics(model, template_path.name, scene, reader)

    graph = model.graph
    set_runtime_inputs(graph, semantics.runtime_inputs)
    set_outputs(graph, semantics.outputs)
    clear_value_info(graph)

    runtime_names = {spec.name for spec in semantics.runtime_inputs}
    clear_external_parameter_inputs(graph, runtime_names)

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for binding in semantics.parameter_bindings:
        tensor = reader.tensor_proto_for_binding(binding, output_dir)
        replace_initializer(graph, tensor)

    onnx.save(model, output_path)

    if shape_enricher is not None:
        shape_enricher(str(output_path))
        refreshed = onnx.load(output_path, load_external_data=False)
        set_runtime_inputs(refreshed.graph, semantics.runtime_inputs)
        set_outputs(refreshed.graph, semantics.outputs)
        onnx.save(refreshed, output_path)


def rewrite_structured_manifest(
    template_manifest_path: Path,
    output_manifest_path: Path,
    output_dir: Path,
) -> None:
    payload = json.loads(template_manifest_path.read_text(encoding="utf-8"))

    def _rewrite(obj: object) -> object:
        if isinstance(obj, dict):
            rewritten: dict[str, object] = {}
            for key, value in obj.items():
                if key == "path" and isinstance(value, str):
                    rewritten[key] = str(output_dir / Path(value).name)
                else:
                    rewritten[key] = _rewrite(value)
            return rewritten
        if isinstance(obj, list):
            return [_rewrite(item) for item in obj]
        return obj

    output_manifest_path.write_text(
        json.dumps(_rewrite(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

