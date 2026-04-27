from __future__ import annotations

from collections.abc import Callable

import onnx


StaticShapePropagator = Callable[[onnx.ModelProto], None]


def enrich_model_shapes(
    model: onnx.ModelProto,
    *,
    static_shape_propagator: StaticShapePropagator | None = None,
) -> onnx.ModelProto:
    model = onnx.shape_inference.infer_shapes(model)
    if static_shape_propagator is not None:
        static_shape_propagator(model)
        model = onnx.shape_inference.infer_shapes(model)
        static_shape_propagator(model)
    return model


def enrich_onnx_file(
    save_path: str,
    *,
    static_shape_propagator: StaticShapePropagator | None = None,
) -> None:
    model = onnx.load(save_path, load_external_data=False)
    model = enrich_model_shapes(model, static_shape_propagator=static_shape_propagator)
    onnx.save(model, save_path)

