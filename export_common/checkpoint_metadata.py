from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from onnx import TensorProto
from transformers import AutoConfig


DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def torch_dtype_to_name(dtype: torch.dtype) -> str:
    for name, mapped in DTYPE_MAP.items():
        if mapped == dtype:
            return name
    raise ValueError(f"Unsupported torch dtype: {dtype!r}")


def parse_config_torch_dtype(model_path: str) -> torch.dtype | None:
    try:
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    except Exception:
        return None

    config_dtype = getattr(config, "torch_dtype", None)
    if isinstance(config_dtype, torch.dtype):
        return config_dtype
    if isinstance(config_dtype, str):
        key = config_dtype.removeprefix("torch.").lower()
        return DTYPE_MAP.get(key)
    return None


def infer_checkpoint_torch_dtype(model_path: str) -> torch.dtype:
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        raise ValueError(f"model_path must be a local directory; got {model_path!r}")

    safetensor_paths = sorted(model_dir.glob("*.safetensors"))
    if safetensor_paths:
        try:
            from safetensors import safe_open

            with safe_open(str(safetensor_paths[0]), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
                if keys:
                    return handle.get_tensor(keys[0]).dtype
        except Exception:
            pass

    bin_candidates = sorted(model_dir.glob("pytorch_model*.bin"))
    for candidate in bin_candidates:
        try:
            state_dict = torch.load(candidate, map_location="cpu", weights_only=True)
        except Exception:
            continue
        if isinstance(state_dict, dict):
            for value in state_dict.values():
                if isinstance(value, torch.Tensor):
                    return value.dtype

    config_dtype = parse_config_torch_dtype(model_path)
    if config_dtype is not None:
        return config_dtype

    raise ValueError(
        "Failed to infer checkpoint dtype from local weights or config "
        f"for {model_path!r}."
    )


def model_float_dtype(model: nn.Module) -> torch.dtype:
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.is_floating_point():
            return tensor.dtype
    return torch.float32


def tensorproto_from_safetensors_dtype(dtype_name: str) -> int:
    key = dtype_name.upper()
    mapping = {
        "BF16": TensorProto.BFLOAT16,
        "F16": TensorProto.FLOAT16,
        "F32": TensorProto.FLOAT,
        "F64": TensorProto.DOUBLE,
        "I64": TensorProto.INT64,
        "I32": TensorProto.INT32,
        "BOOL": TensorProto.BOOL,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported safetensors dtype {dtype_name!r}")
    return mapping[key]


@dataclass(frozen=True)
class CheckpointTensorMeta:
    name: str
    file_name: str
    elem_type: int
    shape: list[int]
    absolute_offset: int
    byte_length: int
    dtype_name: str


@dataclass(frozen=True)
class ParameterBinding:
    graph_name: str
    source_name: str
    elem_type: int
    shape: list[int]
    transform: str = "identity"


class WeightMetadataReader:
    def __init__(self, model_path: str, *, reference_float_weight_name: str) -> None:
        self.model_path = Path(model_path)
        self.config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        self.reference_float_weight_name = reference_float_weight_name
        self._index_path = self.model_path / "model.safetensors.index.json"
        if not self._index_path.exists():
            raise FileNotFoundError(f"Missing safetensors index: {self._index_path}")
        with self._index_path.open("r", encoding="utf-8") as handle:
            self._weight_map = json.load(handle)["weight_map"]
        self._header_cache: dict[str, dict[str, object]] = {}
        self._meta_cache: dict[str, CheckpointTensorMeta] = {}

    def _header_for_file(self, file_name: str) -> tuple[int, dict[str, object]]:
        cached = self._header_cache.get(file_name)
        if cached is not None:
            return cached["base_offset"], cached["header"]  # type: ignore[return-value]

        path = self.model_path / file_name
        with path.open("rb") as handle:
            header_len = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(header_len).decode("utf-8"))
        base_offset = 8 + header_len
        self._header_cache[file_name] = {"base_offset": base_offset, "header": header}
        return base_offset, header

    def get(self, name: str) -> CheckpointTensorMeta:
        cached = self._meta_cache.get(name)
        if cached is not None:
            return cached

        file_name = self._weight_map[name]
        base_offset, header = self._header_for_file(file_name)
        info = header[name]
        dtype_name = info["dtype"]
        shape = [int(x) for x in info["shape"]]
        start, end = info["data_offsets"]
        meta = CheckpointTensorMeta(
            name=name,
            file_name=file_name,
            elem_type=tensorproto_from_safetensors_dtype(dtype_name),
            shape=shape,
            absolute_offset=base_offset + int(start),
            byte_length=int(end) - int(start),
            dtype_name=dtype_name,
        )
        self._meta_cache[name] = meta
        return meta

    def model_float_elem_type(self) -> int:
        meta = self.get(self.reference_float_weight_name)
        return meta.elem_type

    def tensor_proto_for_binding(self, binding: ParameterBinding, output_dir: Path) -> onnx.TensorProto:
        meta = self.get(binding.source_name)
        if (
            binding.transform == "identity"
            and meta.elem_type == binding.elem_type
            and meta.shape == binding.shape
        ):
            tensor = onnx.TensorProto()
            tensor.name = binding.graph_name
            tensor.data_type = binding.elem_type
            tensor.dims.extend(binding.shape)
            tensor.data_location = onnx.TensorProto.EXTERNAL
            tensor_path = self.model_path / meta.file_name
            location = os.path.relpath(tensor_path, output_dir)
            for key, value in (
                ("location", location),
                ("offset", str(meta.absolute_offset)),
                ("length", str(meta.byte_length)),
            ):
                entry = tensor.external_data.add()
                entry.key = key
                entry.value = value
            return tensor

        from safetensors import safe_open

        with safe_open(str(self.model_path / meta.file_name), framework="pt", device="cpu") as handle:
            value = handle.get_tensor(binding.source_name)

        if binding.transform == "transpose2d":
            value = value.t().contiguous()
        elif binding.transform == "add_one_float32":
            value = (1.0 + value.float()).contiguous()
        elif binding.transform == "cast_float32":
            value = value.to(torch.float32)
        elif binding.transform == "identity":
            value = value.contiguous()
        else:
            raise ValueError(f"Unsupported binding transform {binding.transform!r}")

        if binding.elem_type == TensorProto.BFLOAT16:
            value = value.to(torch.bfloat16).contiguous()
            raw_data = value.view(torch.uint16).cpu().numpy().tobytes()
        elif binding.elem_type == TensorProto.FLOAT:
            raw_data = value.to(torch.float32).cpu().numpy().tobytes()
        elif binding.elem_type == TensorProto.INT64:
            raw_data = value.to(torch.int64).cpu().numpy().tobytes()
        else:
            raise ValueError(f"Unsupported materialized elem_type {binding.elem_type}")

        tensor = onnx.TensorProto()
        tensor.name = binding.graph_name
        tensor.data_type = binding.elem_type
        tensor.dims.extend(binding.shape)
        tensor.raw_data = raw_data
        return tensor

