"""Shared helpers for Z-Image ONNX export."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

_EXPORT_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _EXPORT_ROOT.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_EXPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPORT_ROOT))

_DIFFUSERS_SRC = Path(
    os.environ.get("DIFFUSERS_SRC", "/home/zcx/CLionProjects/diffusers/src")
)
if _DIFFUSERS_SRC.is_dir() and str(_DIFFUSERS_SRC) not in sys.path:
    sys.path.insert(0, str(_DIFFUSERS_SRC))

from export_common.checkpoint_metadata import (  # noqa: E402
    infer_checkpoint_torch_dtype,
    model_float_dtype,
    torch_dtype_to_name,
)
from export_common.export_pipeline import (  # noqa: E402
    onnx_export as _common_onnx_export,
    seq_tag as _common_seq_tag,
    shape_enrich_onnx_file as _common_shape_enrich_onnx_file,
    strip_initializers_to_inputs as _common_strip_initializers_to_inputs,
)
from export_common.manifest_utils import write_structured_manifest as _common_write_structured_manifest  # noqa: E402
from export_common.static_shape_propagation import apply_static_shape_propagation  # noqa: E402

from diffusers import AutoencoderKL, ZImageTransformer2DModel  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402


PATCH_KEY = "2-1"
DEFAULT_CAP_SEQ = 128
DEFAULT_IMAGE_SIZE = 512


def seq_tag(seq_len: int) -> str:
    return _common_seq_tag(seq_len)


def img_scene_tag(image_size: int) -> str:
    """Spatial scene tag, e.g. img512 for 512×512 representative export."""
    return f"img{image_size}"


def cap_scene_tag(cap_seq: int) -> str:
    """Caption token-length tag, e.g. cap128."""
    return f"cap{cap_seq}"


def seq_scene_tag(seq_len: int) -> str:
    """DiT token-sequence tag, e.g. seq1k, seq1152."""
    return f"seq{seq_tag(seq_len)}"


def image_latent_hw(image_size: int, vae_scale_factor: int = 8) -> tuple[int, int]:
    """Match ``ZImagePipeline.prepare_latents`` spatial size."""
    vae_scale = vae_scale_factor * 2
    side = 2 * (int(image_size) // vae_scale)
    return side, side


def image_token_seq_len(image_size: int, vae_scale_factor: int = 8, patch_size: int = 2) -> int:
    h_lat, w_lat = image_latent_hw(image_size, vae_scale_factor)
    return (h_lat // patch_size) * (w_lat // patch_size)


def patch_feature_dim(in_channels: int = 16, patch_size: int = 2, f_patch_size: int = 1) -> int:
    return f_patch_size * patch_size * patch_size * in_channels


def load_transformer(model_path: str, device: str = "cpu", dtype: torch.dtype | None = None) -> ZImageTransformer2DModel:
    transformer_dir = Path(model_path) / "transformer"
    if dtype is None:
        dtype = infer_checkpoint_torch_dtype(str(transformer_dir))
    model = ZImageTransformer2DModel.from_pretrained(
        str(transformer_dir),
        torch_dtype=dtype,
        local_files_only=True,
    )
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def load_vae(model_path: str, device: str = "cpu", dtype: torch.dtype | None = None) -> AutoencoderKL:
    vae_dir = Path(model_path) / "vae"
    if dtype is None:
        dtype = infer_checkpoint_torch_dtype(str(vae_dir))
    vae = AutoencoderKL.from_pretrained(
        str(vae_dir),
        torch_dtype=dtype,
        local_files_only=True,
    )
    vae.to(device=device, dtype=dtype)
    vae.eval()
    return vae


def load_text_encoder(model_path: str, device: str = "cpu", dtype: torch.dtype | None = None) -> AutoModel:
    te_dir = Path(model_path) / "text_encoder"
    if dtype is None:
        dtype = infer_checkpoint_torch_dtype(str(te_dir))
    model = AutoModel.from_pretrained(
        str(te_dir),
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def load_tokenizer(model_path: str) -> AutoTokenizer:
    tok_dir = Path(model_path) / "tokenizer"
    te_dir = Path(model_path) / "text_encoder"
    path = tok_dir if tok_dir.is_dir() else te_dir
    return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)


def model_dtype(model: nn.Module) -> torch.dtype:
    return model_float_dtype(model)


def onnx_export(
    module: nn.Module,
    dummy_inputs: tuple,
    save_path: str,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict | None = None,
    *,
    opset: int = 20,
    simplify: bool = True,
    strip_initializers: bool = False,
    custom_opsets: dict[str, int] | None = None,
) -> None:
    _common_onnx_export(
        module,
        dummy_inputs,
        save_path,
        input_names,
        output_names,
        dynamic_axes or {},
        opset=opset,
        simplify=simplify,
        strip_initializers=strip_initializers,
        custom_opsets=custom_opsets,
        static_shape_propagator=apply_static_shape_propagation,
    )


def make_rope_cos_sin(
    transformer: ZImageTransformer2DModel,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos_ids = transformer.create_coordinate_grid(
        size=(1, 16, 16),
        start=(0, 0, 0),
        device=device,
    ).flatten(0, 2)
    if pos_ids.shape[0] < seq_len:
        pad = transformer.create_coordinate_grid(
            size=(1, 1, 1),
            start=(0, 0, 0),
            device=device,
        ).flatten(0, 2).repeat(seq_len - pos_ids.shape[0], 1)
        pos_ids = torch.cat([pos_ids, pad], dim=0)
    pos_ids = pos_ids[:seq_len]
    freqs = transformer.rope_embedder(pos_ids)
    cos = freqs.real.unsqueeze(0).expand(batch_size, -1, -1)
    sin = freqs.imag.unsqueeze(0).expand(batch_size, -1, -1)
    return cos, sin


def make_rope_freqs_cis(
    transformer: ZImageTransformer2DModel,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Legacy helper; prefer ``make_rope_cos_sin`` for block export."""
    cos, sin = make_rope_cos_sin(transformer, batch_size, seq_len, device)
    return torch.complex(cos, sin)


def make_attn_mask(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)


def write_manifest(graph_defs: dict, save_path: str, root_key: str) -> None:
    _common_write_structured_manifest(graph_defs, root_key=root_key, save_path=save_path)
