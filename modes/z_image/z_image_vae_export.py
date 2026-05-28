"""Export Z-Image VAE decode subgraph."""

from __future__ import annotations

import os

import torch

from z_image_export_semantics import ExportScene, default_strip_initializers
from z_image_export_shared import img_scene_tag, load_vae, onnx_export
from z_image_onnx_blocks import VAEDecodeBlock


def export_vae_decode(
    model_path: str,
    out_dir: str,
    scene: ExportScene,
    *,
    opset: int = 20,
    simplify: bool = True,
    strip_initializers: bool | None = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    vae = load_vae(model_path)
    block = VAEDecodeBlock(vae)
    device = next(vae.parameters()).device
    h_lat, w_lat = scene.latent_hw()
    latents = torch.randn(
        scene.batch_size,
        vae.config.latent_channels,
        h_lat,
        w_lat,
        device=device,
        dtype=torch.float32,
    )
    save_path = os.path.join(out_dir, f"vae_decode_{img_scene_tag(scene.image_size)}.onnx")
    strip = default_strip_initializers("vae_decode", cli_override=strip_initializers)
    onnx_export(
        block,
        (latents,),
        save_path,
        input_names=["latents"],
        output_names=["image"],
        dynamic_axes={},
        opset=opset,
        simplify=simplify,
        strip_initializers=strip,
    )
    return save_path
