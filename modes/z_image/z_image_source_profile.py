"""
Ground-truth I/O profile from ``transformer_z_image.py`` forward + real checkpoint.

All shape/dtype contracts for ONNX export must be derivable from this module —
not hand-tuned to pass boundary checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from diffusers import AutoencoderKL, ZImageTransformer2DModel
from transformers import AutoModel


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: list[int]
    dtype: str
    source: str  # e.g. "transformer_z_image.py:945"


@dataclass(frozen=True)
class ForwardStep:
    """One semantic step in ``ZImageTransformer2DModel.forward`` (basic mode)."""

    step_id: str
    source: str
    outputs: tuple[TensorSpec, ...]
    notes: str = ""


@dataclass
class ZImageSourceProfile:
    model_path: str
    image_size: int
    cap_seq: int
    batch_size: int
    patch_size: int
    f_patch_size: int

    # from config.json + safetensors
    dim: int = 0
    in_channels: int = 0
    cap_feat_dim: int = 0
    n_layers: int = 0
    n_refiner_layers: int = 0
    n_text_hidden_layers: int = 0
    n_text_forward_layers: int = 0
    text_rope_dim: int = 0
    t_scale: float = 1000.0
    axes_dims: list[int] = field(default_factory=list)
    axes_lens: list[int] = field(default_factory=list)
    weight_dtype: str = "bfloat16"
    text_encoder_dtype: str = "bfloat16"
    vae_dtype: str = "bfloat16"
    latent_hw: tuple[int, int] = (0, 0)
    image_token_seq: int = 0
    patch_dim: int = 0
    rope_dim: int = 0

    # traced from one forward (basic mode, representative static scene)
    forward_steps: tuple[ForwardStep, ...] = ()

    # pipeline boundaries (pipeline_z_image.py — outside transformer)
    pipeline_steps: tuple[ForwardStep, ...] = ()


def _dtype_name(t: torch.dtype | torch.Tensor) -> str:
    if isinstance(t, torch.Tensor):
        if t.is_complex():
            return "complex64"
        t = t.dtype
    mapping = {
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float16: "float16",
        torch.int64: "int64",
        torch.int32: "int32",
        torch.bool: "bool",
    }
    return mapping.get(t, str(t).replace("torch.", ""))


def _latent_hw(image_size: int, vae_scale_factor: int = 8) -> tuple[int, int]:
    vae_scale = vae_scale_factor * 2
    side = 2 * (int(image_size) // vae_scale)
    return side, side


def build_source_profile(
    model_path: str,
    *,
    image_size: int = 512,
    cap_seq: int = 128,
    batch_size: int = 1,
    patch_size: int = 2,
    f_patch_size: int = 1,
    device: str = "cpu",
) -> ZImageSourceProfile:
    root = Path(model_path)
    tr_dir = root / "transformer"
    cfg = json.loads((tr_dir / "config.json").read_text())

    profile = ZImageSourceProfile(
        model_path=str(root),
        image_size=image_size,
        cap_seq=cap_seq,
        batch_size=batch_size,
        patch_size=patch_size,
        f_patch_size=f_patch_size,
        dim=int(cfg["dim"]),
        in_channels=int(cfg["in_channels"]),
        cap_feat_dim=int(cfg["cap_feat_dim"]),
        n_layers=int(cfg["n_layers"]),
        n_refiner_layers=int(cfg["n_refiner_layers"]),
        t_scale=float(cfg["t_scale"]),
        axes_dims=list(cfg["axes_dims"]),
        axes_lens=list(cfg["axes_lens"]),
    )

    h, w = _latent_hw(image_size)
    profile.latent_hw = (h, w)
    profile.image_token_seq = (h // patch_size) * (w // patch_size)
    profile.patch_dim = f_patch_size * patch_size * patch_size * profile.in_channels
    profile.rope_dim = sum(d // 2 for d in profile.axes_dims)

    dtype = torch.bfloat16
    tr = ZImageTransformer2DModel.from_pretrained(
        str(tr_dir), torch_dtype=dtype, local_files_only=True
    ).to(device).eval()
    te = AutoModel.from_pretrained(
        str(root / "text_encoder"), torch_dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    _ = AutoencoderKL.from_pretrained(
        str(root / "vae"), torch_dtype=dtype, local_files_only=True
    ).to(device).eval()

    profile.weight_dtype = _dtype_name(next(tr.parameters()))
    profile.text_encoder_dtype = _dtype_name(next(te.parameters()))
    te_cfg = te.config
    profile.n_text_hidden_layers = int(te_cfg.num_hidden_layers)
    profile.n_text_forward_layers = profile.n_text_hidden_layers - 1
    head_dim = int(getattr(te_cfg, "head_dim", te_cfg.hidden_size // te_cfg.num_attention_heads))
    profile.text_rope_dim = head_dim

    # --- pipeline boundaries ---
    latents_f32 = torch.randn(batch_size, profile.in_channels, h, w, device=device, dtype=torch.float32)
    latents_bf16 = latents_f32.to(tr.dtype).unsqueeze(2)

    ids = torch.randint(0, 1000, (batch_size, cap_seq), device=device)
    mask = torch.ones(batch_size, cap_seq, device=device, dtype=torch.long)
    te_hidden = te(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-2]

    pipeline_steps = [
        ForwardStep(
            "P0_latents",
            "pipeline_z_image.py:prepare_latents",
            (TensorSpec("latents", [batch_size, profile.in_channels, h, w], "float32", "pipeline"),),
        ),
        ForwardStep(
            "P1_latent_model_input",
            "pipeline_z_image.py:521 latents.to(transformer.dtype).unsqueeze(2)",
            (TensorSpec("latent", [batch_size, profile.in_channels, 1, h, w], profile.weight_dtype, "pipeline"),),
        ),
        ForwardStep(
            "P2_text_hidden",
            "pipeline_z_image.py:_encode_prompt hidden_states[-2]",
            (TensorSpec("prompt_hidden", [batch_size, cap_seq, profile.cap_feat_dim], profile.text_encoder_dtype, "pipeline"),),
        ),
    ]

    # --- transformer forward (basic mode), same order as source ---
    x_list = list(latents_bf16.unbind(0))
    cap_list = [te_hidden[i][mask[i].bool()] for i in range(batch_size)]
    t = torch.tensor([0.5], device=device)
    pk = f"{patch_size}-{f_patch_size}"

    adaln = tr.t_embedder(t * tr.t_scale).type_as(x_list[0])
    x, cap_raw, x_size, x_pos, cap_pos, x_pad, cap_pad = tr.patchify_and_embed(
        x_list, cap_list, patch_size, f_patch_size
    )
    x_cat = tr.all_x_embedder[pk](torch.cat(x, dim=0))
    x_seqlens = [len(xi) for xi in x]
    x_emb, x_freqs, x_mask, _, _ = tr._prepare_sequence(
        list(x_cat.split(x_seqlens, 0)), x_pos, x_pad, tr.x_pad_token, None, x[0].device
    )
    x_ref = x_emb
    for layer in tr.noise_refiner:
        x_ref = layer(x_ref, x_mask, x_freqs, adaln_input=adaln)

    cap_seqlens = [len(c) for c in cap_raw]
    cap_emb = tr.cap_embedder(torch.cat(cap_raw, dim=0))
    cap_emb, cap_freqs, cap_mask, _, _ = tr._prepare_sequence(
        list(cap_emb.split(cap_seqlens, 0)), cap_pos, cap_pad, tr.cap_pad_token, None, cap_raw[0].device
    )
    cap_ref = cap_emb
    for layer in tr.context_refiner:
        cap_ref = layer(cap_ref, cap_mask, cap_freqs)

    unified, unified_freqs, unified_mask, _ = tr._build_unified_sequence(
        x_ref, x_freqs, x_seqlens, None,
        cap_ref, cap_freqs, cap_seqlens, None,
        None, None, None, None, False, cap_raw[0].device,
    )
    for layer in tr.layers[:1]:
        unified = layer(unified, unified_mask, unified_freqs, adaln_input=adaln)
    out_patch = tr.all_final_layer[pk](unified, c=adaln)
    unpatched = tr.unpatchify(list(out_patch.unbind(0)), x_size, patch_size, f_patch_size)
    noise_pred = (-torch.stack(unpatched, 0).float()).squeeze(2)

    def spec(name: str, tensor: torch.Tensor, src: str) -> TensorSpec:
        return TensorSpec(name, list(tensor.shape), _dtype_name(tensor), src)

    forward_steps = (
        ForwardStep(
            "T1_adaln",
            "transformer_z_image.py:945 t_embedder(t*t_scale).type_as(x[0])",
            (spec("adaln_input", adaln, "945"),),
        ),
        ForwardStep(
            "T2_patchify_x",
            "transformer_z_image.py:606 _patchify_image (inside patchify_and_embed)",
            (spec("patch_feats", x[0], "606"),),
            notes="Per-batch-item, before x_embedder; shape [S_x, patch_dim].",
        ),
        ForwardStep(
            "T2_patchify_cap",
            "transformer_z_image.py:598 _pad_with_ids(cap) (inside patchify_and_embed)",
            (spec("cap_feats", cap_raw[0], "598"),),
            notes="Pre cap_embedder; TE output after gather+pad.",
        ),
        ForwardStep(
            "T3_x_embed_prepare",
            "transformer_z_image.py:980-983 all_x_embedder + _prepare_sequence",
            (
                spec("x_tokens", x_emb, "981"),
                spec("x_rope_cos", x_freqs.real, "785"),
            ),
        ),
        ForwardStep(
            "T4_noise_refiner",
            "transformer_z_image.py:985-992 noise_refiner ×n_refiner_layers",
            (spec("x_refined", x_ref, "992"),),
        ),
        ForwardStep(
            "T5_cap_embed_prepare",
            "transformer_z_image.py:996-999 cap_embedder + _prepare_sequence",
            (spec("cap_tokens", cap_emb, "999"),),
        ),
        ForwardStep(
            "T6_context_refiner",
            "transformer_z_image.py:1001-1006 context_refiner ×n_refiner_layers",
            (spec("cap_refined", cap_ref, "1006"),),
        ),
        ForwardStep(
            "T7_unified",
            "transformer_z_image.py:1018 _build_unified_sequence",
            (
                spec("unified_tokens", unified, "859"),
                spec("unified_rope_cos", unified_freqs.real, "860"),
            ),
        ),
        ForwardStep(
            "T8_main_layer",
            "transformer_z_image.py:1048 layers[i]",
            (spec("unified_out", unified, "1054"),),
            notes="Representative: layers[0]; repeat ×n_layers.",
        ),
        ForwardStep(
            "T9_final_layer",
            "transformer_z_image.py:1064 all_final_layer",
            (spec("patch_output", out_patch, "1064"),),
        ),
        ForwardStep(
            "T10_unpatchify",
            "transformer_z_image.py:1068 unpatchify",
            (spec("dit_output", unpatched[0], "1068"),),
            notes="Transformer return; [C,F,H,W] bf16.",
        ),
        ForwardStep(
            "T11_pipeline_noise_pred",
            "pipeline_z_image.py:556-558 -sample.float().squeeze(2)",
            (spec("noise_pred", noise_pred, "556"),),
        ),
    )

    profile.forward_steps = forward_steps
    profile.pipeline_steps = tuple(pipeline_steps)
    return profile


def profile_to_json(profile: ZImageSourceProfile) -> str:
    def step_dict(s: ForwardStep) -> dict:
        return {
            "step_id": s.step_id,
            "source": s.source,
            "notes": s.notes,
            "outputs": [
                {"name": o.name, "shape": o.shape, "dtype": o.dtype, "source": o.source}
                for o in s.outputs
            ],
        }

    payload = {
        "model_path": profile.model_path,
        "scene": {
            "image_size": profile.image_size,
            "cap_seq": profile.cap_seq,
            "batch_size": profile.batch_size,
        },
        "config": {
            "dim": profile.dim,
            "in_channels": profile.in_channels,
            "cap_feat_dim": profile.cap_feat_dim,
            "n_layers": profile.n_layers,
            "n_refiner_layers": profile.n_refiner_layers,
            "t_scale": profile.t_scale,
            "axes_dims": profile.axes_dims,
            "axes_lens": profile.axes_lens,
            "weight_dtype": profile.weight_dtype,
            "latent_hw": list(profile.latent_hw),
            "image_token_seq": profile.image_token_seq,
            "patch_dim": profile.patch_dim,
            "rope_dim": profile.rope_dim,
        },
        "pipeline_steps": [step_dict(s) for s in profile.pipeline_steps],
        "forward_steps": [step_dict(s) for s in profile.forward_steps],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
