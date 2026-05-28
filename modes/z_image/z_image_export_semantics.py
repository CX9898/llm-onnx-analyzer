"""Z-Image ONNX export semantics: contracts derived from source profile + weight policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from onnx import TensorProto

from z_image_source_profile import ZImageSourceProfile, build_source_profile

try:
    from z_image_export_shared import cap_scene_tag, img_scene_tag, seq_scene_tag
except ImportError:
    def img_scene_tag(image_size: int) -> str:
        return f"img{image_size}"

    def cap_scene_tag(cap_seq: int) -> str:
        return f"cap{cap_seq}"

    def seq_scene_tag(seq_len: int) -> str:
        n = f"{seq_len // 1024}k" if seq_len % 1024 == 0 else str(seq_len)
        return f"seq{n}"

# ---------------------------------------------------------------------------
# Source paths (constraint 1)
# ---------------------------------------------------------------------------

DIFFUSERS_TRANSFORMER = Path("diffusers/models/transformers/transformer_z_image.py")
DIFFUSERS_PIPELINE = Path("diffusers/pipelines/z_image/pipeline_z_image.py")
TRANSFORMERS_TEXT_ENCODER = Path("transformers/models/qwen3/modeling_qwen3.py")

SOURCE_ALIGNMENT = {
    "dit": str(DIFFUSERS_TRANSFORMER),
    "pipeline": str(DIFFUSERS_PIPELINE),
    "text_encoder": str(TRANSFORMERS_TEXT_ENCODER),
}

# ONNX 子图 ↔ 源码 forward 步骤的显式映射（分析用分解，不是改源码顺序）
EXPORT_TO_SOURCE = {
    "text_encode_*.onnx": "pipeline_z_image.py:_encode_prompt → hidden_states[-2]",
    "patchify_and_embed_img*.onnx": "transformer_z_image.py:588 patchify_and_embed",
    "x_branch_seq*.onnx": "transformer_z_image.py:980-992 embed_prepare + noise_refiner[0]×n (in-graph)",
    "cap_branch_cap*.onnx": "transformer_z_image.py:996-1006 embed_prepare + context_refiner[0]×n (in-graph)",
    "final_output_img*.onnx": "transformer_z_image.py:1064-1068 + pipeline_z_image.py:556-558",
    "timestep_embed.onnx": "transformer_z_image.py:945 t_embedder(t*t_scale).type_as(x)",
    "noise_refiner_block_repr_seq*.onnx": "transformer_z_image.py:985-992 noise_refiner[0] (legacy; merged into x_branch)",
    "context_refiner_block_repr_cap*.onnx": "transformer_z_image.py:1001-1006 context_refiner[0] (legacy; merged into cap_branch)",
    "sequence_concat_basic_seq*.onnx": "transformer_z_image.py:859 _build_unified_sequence",
    "main_layer_repr_seq*.onnx": "transformer_z_image.py:1048 layers[0]",
    "vae_decode_img*.onnx": "pipeline_z_image.py:583-586 latents(f32)→vae.decode",
}

# ---------------------------------------------------------------------------
# Weight policy (constraint 5)
# ---------------------------------------------------------------------------

WEIGHT_INLINE_MAX_BYTES = 100 * 1024 * 1024  # ~100 MB

GraphKind = Literal[
    "timestep_embed",
    "patchify_and_embed",
    "x_branch",
    "cap_branch",
    "final_output",
    "sequence_concat",
    "main_layer_repr",
    "text_encode",
    "vae_decode",
]

# Weights stay as initializers bound to operators (Netron-friendly).
# ``strip_initializers`` is opt-in legacy only — do not enable by default.
_STRIP_BY_DEFAULT: frozenset[GraphKind] = frozenset()


def default_strip_initializers(kind: GraphKind, *, cli_override: bool | None = None) -> bool:
    if cli_override is not None:
        return cli_override
    return kind in _STRIP_BY_DEFAULT


# ---------------------------------------------------------------------------
# Export scene (static representative shapes)
# ---------------------------------------------------------------------------

PATCH_KEY = "2-1"
SEQ_MULTI_OF = 32
X_PAD_DIM = 64
ADALN_EMBED_DIM = 256


@dataclass(frozen=True)
class ExportScene:
    batch_size: int = 1
    image_size: int = 512
    cap_seq: int = 128
    patch_size: int = 2
    f_patch_size: int = 1
    vae_scale_factor: int = 8

    def latent_hw(self) -> tuple[int, int]:
        vae_scale = self.vae_scale_factor * 2
        side = 2 * (self.image_size // vae_scale)
        return side, side

    def image_token_seq_len(self) -> int:
        h_lat, w_lat = self.latent_hw()
        return (h_lat // self.patch_size) * (w_lat // self.patch_size)

    def unified_seq_len(self) -> int:
        return self.image_token_seq_len() + self.cap_seq

    def patch_feature_dim(self, in_channels: int = 16) -> int:
        return self.f_patch_size * self.patch_size * self.patch_size * in_channels

    def output_dir_name(self) -> str:
        return f"Z_Image_ONNX_{self.image_size}"


@dataclass(frozen=True)
class TensorContract:
    name: str
    shape: list[int | str]
    dtype: str


@dataclass(frozen=True)
class GraphContract:
    kind: GraphKind
    file_name: str
    source_symbol: str
    inputs: tuple[TensorContract, ...]
    outputs: tuple[TensorContract, ...]
    repeat: int = 1
    notes: str = ""


def load_source_profile(model_path: str, scene: ExportScene) -> ZImageSourceProfile:
    """Run one real forward trace; shape/dtype contracts come from here."""
    return build_source_profile(
        model_path,
        image_size=scene.image_size,
        cap_seq=scene.cap_seq,
        batch_size=scene.batch_size,
        patch_size=scene.patch_size,
        f_patch_size=scene.f_patch_size,
    )


def build_denoise_graph_contracts(
    scene: ExportScene,
    profile: ZImageSourceProfile,
) -> list[GraphContract]:
    """Build contracts from ``ZImageSourceProfile`` (real weights + forward trace)."""
    b = scene.batch_size
    s_x = profile.image_token_seq
    s_cap = scene.cap_seq
    s_u = s_x + s_cap
    patch_dim = profile.patch_dim
    h_lat, w_lat = profile.latent_hw
    dim = profile.dim
    cap_feat_dim = profile.cap_feat_dim
    out_channels = profile.in_channels
    rope_dim = profile.rope_dim
    wdtype = profile.weight_dtype

    tag_img = img_scene_tag(scene.image_size)
    tag_x = seq_scene_tag(s_x)
    tag_cap = cap_scene_tag(s_cap)
    tag_u = seq_scene_tag(s_u)

    return [
        GraphContract(
            "patchify_and_embed",
            f"patchify_and_embed_{tag_img}.onnx",
            "transformer_z_image.py:588 patchify_and_embed",
            (
                TensorContract("latent", [b, out_channels, 1, h_lat, w_lat], wdtype),
                TensorContract("cap_feats", [b, s_cap, cap_feat_dim], wdtype),
            ),
            (
                TensorContract("x_patch_feats", [b, s_x, patch_dim], wdtype),
                TensorContract("cap_feats_padded", [b, s_cap, cap_feat_dim], wdtype),
                TensorContract("x_pos_ids", [b, s_x, 3], "int32"),
                TensorContract("cap_pos_ids", [b, s_cap, 3], "int32"),
                TensorContract("x_pad_mask", [b, s_x], "bool"),
                TensorContract("cap_pad_mask", [b, s_cap], "bool"),
            ),
            notes="T2：含 _patchify_image + cap _pad_with_ids。",
        ),
        GraphContract(
            "timestep_embed",
            "timestep_embed.onnx",
            "transformer_z_image.py:945",
            (TensorContract("timestep", [b], "float32"),),
            (TensorContract("adaln_input", [b, ADALN_EMBED_DIM], wdtype),),
        ),
        GraphContract(
            "x_branch",
            f"x_branch_{tag_x}.onnx",
            "transformer_z_image.py:980-992",
            (
                TensorContract("x_patch_feats", [b, s_x, patch_dim], wdtype),
                TensorContract("x_pos_ids", [b, s_x, 3], "int32"),
                TensorContract("x_pad_mask", [b, s_x], "bool"),
                TensorContract("adaln_input", [b, ADALN_EMBED_DIM], wdtype),
            ),
            (
                TensorContract("x_tokens", [b, s_x, dim], wdtype),
                TensorContract("x_rope_cos", [b, s_x, rope_dim], "float32"),
                TensorContract("x_rope_sin", [b, s_x, rope_dim], "float32"),
                TensorContract("x_attn_mask", [b, s_x], "bool"),
            ),
            notes=f"T3+T4 merged: x_embed_prepare + noise_refiner[0]×{profile.n_refiner_layers} in-graph.",
        ),
        GraphContract(
            "cap_branch",
            f"cap_branch_{tag_cap}.onnx",
            "transformer_z_image.py:996-1006",
            (
                TensorContract("cap_feats_padded", [b, s_cap, cap_feat_dim], wdtype),
                TensorContract("cap_pos_ids", [b, s_cap, 3], "int32"),
                TensorContract("cap_pad_mask", [b, s_cap], "bool"),
            ),
            (
                TensorContract("cap_tokens", [b, s_cap, dim], wdtype),
                TensorContract("cap_rope_cos", [b, s_cap, rope_dim], "float32"),
                TensorContract("cap_rope_sin", [b, s_cap, rope_dim], "float32"),
                TensorContract("cap_attn_mask", [b, s_cap], "bool"),
            ),
            notes=f"T5+T6 merged: cap_embed_prepare + context_refiner[0]×{profile.n_refiner_layers} in-graph.",
        ),
        GraphContract(
            "sequence_concat",
            f"sequence_concat_basic_{tag_u}.onnx",
            "transformer_z_image.py:859",
            (
                TensorContract("x_tokens", [b, s_x, dim], wdtype),
                TensorContract("cap_tokens", [b, s_cap, dim], wdtype),
                TensorContract("x_rope_cos", [b, s_x, rope_dim], "float32"),
                TensorContract("x_rope_sin", [b, s_x, rope_dim], "float32"),
                TensorContract("cap_rope_cos", [b, s_cap, rope_dim], "float32"),
                TensorContract("cap_rope_sin", [b, s_cap, rope_dim], "float32"),
            ),
            (
                TensorContract("unified_tokens", [b, s_u, dim], wdtype),
                TensorContract("unified_rope_cos", [b, s_u, rope_dim], "float32"),
                TensorContract("unified_rope_sin", [b, s_u, rope_dim], "float32"),
            ),
        ),
        GraphContract(
            "main_layer_repr",
            f"main_layer_repr_{tag_u}.onnx",
            "transformer_z_image.py:1048 layers[0]",
            (
                TensorContract("hidden_states", [b, s_u, dim], wdtype),
                TensorContract("attn_mask", [b, s_u], "bool"),
                TensorContract("rope_cos", [b, s_u, rope_dim], "float32"),
                TensorContract("rope_sin", [b, s_u, rope_dim], "float32"),
                TensorContract("adaln_input", [b, ADALN_EMBED_DIM], wdtype),
            ),
            (TensorContract("output", [b, s_u, dim], wdtype),),
            repeat=profile.n_layers,
        ),
        GraphContract(
            "final_output",
            f"final_output_{tag_img}.onnx",
            "transformer_z_image.py:1064-1068 + pipeline_z_image.py:556-558",
            (
                TensorContract("hidden_states", [b, s_u, dim], wdtype),
                TensorContract("adaln_input", [b, ADALN_EMBED_DIM], wdtype),
            ),
            (TensorContract("noise_pred", [b, out_channels, h_lat, w_lat], "float32"),),
            notes="T9 merged: final_layer + slice image tokens + unpatchify.",
        ),
    ]


def denoise_repeat_for_stem(stem: str, profile: ZImageSourceProfile) -> int:
    if stem.startswith("main_layer_repr"):
        return profile.n_layers
    return 1


def text_encode_repeat_for_stem(stem: str, profile: ZImageSourceProfile) -> int:
    if stem.startswith("text_decoder_layer_repr"):
        return profile.n_text_forward_layers
    return 1


def build_text_encode_graph_contracts(scene: ExportScene, profile: ZImageSourceProfile) -> list[GraphContract]:
    b = scene.batch_size
    s = scene.cap_seq
    h = profile.cap_feat_dim
    rope = profile.text_rope_dim
    wdtype = profile.text_encoder_dtype
    tag = cap_scene_tag(s)
    n_fwd = profile.n_text_forward_layers

    return [
        GraphContract(
            "text_embed_prepare",
            f"text_embed_prepare_{tag}.onnx",
            "pipeline_z_image.py:_encode_prompt (embed + RoPE + mask)",
            (
                TensorContract("input_ids", [b, s], "int64"),
                TensorContract("attention_mask", [b, s], "int64"),
            ),
            (
                TensorContract("hidden_states", [b, s, h], wdtype),
                TensorContract("rope_cos", [b, s, rope], wdtype),
                TensorContract("rope_sin", [b, s, rope], wdtype),
                TensorContract("attn_mask", [b, 1, s, s], wdtype),
            ),
        ),
        GraphContract(
            "text_decoder_layer_repr",
            f"text_decoder_layer_repr_{tag}.onnx",
            "Qwen3Model.layers[i] (repr: layers[0])",
            (
                TensorContract("hidden_states", [b, s, h], wdtype),
                TensorContract("rope_cos", [b, s, rope], wdtype),
                TensorContract("rope_sin", [b, s, rope], wdtype),
                TensorContract("attn_mask", [b, 1, s, s], wdtype),
            ),
            (TensorContract("hidden_states_out", [b, s, h], wdtype),),
            repeat=n_fwd,
            notes="Host 串联 ×repeat；末段输出即 prompt_hidden（对应 hidden_states[-2]）。",
        ),
    ]


def build_text_encode_contract(scene: ExportScene, profile: ZImageSourceProfile) -> GraphContract:
    """Legacy single-graph name; boundary 校验用 decoder 末段输出。"""
    contracts = build_text_encode_graph_contracts(scene, profile)
    return contracts[-1]


def build_vae_decode_contract(scene: ExportScene, profile: ZImageSourceProfile) -> GraphContract:
    b = scene.batch_size
    h_lat, w_lat = profile.latent_hw
    rgb = scene.image_size
    return GraphContract(
        "vae_decode",
        f"vae_decode_{img_scene_tag(scene.image_size)}.onnx",
        "pipeline_z_image.py:583-586",
        (TensorContract("latents", [b, profile.in_channels, h_lat, w_lat], "float32"),),
        (TensorContract("image", [b, 3, rgb, rgb], profile.vae_dtype),),
    )


def dtype_name_to_tensorproto(name: str) -> int:
    mapping = {
        "bf16": TensorProto.BFLOAT16,
        "float32": TensorProto.FLOAT,
        "float16": TensorProto.FLOAT16,
        "int64": TensorProto.INT64,
        "int32": TensorProto.INT32,
        "bool": TensorProto.BOOL,
    }
    return mapping[name]
