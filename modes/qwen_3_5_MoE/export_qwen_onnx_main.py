#!/usr/bin/env python3
"""
Export a representative merged ONNX set for Qwen3.5-MoE layers.

Exported files:

- embedding_<seq>.onnx
- layer_00_linear_attn_block.onnx  or structured prefill bundle
- layer_00_moe_block_<seq>.onnx
- layer_03_full_attn_block_<seq>.onnx
- layer_03_moe_block_<seq>.onnx
- norm_<seq>.onnx
- lm_head_<seq>.onnx
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qwen_merged_block_export import (
    _load_model,
    export_linear_attn_decode_bundle,
    export_linear_attn_prefill_bundle,
    export_full_attn_block,
    export_moe_block,
)
from qwen_export_shared import (
    _layer_type,
    _seq_tag,
    _text_config,
    export_embedding,
    export_lm_head,
    export_norm,
)
from qwen_vision_export import (
    export_vision_block_repr,
    export_vision_cu_seqlens,
    export_vision_patch_embed,
    export_vision_patch_merger,
    export_vision_pos_embed_interp,
    export_vision_rot_pos_emb,
)
from qwen_mm_export import (
    export_image_mask_build,
    export_mm_inject,
    export_mrope_position_ids_decode,
    export_mrope_position_ids_prefill,
)
from export_common.onnx_graph_utils import print_simplification_report, reset_records, save_stats_json


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_path", help="Local path to model directory")
    ap.add_argument(
        "--output_dir",
        default="./conservative_merged_moe_onnx",
        help="Output directory",
    )
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--opset", type=int, default=20)
    ap.add_argument("--no_simplify", action="store_true")
    ap.add_argument(
        "--phase",
        choices=["prefill", "decode"],
        required=True,
        help="Canonical merged export phase.",
    )
    ap.add_argument(
        "--variant",
        choices=["text", "vl"],
        default="text",
        help=(
            "Top-level loaded class. 'text' loads Qwen3_5MoeForCausalLM and only "
            "exports the text-side ONNX graphs (current default). 'vl' loads "
            "Qwen3_5MoeForConditionalGeneration and additionally exports the "
            "vision tower (patch_embed / vision_block / patch_merger) and the "
            "multimodal injection graph (mm_inject)."
        ),
    )
    ap.add_argument(
        "--export_scope",
        choices=["representative", "full"],
        default="representative",
        help="Export only representative layers or every layer on the canonical path.",
    )
    ap.add_argument("--seq_len", type=int, default=8192)
    ap.add_argument(
        "--decode_context_len",
        type=int,
        default=8192,
        help="History KV-cache length used by --phase decode full attention export.",
    )
    ap.add_argument("--linear_layer", type=int, default=0)
    ap.add_argument("--full_layer", type=int, default=3)
    ap.add_argument(
        "--linear_prefill_chunk_size",
        type=int,
        default=64,
        help="Chunk size used by the prefill structured linear delta_net export (default: 64).",
    )
    ap.add_argument(
        "--vision_block_layer",
        type=int,
        default=0,
        help=(
            "Vision Transformer layer index used as the representative ViT block "
            "(only used when --variant vl)."
        ),
    )
    ap.add_argument(
        "--vision_token_seq_len",
        type=int,
        default=1024,
        help=(
            "Number of vision patch tokens used as the static seq_len for the "
            "vision-tower ONNX dummies (only used when --variant vl). The default "
            "1024 corresponds to a 32x32 grid (single image, post patch_embed)."
        ),
    )
    ap.add_argument(
        "--mm_image_token_count",
        type=int,
        default=256,
        help=(
            "Number of image tokens injected into the text sequence by mm_inject "
            "(only used when --variant vl). Equals vision_token_seq_len // (spatial_merge_size**2)."
        ),
    )
    ap.add_argument(
        "--mrope_text_pre_len",
        type=int,
        default=64,
        help=(
            "Number of *text* tokens preceding the single image segment in the "
            "representative prefill request used by mrope_position_ids_prefill_*.onnx "
            "(only used when --variant vl --phase prefill). The remaining text "
            "tokens fill the suffix so that "
            "text_pre + mm_image_token_count + text_post == seq_len."
        ),
    )
    ap.add_argument(
        "--vision_grid_mode",
        choices=["dynamic", "static"],
        default="dynamic",
        help=(
            "Trade-off mode for the three H/W-sensitive subgraphs "
            "(vision_rot_pos_emb / vision_pos_embed_interp / "
            "mrope_position_ids_prefill).\n"
            "  dynamic (default) — H/W are real tensor inputs read from "
            "``grid_thw``; every source op is visible in ONNX, but each "
            "subgraph carries unk__N for H/W-derived dims.\n"
            "  static — H/W are captured as Python ints from "
            "``--vision_token_seq_len`` (and ``--vision_grid_t``); the "
            "tracer folds the entire arange/cos/sin/repeat chain to "
            "constants for that grid; 0 unk__N but those source ops "
            "collapse to a pre-computed lookup table. Re-export per "
            "resolution if you need multi-bucket coverage."
        ),
    )
    return ap.parse_args()


def _default_output_dir_for_phase(
    phase: str,
    variant: str = "text",
    grid_mode: str = "dynamic",
) -> str:
    model_tag = "35B_A3B"
    variant_tag = "_VL" if variant == "vl" else ""
    grid_tag = "_static" if grid_mode == "static" else ""
    if phase == "prefill":
        return f"./Qwen3_5_{model_tag}{variant_tag}_Merged_ONNX_Prefill{grid_tag}"
    if phase == "decode":
        return f"./Qwen3_5_{model_tag}{variant_tag}_Merged_ONNX_Decode{grid_tag}"
    return "./conservative_merged_moe_onnx"


def _apply_phase_preset(args: argparse.Namespace) -> None:
    if args.phase == "prefill":
        if args.seq_len <= 0:
            raise ValueError("--seq_len must be > 0 for --phase prefill")
    elif args.phase == "decode":
        args.seq_len = 1

    if args.output_dir == "./conservative_merged_moe_onnx":
        args.output_dir = _default_output_dir_for_phase(
            args.phase, args.variant, args.vision_grid_mode,
        )


def _target_layer_sets(model, export_scope: str, linear_layer: int, full_layer: int) -> tuple[list[int], list[int]]:
    if export_scope == "representative":
        return [linear_layer], [full_layer]

    text_cfg = _text_config(model)
    linear_layers = [
        idx for idx in range(text_cfg.num_hidden_layers)
        if _layer_type(model, idx) == "linear_attention"
    ]
    full_layers = [
        idx for idx in range(text_cfg.num_hidden_layers)
        if _layer_type(model, idx) == "full_attention"
    ]
    return linear_layers, full_layers


def main() -> None:
    args = _parse_args()
    _apply_phase_preset(args)

    os.makedirs(args.output_dir, exist_ok=True)
    reset_records()

    model = _load_model(args.model_path, variant=args.variant)

    if _layer_type(model, args.linear_layer) != "linear_attention":
        raise ValueError(f"layer {args.linear_layer} is not linear_attention")
    if _layer_type(model, args.full_layer) != "full_attention":
        raise ValueError(f"layer {args.full_layer} is not full_attention")

    simplify = not args.no_simplify
    seq_tag = _seq_tag(args.seq_len)
    is_decode_phase = args.phase == "decode"
    is_prefill_phase = args.phase == "prefill"
    token_seq_len = args.seq_len if is_prefill_phase else 1
    linear_layers, full_layers = _target_layer_sets(
        model,
        args.export_scope,
        args.linear_layer,
        args.full_layer,
    )
    fold_pure_shape_chains = True
    is_vl = args.variant == "vl"
    static_grid = args.vision_grid_mode == "static"
    # Vision tower + multimodal-flow graphs are only meaningful during
    # prefill: image tokens are produced once at prefill time and merged
    # into inputs_embeds by mm_inject. The decode phase is single-token
    # autoregressive over the KV cache, with no image-token injection — so
    # the vision-side graphs and image_mask / mm_inject are intentionally
    # not re-exported in decode. The text decoder still needs the M-RoPE
    # 3D ``position_ids`` constructor on every decode step, however, so
    # ``mrope_position_ids_decode_ctx<N>.onnx`` *is* exported in vl-decode.
    export_vision_prefill_pipeline = is_vl and is_prefill_phase
    export_mrope_decode = is_vl and is_decode_phase
    # Total step count:
    #   text path                                        = 7 steps
    #   vl-prefill adds 9 multimodal/vision-side steps   = +9
    #   vl-decode adds 1 mrope decode step               = +1
    total_steps = (
        7
        + (9 if export_vision_prefill_pipeline else 0)
        + (1 if export_mrope_decode else 0)
    )
    step = [0]

    def _bump() -> str:
        step[0] += 1
        return f"[{step[0]}/{total_steps}]"

    print(f"variant: {args.variant}")
    print(f"phase preset: {args.phase}")
    print(f"export scope: {args.export_scope}")

    if export_vision_prefill_pipeline:
        # ── vision tower (prefill only) ───────────────────────────────
        # Vision tokens are produced once at prefill time, then their merged
        # image_embeds are scatter-injected into inputs_embeds via mm_inject.
        # The decode phase reuses the resulting KV cache and never invokes
        # any of these graphs, so we skip exporting them in decode.
        print(f"{_bump()} vision_patch_embed_{_seq_tag(args.vision_token_seq_len)}")
        export_vision_patch_embed(
            model,
            args.output_dir,
            args.opset,
            simplify,
            args.vision_token_seq_len,
            fold_pure_shape_chains=fold_pure_shape_chains,
        )

        print(
            f"{_bump()} vision_pos_embed_interp_{_seq_tag(args.vision_token_seq_len)}"
        )
        export_vision_pos_embed_interp(
            model,
            args.output_dir,
            args.opset,
            simplify,
            args.vision_token_seq_len,
            static_grid=static_grid,
            fold_pure_shape_chains=fold_pure_shape_chains,
        )

        print(f"{_bump()} vision_rot_pos_emb_{_seq_tag(args.vision_token_seq_len)}")
        export_vision_rot_pos_emb(
            model,
            args.output_dir,
            args.opset,
            simplify,
            args.vision_token_seq_len,
            static_grid=static_grid,
            fold_pure_shape_chains=fold_pure_shape_chains,
        )

        print(f"{_bump()} vision_cu_seqlens_{_seq_tag(args.vision_token_seq_len)}")
        export_vision_cu_seqlens(
            model,
            args.output_dir,
            args.opset,
            simplify,
            args.vision_token_seq_len,
            fold_pure_shape_chains=fold_pure_shape_chains,
        )

        print(
            f"{_bump()} vision_block_{args.vision_block_layer:02d}_repr_"
            f"{_seq_tag(args.vision_token_seq_len)}"
        )
        export_vision_block_repr(
            model,
            args.vision_block_layer,
            args.output_dir,
            args.opset,
            simplify,
            args.vision_token_seq_len,
            fold_pure_shape_chains=fold_pure_shape_chains,
        )

        print(
            f"{_bump()} vision_patch_merger_{_seq_tag(args.vision_token_seq_len)}"
        )
        export_vision_patch_merger(
            model,
            args.output_dir,
            args.opset,
            simplify,
            args.vision_token_seq_len,
            fold_pure_shape_chains=fold_pure_shape_chains,
        )

        # ── multimodal flow (image_mask -> mm_inject -> mrope) ───────
        print(f"{_bump()} image_mask_build_{_seq_tag(token_seq_len)}")
        export_image_mask_build(
            model,
            args.output_dir,
            args.batch_size,
            args.opset,
            simplify,
            token_seq_len,
            fold_pure_shape_chains=fold_pure_shape_chains,
        )

        print(f"{_bump()} mm_inject_{_seq_tag(token_seq_len)}")
        export_mm_inject(
            model,
            args.output_dir,
            args.batch_size,
            args.opset,
            simplify,
            token_seq_len,
            image_token_count=args.mm_image_token_count,
            fold_pure_shape_chains=fold_pure_shape_chains,
        )

        # Derive the representative image grid (T=1, square H==W) from
        # the user-facing ``--vision_token_seq_len``. The same convention
        # is already used by every vision-tower export above.
        from qwen_vision_export import _representative_grid_thw  # local import to avoid cycle
        grid_t, grid_h, grid_w = _representative_grid_thw(args.vision_token_seq_len)

        print(f"{_bump()} mrope_position_ids_prefill_{_seq_tag(token_seq_len)}")
        export_mrope_position_ids_prefill(
            model,
            args.output_dir,
            args.batch_size,
            args.opset,
            simplify,
            token_seq_len,
            text_pre_len=args.mrope_text_pre_len,
            image_token_count=args.mm_image_token_count,
            image_grid_t=grid_t,
            image_grid_h=grid_h,
            image_grid_w=grid_w,
            static_grid=static_grid,
            fold_pure_shape_chains=fold_pure_shape_chains,
        )

    if export_mrope_decode:
        # vl-decode still needs the 3D ``position_ids`` builder on every
        # step (compute_3d_position_ids elif branch). No vision tower or
        # mm_inject in decode, just the M-RoPE index recompute.
        print(
            f"{_bump()} mrope_position_ids_decode_ctx{_seq_tag(args.decode_context_len)}"
        )
        export_mrope_position_ids_decode(
            model,
            args.output_dir,
            args.batch_size,
            args.opset,
            simplify,
            decode_context_len=args.decode_context_len,
            fold_pure_shape_chains=fold_pure_shape_chains,
        )

    print(f"{_bump()} embedding_{_seq_tag(token_seq_len)}")
    export_embedding(
        model,
        args.output_dir,
        args.batch_size,
        args.opset,
        simplify,
        token_seq_len,
        fold_pure_shape_chains=fold_pure_shape_chains,
    )

    print(f"{_bump()} linear attention blocks ({len(linear_layers)})")
    for layer_idx in linear_layers:
        if is_decode_phase:
            print(f"  layer_{layer_idx:02d}_linear_attn_block")
            export_linear_attn_decode_bundle(
                model,
                layer_idx,
                args.output_dir,
                args.batch_size,
                args.opset,
                simplify,
                fold_pure_shape_chains=fold_pure_shape_chains,
            )
        else:
            print(f"  layer_{layer_idx:02d}_linear_attn_block_{seq_tag}")
            export_linear_attn_prefill_bundle(
                model,
                layer_idx,
                args.output_dir,
                args.batch_size,
                args.opset,
                simplify,
                args.seq_len,
                chunk_size=args.linear_prefill_chunk_size,
                fold_pure_shape_chains=fold_pure_shape_chains,
            )

    moe_layers = sorted(set(linear_layers + full_layers))
    print(f"{_bump()} moe blocks ({len(moe_layers)})")
    for layer_idx in moe_layers:
        print(f"  layer_{layer_idx:02d}_moe_block_{_seq_tag(token_seq_len)}")
        export_moe_block(
            model,
            layer_idx,
            args.output_dir,
            args.batch_size,
            args.opset,
            simplify,
            token_seq_len,
            fold_pure_shape_chains=fold_pure_shape_chains,
        )

    # In the ``vl`` (multimodal) variant the upstream
    # ``mrope_position_ids_*.onnx`` graphs emit 3D ``position_ids[3, B, S]``
    # (true M-RoPE T/H/W axes). Trace the matching 3D branch of
    # ``RotaryEmbeddingBlockMoE.forward`` so the IO contract of
    # ``layer_*_full_attn_block_*.onnx`` accepts that shape directly.
    # Pure-text variant keeps the 2D contract (T=H=W static repeat).
    full_attn_position_ids_ndim = 3 if args.variant == "vl" else 2
    print(f"{_bump()} full attention blocks ({len(full_layers)})")
    for layer_idx in full_layers:
        if is_decode_phase:
            print(f"  layer_{layer_idx:02d}_full_attn_block_decode_ctx{_seq_tag(args.decode_context_len)}")
            export_full_attn_block(
                model,
                layer_idx,
                args.output_dir,
                args.batch_size,
                args.opset,
                simplify,
                1,
                past_seq_len=args.decode_context_len,
                fold_pure_shape_chains=fold_pure_shape_chains,
                position_ids_ndim=full_attn_position_ids_ndim,
            )
        else:
            print(f"  layer_{layer_idx:02d}_full_attn_block_{seq_tag}")
            export_full_attn_block(
                model,
                layer_idx,
                args.output_dir,
                args.batch_size,
                args.opset,
                simplify,
                args.seq_len,
                fold_pure_shape_chains=fold_pure_shape_chains,
                position_ids_ndim=full_attn_position_ids_ndim,
            )

    print(f"{_bump()} exported moe blocks reused above")

    print(f"{_bump()} norm_{_seq_tag(token_seq_len)}")
    export_norm(
        model,
        args.output_dir,
        args.batch_size,
        args.opset,
        simplify,
        token_seq_len,
        fold_pure_shape_chains=fold_pure_shape_chains,
    )

    print(f"{_bump()} lm_head_{_seq_tag(token_seq_len)}")
    export_lm_head(
        model,
        args.output_dir,
        args.batch_size,
        args.opset,
        simplify,
        token_seq_len,
        fold_pure_shape_chains=fold_pure_shape_chains,
    )

    print_simplification_report()
    json_path = save_stats_json(args.output_dir)
    print(f"\nStats saved -> {json_path}")


if __name__ == "__main__":
    main()

