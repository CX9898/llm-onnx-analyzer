#!/usr/bin/env python3
"""Export representative ONNX subgraphs for Z-Image (diffusers ZImagePipeline)."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ZIMAGE_DIR = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_ZIMAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_ZIMAGE_DIR))

from export_common.onnx_graph_utils import reset_records, save_stats_json  # noqa: E402

from audit_z_image_source_alignment import write_audit  # noqa: E402
from z_image_boundary_validate import validate_onnx_directory  # noqa: E402
from z_image_dit_export import export_denoise_bundle  # noqa: E402
from z_image_export_semantics import ExportScene, load_source_profile  # noqa: E402
from z_image_source_profile import profile_to_json  # noqa: E402
from z_image_export_shared import (  # noqa: E402
    DEFAULT_CAP_SEQ,
    DEFAULT_IMAGE_SIZE,
    _DIFFUSERS_SRC,
)
from z_image_text_export import export_text_encode  # noqa: E402
from z_image_vae_export import export_vae_decode  # noqa: E402
from z_image_flow_analyze import analyze_export_output  # noqa: E402


def _clean_output_root(out_dir: Path, phase: str) -> None:
    """Remove prior export artifacts so re-export starts from a clean tree."""
    if phase == "all" and out_dir.is_dir():
        shutil.rmtree(out_dir)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_path", help="Local Z-Image diffusers checkpoint directory")
    ap.add_argument(
        "--phase",
        choices=["denoise", "text_encode", "vae_decode", "all"],
        default="all",
        help="Export phase (default: all)",
    )
    ap.add_argument("--output_dir", default=None, help="Output directory")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    ap.add_argument("--cap_seq", type=int, default=DEFAULT_CAP_SEQ)
    ap.add_argument("--opset", type=int, default=20)
    ap.add_argument("--no_simplify", action="store_true")
    ap.add_argument(
        "--strip_initializers",
        action="store_true",
        help="Legacy: move initializers to graph inputs (harms Netron visualization; not recommended)",
    )
    ap.add_argument(
        "--no_strip_initializers",
        action="store_true",
        help="Explicitly keep initializers in-graph (default; large weights use .onnx.data sidecar)",
    )
    ap.add_argument("--num_steps", type=int, default=28, help="E2E rollup: scheduler steps")
    ap.add_argument("--cfg_scale", type=float, default=5.0, help="E2E rollup: CFG scale")
    ap.add_argument(
        "--skip_flow_analysis",
        action="store_true",
        help="Skip post-export MACs batch + e2e rollup",
    )
    return ap.parse_args()


def _strip_override(args: argparse.Namespace) -> bool | None:
    if args.strip_initializers:
        return True
    if args.no_strip_initializers:
        return False
    return None


def main() -> None:
    args = _parse_args()
    scene = ExportScene(
        batch_size=args.batch_size,
        image_size=args.image_size,
        cap_seq=args.cap_seq,
    )
    out_dir = args.output_dir or os.path.join("output", scene.output_dir_name())
    out_path = Path(out_dir)
    _clean_output_root(out_path, args.phase)
    os.makedirs(out_dir, exist_ok=True)
    simplify = not args.no_simplify
    strip = _strip_override(args)

    print(f"[z-image] model={args.model_path}")
    print(f"[z-image] phase={args.phase} -> {out_dir}")

    profile = load_source_profile(args.model_path, scene)
    profile_path = Path(out_dir) / "source_profile.json"
    profile_path.write_text(profile_to_json(profile), encoding="utf-8")
    print(f"[z-image] source profile -> {profile_path} (from real weights + forward trace)")

    if args.phase in ("denoise", "all"):
        reset_records()
        print("\n=== denoise subgraphs ===")
        denoise_dir = out_dir if args.phase == "denoise" else os.path.join(out_dir, "denoise")
        export_denoise_bundle(
            args.model_path,
            denoise_dir,
            scene=scene,
            opset=args.opset,
            simplify=simplify,
            strip_initializers=strip,
        )
        save_stats_json(denoise_dir)
        write_audit(Path(denoise_dir), diffusers_src=_DIFFUSERS_SRC, mode_dir=_ZIMAGE_DIR)

    if args.phase in ("text_encode", "all"):
        reset_records()
        te_dir = out_dir if args.phase == "text_encode" else os.path.join(out_dir, "text_encode")
        print("\n=== text_encode ===")
        export_text_encode(
            args.model_path,
            te_dir,
            scene,
            opset=args.opset,
            simplify=simplify,
            strip_initializers=strip,
        )
        save_stats_json(te_dir)

    if args.phase in ("vae_decode", "all"):
        reset_records()
        vd_dir = out_dir if args.phase == "vae_decode" else os.path.join(out_dir, "vae_decode")
        print("\n=== vae_decode ===")
        export_vae_decode(
            args.model_path,
            vd_dir,
            scene,
            opset=args.opset,
            simplify=simplify,
            strip_initializers=strip,
        )
        save_stats_json(vd_dir)

    if args.phase == "all":
        boundary_errors = validate_onnx_directory(Path(out_dir))
        if boundary_errors:
            print("\n[z-image] BOUNDARY VALIDATION FAILED:")
            for err in boundary_errors:
                print(f"  - {err}")
            raise SystemExit(1)
        print("\n[z-image] boundary validation: all adjacent I/O dtype/shape OK")

        if not args.skip_flow_analysis:
            analyze_export_output(
                Path(out_dir),
                args.model_path,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                image_size=args.image_size,
                cap_seq=args.cap_seq,
            )

    print("\n[z-image] done.")


if __name__ == "__main__":
    main()
