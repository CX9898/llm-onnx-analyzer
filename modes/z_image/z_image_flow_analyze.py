"""Run flow-stats batch analysis + e2e rollup for Z-Image export output."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"
_ZIMAGE_DIR = Path(__file__).resolve().parent


def run_flow_stats_batch(out_root: Path) -> list[Path]:
    written: list[Path] = []
    for phase in ("text_encode", "denoise", "vae_decode"):
        phase_dir = out_root / phase
        if not phase_dir.is_dir() or not any(phase_dir.glob("*.onnx")):
            continue
        out_xlsx = phase_dir / "onnx_flow_stats_multi.xlsx"
        out_json = phase_dir / "onnx_flow_stats_multi.json"
        cmd = [
            sys.executable,
            str(_SCRIPTS / "analyze_onnx_flow_stats_batch.py"),
            str(phase_dir),
            "--out_xlsx",
            str(out_xlsx),
            "--out_json",
            str(out_json),
        ]
        subprocess.run(cmd, check=True)
        written.extend([out_xlsx, out_json])
    return written


def run_e2e_rollup(
    out_root: Path,
    model_path: str,
    *,
    num_steps: int = 28,
    cfg_scale: float = 5.0,
    image_size: int = 512,
    cap_seq: int = 128,
) -> Path:
    cmd = [
        sys.executable,
        str(_ZIMAGE_DIR / "z_image_e2e_rollup.py"),
        str(out_root),
        "--model_path",
        model_path,
        "--num_steps",
        str(num_steps),
        "--cfg_scale",
        str(cfg_scale),
        "--image_size",
        str(image_size),
        "--cap_seq",
        str(cap_seq),
    ]
    subprocess.run(cmd, check=True)
    return out_root / "z_image_e2e_rollup.json"


def analyze_export_output(
    out_root: Path,
    model_path: str,
    *,
    num_steps: int = 28,
    cfg_scale: float = 5.0,
    image_size: int = 512,
    cap_seq: int = 128,
) -> None:
    print("\n=== flow stats (per-phase batch) ===")
    artifacts = run_flow_stats_batch(out_root)
    for path in artifacts:
        print(f"  -> {path}")

    print("\n=== e2e rollup ===")
    rollup_path = run_e2e_rollup(
        out_root,
        model_path,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
        image_size=image_size,
        cap_seq=cap_seq,
    )
    print(f"  -> {rollup_path}")
    print(f"  -> {rollup_path.with_suffix('.md')}")
