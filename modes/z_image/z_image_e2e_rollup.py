"""End-to-end MACs / memory rollup for Z-Image ONNX export (text + denoise×steps×CFG + vae)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from z_image_export_semantics import denoise_repeat_for_stem, text_encode_repeat_for_stem, load_source_profile, ExportScene


def _load_summary(onnx_path: Path) -> dict | None:
    summary_path = onnx_path.with_name(onnx_path.stem + ".flow_stats.summary.json")
    if not summary_path.is_file():
        return None
    with open(summary_path, encoding="utf-8") as f:
        return json.load(f)


def _find_onnx(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def _denoise_graph_summaries(denoise_dir: Path, profile) -> list[dict]:
    allow: set[str] | None = None
    meta_path = denoise_dir / "denoise_export_meta.json"
    if meta_path.is_file():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        allow = set(meta.get("exports", []))

    rows: list[dict] = []
    for onnx_path in sorted(denoise_dir.glob("*.onnx")):
        if allow is not None and onnx_path.name not in allow:
            continue
        summary = _load_summary(onnx_path)
        if summary is None:
            continue
        stem = onnx_path.stem
        repeat = denoise_repeat_for_stem(stem, profile)
        rows.append(
            {
                "file": onnx_path.name,
                "stem": stem,
                "repeat_per_step": repeat,
                "forward_macs": int(summary.get("total_forward_macs", 0)),
                "memory_bytes": int(summary.get("total_output_memory_bytes", 0)),
                "params": int(summary.get("total_params", summary.get("total_params_initializer", 0))),
            }
        )
    return rows


def _text_encode_graph_summaries(text_dir: Path, profile) -> list[dict]:
    allow: set[str] | None = None
    meta_path = text_dir / "text_encode_export_meta.json"
    if meta_path.is_file():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        allow = set(meta.get("exports", []))

    rows: list[dict] = []
    for onnx_path in sorted(text_dir.glob("*.onnx")):
        if allow is not None and onnx_path.name not in allow:
            continue
        summary = _load_summary(onnx_path)
        if summary is None:
            continue
        stem = onnx_path.stem
        repeat = text_encode_repeat_for_stem(stem, profile)
        rows.append(
            {
                "file": onnx_path.name,
                "stem": stem,
                "repeat_per_prompt": repeat,
                "forward_macs": int(summary.get("total_forward_macs", 0)),
                "memory_bytes": int(summary.get("total_output_memory_bytes", 0)),
                "params": int(summary.get("total_params", summary.get("total_params_initializer", 0))),
            }
        )
    return rows


def compute_e2e_rollup(
    out_root: Path,
    *,
    scene: ExportScene,
    profile,
    num_steps: int,
    cfg_scale: float,
) -> dict:
    cfg_factor = 2 if cfg_scale > 1.0 else 1
    text_dir = out_root / "text_encode"
    denoise_dir = out_root / "denoise"
    vae_dir = out_root / "vae_decode"

    text_rows = _text_encode_graph_summaries(text_dir, profile)
    text_macs = sum(r["forward_macs"] * r["repeat_per_prompt"] for r in text_rows)
    text_mem = sum(r["memory_bytes"] * r["repeat_per_prompt"] for r in text_rows)
    text_params = sum(r["params"] * r["repeat_per_prompt"] for r in text_rows)

    vae_onnx = _find_onnx(vae_dir, "vae_decode_*.onnx")
    vae_summary = _load_summary(vae_onnx) if vae_onnx else None

    denoise_rows = _denoise_graph_summaries(denoise_dir, profile)
    denoise_macs_per_step = sum(r["forward_macs"] * r["repeat_per_step"] for r in denoise_rows)
    denoise_mem_per_step = sum(r["memory_bytes"] * r["repeat_per_step"] for r in denoise_rows)
    denoise_params = sum(r["params"] * r["repeat_per_step"] for r in denoise_rows)

    vae_macs = int(vae_summary["total_forward_macs"]) if vae_summary else 0
    vae_mem = int(vae_summary["total_output_memory_bytes"]) if vae_summary else 0
    vae_params = int(vae_summary.get("total_params", vae_summary.get("total_params_initializer", 0))) if vae_summary else 0

    denoise_total_macs = denoise_macs_per_step * num_steps * cfg_factor
    denoise_total_mem = denoise_mem_per_step * num_steps * cfg_factor

    return {
        "scene": {
            "image_size": scene.image_size,
            "cap_seq": scene.cap_seq,
            "batch_size": scene.batch_size,
        },
        "assumptions": {
            "num_inference_steps": num_steps,
            "cfg_scale": cfg_scale,
            "cfg_dit_multiplier": cfg_factor,
            "notes": (
                "text_encode once per prompt (layer repr ×N from text_encode_layer_manifest); "
                "denoise chain × steps × CFG; vae_decode once."
            ),
        },
        "per_graph_text_encode": text_rows,
        "per_graph_denoise": denoise_rows,
        "denoise_per_step": {
            "forward_macs": denoise_macs_per_step,
            "memory_bytes": denoise_mem_per_step,
            "params": denoise_params,
        },
        "phases": {
            "text_encode": {
                "forward_macs": text_macs,
                "memory_bytes": text_mem,
                "params": text_params,
                "runs": 1,
            },
            "denoise": {
                "forward_macs": denoise_total_macs,
                "memory_bytes": denoise_total_mem,
                "params": denoise_params,
                "runs": num_steps * cfg_factor,
                "per_step_macs": denoise_macs_per_step,
            },
            "vae_decode": {
                "forward_macs": vae_macs,
                "memory_bytes": vae_mem,
                "params": vae_params,
                "runs": 1,
            },
        },
        "e2e_total": {
            "forward_macs": text_macs + denoise_total_macs + vae_macs,
            "memory_bytes_peak_estimate": max(
                text_mem,
                denoise_mem_per_step * cfg_factor,
                vae_mem,
            ),
            "params_unique": text_params + denoise_params + vae_params,
        },
    }


def write_rollup_report(out_root: Path, rollup: dict) -> tuple[Path, Path]:
    json_path = out_root / "z_image_e2e_rollup.json"
    md_path = out_root / "z_image_e2e_rollup.md"
    json_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=False), encoding="utf-8")

    a = rollup["assumptions"]
    e2e = rollup["e2e_total"]
    lines = [
        "# Z-Image 端到端定量汇总",
        "",
        f"- 推理步数: **{a['num_inference_steps']}**",
        f"- CFG scale: **{a['cfg_scale']}** (DiT ×{a['cfg_dit_multiplier']})",
        "",
        "## 阶段 MACs",
        "",
        "| 阶段 | 次数 | Forward MACs |",
        "|------|------|--------------|",
    ]
    for name, phase in rollup["phases"].items():
        lines.append(f"| {name} | {phase['runs']} | {phase['forward_macs']:,} |")
    lines.extend(
        [
            f"| **合计** | — | **{e2e['forward_macs']:,}** |",
            "",
            "## Denoise 单步子图（含 repeat）",
            "",
            "| 子图 | repeat | MACs |",
            "|------|--------|------|",
        ]
    )
    for row in rollup["per_graph_denoise"]:
        macs = row["forward_macs"] * row["repeat_per_step"]
        lines.append(f"| {row['file']} | ×{row['repeat_per_step']} | {macs:,} |")
    lines.append(f"| **denoise 单步合计** | — | **{rollup['denoise_per_step']['forward_macs']:,}** |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_root", help="Export root, e.g. output/Z_Image_ONNX_512")
    ap.add_argument("--model_path", required=True, help="Z-Image checkpoint for profile/repeat")
    ap.add_argument("--num_steps", type=int, default=28, help="Scheduler inference steps")
    ap.add_argument("--cfg_scale", type=float, default=5.0, help="CFG scale (>1 enables 2× DiT)")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--cap_seq", type=int, default=128)
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    scene = ExportScene(image_size=args.image_size, cap_seq=args.cap_seq)
    profile = load_source_profile(args.model_path, scene)
    rollup = compute_e2e_rollup(
        out_root,
        scene=scene,
        profile=profile,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
    )
    json_path, md_path = write_rollup_report(out_root, rollup)
    print(f"Rollup JSON: {json_path}")
    print(f"Rollup MD  : {md_path}")
    print(json.dumps(rollup["e2e_total"], indent=2))


if __name__ == "__main__":
    main()
