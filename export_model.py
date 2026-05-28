#!/usr/bin/env python3
"""统一 ONNX 导出入口。

只需指定模型与本地真实权重路径，就能拿到一套完整子图：

    python export_model.py qwen3.5moe    --model_path /path/to/Qwen3.5-35B-A3B
    python export_model.py qwen3.5moe-vl --model_path /path/to/Qwen3.5-35B-A3B
    python export_model.py z-image         --model_path /path/to/Tongyi-MAI/Z-Image

    # 导出根目录默认 ./output，可用 --output_dir 覆盖。

约定（不暴露给 CLI，避免膨胀）：
- qwen3.5moe：8k 上下文 text-only 子集，加载 ``Qwen3_5MoeForCausalLM``，
  自动导出 prefill + decode 两套，分别落在
  ``<output_dir>/Qwen3_5_35B_A3B_ONNX_Prefill_8k`` 与
  ``<output_dir>/Qwen3_5_35B_A3B_ONNX_Decode_8k``。
- qwen3.5moe-vl：完整多模态版本，加载
  ``Qwen3_5MoeForConditionalGeneration``，在 text 子图基础上多导出 4 个
  vision/MM 子图（patch_embed / 代表 ViT block / patch_merger / mm_inject），
  分别落在
  ``<output_dir>/Qwen3_5_35B_A3B_VL_ONNX_Prefill_8k`` 与
  ``<output_dir>/Qwen3_5_35B_A3B_VL_ONNX_Decode_8k``。
- 其它高级旋钮（export_scope、batch_size、opset、各代表层索引、
  linear prefill chunk size、vision_token_seq_len、mm_image_token_count
  等）一律沿用各模型内部主入口的默认值。
  需要调时直接去对应的 ``modes/<model>/`` 主入口里调。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_MODES_ROOT = _REPO_ROOT / "modes"

_QWEN_MODEL_TAG = "35B_A3B"
_QWEN_CONTEXT_LEN = 8192
_QWEN_PHASES = ("prefill", "decode")


def _seq_tag(seq_len: int) -> str:
    return f"{seq_len // 1024}k" if seq_len % 1024 == 0 else str(seq_len)


def _run(cmd: list[str], cwd: Path, extra_pythonpath: list[Path]) -> None:
    env = os.environ.copy()
    paths = [str(p) for p in extra_pythonpath]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)

    print(f">>> {' '.join(cmd)}")
    if subprocess.run(cmd, cwd=str(cwd), env=env).returncode != 0:
        raise SystemExit(f"sub-export failed: {' '.join(cmd)}")


def _run_qwen_3_5_moe(model_path: str, output_dir: Path, *, variant: str) -> None:
    """Drive the Qwen3.5-MoE per-mode entry script across both phases.

    ``variant`` selects the top-level loaded class:
    - ``"text"`` -> ``Qwen3_5MoeForCausalLM`` (text-only ONNX subset).
    - ``"vl"``   -> ``Qwen3_5MoeForConditionalGeneration`` (text + vision tower
      + multimodal injection).
    """
    if variant not in ("text", "vl"):
        raise ValueError(f"variant must be 'text' or 'vl', got {variant!r}")

    qwen_dir = _MODES_ROOT / "qwen_3_5_MoE"
    main_script = qwen_dir / "export_qwen_onnx_main.py"
    if not main_script.is_file():
        raise FileNotFoundError(f"missing entry: {main_script}")

    ctx_tag = _seq_tag(_QWEN_CONTEXT_LEN)
    variant_tag = "_VL" if variant == "vl" else ""
    log_tag = f"qwen3.5moe{'-vl' if variant == 'vl' else ''}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for phase in _QWEN_PHASES:
        phase_dir = output_dir / (
            f"Qwen3_5_{_QWEN_MODEL_TAG}{variant_tag}_ONNX_{phase.capitalize()}_{ctx_tag}"
        )
        phase_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[export_model] {log_tag} phase={phase} ctx={ctx_tag} -> {phase_dir}")
        _run(
            [
                sys.executable,
                str(main_script),
                model_path,
                "--phase", phase,
                "--variant", variant,
                "--output_dir", str(phase_dir),
            ],
            cwd=qwen_dir,
            extra_pythonpath=[_REPO_ROOT, qwen_dir],
        )

    print(f"\n[export_model] {log_tag} done.")


def _export_qwen_3_5_moe(model_path: str, output_dir: Path) -> None:
    _run_qwen_3_5_moe(model_path, output_dir, variant="text")


def _export_qwen_3_5_moe_vl(model_path: str, output_dir: Path) -> None:
    _run_qwen_3_5_moe(model_path, output_dir, variant="vl")


def _export_llada_2_1(model_path: str, output_dir: Path) -> None:
    raise NotImplementedError(
        f"LLaDA-2.1 导出尚未接入：请在 {_MODES_ROOT / 'llada_2_1'} 下添加实现，"
        " 并在 export_model.py 的 _DISPATCH 中注册。"
    )


def _export_z_image(model_path: str, output_dir: Path) -> None:
    zimage_dir = _MODES_ROOT / "z_image"
    main_script = zimage_dir / "export_z_image_onnx_main.py"
    if not main_script.is_file():
        raise FileNotFoundError(f"missing entry: {main_script}")

    output_dir.mkdir(parents=True, exist_ok=True)
    phases = ("all",)
    phase_dir = output_dir / "Z_Image_ONNX_512"
    phase_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[export_model] z-image -> {phase_dir}")
    _run(
        [
            sys.executable,
            str(main_script),
            model_path,
            "--phase",
            "all",
            "--output_dir",
            str(phase_dir),
            "--image_size",
            "512",
            "--cap_seq",
            "128",
        ],
        cwd=zimage_dir,
        extra_pythonpath=[_REPO_ROOT, zimage_dir],
    )
    print("\n[export_model] z-image done.")


_DISPATCH = {
    "qwen3.5moe": _export_qwen_3_5_moe,
    "qwen3.5moe-vl": _export_qwen_3_5_moe_vl,
    "llada2.1": _export_llada_2_1,
    "z-image": _export_z_image,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="export_model",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model", choices=sorted(_DISPATCH), help="要导出的模型")
    parser.add_argument("--model_path", required=True, help="本地真实权重目录")
    parser.add_argument("--output_dir", default="./output", help="导出根目录（默认 ./output）")
    args = parser.parse_args()

    _DISPATCH[args.model](args.model_path, Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
