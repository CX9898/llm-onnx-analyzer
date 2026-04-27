#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from qwen_export_semantics import build_dtype_audit_report, render_dtype_audit_markdown


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional markdown output path for the audit report.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    export_wrapper = repo_root / "export_qwen_onnx" / "qwen_onnx_blocks.py"
    source_model = Path("/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py")

    findings = build_dtype_audit_report(_read(export_wrapper), _read(source_model))
    markdown = render_dtype_audit_markdown(findings)

    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")


if __name__ == "__main__":
    main()
