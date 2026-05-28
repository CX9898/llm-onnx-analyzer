"""Source alignment audit for Z-Image export wrappers."""

from __future__ import annotations

from pathlib import Path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def build_audit_report(
    *,
    diffusers_src: Path,
    export_blocks_path: Path,
    export_rope_path: Path,
) -> str:
    source_transformer = diffusers_src / "diffusers/models/transformers/transformer_z_image.py"
    source_text = _read(source_transformer)
    blocks_text = _read(export_blocks_path)
    rope_text = _read(export_rope_path)

    findings: list[tuple[str, str, str]] = []

    checks = [
        (
            "block_adaln_modulation",
            "adaLN_modulation(adaln_input) + static split(4) on dim=2",
            "ZImageTransformerBlockExportBlock 复制源码 AdaLN 分支（固定 slice 替代 chunk）",
        ),
        (
            "block_plain_path",
            "x + block.attention_norm2(attn_out)",
            "Plain refiner 路径与源码一致",
        ),
        (
            "rope_real_math",
            "x0 * cos - x1 * sin",
            "apply_rotary_emb_onnx：标准 Mul/Add 展开，等价于源码 view_as_complex（非 custom op）",
        ),
        (
            "text_encode_hidden",
            "hidden_states[-2]",
            "TextEncodePaddedBlock 对齐 pipeline._encode_prompt 编码路径",
        ),
        (
            "vae_scaling",
            "scaling_factor",
            "VAEDecodeBlock 保留 decode 前 scaling",
        ),
        (
            "sequence_concat_basic",
            "torch.cat([x_tokens, cap_tokens]",
            "SequenceConcatBasicBlock 对齐 _build_unified_sequence basic 模式",
        ),
    ]

    for key, needle, note in checks:
        in_blocks = needle in blocks_text
        in_rope = needle in rope_text
        in_source = needle in source_text
        if in_rope or in_blocks:
            status = "ok"
        elif in_source:
            status = "source_only"
        else:
            status = "missing"
        if key == "rope_real_math" and in_rope:
            status = "ok (ONNX 无 complex，实数等价展开)"
        findings.append((key, status, note))

    lines = [
        "# Z-Image source alignment audit",
        "",
        f"- Source: `{source_transformer}`",
        "",
        "Z-Image **不使用 custom op**：约束 6 针对 Qwen 的 RecurrentGatedDeltaRule 等带状态循环结构；",
        "Z-Image RoPE 可直接展开为标准 ONNX 算子。",
        "",
        "| Check | Status | Note |",
        "| --- | --- | --- |",
    ]
    for key, status, note in findings:
        lines.append(f"| `{key}` | {status} | {note} |")

    return "\n".join(lines) + "\n"


def write_audit(out_dir: Path, *, diffusers_src: Path, mode_dir: Path) -> Path:
    report = build_audit_report(
        diffusers_src=diffusers_src,
        export_blocks_path=mode_dir / "z_image_onnx_blocks.py",
        export_rope_path=mode_dir / "z_image_onnx_rope.py",
    )
    path = out_dir / "source_alignment_audit.md"
    path.write_text(report, encoding="utf-8")
    return path
