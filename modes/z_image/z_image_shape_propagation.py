"""Static shape propagation for Z-Image denoise chain validation (constraint 4)."""

from __future__ import annotations

from dataclasses import dataclass

from z_image_export_semantics import ExportScene, GraphContract, build_denoise_graph_contracts


@dataclass(frozen=True)
class ChainValidation:
    ok: bool
    errors: list[str]


def validate_denoise_chain(scene: ExportScene, profile) -> ChainValidation:
    contracts = build_denoise_graph_contracts(scene, profile)
    by_kind = {c.kind: c for c in contracts}
    errors: list[str] = []

    def out_shape(kind: str, idx: int = 0) -> list:
        return list(by_kind[kind].outputs[idx].shape)

    def in_shape(kind: str, idx: int = 0) -> list:
        return list(by_kind[kind].inputs[idx].shape)

    # T2 patchify_and_embed → x/cap branches
    if out_shape("patchify_and_embed", 0) != in_shape("x_branch", 0):
        errors.append("patchify x_patch_feats != x_branch input")
    if out_shape("patchify_and_embed", 1) != in_shape("cap_branch", 0):
        errors.append("patchify cap_feats_padded != cap_branch input")
    if out_shape("patchify_and_embed", 2) != in_shape("x_branch", 1):
        errors.append("patchify x_pos_ids != x_branch pos_ids")
    if out_shape("patchify_and_embed", 4) != in_shape("x_branch", 2):
        errors.append("patchify x_pad_mask != x_branch pad_mask")
    if out_shape("patchify_and_embed", 3) != in_shape("cap_branch", 1):
        errors.append("patchify cap_pos_ids != cap_branch pos_ids")
    if out_shape("patchify_and_embed", 5) != in_shape("cap_branch", 2):
        errors.append("patchify cap_pad_mask != cap_branch pad_mask")

    if out_shape("timestep_embed") != in_shape("x_branch", 3):
        errors.append("timestep adaln != x_branch adaln_input")

    # branches → concat
    if out_shape("x_branch", 0) != in_shape("sequence_concat", 0):
        errors.append("x_branch x_tokens != sequence_concat x_tokens")
    if out_shape("cap_branch", 0) != in_shape("sequence_concat", 1):
        errors.append("cap_branch cap_tokens != sequence_concat cap_tokens")
    if out_shape("x_branch", 1) != in_shape("sequence_concat", 2):
        errors.append("x_branch x_rope_cos != sequence_concat x_rope_cos")
    if out_shape("x_branch", 2) != in_shape("sequence_concat", 3):
        errors.append("x_branch x_rope_sin != sequence_concat x_rope_sin")
    if out_shape("cap_branch", 1) != in_shape("sequence_concat", 4):
        errors.append("cap_branch cap_rope_cos != sequence_concat cap_rope_cos")
    if out_shape("cap_branch", 2) != in_shape("sequence_concat", 5):
        errors.append("cap_branch cap_rope_sin != sequence_concat cap_rope_sin")

    # unified → main → final
    if out_shape("sequence_concat", 0) != in_shape("main_layer_repr", 0):
        errors.append("sequence_concat unified_tokens != main_layer input")
    if out_shape("main_layer_repr") != in_shape("final_output", 0):
        errors.append("main_layer output != final_output hidden_states")
    if out_shape("timestep_embed") != in_shape("final_output", 1):
        errors.append("timestep adaln != final_output adaln_input")

    if out_shape("sequence_concat", 1) != in_shape("main_layer_repr", 2):
        errors.append("sequence_concat unified_rope_cos != main_layer rope_cos")
    if out_shape("sequence_concat", 2) != in_shape("main_layer_repr", 3):
        errors.append("sequence_concat unified_rope_sin != main_layer rope_sin")

    return ChainValidation(ok=not errors, errors=errors)


def contract_io_summary(contract: GraphContract) -> str:
    ins = ", ".join(f"{t.name}:{t.shape}{t.dtype}" for t in contract.inputs)
    outs = ", ".join(f"{t.name}:{t.shape}{t.dtype}" for t in contract.outputs)
    repeat = f" ×{contract.repeat}" if contract.repeat > 1 else ""
    return f"{contract.file_name}{repeat}  ({ins}) -> ({outs})"
