from __future__ import annotations

import json
import os
from collections import Counter

import onnx


def structured_load_ops(path: str) -> Counter[str]:
    model = onnx.load(path)
    return Counter(
        node.op_type if not node.domain else f"{node.domain}::{node.op_type}"
        for node in model.graph.node
    )


def structured_counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return {op: int(count) for op, count in sorted(counter.items())}


def structured_expand_counts(
    key: str,
    manifest: dict[str, dict],
    node_type_to_key: dict[str, str],
    memo: dict[str, Counter[str]],
) -> Counter[str]:
    if key in memo:
        return memo[key]
    direct = Counter(manifest[key]["direct_ops"])
    expanded: Counter[str] = Counter()
    for op, count in direct.items():
        child_key = node_type_to_key.get(op)
        if child_key is None:
            expanded[op] += count
            continue
        child_counts = structured_expand_counts(child_key, manifest, node_type_to_key, memo)
        for child_op, child_count in child_counts.items():
            expanded[child_op] += child_count * count
    memo[key] = expanded
    return expanded


def write_structured_manifest(
    graph_defs: dict[str, dict[str, str | None]],
    *,
    root_key: str,
    save_path: str,
) -> None:
    manifest: dict[str, dict] = {}
    node_type_to_key: dict[str, str] = {}
    for key, cfg in graph_defs.items():
        path = str(cfg["path"])
        node_type = cfg.get("node_type")
        direct_ops = structured_load_ops(path)
        manifest[key] = {
            "path": path,
            "node_type": node_type,
            "inputs": cfg.get("inputs", []),
            "outputs": cfg.get("outputs", []),
            "size_mb": round(os.path.getsize(path) / (1 << 20), 6),
            "node_count": int(sum(direct_ops.values())),
            "direct_ops": structured_counter_to_dict(direct_ops),
        }
        if node_type is not None:
            node_type_to_key[str(node_type)] = key

    memo: dict[str, Counter[str]] = {}
    for key in manifest:
        manifest[key]["expanded_ops"] = structured_counter_to_dict(
            structured_expand_counts(key, manifest, node_type_to_key, memo)
        )
        direct_ops = Counter(manifest[key]["direct_ops"])
        children = []
        for op, count in direct_ops.items():
            child_key = node_type_to_key.get(op)
            if child_key is None:
                continue
            children.append(
                {
                    "node_type": op,
                    "graph_key": child_key,
                    "path": manifest[child_key]["path"],
                    "occurrences": int(count),
                }
            )
        manifest[key]["children"] = children

    payload = {
        "root_graph": root_key,
        "graphs": manifest,
        "root_expanded_ops": manifest[root_key]["expanded_ops"],
    }
    with open(save_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

