"""小规模端到端冒烟：跑通 run_pipeline -> write_outputs，打印所有核对表。

不是测试。只是「交付前实际跑一遍」，因为这几百行里绝大多数错误不会在阅读时暴露。
"""
import json
import sys
import tempfile
from pathlib import Path

from finetunefactory import pipeline

CELLS = [
    {"kind": "full", "note": "上界"},
    {"kind": "freeze", "note": "冻结"},
    {"kind": "lora", "rank": 8, "targets": "qv", "note": "LoRA qv"},
    {"kind": "lora", "rank": 8, "targets": "qv", "objective": "sft_response_only",
     "note": "LoRA qv 只学回答"},
    {"kind": "qlora", "rank": 8, "targets": "qv", "bits": 4, "note": "4bit"},
    {"kind": "qlora", "rank": 8, "targets": "qv", "bits": 8, "note": "8bit"},
    {"kind": "lora", "rank": 32, "targets": "qv", "note": "r32"},
]

ov = {
    "sweep": {
        "cells": CELLS,
        "capabilities": ["upper", "sort", "shift"],
        "objectives": ["sft"],
    },
    "train": {"steps": 4, "batch": 4, "n_train": 16, "n_eval": 8, "seq_len": 16,
              "warmup_steps": 2},
}

out = Path(tempfile.mkdtemp(prefix="ff-smoke-"))
res = pipeline.run_pipeline(root=".", **ov)

print("=" * 78)
print("specs:", len(res.specs), " frontier:", len(res.frontier))
for s in res.frontier:
    print("   front:", s.spec_id)
print("pareto_stats:", json.dumps(res.counters.get("pareto_stats"), ensure_ascii=False))
print("skipped:", json.dumps(res.counters.get("skipped"), ensure_ascii=False)[:300])
print()
print("checks:")
for k, v in res.counters["checks"].items():
    ok = v.get("ok")
    mark = "OK  " if ok else "FAIL"
    extra = ""
    if k == "macs":
        extra = f" {v.get('n_checked')} spec 逐算子比对"
    if k == "merge":
        extra = f" max_abs_diff={v.get('max_abs_diff')}"
    if k == "gradients":
        w = v.get("worst_rel_error")
        extra = f" {v.get('n_applicable')} 个计划, 最差相对误差={'n/a' if w is None else f'{w:.3g}'}"
    if k == "quant_error":
        extra = f" {str(v.get('reason', ''))[:60]}"
    print(f"  [{mark}] {k:<16}{extra}")
    if not ok:
        print("        ", json.dumps(v, ensure_ascii=False, default=str)[:500])

print()
print("verdicts:")
for v in res.verdicts:
    print(f"  {v['verdict']:<10} {v['name']}  ({len(v.get('observed', []))} 条证据)")
    print(f"      {str(v.get('reason', ''))[:160]}")

print()
paths = pipeline.write_outputs(res, out)
print("outputs ->", out)
for k, p in paths.items():
    print(f"  {k:<14} {p.name:<18} {p.stat().st_size} B")

# 落盘再读回来：序列化是另一条独立的失败路径。
back = pipeline.load_result(out)
print()
print("roundtrip: specs", len(back.specs), "frontier", len(back.frontier),
      "dag nodes", len(back.dag.nodes()), "edges", len(back.dag.edges()))
print("payloads roundtrip:", sorted(pipeline.iter_artifact_specs(out)) == sorted(res.specs))
kinds = {}
for n in back.dag.nodes():
    kinds[n.kind] = kinds.get(n.kind, 0) + 1
print("dag 节点种类:", kinds)
print("pareto 的父节点数:", len(back.dag.parents("pareto")))
