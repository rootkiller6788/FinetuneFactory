"""报告渲染冒烟：跑一遍小的，再从磁盘 load_full 回来渲染一次，逐行对比。"""
import sys, tempfile
from pathlib import Path
from finetunefactory import pipeline, report

CELLS = [
    {"kind": "full", "note": "上界"},
    {"kind": "freeze", "note": "冻结"},
    {"kind": "lora", "rank": 8, "targets": "qv"},
    {"kind": "lora", "rank": 8, "targets": "qv", "objective": "sft_response_only"},
    {"kind": "qlora", "rank": 8, "targets": "qv", "bits": 4},
    {"kind": "qlora", "rank": 8, "targets": "qv", "bits": 8},
]
ov = {
    "sweep": {"cells": CELLS, "capabilities": ["upper", "shift", "sort"], "objectives": ["sft"]},
    "train": {"steps": 4, "batch": 4, "n_train": 16, "n_eval": 8, "seq_len": 16, "warmup_steps": 2},
}
out = Path(tempfile.mkdtemp(prefix="ff-rep-"))
res = pipeline.run_pipeline(root=".", **ov)
paths = pipeline.write_outputs(res, out)
md = report.render_markdown(res)
print("== 章节 ==")
for line in md.splitlines():
    if line.startswith("#"):
        print("  ", line)
print("报告字节:", len(md))
print()
print("== 基准表 ==")
print(report.markdown_table(("能力","chance","基座","基座-chance","读作"), report.baseline_rows(res)))
print()
print("== 矩阵前 4 行 ==")
rows = report.matrix_rows(res)
print(report.markdown_table(report.matrix_header(), rows[:4]))
print("矩阵行数:", len(rows))
print()
print("== 显存分解前 3 行 ==")
print(report.markdown_table(report.cost_header(), report.cost_rows(res)[:3]))
print()
print("== 核对 ==")
print(report.markdown_table(("项","结果","说明"), report.check_rows(res)))
print()
print("== 裁决 ==")
print(report.markdown_table(("期望","裁决","条数","说明"), report.verdict_rows(res)))
print(report.markdown_table(("期望","配置要求","实际裁决","备注"), report.expectation_status_rows(res)))
print()
print("== 摘要 ==")
print(report.summary_line(res))
print("meta:", report.meta_payload(res))
print()
# ---- 落盘再读回来渲染，两条路径必须逐行一样
back = pipeline.load_full(out)
print("load_full: runs", len(back.runs), "evals", len(back.evals), "artifacts", len(back.artifacts),
      "dag", back.dag is not None)
md2 = report.render_markdown(back)
print("两条路径的报告逐字相同:", md == md2)
if md != md2:
    a, b = md.splitlines(), md2.splitlines()
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            print(f"  第 {i} 行不同:\n    A: {x}\n    B: {y}")
            break
    print("  行数:", len(a), len(b))
print()
# ---- explain 两个：一个前沿点，一个落选点
front = [s.spec_id for s in res.frontier]
allids = sorted(res.specs)
print("explain 前沿:", front[0] if front else None)
if front:
    txt = report.explain(back, front[0])
    print(txt[:1800])
dominated = [s for s in allids if s not in front]
print()
print("explain 落选:", dominated[0] if dominated else None)
if dominated:
    txt = report.explain(back, dominated[0])
    i = txt.find("## 三")
    print(txt[i:i+1400])
print()
try:
    report.explain(res, "no-such-spec")
except KeyError as e:
    print("未知 spec 的报错 OK:", str(e)[:90])
