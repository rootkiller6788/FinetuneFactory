"""lr 敏感性重测。**旧数字是在梯度引擎有两处静默 bug 时测的，全部作废。**

不是测试。是「配置里那句 lr 标定话术到底还成不成立」的一次实测。
"""
import json
from finetunefactory import pipeline

CELLS = [
    {"kind": "full"},
    {"kind": "freeze"},
    {"kind": "lora", "rank": 8, "targets": "qv"},
    {"kind": "qlora", "rank": 8, "targets": "qv", "bits": 4},
]
LRS = [0.01, 0.03, 0.1, 0.3]
print(f"{'lr':>6}  {'策略':<8}{'loss 首→末':>22}{'acc':>8}{'gain':>9}{'干扰':>9}")
rows = []
for lr in LRS:
    res = pipeline.run_pipeline(
        root=".",
        sweep={"cells": CELLS, "capabilities": ["upper"], "objectives": ["sft"]},
        train={"lr": lr, "steps": 48, "batch": 8, "n_train": 160, "n_eval": 24,
               "seq_len": 16, "warmup_steps": 8},
    )
    for sid, spec in sorted(res.specs.items(), key=lambda kv: kv[1].finetuning.kind):
        run = res.runs[sid]
        lo, hi = run.losses[0], run.losses[-1]
        acc = res.evals[sid].target_acc
        g = res.gain_of(sid)
        it = res.interference_of(sid)
        rows.append({"lr": lr, "kind": spec.finetuning.kind, "loss0": lo, "loss1": hi,
                     "acc": acc, "gain": g, "interf": it})
        print(f"{lr:>6}  {spec.finetuning.kind:<8}{lo:>10.3f} →{hi:>10.3f}{acc:>8.4f}{g:>+9.4f}{it:>+9.4f}")
    print()
json.dump(rows, open(".scratch/lr.json", "w"), indent=2)
