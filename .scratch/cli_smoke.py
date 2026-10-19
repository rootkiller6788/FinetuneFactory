"""CLI 冒烟：跑一次小规模 run，再逐个跑只读子命令。"""
import json, shutil, sys, tempfile
from pathlib import Path
sys.path.insert(0, "src")
from finetunefactory import cli

OUT = Path(tempfile.mkdtemp(prefix="ff-cli-"))
CELLS = [
    {"kind": "full"}, {"kind": "freeze"},
    {"kind": "lora", "rank": 8, "targets": "qv"},
    {"kind": "lora", "rank": 8, "targets": "qv", "objective": "sft_response_only"},
    {"kind": "qlora", "rank": 8, "targets": "qv", "bits": 4},
    {"kind": "qlora", "rank": 8, "targets": "qv", "bits": 8},
]
cfg = json.loads(Path("configs/default.json").read_text(encoding="utf-8"))
cfg["sweep"].update({"cells": CELLS, "capabilities": ["upper", "shift", "sort"], "objectives": ["sft"]})
cfg["train"].update({"steps": 4, "batch": 4, "n_train": 16, "n_eval": 8, "seq_len": 16, "warmup_steps": 2})
cfg.pop("_comment", None)
cfgpath = OUT / "cfg.json"
cfgpath.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

def run(args):
    print("\n" + "=" * 78)
    print("$", " ".join(args))
    print("=" * 78)
    code = cli.main(args)
    print("[exit]", code)
    return code

run(["run", "--config", str(cfgpath), "--out", str(OUT), "--quiet"])
run(["compare", "--out", str(OUT)])
run(["frontier", "--out", str(OUT)])
run(["list", "--out", str(OUT)])
run(["explain", "sft-full-1747d62bc934", "--out", str(OUT)])
run(["explain", "shift/lora/r8/qv/sft", "--out", str(OUT)])
run(["artifact", "shift/lora/r8/qv/sft", "--out", str(OUT)])
run(["artifact", "shift/full/sft", "--out", str(OUT)])
run(["dag", "--out", str(OUT), "--dot", "--frontier-only"])
run(["dag", "--out", str(OUT), "--limit", "6"])
run(["explain", "no-such", "--out", str(OUT)])
run(["compare", "--out", str(OUT / "nope")])
run(["explain", "x", "--out", str(OUT / "nope")])
print("\nOUT =", OUT)
