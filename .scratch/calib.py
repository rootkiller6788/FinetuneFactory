"""标定：一次训练到底学不学得动、跑多久、四种策略差在哪。**不是测试**，跑完就删。"""

import json
import sys
import time

from finetunefactory.config import load_config
from finetunefactory.data import (
    baselines,
    build_eval_sets,
    build_train_set,
    build_vocab,
    load_index,
)
from finetunefactory.eval import evaluate, evaluate_target_only
from finetunefactory.ir import FinetuneStrategy, ModelSpec, make_spec
from finetunefactory.model import init_base_weights
from finetunefactory.plan import build_plan
from finetunefactory.train import TrainSettings, train

cfg = load_config(None, root=".")
ts = TrainSettings.from_config(cfg)
ms = ModelSpec.from_dict(json.load(open("data/models/models.json", encoding="utf-8"))["models"][0]["spec"])
bw = init_base_weights(ms, 0)
idx = load_index(cfg["dataset"]["index"], root=".")
v = build_vocab(idx)
caps = list(idx.capabilities)
eval_sets = build_eval_sets(idx, v, template=ts.template, seq_len=ts.seq_len, n_eval=ts.n_eval, seed=0)
train_sets = {
    c: build_train_set(
        idx, v, capability=c, template=ts.template, seq_len=ts.seq_len,
        n_train=ts.n_train, seed=0, n_eval=ts.n_eval,
    )
    for c in caps
}
ref = baselines(idx, train_sets, eval_sets)
print(
    "平凡基准 chance/heuristic/majority:",
    {c: (round(ref[c]["chance"], 3), round(ref[c]["heuristic"], 3), round(ref[c]["majority"], 3)) for c in caps},
    flush=True,
)

TARGET_SETS = {
    "qv": ("q_proj", "v_proj"),
    "qkvo": ("q_proj", "k_proj", "v_proj", "o_proj"),
}


def mkspec(kind, cap, steps, *, rank=8, targets="qv", lr=None, objective="sft", **kw):
    lr = ts.lr if lr is None else lr
    if kind in ("lora", "qlora"):
        kw = {"rank": rank, "alpha": 2 * rank, "target_modules": TARGET_SETS[targets], **kw}
    elif kind == "freeze":
        kw = {"freeze_layers": 1, **kw}
    ft = FinetuneStrategy.canonical(kind, n_layers=ms.n_layers, **kw)
    return make_spec(
        ms,
        {"capability": cap, "template": ts.template, "seq_len": ts.seq_len,
         "n_train": ts.n_train, "n_eval": ts.n_eval, "seed": 0},
        finetuning=ft,
        objective=objective,
        optimizer={"name": ts.optimizer, "lr": lr, "weight_decay": ts.weight_decay},
        scheduler={"name": ts.scheduler, "warmup_steps": ts.warmup_steps, "total_steps": steps},
        batch={"size": ts.batch, "grad_accum": ts.grad_accum},
        evaluation={"every_n_steps": max(1, steps // 2), "batch_size": 32},
        checkpoint={"every_n_steps": max(1, steps // 2), "keep_last": 6},
        seed=0,
    )


# 微调前的模型在四项能力上的**实测**值 —— 这才是「干扰」的基准点。
# 用 chance 当基准会把「本来就只有 0.09、微调后 0.13」也算成干扰。
_sp = mkspec("full", caps[0], 48)
_base_plan = build_plan(_sp, bw, cost=cfg["cost"])
BASE = dict(evaluate(_base_plan, _sp, eval_sets, batch_size=32).per_capability)
print("微调前模型实测:", {c: round(BASE[c], 3) for c in caps}, flush=True)


def cell(kind, cap, *, rank=8, targets="qv", steps=48, lr=None, objective="sft", **kw):
    sp = mkspec(kind, cap, steps, rank=rank, targets=targets, lr=lr, objective=objective, **kw)
    plan = build_plan(sp, bw, cost=cfg["cost"])
    t0 = time.perf_counter()
    rec = train(
        sp, plan, train_sets[cap],
        settings=TrainSettings(
            steps=steps, batch=ts.batch, grad_accum=ts.grad_accum, lr=sp.optimizer.lr,
            warmup_steps=ts.warmup_steps, seq_len=ts.seq_len, template=ts.template,
            n_train=ts.n_train, n_eval=ts.n_eval, eval_every=max(1, steps // 2),
            checkpoint_every=max(1, steps // 2), optimizer=ts.optimizer,
            weight_decay=ts.weight_decay, scheduler=ts.scheduler, max_grad_norm=ts.max_grad_norm,
        ),
        probe=lambda st, pl: evaluate_target_only(pl, sp, eval_sets[cap], batch_size=32),
    )
    dt = time.perf_counter() - t0
    ev = evaluate(plan, sp, eval_sets, batch_size=32)
    macs = plan.forward_macs_total(seq_len=ts.seq_len)
    label = f"{kind}/r{rank}/{targets}" if kind in ("lora", "qlora") else kind
    print(
        f"{label:16s} {cap:8s} {objective:16s} loss {rec.first_loss:6.3f}->{rec.final_loss:6.3f} "
        f"acc {ev.target_acc:6.3f} gain {ev.gain(BASE):+.3f} "
        f"(chance {ref[cap]['chance']:.3f}/heur {ref[cap]['heuristic']:.3f}) "
        f"interf {ev.interference(BASE):+.3f} "
        f"tr {plan.n_trainable_params:5d}p vram {plan.cost(seq_len=ts.seq_len, batch=ts.batch)['vram_bytes']:7d} "
        f"macs {macs:6d} {dt:5.1f}s",
        flush=True,
    )
    print("      ckpt:", [(c.step, round(c.loss, 3), None if c.eval_acc is None else round(c.eval_acc, 3)) for c in rec.checkpoints], flush=True)
    return rec, ev, dt


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "matrix":
        steps = int(sys.argv[2]) if len(sys.argv) > 2 else 48
        for cap in ("upper", "sort"):
            for kind in ("full", "freeze", "lora", "qlora"):
                cell(kind, cap, steps=steps)
        for objective in ("sft", "sft_response_only"):
            cell("full", "upper", steps=steps, objective=objective)
            cell("lora", "upper", steps=steps, objective=objective, rank=8)
        for rank in (4, 16):
            cell("lora", "upper", steps=steps, rank=rank)
    else:
        cap = sys.argv[1] if len(sys.argv) > 1 else "upper"
        steps = int(sys.argv[2]) if len(sys.argv) > 2 else 48
        lr = float(sys.argv[3]) if len(sys.argv) > 3 else ts.lr
        cell("full", cap, steps=steps, lr=lr)
