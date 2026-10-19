"""流水线：扇出 64 个规格，收拢成一个前沿。

```
Base Model + Dataset ──┐
                       ├──► Validate ──► Prepare ──► Build Strategy ──► Train ──► Evaluate ──► Compare ──► Export
Training Spec (×64) ───┘                             (per cell)        (per cell)
```

### 扇出的那一侧：64 个独立世界的共享前提

每个规格都从**同一份基座权重**出发。这件事比它听起来脆弱：
`plan.build_plan` 把权重包成张量，而训练是**原地**改这些张量的 ——
所以如果 64 个计划共享同一批 list，第二个规格就是从第一个训练完的地方接着训的。
`plan._copy_nested` 与 `build_plan` 里那条别名断言一起堵住了它。这里之所以还要
写一遍，是因为**扇出型项目的整个对照关系都建立在这条不变量上**，
而它失效时每条曲线单独看都完全正常。

### 收拢的那一侧：唯一的 merge 节点

`pareto` 是整个 DAG 里唯一一个「吃很多、吐一个」的节点。它收的是前沿候选的
产物，出的是一条可以被引用、可以被反驳的前沿。四条设计期望的裁决
（:func:`compare.judge_expectations`）也在这一侧 —— 它们**允许被推翻**，
被推翻的那几条是这个项目最值钱的输出。
"""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import compare
from .config import (
    cell_label,
    expectations_of,
    load_config,
    protocol_digest,
    sweep_cells,
    version_digest,
)
from .data import (
    Example,
    baselines,
    build_eval_sets,
    build_train_set,
    build_vocab,
    load_index,
)
from .eval import evaluate, evaluate_target_only
from .ir import FinetuneStrategy, ModelSpec, make_spec
from .lineage import build_lineage, dumps as dumps_lineage
from .model import init_base_weights
from .plan import ParamPlan, build_plan
from .schema import Artifact, ModelEntry, PipelineResult
from .train import TrainSettings, train

__all__ = [
    "run_pipeline",
    "write_outputs",
    "load_result",
    "load_full",
    "load_runs",
    "load_evals",
    "load_artifacts",
    "load_artifact",
    "load_lineage",
    "iter_artifact_specs",
    "load_model_registry",
    "ARTIFACT_DIR",
]


ARTIFACT_DIR = "artifacts"

#: 进度回调：`(spec_id, label, index, total)`。给 CLI 用。
Progress = Callable[[str, str, int, int], None]


# ----------------------------------------------------------------------
# 输入
# ----------------------------------------------------------------------
def load_model_registry(path: str | Path, *, root: Path | str = ".") -> list[ModelEntry]:
    """读模型注册表。

    结构事实（`:class:`~finetunefactory.ir.ModelSpec`）在文件里，**权重不在** ——
    权重由 :func:`model.init_base_weights` 从 Spec + 种子确定性生成。
    这不是省事：落盘几万个浮点只会给「同一份 Spec」多出一个能漂移的自由度。
    """
    p = Path(path)
    if not p.is_absolute():
        p = Path(root) / p
    raw = json.loads(p.read_text(encoding="utf-8"))
    entries = [ModelEntry.from_dict(m) for m in raw.get("models", [])]
    if not entries:
        raise ValueError(f"模型注册表 {p} 里一个模型都没有")
    return entries


def _strategy_from_cell(cell: Mapping[str, Any], *, n_layers: int, alpha_mult: float) -> FinetuneStrategy:
    """把网格里的一格翻成 :class:`FinetuneStrategy`。

    ``alpha = alpha_mult × rank``：这是 LoRA 的常见约定，而 ``2`` 这个倍数必须来自
    配置。写死在代码里的话，「alpha 与 rank 的对比实验」就不是单变量的了 ——
    改 rank 会同时改有效学习率，而报告里只会看到 rank 那一列在动。
    """
    kind = str(cell["kind"])
    rank = int(cell.get("rank") or 0)
    kw: dict[str, Any] = {}
    if kind in ("lora", "qlora"):
        kw = {
            "rank": rank,
            "alpha": float(alpha_mult) * rank,
            "target_modules": tuple(cell.get("target_modules") or ()),
        }
        if kind == "qlora":
            kw["quant_bits"] = int(cell.get("quant_bits") or 4)
            kw["quant_group"] = int(cell.get("quant_group") or 32)
    return FinetuneStrategy.canonical(kind, n_layers=n_layers, **kw)


# ----------------------------------------------------------------------
# 产物字节
# ----------------------------------------------------------------------
def _payload_bytes(plan: ParamPlan) -> tuple[bytes, dict[str, Any]]:
    """把产物序列化成**裸 float32 小端**，返回 ``(字节, 旁证元数据)``。

    为什么是裸 float32、不带任何头部：``Artifact.payload_bytes`` 是**算术值**
    （``Σ 参数量 × 4``，由参数表推出），而这条等式只有在文件里**只**放浮点数时
    才成立。加一个魔数、一个长度前缀、或者一点点对齐，等式就会差出固定的几十字节 ——
    而那个差看起来完全像「正常的序列化开销」。元数据因此全部放进旁边的 JSON。
    """
    names: list[str] = []
    shapes: list[list[int]] = []
    offsets: list[int] = []
    chunks: list[bytes] = []
    off = 0
    for name, value in plan.payload_tensors():
        flat: list[float] = []
        _flatten(value, flat)
        blob = struct.pack(f"<{len(flat)}f", *flat)
        names.append(name)
        shapes.append(_shape_of(value))
        offsets.append(off)
        chunks.append(blob)
        off += len(blob)
    payload = b"".join(chunks)
    return payload, {
        "names": names,
        "shapes": shapes,
        "offsets": offsets,
        "dtype": "float32",
        "endian": "little",
        "layout": "concatenated row-major, no header",
        "n_floats": sum(len(c) // 4 for c in chunks),
    }


def _flatten(value: Any, out: list[float]) -> None:
    """展平成 float 序列（行主序）。只认到两层 —— 参数一定是 1D 或 2D。"""
    if value and isinstance(value[0], (list, tuple)):
        for row in value:
            for v in row:
                out.append(float(v))
    else:
        for v in value:
            out.append(float(v))


def _shape_of(value: Any) -> list[int]:
    if value and isinstance(value[0], (list, tuple)):
        return [len(value), len(value[0])]
    return [len(value)]


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def run_pipeline(
    cfg: Mapping[str, Any] | None = None,
    *,
    root: Path | str = ".",
    progress: Progress | None = None,
    **overrides: Any,
) -> PipelineResult:
    """跑完整条流水线。`cfg` 为 `None` 时读 ``configs/default.json``。

    `overrides` 会覆盖配置里的顶层键（例如 ``train={"steps": 8}``）——
    它让短跑（冒烟）与长跑（出报告）走的是**同一条代码路径**，
    只有数字不同。另写一条「快速路径」是这类项目里最贵的省事：
    两条路径漂移之后，短跑全绿而长跑出来的报告是另一回事。
    """
    root = Path(root)
    t_start = time.perf_counter()
    if cfg is None:
        cfg = load_config(None, root=root)

    # `overrides` 必须在**两条**分支上都生效。
    #
    # 之前写成 `if cfg is None: ... elif overrides: ...`，于是 `cfg=None`（也就是
    # 读默认配置——最常用的那一种）时 overrides 被**静默丢弃**。而冒烟短跑正是
    # 这么调的：`run_pipeline(root=".", train={"steps": 4})`。结果它拿着生产配置
    # 跑了一遍，唯一的线索是打印出来的 spec 数量对不上——如果我没数那一眼，
    # 「冒烟通过」这句话就是在说另一份配置。
    if overrides:
        from .config import deep_merge, validate_config

        cfg = deep_merge(dict(cfg), overrides)
        validate_config(cfg)

    train_cfg = dict(cfg.get("train") or {})
    settings = TrainSettings.from_config(cfg)
    cost_cfg = dict(cfg.get("cost") or {})
    acceptance = dict(cfg.get("acceptance") or {})
    tol = float(acceptance.get("merge_tolerance", 1e-5))

    # ---- 输入
    models = load_model_registry((cfg.get("model") or {}).get("registry", ""), root=root)
    base_entry = models[0]
    model_spec: ModelSpec = base_entry.spec
    base_weights = init_base_weights(model_spec, 0)

    idx = load_index((cfg.get("dataset") or {}).get("index", ""), root=str(root))
    vocab = build_vocab(idx)
    caps = list(idx.capabilities)

    eval_sets = build_eval_sets(
        idx, vocab, template=settings.template, seq_len=settings.seq_len,
        n_eval=settings.n_eval, seed=0,
    )
    train_sets = {
        c: build_train_set(
            idx, vocab, capability=c, template=settings.template, seq_len=settings.seq_len,
            n_train=settings.n_train, seed=0, n_eval=settings.n_eval,
        )
        for c in caps
    }

    # ---- 基准点（两个，见 schema.PipelineResult 的字段说明）
    ref = baselines(idx, train_sets, eval_sets)
    chance_accs = {c: float(ref[c]["chance"]) for c in caps}
    base_accs, base_eval_seconds = _evaluate_base(
        model_spec, base_weights, caps, eval_sets, cost_cfg, settings
    )

    # ---- 扇出
    cells = sweep_cells(cfg)
    alpha_mult = float((cfg.get("sweep") or {}).get("alpha_mult", 2.0))
    result = PipelineResult(
        cfg=dict(cfg),
        models=[base_entry],
        base_accs=base_accs,
        chance_accs=chance_accs,
        protocol_digest=protocol_digest(cfg),
        version_digest=version_digest(
            cfg,
            [
                ("model", model_spec.to_dict()),
                ("dataset", idx.to_dict()),
                ("train", train_cfg),
                ("cost", cost_cfg),
            ],
        ),
    )

    costs: dict[str, dict[str, Any]] = {}
    macs: dict[str, Any] = {}
    checks: dict[str, Any] = {"merge": {}, "gradients": {}, "macs": {}}
    timings: dict[str, float] = {}

    for cell in cells:
        spec = _make_spec(
            model_spec, cell, settings,
            alpha_mult=alpha_mult,
            seed=int(train_cfg.get("seed", 0)),
        )
        sid = spec.spec_id
        if sid in result.specs:
            raise ValueError(
                f"两个格子得到了同一个 spec_id={sid}（label={cell['label']}）。"
                "它们实际上是同一份配置 —— 后一份会静默覆盖前一份。"
            )
        plan = build_plan(spec, base_weights, cost=cost_cfg)
        capability = spec.dataset.capability

        if progress is not None:
            progress(sid, cell["label"], int(cell["index"]), len(cells))

        # 结构核对：每个规格都做，因为它是「计划说的是不是真的」的**唯一**依据。
        if cfg.get("acceptance", {}).get("require_cost_three_way_agree", True):
            checks["macs"][sid] = compare.check_macs(plan, seq_len=settings.seq_len)

        t0 = time.perf_counter()
        run = train(
            spec, plan, train_sets[capability],
            settings=settings,
            probe=lambda _st, pl: evaluate_target_only(
                pl, spec, eval_sets[capability], batch_size=32
            ),
        )
        run_seconds = time.perf_counter() - t0
        run.train_seconds = run_seconds
        timings[sid] = run_seconds

        ev = evaluate(plan, spec, eval_sets, batch_size=32)

        # 合并核验与梯度核对：都在**训练之后**做 —— 那时适配器已经不是全零，
        # 于是「B=0 时两边都是 0」这种退化解（差分与解析都是 0，天然相等）
        # 不可能让检查静默通过。
        merge = compare.verify_merge(
            plan, tolerance=tol, seq_len=settings.seq_len, seed=spec.seed
        )
        checks["merge"][sid] = merge
        if cfg.get("acceptance", {}).get("require_cost_three_way_agree", True):
            # ② 必须取 **一次单样本前向** 的实测值，也就是 `check_macs` 刚算出来的那个。
            #
            # 这里曾经取的是 `run.macs["total"]` —— 那是**整个训练跑**累加出来的 MAC，
            # 而 ① 是单样本一次前向。两者同名（都叫 macs）但差着 steps×batch 倍，
            # 于是这个核对**对每一个 spec 都失败**，而失败的原因不是有任何一处算错，
            # 是拿两把尺子在量同一根棍子。教训：同名不等于同量；这一处如果
            # `steps×batch` 恰为 1，它会静默通过，而那时它同样什么都没验证。
            checks.setdefault("three_way", {})[sid] = compare.verify_cost_three_way(
                plan,
                seq_len=settings.seq_len,
                batch=settings.batch,
                steps=settings.steps,
                effective_batch=settings.effective_batch,
                measured_macs=int(checks["macs"][sid]["measured_total"]),
            )
        checks["gradients"][sid] = compare.check_gradients(
            plan, example=train_sets[capability][0]
        )

        payload, payload_meta = _payload_bytes(plan)
        n_artifact = plan.artifact_bytes()
        art = Artifact(
            spec_id=sid,
            kind=spec.finetuning.kind,
            n_trainable_params=plan.n_trainable_params,
            n_bytes=n_artifact,
            digest=_digest(payload),
            payload_bytes=len(payload),
            merge_max_abs_diff=merge["max_abs_diff"],
            quant_error=float(plan.quant.rel_error) if plan.quant is not None else 0.0,
            meta={
                "label": cell["label"],
                "rank": int(spec.finetuning.rank),
                "alpha": float(spec.finetuning.alpha),
                "target_modules": list(spec.finetuning.target_modules),
                "quant_bits": int(spec.finetuning.quant_bits),
                "quant_group": int(spec.finetuning.quant_group),
                "objective": spec.objective,
                "capability": capability,
                "payload": payload_meta,
                "merge": merge,
                "n_updated_params": run.n_updated_params,
                "train_seconds": run_seconds,
            },
        )
        if art.payload_bytes != n_artifact:
            raise ValueError(
                f"产物字节数与算术值不等：实测 {art.payload_bytes} vs 声明 {n_artifact}（{sid}）。"
                "这条等式是整个项目里唯一一处「声明值 vs 实测值」的强核对，它不成立就不能发报告。"
            )

        result.specs[sid] = spec
        result.runs[sid] = run
        result.evals[sid] = ev
        result.artifacts[sid] = art
        result.payloads[sid] = payload
        costs[sid] = plan.cost(seq_len=settings.seq_len, batch=settings.batch)
        macs[sid] = dict(run.macs or {})

    # ---- 收拢：验收核对
    verdicts_checks = {
        "ir_roundtrip": compare.check_ir_roundtrip(result.specs.values()),
        "single_call_path": compare.check_single_call_path(root),
        "merge": _summarize_merge(checks["merge"], tol),
        "macs": _summarize_macs(checks["macs"]),
        "three_way": _summarize_three_way(checks.get("three_way", {})),
        "gradients": _summarize_gradients(checks["gradients"]),
        "quant_error": compare.verify_quant_error_decreasing(result.artifacts),
        "artifact_bytes": _summarize_artifact_bytes(result.artifacts),
    }

    # `counters` 必须在**取点之前**落到 `result` 上。
    #
    # `compare.objective_values(result, sid)` 是从 `result.counters["costs"]` 里读
    # vram 的。之前这里的顺序是「先取点、后赋 counters」，于是取点那一刻
    # `result.counters` 还是空的，`_cost_of` 对每个 spec 都返回 None，
    # `vram_bytes` 全变 NaN —— 而 `pareto_frontier` 遇到 NaN 是**按设计跳过**的。
    # 结果是：64 个（或 21 个）点一个不剩地进了 `skipped_nan`，前沿恒为空，
    # 报告照样生成得出来，只是「前沿」那一节永远是空的。
    #
    # 这正是本项目最警惕的那种失败：**没有异常、没有 NaN 泄漏到报告里、
    # 每一张表都长得像样**，只是收拢那一半整个没在算。所以顺序在这里是实质的，
    # 不是风格问题：先让 result 自洽，再问它问题。
    result.counters = {
        "costs": costs,
        "macs": macs,
        "baselines": {c: dict(ref[c]) for c in caps},
        "checks": verdicts_checks,
        # `skipped` 与 `pareto_stats` 先占位，下面拿到前贤统计后回填 ——
        # 键必须在，因为报告和 CLI 都会无条件读它们。
        "skipped": {"pareto": []},
        "pareto_stats": {},
        "timings": {"per_spec": timings, "base_eval": base_eval_seconds, "total": 0.0},
        "n_specs": len(result.specs),
        "n_cells": len(cells),
    }

    # ---- 收拢：前沿
    points = {sid: compare.objective_values(result, sid) for sid in result.specs}
    front, front_stats = compare.pareto_frontier(points)
    result.frontier = [result.specs[sid] for sid in front]
    result.frontier_keys = compare.frontier_keys(points, front)
    result.counters["pareto_stats"] = front_stats
    result.counters["skipped"] = {
        "pareto": list(front_stats.get("skipped_nan", [])) + list(front_stats.get("skipped_missing", []))
    }

    # 声明与裁决一起落盘。分开存是因为**「声明了但没裁决」本身是一条结论**：
    # 只存裁决的话，那条被漏掉的期望在报告里就是「不存在」，而不是「缺失」。
    result.expectations = expectations_of(cfg)
    result.verdicts = compare.judge_expectations(result, cfg)
    result.dag = build_lineage(result)

    result.counters["timings"]["total"] = time.perf_counter() - t_start
    return result


def _make_spec(
    model_spec: ModelSpec,
    cell: Mapping[str, Any],
    settings: TrainSettings,
    *,
    alpha_mult: float,
    seed: int,
):
    """从一格展开出一份 :class:`TrainingSpec`。

    **每个子 Spec 都由 `settings` 显式给出，不靠 `make_spec` 的默认值。**
    靠默认值的话，「报告里的数字」就会依赖于一个不在配置里的数 ——
    而 `TrainSettings` 与 IR 各自的默认值**是两份**（一个是 dataclass 字段，
    一个是 `DatasetSpec`/`OptimizerSpec` 的字段），它们今天恰好一致，明天不一定。
    """
    finetuning = _strategy_from_cell(cell, n_layers=model_spec.n_layers, alpha_mult=alpha_mult)
    return make_spec(
        model_spec,
        {
            "capability": cell["capability"],
            "template": settings.template,
            "seq_len": settings.seq_len,
            "n_train": settings.n_train,
            "n_eval": settings.n_eval,
            "seed": seed,
        },
        finetuning=finetuning,
        objective=str(cell["objective"]),
        optimizer={
            "name": settings.optimizer,
            "lr": settings.lr,
            "weight_decay": settings.weight_decay,
        },
        scheduler={
            "name": settings.scheduler,
            "warmup_steps": settings.warmup_steps,
            "total_steps": settings.steps,
        },
        batch={"size": settings.batch, "grad_accum": settings.grad_accum},
        evaluation={"every_n_steps": settings.eval_every, "batch_size": 32},
        checkpoint={"every_n_steps": settings.checkpoint_every, "keep_last": 6},
        seed=seed,
    )


def _evaluate_base(
    model_spec: ModelSpec,
    base_weights: Mapping[str, Any],
    caps: Sequence[str],
    eval_sets: Mapping[str, Sequence[Example]],
    cost_cfg: Mapping[str, Any],
    settings: TrainSettings,
) -> tuple[dict[str, float], float]:
    """微调前那个模型在**全部**能力上的实测值 —— gain 的基准点。

    它必须是实测而不是解析的：随机初始化的模型在每项能力上的准确率**都不等于
    chance**（实测 upper 0.042 / sort 0.073，而 chance 是 0.125），因为
    未训练模型的 argmax 被「哪个 token 的 unembedding 范数最大」主导，
    不是均匀分布。「用 chance 当基准」因此在数据上是**错的**，
    而它看起来只是保守一点。
    """
    t0 = time.perf_counter()
    spec = make_spec(
        model_spec,
        {
            "capability": caps[0],
            "template": settings.template,
            "seq_len": settings.seq_len,
            "n_train": settings.n_train,
            "n_eval": settings.n_eval,
            "seed": 0,
        },
        finetuning=FinetuneStrategy.canonical("full"),
        objective="sft",
        seed=0,
    )
    # 不训练，直接评 —— 这就是「Base」那一行。
    plan = build_plan(spec, base_weights, cost=cost_cfg)
    ev = evaluate(plan, spec, eval_sets, batch_size=32)
    return {c: float(ev.per_capability.get(c, float("nan"))) for c in caps}, time.perf_counter() - t0


def _digest(payload: bytes) -> str:
    from .rng import sha256_bytes

    return sha256_bytes(payload)


# ----------------------------------------------------------------------
# 验收汇总
# ----------------------------------------------------------------------
def _summarize_merge(per_spec: Mapping[str, Mapping[str, Any]], tol: float) -> dict[str, Any]:
    applicable = {k: v for k, v in per_spec.items() if v.get("applicable")}
    bad = {k: v for k, v in applicable.items() if not v.get("ok")}
    identical = sorted(k for k, v in applicable.items() if v.get("identical"))
    diffs = [float(v["max_abs_diff"]) for v in applicable.values() if v.get("max_abs_diff") is not None]
    return {
        "n_applicable": len(applicable),
        "n_not_applicable": len(per_spec) - len(applicable),
        "max_abs_diff": max(diffs) if diffs else None,
        "tolerance": tol,
        "failures": sorted(bad),
        # 「恰好为 0」被单独报出来：两份独立实现给出逐位相同的结果，
        # 通常意味着其中一份被换成了另一份（oracle 复用），而不是完美。
        "identical_suspect": identical,
        "ok": not bad and not identical,
    }


def _summarize_macs(per_spec: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    bad = {k: v["mismatched_ops"] for k, v in per_spec.items() if not v.get("ok")}
    return {"n_checked": len(per_spec), "mismatches": bad, "ok": not bad}


def _summarize_three_way(per_spec: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    bad = {k: v for k, v in per_spec.items() if not v.get("ok")}
    return {"n_checked": len(per_spec), "failures": sorted(bad), "ok": not bad}


def _summarize_gradients(per_spec: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    applicable = {k: v for k, v in per_spec.items() if v.get("applicable")}
    bad = {k: v.get("max_rel_error") for k, v in applicable.items() if not v.get("ok")}
    worst = max((float(v["max_rel_error"]) for v in applicable.values()
                 if v.get("max_rel_error") is not None), default=None)
    return {
        "n_applicable": len(applicable),
        "worst_rel_error": worst,
        "failures": bad,
        "ok": not bad,
    }


def _summarize_artifact_bytes(artifacts: Mapping[str, Artifact]) -> dict[str, Any]:
    bad = {
        sid: {"declared": a.n_bytes, "measured": a.payload_bytes}
        for sid, a in artifacts.items()
        if a.n_bytes != a.payload_bytes
    }
    return {"n_checked": len(artifacts), "mismatches": bad, "ok": not bad}


# ----------------------------------------------------------------------
# 输出
# ----------------------------------------------------------------------
def write_outputs(result: PipelineResult, out: Path | str) -> dict[str, Path]:
    """落盘。返回写出来的路径集合。

    产出五样东西：

    - ``result.json`` —— 摘要（Spec、前沿、裁决、验收）。**不含**产物字节；
    - ``runs.jsonl`` —— 每行一次训练运行。行式而不是一个大数组：
      64 行里加一行 / 删一行，diff 只动一行；
    - ``evals.jsonl`` —— 每行一次评测。**报告的 gain / interference 两列靠它**；
    - ``artifacts/<spec_id>.bin`` + ``.json`` —— 裸 float32 + 旁证元数据。
      写完**必须核对文件大小**与声明值逐位相等；
    - ``lineage.json`` —— 血缘 DAG。
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    adir = out / ARTIFACT_DIR
    adir.mkdir(exist_ok=True)
    written: dict[str, Path] = {}

    summary = result.to_dict()
    summary["counters"] = _jsonable(result.counters)
    summary["dag"] = result.dag.to_dict() if result.dag is not None else None
    p = out / "result.json"
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    written["result"] = p

    p = out / "runs.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for sid in sorted(result.runs):
            fh.write(json.dumps(result.runs[sid].to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    written["runs"] = p

    # `evals.jsonl` 和 `runs.jsonl` 分开，理由与 L1 那边分开 `runs` / `artifacts` 一样：
    # 它们是**两种**按规格增长的东西，合成一个文件之后，「只读评测」要先解出全部运行记录。
    #
    # 而它**必须落盘**：`result.json` 是刻意只放摘要的（`to_dict` 里没有 evals），
    # 于是没有这个文件的话，`load_result` 拼出来的对象上 `gain_of` / `interference_of`
    # 会对每一个 spec 都返回 NaN —— 报告的对比矩阵会整个变成「未知」，
    # 而报告照样生成得出来、每一节都在，只是没有任何一行有数字。
    p = out / "evals.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for sid in sorted(result.evals):
            fh.write(json.dumps(result.evals[sid].to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    written["evals"] = p

    for sid, art in sorted(result.artifacts.items()):
        payload = result.payloads.get(sid)
        if payload is None:
            raise ValueError(
                f"产物 {sid} 的字节不在内存里（payloads 缺失）。"
                "write_outputs 需要 run_pipeline 的同一次调用结果 —— "
                "从 result.json 载入的摘要**不含**产物字节，那是刻意的（JSON 装不下 bytes）。"
            )
        bp = adir / f"{sid}.bin"
        bp.write_bytes(payload)
        actual = bp.stat().st_size
        if actual != art.payload_bytes:
            raise ValueError(
                f"落盘字节数 {actual} 与声明 {art.payload_bytes} 不等（{sid}）。"
                "这条等式不成立说明产物里混进了头部 / 对齐 / 格式化开销。"
            )
        meta = art.to_dict()
        meta["file"] = bp.name
        (adir / f"{sid}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        written[f"artifact:{sid}"] = bp

    lp = out / "lineage.json"
    dumps_lineage(result.dag, lp)
    written["lineage"] = lp
    return written


def _jsonable(value: Any) -> Any:
    """把 counters 里的非 JSON 类型收敛掉。**遇到不认识的类型报错而不是转 str** ——
    转 str 会让一个本该是数的字段在产物里变成字符串，而读它的人只会看到引号。"""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"产物里有不能序列化的类型 {type(value).__name__}：{value!r}")


def load_result(out: Path | str) -> PipelineResult:
    """载入一份 ``result.json``（**只载摘要**，快，且不依赖任何旁证）。

    载回来的结果**没有** `evals` / `runs` / `artifacts`（那些在
    ``runs.jsonl`` / ``evals.jsonl`` / ``artifacts/*.json`` 里），也**没有**
    `payloads`（JSON 装不下 bytes）。所以 ``gain_of`` / ``interference_of`` 这类查询
    在载入的对象上会返回 NaN —— 报告里因此把「未知」和「0」分得开。

    要一个能渲染完整报告的载入器，用 :func:`load_full`。两者的差别是**缺失的语义**：
    这里缺的是「旁证」，那里缺的是「结论」。

    `dag` 是例外：`lineage.json` 是一份**自足**的摘要（节点 + 边），
    载回来就是完整的，所以这里顺手接上。不接的话，`result.dag` 是 None ——
    而 None 和「图是空的」在调用方看来长得一样，CLI 的 `dag` 子命令就得
    再写一条只有它自己走的载入路径。
    """
    out = Path(out)
    p = out / "result.json"
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p}")
    d = json.loads(p.read_text(encoding="utf-8"))
    result = PipelineResult.from_dict(d)
    result.cfg = {}
    result.dag = load_lineage(out)
    return result


def load_lineage(out: Path | str) -> Any:
    """读 ``lineage.json``。**文件不在就返回 None**，不抛错。

    血缘图是**旁证**而不是主产物：一份早期的输出目录里可能没有它。
    因为「图没写出来」就让整个 `result.json` 读不进来，是把旁证的缺失
    升级成了主结论的失败。
    """
    from .lineage import Lineage

    p = Path(out) / "lineage.json"
    if not p.exists():
        return None
    return Lineage.from_dict(json.loads(p.read_text(encoding="utf-8")))


def load_runs(out: Path | str) -> dict[str, Any]:
    """读 ``runs.jsonl``。它和 ``result.json`` 分开，是因为它的行数按规格数增长。"""
    from .schema import RunRecord

    return _load_jsonl(out, "runs.jsonl", RunRecord)


def load_evals(out: Path | str) -> dict[str, Any]:
    """读 ``evals.jsonl``。**报告里 gain / interference 两列的唯一来源。**

    没有它，``result.json`` 里剩下的只是「跑了哪些规格」，
    「跑成什么样」一行都没有 —— 而 ``gain`` 缺失时的默认表现是空列或 NaN，
    两者看起来都像「训练没用」，不像「数据没载入」。
    """
    from .schema import EvalResult

    return _load_jsonl(out, "evals.jsonl", EvalResult)


def _load_jsonl(out: Path | str, name: str, cls: Any) -> dict[str, Any]:
    p = Path(out)
    if p.is_dir():
        p = p / name
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p}")
    recs: dict[str, Any] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = cls.from_dict(json.loads(line))
        recs[rec.spec_id] = rec
    return recs


def load_artifacts(out: Path | str) -> dict[str, Artifact]:
    """读 ``artifacts/*.json``（**只读元数据，不读 .bin**）。

    产品字节在 ``.bin`` 里，而报告只需要「多大 / 摘要是什么 / 合并差多少 /
    量化误差多少」—— 那四件事全在 JSON 里。把 21 份 float32 读进内存来做
    一张表，是把「看报告」变成一次昂贵的副作用。
    """
    adir = Path(out) / ARTIFACT_DIR
    if not adir.is_dir():
        return {}
    arts: dict[str, Artifact] = {}
    for p in sorted(adir.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        d.pop("file", None)  # 展示用的字段，不在 Artifact 的字段表里
        art = Artifact.from_dict(d)
        arts[art.spec_id] = art
    return arts


def load_full(out: Path | str) -> PipelineResult:
    """载入一个**完整**的结果目录：result.json + runs + evals + artifacts + lineage。

    这才是只读子命令该用的载入器，而 :func:`load_result` 只读摘要（它快、它不需要
    任何旁证、它在旁证缺失时仍然能工作）。

    两者分开而不是合并成一个「能读的都读」的函数，是因为**缺失的语义不同**：

    - ``result.json`` 不在 ⇒ 这个目录根本不是一次运行的结果，报错；
    - ``runs.jsonl`` / ``evals.jsonl`` 不在 ⇒ 这是一份早期输出，**降级**：
      报告里那两列写「未知」，其余照常。因为缺了它们就说「读不进来」，
      是把旁证的缺失升级成了主结论的失败。

    ``payloads`` 永远是空的 —— 载入路径**不假装**自己能重新落盘。真去调
    :func:`write_outputs` 会拿到一条明确的错误，而不是一堆 0 字节的产物。
    """
    result = load_result(out)
    for name, loader in (("runs", load_runs), ("evals", load_evals)):
        try:
            setattr(result, name, loader(out))
        except FileNotFoundError:
            setattr(result, name, {})
    result.artifacts = load_artifacts(out)
    return result


def load_artifact(out: Path | str, spec_id: str) -> dict[str, Any]:
    """读一个产物的元数据 JSON（含 ``file`` / ``names`` / ``shapes`` / ``offsets``）。"""
    p = Path(out) / ARTIFACT_DIR / f"{spec_id}.json"
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def iter_artifact_specs(out: Path | str) -> Iterable[str]:
    """列出已落盘的产物 spec_id。"""
    adir = Path(out) / ARTIFACT_DIR
    if not adir.exists():
        return ()
    return tuple(sorted(p.stem for p in adir.glob("*.bin")))
