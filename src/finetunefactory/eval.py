"""评测：目标能力涨了多少、别的能力掉了多少。

它跑**全部**能力，不只是被微调的那一个。这是「干扰」这个词有意义的唯一条件：
如果评测只跑目标能力，那么「LoRA 的干扰比 full 小」这句话在数据上根本无法出现。

### 三个口径，必须分清

同一个模型在同一个能力上可以算出三个不同的数，本模块三个都给，**各有各的字段名**：

- ``per_capability[cap]``：**回答内容上的逐 token 准确率**。主指标，
  用于 gain / interference / Pareto。平滑，且正好是训练优化量的离散对应物。
- ``meta["exact_match"][cap]``：**整条回答全对**的比率。接近人的直觉，
  但 4 个 token 全对是 all-or-nothing，小模型会长期停在 0，用它做 Pareto 会
  得到一条几乎全是 0 的前沿。
- ``meta["eos_acc"]`` / ``meta["prompt_acc"]``：EOS 位置与 prompt 位置上的准确率。
  EOS 必然接近 1（它是个确定性 token），prompt 位置测的是「模型有没有记住
  数据的框架」。**这两个数都不进主指标** —— 把它们混进去会让所有能力的准确率
  被同一个常数抬高，于是「涨了多少」这个差里混进一个与能力无关的偏移。

评测**永远走 :func:`model.forward`**，与训练同一条路径、同一份 ``plan.build()``。
评测另写一条「纯数值前向」是这类项目里最贵的省事：两条路径一旦漂移，
报告里的准确率就不再是被训练出来的那个模型的表现，而没有任何东西会报错。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .data import Example, content_positions, response_positions
from .model import forward
from .plan import ParamPlan
from .schema import EvalResult

__all__ = ["evaluate", "evaluate_capability", "token_accuracy"]


def token_accuracy(
    logits: Sequence[Sequence[float]],
    ex: Example,
    *,
    positions: Sequence[int] | None = None,
) -> tuple[int, int]:
    """在给定位置上数正确的 token。返回 ``(命中, 总数)``。

    位置的口径由 `positions` 决定，默认用 :func:`content_positions`（回答内容、**不含 EOS**）。
    把「哪些位置算数」做成参数而不是写死，是因为本项目要同时汇报内容准确率、
    EOS 命中与 prompt 准确率 —— 三者在同一批 logits 上用不同的位置集合算出来。
    """
    if positions is None:
        positions = content_positions(ex)
    ok = 0
    for t in positions:
        tgt = ex.targets[t]
        if tgt < 0:
            continue
        row = logits[t]
        # argmax 平局取小的 id（index 返回首个命中）—— 与 data._argmax 的约定一致。
        best = row.index(max(row))
        if best == tgt:
            ok += 1
    return ok, len(positions)


def _prompt_positions(ex: Example) -> list[int]:
    """prompt 上的监督位置（回答之前的那些）。用于 `prompt_acc`，**不进主指标**。"""
    return [t for t in range(len(ex.targets)) if ex.mask_all[t] > 0 and t + 1 < ex.response_start]


def evaluate_capability(
    plan: ParamPlan,
    spec: Any,
    examples: Sequence[Example],
    capability: str,
    *,
    batch_size: int = 32,
) -> tuple[dict[str, float], dict[str, Any]]:
    """在一个能力上评一次，返回 ``(主指标, 明细)``。

    `batch_size` 在这里**只是分块调度，对数值没有任何影响**，这是刻意的：
    每条样本独立前向，块与块之间没有共享状态。如果为了吞吐把多条样本拼成一条长序列，
    注意力会跨样本泄漏，准确率就会依赖 batch 的组成 —— 于是「换了 batch_size
    数字变了」这种事会静默发生，而它在报告里看起来完全正常。
    `ir.EvalSpec.batch_size` 因此被解释成「一次调度多少条」，而不是
    「一次前向多少条」。报告的口径一节会把这一条写出来。
    """
    n_ok = 0
    n_tot = 0
    n_exact = 0
    n_eos = 0
    n_prompt_ok = 0
    n_prompt_tot = 0
    losses: list[float] = []
    step = max(1, int(batch_size))
    for start in range(0, len(examples), step):
        # `build()` 提到内层循环**外面**。它在评测期间是常量（没有优化器在动叶子），
        # 而 LoRA 的 `build()` 每个挂载点都要造一张 2D 中间张量 —— 在 96 次前向里
        # 重复造 96 遍纯属浪费。提到外面还有一层好处：它把「评测期间权重不变」
        # 这件事写进了控制流，读的人不必再去确认叶子有没有被改。
        weights = plan.build()
        for ex in examples[start : start + step]:
            logits = forward(weights, ex.ids, spec.model).data
            pos = content_positions(ex)
            ok, tot = token_accuracy(logits, ex, positions=pos)
            n_ok += ok
            n_tot += tot
            if tot and ok == tot:
                n_exact += 1
            if ex.eos_t >= 0:
                ok_e, _ = token_accuracy(logits, ex, positions=[ex.eos_t])
                n_eos += ok_e
            pp = _prompt_positions(ex)
            if pp:
                ok_p, _ = token_accuracy(logits, ex, positions=pp)
                n_prompt_ok += ok_p
                n_prompt_tot += len(pp)
            losses.append(_masked_ce(logits, ex))
    detail = {
        "n_eval": len(examples),
        "n_positions": n_tot,
        "n_correct": n_ok,
        "exact_match": n_exact / len(examples) if examples else 0.0,
        "eos_acc": n_eos / len(examples) if examples else 0.0,
        "prompt_acc": n_prompt_ok / n_prompt_tot if n_prompt_tot else 0.0,
        "loss": sum(losses) / len(losses) if losses else float("nan"),
        "chunk": step,
    }
    acc = n_ok / n_tot if n_tot else 0.0
    return {"acc": acc}, detail


def _masked_ce(logits: Sequence[Sequence[float]], ex: Example) -> float:
    """一条样本在**回答掩码**下的交叉熵。它必须与训练时的损失同口径
    （同一个掩码、同样的「除以 sum(mask)」），否则「训练损失降到 x 但评测损失是 y」
    这种对照就没法做。"""
    tot = 0.0
    denom = 0.0
    for t in response_positions(ex):
        row = logits[t]
        m = max(row)
        ex_ = [math.exp(v - m) for v in row]
        s = sum(ex_)
        tot += -math.log(max(ex_[ex.targets[t]] / s, 1e-30))
        denom += 1.0
    return tot / denom if denom > 0 else float("nan")


def evaluate(
    plan: ParamPlan,
    spec: Any,
    eval_sets: Mapping[str, Sequence[Example]],
    *,
    batch_size: int = 32,
) -> EvalResult:
    """在**全部**能力上评一次，返回 :class:`~finetunefactory.schema.EvalResult`。

    主指标是目标能力上的内容准确率；`per_capability` 里含全部能力，
    于是 :meth:`EvalResult.interference` 才有东西可算。

    **全部能力的评测集大小必须一致**（同一个 `n_eval`）。不一致的话，
    各能力的准确率方差不同，而 interference 是它们的**简单平均** ——
    方差大的那一个会主导这个平均，于是「干扰」这个数主要反映的是采样噪声。
    配置校验保证 `train.n_eval` 是唯一的，所以这一点在数据侧就成立了。
    """
    per_capability: dict[str, float] = {}
    detail: dict[str, Any] = {"exact_match": {}, "eos_acc": {}, "prompt_acc": {}, "loss": {}}
    target_n_correct = 0
    target_n_positions = 0
    target = spec.dataset.capability
    for cap in sorted(eval_sets):
        main, d = evaluate_capability(plan, spec, eval_sets[cap], cap, batch_size=batch_size)
        per_capability[cap] = main["acc"]
        detail["exact_match"][cap] = d["exact_match"]
        detail["eos_acc"][cap] = d["eos_acc"]
        detail["prompt_acc"][cap] = d["prompt_acc"]
        detail["loss"][cap] = d["loss"]
        if cap == target:
            target_n_correct = int(d["n_correct"])
            target_n_positions = int(d["n_positions"])
    return EvalResult(
        spec_id=spec.spec_id,
        capability=target,
        per_capability=per_capability,
        n_eval=len(eval_sets.get(target, ())),
        n_correct=target_n_correct,
        meta={
            "objective": spec.objective,
            "template": spec.dataset.template,
            "seq_len": spec.dataset.seq_len,
            "kind": spec.finetuning.kind,
            "metric": "content_token_accuracy",
            "metric_note": "回答内容位置上的逐 token 准确率；EOS 与 prompt 位置不计入",
            "target_n_positions": target_n_positions,
            "n_eval_per_capability": {c: len(v) for c, v in sorted(eval_sets.items())},
            **detail,
        },
    )


def evaluate_target_only(
    plan: ParamPlan, spec: Any, examples: Sequence[Example], *, batch_size: int = 32
) -> float:
    """只评目标能力，返回准确率。**供训练中途的 `probe` 用**（检查点曲线）。

    它故意比 :func:`evaluate` 便宜得多：检查点每 24 步落一次，全程评四个能力
    会让评测开销超过训练本身。曲线只需要知道「在目标能力上有没有在动」。
    """
    main, _ = evaluate_capability(plan, spec, examples, spec.dataset.capability, batch_size=batch_size)
    return main["acc"]
