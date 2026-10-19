"""训练引擎：**一段代码跑四种策略**。

这个模块里没有 ``if kind ==``。它拿到的是一张
:class:`~finetunefactory.plan.ParamPlan`，而计划已经决定了「哪些张量带梯度」。
于是四种策略走的是同一段循环、同一段前向、同一个损失、同一个优化器步：

    for step in range(steps):
        weights = plan.build()              # 计划决定这一步的权重长什么样
        loss    = batch_cross_entropy([...forward(w, ids) for ...])
        loss.backward()
        optimizer.step(plan.trainable_leaves())

**这条性质必须是可检验的，而不是一句口号。** 它的检验方式是
:meth:`~finetunefactory.plan.ParamPlan.self_check`（可训练集合与真实叶子一致）
加上 ``grep -n 'kind ==' train.py`` 返回空。后者听起来像个玩笑，
但它是这个项目里唯一能防止「某天有人为了赶进度在 train 里加一个特判」的东西 ——
那一刻起，「同一份 Spec 四种产物」就不再是同一个实验的四个格子了。

### 三个反直觉的地方

1. **损失一律取平均，绝不取和**（skill 坑 48）。取和会让梯度范数正比于 batch
   大小，于是「改 batch 大小」等效于「改学习率」，而症状是 loss 曲线看起来
   完全正常。
2. **每个叶子在每一步前必须显式 `zero_grad`**。引擎是累加进 `.grad` 的
   （这是让梯度累积能成立的前提），漏掉清零不会报错，只会让梯度越滚越大，
   表现为「跑几十步之后突然发散」。
3. **`train_flops` 是估算，`macs` 是实测**。前者按「反向 ≈ 2× 前向」的通用系数
   外推，后者是计数器真的数出来的。报告里两列必须分开标，混在一起会让
   「实测」这个说法失去意义。
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .autograd import Tensor, batch_cross_entropy, mac_counter
from .data import Example, batch_of
from .ir import TrainingSpec
from .model import forward
from .plan import ParamPlan
from .rng import derive_seed
from .schema import CheckpointRecord, RunRecord

__all__ = [
    "TrainSettings",
    "AdamW",
    "SGD",
    "make_optimizer",
    "clip_grad_norm_",
    "train",
    "batch_order",
]


@dataclass(frozen=True)
class TrainSettings:
    """`cfg["train"]` 的类型化视图。

    **为什么不让 train 直接读那个 dict**：`tr["lr"]` 与 `tr["LR"]` 在 dict 里
    是两个不同的东西，而后者只会抛 `KeyError`（好）或者在一个深合并过的配置里
    悄悄读到默认值（坏）。字段化之后拼错是 `AttributeError`，而且是 import 期就发现。
    """

    steps: int = 48
    batch: int = 8
    grad_accum: int = 1
    lr: float = 3e-2
    warmup_steps: int = 8
    seq_len: int = 16
    template: str = "plain"
    n_train: int = 160
    n_eval: int = 24
    eval_every: int = 24
    checkpoint_every: int = 24
    optimizer: str = "adamw"
    weight_decay: float = 0.0
    scheduler: str = "cosine"
    max_grad_norm: float = 1.0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "TrainSettings":
        tr = dict(cfg.get("train") or {})
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in tr.items() if k in fields})

    @property
    def effective_batch(self) -> int:
        return self.batch * self.grad_accum


# ----------------------------------------------------------------------
# 优化器
# ----------------------------------------------------------------------
class _Optimizer:
    """优化器基类。**它只认识「叶子列表」，不认识策略。**

    状态按**索引**建，顺序沿用 `plan.trainable_leaves()` 的顺序 ——
    顺序一变，同一份 Spec 的两次运行就会更新到不同的张量上，
    而这个错误不会有任何症状（损失照样降）。
    """

    name = "base"

    def __init__(self, leaves: Sequence[tuple[str, Tensor]], *, lr: float, weight_decay: float = 0.0) -> None:
        self.leaves = list(leaves)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.n_steps = 0

    def state_bytes(self) -> int:
        raise NotImplementedError

    def step(self, lr: float) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        """**每一步都必须调**。引擎是累加进 `.grad` 的（梯度累积就靠这一点），
        漏掉清零不会报错，只会让梯度越滚越大，表现为「跑几十步之后突然发散」。"""
        for _name, t in self.leaves:
            t.grad = None


class AdamW(_Optimizer):
    """Adam + 解耦权重衰减。

    **权重衰减是解耦的**（`p ← p − lr·wd·p`），不是加进梯度里。这两者的区别在
    有自适应缩放时是实质性的：加进梯度会被 `1/sqrt(v)` 缩放，于是「同一份 wd」
    对不同量级的参数效果不同。报告里如果写的是 AdamW，那它就必须是解耦的那一个。

    状态是每个参数两份（一阶矩 `m` 与二阶矩 `v`），于是 ``state_multiplier == 2`` ——
    这个「2」是 `ir.OptimizerSpec.state_multiplier` 里那个 2 的**实现侧对应物**，
    两处必须一致（显存表就是按它算的）。
    """

    name = "adamw"

    def __init__(
        self,
        leaves: Sequence[tuple[str, Tensor]],
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(leaves, lr=lr, weight_decay=weight_decay)
        self.b1, self.b2 = betas
        self.eps = float(eps)
        self.m: list[Any] = [_zeros_like(t.data) for _n, t in self.leaves]
        self.v: list[Any] = [_zeros_like(t.data) for _n, t in self.leaves]

    def state_bytes(self) -> int:
        return 2 * 4 * sum(_numel(t.data) for _n, t in self.leaves)

    def step(self, lr: float) -> None:
        self.n_steps += 1
        t_ = self.n_steps
        bc1 = 1.0 - self.b1**t_
        bc2 = 1.0 - self.b2**t_
        for idx, (_name, leaf) in enumerate(self.leaves):
            if leaf.grad is None:
                continue
            m, v = self.m[idx], self.v[idx]
            g = leaf.grad
            wd = self.weight_decay

            def upd(p: float, gi: float, i: int, j: int) -> float:
                key = (i, j)
                mi = m[key]
                vi = v[key]
                mi = self.b1 * mi + (1.0 - self.b1) * gi
                vi = self.b2 * vi + (1.0 - self.b2) * gi * gi
                m[key] = mi
                v[key] = vi
                mh = mi / bc1
                vh = vi / bc2
                p2 = p - lr * wd * p if wd else p
                return p2 - lr * mh / (math.sqrt(vh) + self.eps)

            _apply(leaf, upd)


class SGD(_Optimizer):
    """带动量的 SGD。它存在的唯一理由是让「优化器」这条轴不至于是硬编码的。"""

    name = "sgd"

    def __init__(
        self,
        leaves: Sequence[tuple[str, Tensor]],
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(leaves, lr=lr, weight_decay=weight_decay)
        self.momentum = betas[0]
        self.buf: list[Any] = [_zeros_like(t.data) for _n, t in self.leaves]

    def state_bytes(self) -> int:
        return 4 * sum(_numel(t.data) for _n, t in self.leaves)

    def step(self, lr: float) -> None:
        self.n_steps += 1
        for idx, (_name, leaf) in enumerate(self.leaves):
            if leaf.grad is None:
                continue
            buf = self.buf[idx]
            wd = self.weight_decay

            def upd(p: float, gi: float, i: int, j: int) -> float:
                key = (i, j)
                b = self.momentum * buf[key] + gi
                buf[key] = b
                return p - lr * (b + wd * p if wd else b)

            _apply(leaf, upd)


def _apply(leaf: Tensor, fn: Callable[[float, float, int, int], float]) -> None:
    """把 `fn(值, 梯度, i, j)` 逐元素作用到叶子上。**1D 用 j 作下标、i 恒为 0。**

    1D 与 2D 共用一个回调签名，是为了让「同一个优化器两种形状都能跑」这件事
    只写一遍。代价是 1D 那边必须固定 i=0（否则 2D 的 (i,j) 键会与 1D 的 j 撞上）。
    """
    g = leaf.grad
    d = leaf.data
    if d and isinstance(d[0], list):
        if g is None:
            return
        for i, row in enumerate(d):
            gi = g[i]
            for j in range(len(row)):
                row[j] = fn(row[j], gi[j], i, j)
    else:
        for j in range(len(d)):
            d[j] = fn(d[j], 0.0 if g is None else g[j], 0, j)


def _numel(data: Any) -> int:
    if data and isinstance(data[0], list):
        return len(data) * len(data[0])
    return len(data)


def _zeros_like(data: Any) -> Any:
    """按 ``(i, j)`` 建零状态。**1D 也用 ``(0, j)`` 而不是 ``j``。**

    这个细节看着无关紧要，但两处只要有一处不一致，优化器的 `m` / `v`
    就会在 1D 参数上读到 `KeyError`（好）或者——如果那里恰好用了 `defaultdict`
    ——每次都从 0 开始，于是 `norm_f` 的那 16 个参数永远走的是「第一步」的更新量。
    后者的症状是「归一化的缩放学不动」，而它离原因隔了十万八千里。
    所以 `_apply` 与 `_zeros_like` 必须共用同一个键格式。
    """
    if data and isinstance(data[0], list):
        return {(i, j): 0.0 for i in range(len(data)) for j in range(len(data[0]))}
    return {(0, j): 0.0 for j in range(len(data))}


def make_optimizer(plan: ParamPlan, spec: TrainingSpec, *, lr: float, weight_decay: float = 0.0) -> _Optimizer:
    """按 Spec 造优化器。名字不认识就报错 —— **绝不静默退回一个默认值**。

    静默退回的后果是「Spec 里写的是 sgd，实际跑的是 adamw」，而报告里
    `optimizer: sgd` 那一行看上去很正常。`ir.OPTIMIZERS` 是唯一的名字来源。
    """
    opt = spec.optimizer
    cls = {"adamw": AdamW, "sgd": SGD}.get(opt.name)
    if cls is None:
        raise ValueError(f"未知优化器 {opt.name!r}（可选：{sorted(('adamw', 'sgd'))}）")
    return cls(
        plan.trainable_leaves(),
        lr=lr,
        betas=opt.betas,
        eps=opt.eps,
        weight_decay=weight_decay,
    )


def clip_grad_norm_(leaves: Sequence[tuple[str, Tensor]], max_norm: float) -> float:
    """全局范数裁剪，返回裁剪前的范数。

    **返回裁剪前的范数不是为了好看**：它是「训练是否在正常范围」的第一手信号，
    而它必须进 `RunRecord.meta`。只裁剪不记录的话，一个满员被裁的训练过程
    与一个健康的训练过程在报告里长得一模一样 —— 前者的 loss 也会降，
    只是降得比它能达到的慢。`max_norm <= 0` 表示关闭，此时返回 0 并**不做任何事**。
    """
    if max_norm <= 0:
        return 0.0
    total = 0.0
    for _name, leaf in leaves:
        g = leaf.grad
        if g is None:
            continue
        if g and isinstance(g[0], list):
            for row in g:
                for v in row:
                    total += v * v
        else:
            for v in g:
                total += v * v
    norm = math.sqrt(total)
    if norm <= max_norm or norm == 0.0:
        return norm
    factor = max_norm / norm
    for _name, leaf in leaves:
        g = leaf.grad
        if g is None:
            continue
        if g and isinstance(g[0], list):
            for row in g:
                for j in range(len(row)):
                    row[j] *= factor
        else:
            for j in range(len(g)):
                g[j] *= factor
    return norm


# ----------------------------------------------------------------------
# 批的顺序
# ----------------------------------------------------------------------
def batch_order(n: int, batch: int, *, seed: int, steps: int) -> list[list[int]]:
    """确定性的批顺序，**每个 epoch 洗一次牌**。

    用 `derive_seed("batch", seed, epoch)` 而不是一个跨 epoch 连续的随机数流：
    后者在「两次运行步数不同」时会让前面的批完全一样、只在结尾分叉，
    于是「改了 steps 之后前面的数字应该逐位不变」这条直觉成立，
    而它其实反过来说明洗牌没有真的覆盖到全部样本。
    """
    out: list[list[int]] = []
    epoch = 0
    while len(out) < steps:
        rng = random.Random(derive_seed("batch", seed, epoch))
        order = list(range(n))
        rng.shuffle(order)
        for start in range(0, n - batch + 1, batch):
            if len(out) >= steps:
                break
            out.append(order[start : start + batch])
        epoch += 1
    return out


# ----------------------------------------------------------------------
# 训练
# ----------------------------------------------------------------------
def train(
    spec: TrainingSpec,
    plan: ParamPlan,
    examples: Sequence[Example],
    *,
    settings: TrainSettings,
    probe: Callable[[int, ParamPlan], float | None] | None = None,
) -> RunRecord:
    """跑完 `settings.steps` 步，返回 :class:`~finetunefactory.schema.RunRecord`。

    `probe` 是「在某一步上量一下准确率」的回调，由 :mod:`pipeline` 提供
    （它一般是拿当前计划在目标能力上评一次）。**为什么用回调而不是在这里 import
    eval**：训练不该知道「准确率」是怎么算的。它只负责在
    `evaluation.every_n_steps` 的倍数上问一次，并把答回来的数字记进检查点。
    这样「训练在优化什么」与「我们在量什么」是两个可以分别改的口径。

    ⚠️ `probe` **会看到 `plan` 的实时状态**（叶子已经被更新过），所以它量的
    是「这一步的模型」，不是「训练结束后的模型」。这是检查点曲线的意义所在。
    """
    leaves = plan.trainable_leaves()
    if not leaves:
        raise ValueError(f"计划 {spec.spec_id} 没有任何可训练叶子 —— 这份训练什么都不会改")
    optimizer = make_optimizer(plan, spec, lr=settings.lr, weight_decay=settings.weight_decay)
    order = batch_order(len(examples), settings.batch, seed=spec.seed, steps=settings.steps)
    eval_every = max(1, spec.evaluation.every_n_steps)
    ckpt_every = max(1, spec.checkpoint.every_n_steps)

    losses: list[float] = []
    lr_trace: list[float] = []
    checkpoints: list[CheckpointRecord] = []
    grad_norms: list[float] = []
    macs: dict[str, Any] = {}
    t0 = time.perf_counter()

    for step in range(settings.steps):
        indices = order[step]
        optimizer.zero_grad()
        micro_losses: list[float] = []
        n_accum = settings.grad_accum
        weight = 1.0 / n_accum
        for mi in range(n_accum):
            start = mi * settings.batch
            chunk = indices[start : start + settings.batch]
            if not chunk:
                # 最后一步凑不满累积组：跳过微批但不改变 weight。
                # 用「补齐」而不是「改 weight」——改 weight 会让这一步的学习率
                # 与别步不同，而它表现为「最后一步之后指标抖一下」。
                continue
            ids, targets, masks = batch_of(examples, chunk, spec.objective)
            weights = plan.build()
            if step == 0 and mi == 0:
                # 只量第一个微批：它包含 build() + 前向的全部算子，
                # 而累积的其余微批跑的是同一段代码（量一次就够了，
                # 量全部会让计数器把同一段代码数 n_accum 遍）。
                with mac_counter() as mc:
                    pairs = [
                        (forward(weights, x, spec.model), tg, mk)
                        for x, tg, mk in zip(ids, targets, masks)
                    ]
                    loss = batch_cross_entropy(pairs, weight=weight)
                macs = mc.to_dict()
            else:
                pairs = [
                    (forward(weights, x, spec.model), tg, mk) for x, tg, mk in zip(ids, targets, masks)
                ]
                loss = batch_cross_entropy(pairs, weight=weight)
            micro_losses.append(_as_float(loss.data) / weight)
            loss.backward()

        lr = settings.lr * spec.scheduler.lr_at(step)
        lr_trace.append(lr)
        grad_norms.append(clip_grad_norm_(leaves, settings.max_grad_norm))
        optimizer.step(lr)
        losses.append(sum(micro_losses) / len(micro_losses) if micro_losses else float("nan"))

        is_last = step == settings.steps - 1
        if is_last or (step + 1) % ckpt_every == 0:
            acc = probe(step, plan) if probe is not None and (is_last or (step + 1) % eval_every == 0) else None
            checkpoints.append(CheckpointRecord(step=step + 1, loss=losses[-1], eval_acc=acc))

    elapsed = time.perf_counter() - t0
    # 只保留最后 keep_last 个检查点：一次 48 步的训练能落十几个点，
    # 全存下来会让报告里的曲线比结论还长。**但最后一个必须留下** ——
    # 它是「最终模型」的唯一凭据。
    keep = spec.checkpoint.keep_last
    if keep > 0 and len(checkpoints) > keep:
        checkpoints = checkpoints[-keep:]

    per_sample = macs.get("total", 0)
    seen = settings.steps * settings.effective_batch
    # 反向 ≈ 2× 前向，乘加 ×2 得浮点运算。**这是估算**，所以它单独一列，
    # 绝不去覆盖那个实测的 per_sample。
    train_macs = per_sample * seen * 3
    return RunRecord(
        spec_id=spec.spec_id,
        steps=settings.steps,
        batch_size=settings.batch,
        effective_batch=settings.effective_batch,
        n_train_examples=len(examples),
        train_flops=train_macs * 2,
        losses=losses,
        checkpoints=checkpoints,
        macs={
            "per_sample": per_sample,
            "by_op": dict(macs.get("by_op", {})),
            "analytic": plan.forward_macs(seq_len=settings.seq_len),
            "analytic_total": plan.forward_macs_total(seq_len=settings.seq_len),
            "n_samples_seen": seen,
            "backward_factor": 3,
            "estimated": True,
        },
        lr_trace=lr_trace,
        train_seconds=elapsed,
        n_updated_params=_count_updated(leaves),
        meta={
            "optimizer": optimizer.name,
            "optimizer_state_bytes": optimizer.state_bytes(),
            "objective": spec.objective,
            "template": settings.template,
            "seq_len": settings.seq_len,
            "grad_norm_first": grad_norms[0] if grad_norms else 0.0,
            "grad_norm_last": grad_norms[-1] if grad_norms else 0.0,
            "grad_norm_max": max(grad_norms) if grad_norms else 0.0,
            "max_grad_norm": settings.max_grad_norm,
            "template_note": "template 只改 prompt 的拼法，回答段两种模板逐位相同",
        },
    )


def _as_float(data: Any) -> float:
    """`batch_cross_entropy` 的产物是 ``[loss]``。**取 [0] 而不是把 list 当数用** ——
    后者在 Python 里会静默地比出一个恒为 True 的布尔值。"""
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError(f"期望单元素损失，得到长度 {len(data)}")
        return float(data[0])
    return float(data)


def _count_updated(leaves: Sequence[tuple[str, Tensor]]) -> int:
    """**数出来的**「真的拿到梯度的参数个数」。

    它与 `plan.n_trainable_params`（算出来的）互相印证。两者不等说明
    「计划里说是可训练的」与「引擎真的更新了它」对不上 —— 而后者才是报告的
    显存与产物列所依据的东西。
    """
    total = 0
    for _name, leaf in leaves:
        if leaf.grad is not None:
            total += _numel(leaf.grad)
    return total
