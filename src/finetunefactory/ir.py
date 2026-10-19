"""统一 Training IR。

**这个模块是本项目的全部内容。** README 的核心命题是：

    训练策略 → IR → Runtime

    SFT / LoRA / QLoRA 不再是三套并列的代码分支，而是同一个 IR 上 `finetuning`
    字段的三种取值。

所以这份 IR 必须满足一条很硬的性质：**Config Builder 是 IR 的纯函数**，而
Train Engine 只认 Builder 的产物。任何「如果是 LoRA 就…」的分支都必须活在
:mod:`plan` 里（把策略翻译成一张参数计划表），不许活在 :mod:`train` 里。

### 三条设计取舍

1. **策略字段只有一个，其余字段由它派生。**
   `FinetuneStrategy` 同时带着 `rank` / `quant_bits` / `freeze_layers`，
   但 :meth:`FinetuneStrategy.normalized` 会把与当前 `kind` 无关的字段**清零**。
   于是「切换 finetuning 字段」是一个纯粹的操作（:meth:`TrainingSpec.with_kind`），
   而「`kind=full` 却带着 `rank=8`」这种自相矛盾的 IR 会被**拒绝**而不是被忽略 ——
   一个被静默忽略的字段比一个报错的字段危险得多，因为它会让报告里的
   「同一份 Spec」变成一句假话。

2. **`objective` 与 `template` 也是真有行为的字段，不是装饰。**
   `objective` 决定损失落在哪些 token 上（`sft` = 全部，`sft_response_only` =
   只有回答部分）；`template` 决定指令怎么拼。两个都会改变数字，所以它们必须
   进 Spec 摘要 —— 否则「同一份 Spec 得到同一个 Artifact」会因为口径不同而失真。

3. **`distributed` 只接受 `single`。**
   落地约束是 4060 8G 单机。IR 里保留这个字段是为了让「本机跑不了的组合」
   有一个**明确的拒绝点**；接受一个跑不了的组合比拒绝它危险得多。

### 浮点不进指纹

`alpha` / `lr` / `weight_decay` 都是浮点，它们进 :meth:`TrainingSpec.digest`
之前必须先过 :func:`rng.quantize_for_hash`。这不是洁癖：`repr(0.1 + 0.2)` 的
稳定性是 CPython 的实现细节，而指纹要跨进程、跨机器成立。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .rng import canonical_json, quantize_for_hash, sha256_text, short_digest

__all__ = [
    "STRATEGY_KINDS",
    "OBJECTIVES",
    "TEMPLATES",
    "PRECISIONS",
    "OPTIMIZERS",
    "SCHEDULERS",
    "DISTRIBUTED_MODES",
    "DEFAULT_LORA_TARGETS",
    "LORA_ABLE_MODULES",
    "MEANINGFUL",
    "IRError",
    "ModelSpec",
    "DatasetSpec",
    "FinetuneStrategy",
    "OptimizerSpec",
    "SchedulerSpec",
    "CheckpointSpec",
    "EvalSpec",
    "BatchSpec",
    "TrainingSpec",
    "make_spec",
]

#: 四种策略。顺序即报告里的顺序，也是「同一份 Spec 切换字段」的默认顺序。
STRATEGY_KINDS = ("full", "freeze", "lora", "qlora")

#: 目标函数。两个都是真的会影响数字的取值，不是装饰。
OBJECTIVES = ("sft", "sft_response_only")

#: 指令模板。决定 prompt 怎么拼。
TEMPLATES = ("plain", "chat")

PRECISIONS = ("fp32", "bf16")

OPTIMIZERS = ("adamw", "sgd")

SCHEDULERS = ("constant", "cosine", "linear")

#: 落地约束：4060 8G 单机。
DISTRIBUTED_MODES = ("single",)

#: LoRA 的经典默认挂载点。README 的 M0 只要求「跑通」，所以默认给最小的那一对。
DEFAULT_LORA_TARGETS: tuple[str, ...] = ("q_proj", "v_proj")

#: 模型里所有可以被 LoRA 挂载的线性层名。`target_modules` 必须是它的子集。
LORA_ABLE_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

#: **每一种策略认得哪些字段。** 这一张表是「策略是数据」的边界线：
#: 表外的字段在那种策略下不是「默认值」，而是**不存在** ——
#: 传进来会被拒，而不是被静默丢掉。
MEANINGFUL: dict[str, tuple[str, ...]] = {
    "full": (),
    "freeze": ("freeze_layers",),
    "lora": ("rank", "alpha", "target_modules"),
    "qlora": ("rank", "alpha", "target_modules", "quant_bits", "quant_group"),
}


class IRError(ValueError):
    """IR 自相矛盾 / 本机跑不了。**必须在展开成 RuntimeConfig 之前炸。**"""


# ------------------------------------------------------------------ 校验助手


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise IRError(msg)


def _require_int(name: str, value: Any, *, lo: int, hi: int) -> None:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{name} 必须是整数，得到 {value!r}")
    _require(lo <= value <= hi, f"{name} 应在 [{lo}, {hi}]，得到 {value!r}")


def _require_finite(name: str, value: Any, *, lo: float, hi: float) -> None:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{name} 必须是数字，得到 {value!r}")
    _require(math.isfinite(float(value)), f"{name} 必须是有限数，得到 {value!r}")
    _require(lo <= float(value) <= hi, f"{name} 应在 [{lo}, {hi}]，得到 {value!r}")


def _require_choice(name: str, value: Any, choices: Sequence[str]) -> None:
    _require(value in choices, f"{name} 必须是 {list(choices)} 之一，得到 {value!r}")


# ------------------------------------------------------------------ ModelSpec


@dataclass(frozen=True)
class ModelSpec:
    """基座模型的**结构事实**。

    它只需要描述到「参数计划算得出来」的粒度：每个 block 有哪些线性层、
    形状是多少。真正的权重在 :mod:`model` 里，而两者的参数量必须**逐位相等** ——
    这是一条互校（见 :func:`ModelSpec.param_count` 的文档）。
    """

    name: str
    vocab: int = 256
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128
    max_len: int = 64
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        _require(self.name, "model.name 不能为空")
        _require_int("model.vocab", self.vocab, lo=2, hi=1 << 20)
        _require_int("model.d_model", self.d_model, lo=1, hi=1 << 16)
        _require_int("model.n_layers", self.n_layers, lo=1, hi=1 << 12)
        _require_int("model.n_heads", self.n_heads, lo=1, hi=1 << 12)
        _require_int("model.d_ff", self.d_ff, lo=1, hi=1 << 20)
        _require_int("model.max_len", self.max_len, lo=1, hi=1 << 16)
        _require(
            self.d_model % self.n_heads == 0,
            f"model.d_model({self.d_model}) 必须能被 n_heads({self.n_heads}) 整除",
        )
        _require(
            self.head_dim % 2 == 0 or self.head_dim == 1,
            f"head_dim({self.head_dim}) 必须是偶数（RoPE 两两配对）",
        )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    # ---- 解析参数量（与 model.py 的实测值互校）

    def param_count(self, *, quant_bits: int = 0, quant_group: int = 0) -> int:
        """解析公式算参数量。

        **`quant_bits` 不改变参数量** —— 量化只是把同一批参数用更少的比特存下来，
        参数的**个数**不变。这个区分很重要：把量化当成「参数变少了」是量化类项目
        最常见的一句错话。比特数进的是 :func:`ModelSpec.bytes_per_param`，
        不是参数量。
        """
        emb = self.vocab * self.d_model
        head = 0 if self.tie_embeddings else self.vocab * self.d_model
        per_layer = (
            self.d_model * self.d_model * 4  # q k v o
            + self.d_model * self.d_ff * 3  # gate up down
            + self.d_model * 2  # 两个 norm 的缩放
        )
        norm_final = self.d_model
        return emb + head + self.n_layers * per_layer + norm_final

    @property
    def bytes_per_param(self) -> int:
        """基座权重每参数占几字节（fp32）。量化后的字节数见 :meth:`quant_bytes`。"""
        return 4

    def quant_bytes(self, *, bits: int, group: int) -> int:
        """低比特存储的**实际字节数**（含每组一个 fp32 缩放与一个零点）。

        这是 QLoRA 省显存的全部来源，所以它必须是一个能被逐位核对的公式，
        而不是一句「4-bit 所以是 1/8」。分组是有代价的：group=32 时每 32 个参数
        多出 8 字节（scale + zero），于是实际是 ``4/8 + 8/32 = 0.75`` 字节/参数，
        而不是 ``0.5`` —— 差 50%。报告里如果写 0.5，显存表就对不上了。
        """
        n = self.param_count()
        if bits <= 0:
            return n * self.bytes_per_param
        payload = n * bits / 8.0
        n_groups = math.ceil(n / group)
        overhead = n_groups * 8.0  # fp32 scale + fp32 zero
        return int(math.ceil(payload + overhead))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vocab": self.vocab,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "max_len": self.max_len,
            "tie_embeddings": self.tie_embeddings,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ModelSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ------------------------------------------------------------------ DatasetSpec


@dataclass(frozen=True)
class DatasetSpec:
    """微调数据。

    `capability` 是这次微调**针对**的能力（数据只从它里面出），而评测会跑
    **全部**能力 —— 于是「涨了多少」与「别的能力掉了多少」都能量出来。
    """

    capability: str
    template: str = "plain"
    n_train: int = 192
    n_eval: int = 64
    seq_len: int = 48
    seed: int = 0

    def __post_init__(self) -> None:
        _require(self.capability, "dataset.capability 不能为空")
        _require_choice("dataset.template", self.template, TEMPLATES)
        _require_int("dataset.n_train", self.n_train, lo=1, hi=1 << 20)
        _require_int("dataset.n_eval", self.n_eval, lo=1, hi=1 << 20)
        _require_int("dataset.seq_len", self.seq_len, lo=8, hi=1 << 14)
        _require_int("dataset.seed", self.seed, lo=0, hi=1 << 31)

    def to_dict(self) -> dict[str, Any]:
        return dict(
            capability=self.capability,
            template=self.template,
            n_train=self.n_train,
            n_eval=self.n_eval,
            seq_len=self.seq_len,
            seed=self.seed,
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DatasetSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ------------------------------------------------------- FinetuneStrategy（核心）


@dataclass(frozen=True)
class FinetuneStrategy:
    """四种策略，一份字段表。

    **字段是并列的，语义是互斥的** —— 这正是「策略是数据」的形状：
    没有四个类，只有一张表加一个 `kind`。:meth:`normalized` 负责把与 `kind`
    无关的字段清零，:meth:`validate` 负责在**归一化之后**拒绝任何残留的无关字段。
    两条一起，保证「同一份 Spec」这句话是有意义的。
    """

    kind: str = "lora"
    # lora / qlora
    rank: int = 8
    alpha: float = 16.0
    target_modules: tuple[str, ...] = DEFAULT_LORA_TARGETS
    # freeze
    freeze_layers: int = 0
    # qlora。默认给的就是 4-bit / group=32 —— 一份 `FinetuneStrategy(kind="qlora")`
    # 应当是**开箱可用**的，而不是「还得自己想起来填两个字段」。
    quant_bits: int = 4
    quant_group: int = 32

    def __post_init__(self) -> None:
        _require_choice("finetuning.kind", self.kind, STRATEGY_KINDS)
        # 先把 tuple 归一化：手写 JSON 里很自然会写成 list。
        object.__setattr__(self, "target_modules", tuple(self.target_modules))

    # ---- 归一化：把与 kind 无关的字段清零

    def normalized(self) -> "FinetuneStrategy":
        """返回与 `kind` 相称的规范形式。

        这是「切换 finetuning 字段」能成立的**前提**：切换之后剩下的字段必须是
        确定的，否则「同一份 Spec」会带上上一份 Spec 的残留。

        ⚠️ 它**是**有损的（`lora` 会把 `quant_bits` 清零），所以 :meth:`with_kind`
        不能简单地「先 normalized 再改 kind」—— 那样 `full → qlora` 会丢掉
        `quant_bits`。正确的做法是 :meth:`with_kind` 自己负责「带上有意义的字段、
        给新字段填默认值」。
        """
        if self.kind == "full":
            return replace(
                self, rank=0, alpha=0.0, target_modules=(), freeze_layers=0, quant_bits=0, quant_group=0
            )
        if self.kind == "freeze":
            return replace(
                self, rank=0, alpha=0.0, target_modules=(), quant_bits=0, quant_group=0
            )
        if self.kind == "lora":
            return replace(self, freeze_layers=0, quant_bits=0, quant_group=0)
        # qlora
        return replace(self, freeze_layers=0)

    def is_canonical(self) -> bool:
        return self == self.normalized()

    def _non_canonical_fields(self) -> list[str]:
        """列出「与 kind 不符但非零」的字段名。错误信息里要说清差在哪。"""
        n = self.normalized()
        return [f for f in self.__dataclass_fields__ if getattr(self, f) != getattr(n, f)]

    # ---- 从零构造某一种策略的规范形式（唯一的「默认值」来源）

    @classmethod
    def canonical(
        cls,
        kind: str,
        *,
        rank: int = 0,
        alpha: float = 0.0,
        target_modules: Sequence[str] = (),
        freeze_layers: int = 0,
        quant_bits: int = 0,
        quant_group: int = 0,
        n_layers: int = 2,
    ) -> "FinetuneStrategy":
        """构造 `kind` 的规范形式，缺的字段填**这一种策略的默认值**。

        约定：``0`` / ``0.0`` / ``()`` 一律解释成「没给」而不是「给了 0」——
        因为 `0` 对每种策略来说都不是一个有意义的取值（rank=0 的 LoRA 不移动任何
        权重，freeze_layers=0 的 freeze 等于 full）。这个约定让
        :meth:`TrainingSpec.with_kind` 可以无条件地把「上一份 Spec 的残留」
        当成「没给」。

        **给了但没意义**（`kind="full"` 却 `rank=8`）会**报错**，不会被丢掉。
        一个被静默丢掉的字段比一个报错的字段危险得多：调用方会以为自己配了它。
        """
        _require_choice("kind", kind, STRATEGY_KINDS)
        given = {
            "rank": rank,
            "alpha": alpha,
            "target_modules": tuple(target_modules),
            "freeze_layers": freeze_layers,
            "quant_bits": quant_bits,
            "quant_group": quant_group,
        }
        meaningless: list[str] = []
        for field, value in given.items():
            if field in MEANINGFUL[kind]:
                continue
            if value not in (0, 0.0, ()):
                meaningless.append(field)
        if meaningless:
            raise IRError(
                f"kind={kind!r} 收到对它没有意义的字段 {meaningless}"
                f"（{kind} 只认 {list(MEANINGFUL[kind]) or '无'}）。"
                "要么去掉它们，要么换一个 kind —— 静默丢掉会让调用方以为自己配上了。"
            )
        if kind == "full":
            return cls(kind="full").normalized()
        if kind == "freeze":
            return cls(kind="freeze", freeze_layers=freeze_layers or max(1, n_layers - 1)).normalized()
        if kind == "lora":
            return cls(
                kind="lora",
                rank=rank or 8,
                alpha=alpha or 16.0,
                target_modules=tuple(target_modules) or DEFAULT_LORA_TARGETS,
            ).normalized()
        return cls(
            kind="qlora",
            rank=rank or 8,
            alpha=alpha or 16.0,
            target_modules=tuple(target_modules) or DEFAULT_LORA_TARGETS,
            quant_bits=quant_bits or 4,
            quant_group=quant_group or 32,
        ).normalized()

    def validate(self, model: ModelSpec) -> None:
        """校验。**只接受规范形式** —— 自相矛盾的 IR 在这里被拒。"""
        k = self.kind
        if k == "full":
            _require(self.rank == 0, f"kind=full 不许带 rank={self.rank}（这是自相矛盾的 IR）")
            _require(self.alpha == 0.0, f"kind=full 不许带 alpha={self.alpha}")
            _require(not self.target_modules, f"kind=full 不许带 target_modules={self.target_modules}")
            _require(self.freeze_layers == 0, f"kind=full 不许带 freeze_layers={self.freeze_layers}")
            _require(self.quant_bits == 0, f"kind=full 不许带 quant_bits={self.quant_bits}")
        elif k == "freeze":
            _require(self.rank == 0, f"kind=freeze 不许带 rank={self.rank}")
            _require(self.quant_bits == 0, f"kind=freeze 不许带 quant_bits={self.quant_bits}")
            _require(
                0 < self.freeze_layers < model.n_layers,
                f"kind=freeze 的 freeze_layers 应在 (0, {model.n_layers})，得到 {self.freeze_layers}"
                "（冻结全部层等于什么都没训，冻结 0 层等于 full）",
            )
        else:  # lora / qlora
            _require(self.rank > 0, f"kind={k} 的 rank 必须 > 0，得到 {self.rank}")
            _require_int(f"finetuning.rank（{k}）", self.rank, lo=1, hi=max(model.d_model, model.d_ff))
            _require_finite("finetuning.alpha", self.alpha, lo=1e-6, hi=1e6)
            _require(self.target_modules, f"kind={k} 的 target_modules 不能为空")
            unknown = [m for m in self.target_modules if m not in LORA_ABLE_MODULES]
            _require(not unknown, f"target_modules 里有不存在的模块：{unknown}（可选：{list(LORA_ABLE_MODULES)}）")
            _require(
                len(set(self.target_modules)) == len(self.target_modules),
                f"target_modules 有重复：{self.target_modules}（重复会让参数量算重）",
            )
            _require(self.freeze_layers == 0, f"kind={k} 不许带 freeze_layers={self.freeze_layers}")
            if k == "lora":
                _require(self.quant_bits == 0, f"kind=lora 不许带 quant_bits={self.quant_bits}（那是 qlora）")
            else:
                _require(
                    self.quant_bits in (4, 8),
                    f"kind=qlora 的 quant_bits 只支持 4 / 8，得到 {self.quant_bits}",
                )
                _require(
                    self.quant_group > 0 and self.quant_group % 2 == 0,
                    f"kind=qlora 的 quant_group 必须是正偶数（一个组要能对半切开），得到 {self.quant_group}",
                )

    # ---- 派生量

    @property
    def scale(self) -> float:
        """LoRA 的缩放因子 ``alpha / r``。

        它是**乘在 BA 上**的（不是乘在 A 或 B 上）：`W' = W + (alpha/r)·B@A`。
        写成 `alpha/sqrt(r)` 的实现也存在，但那种实现下「改 rank 会同时改有效学习率」，
        于是「rank 与 lr 的对比实验」就不是单变量了。
        """
        if self.kind not in ("lora", "qlora"):
            return 0.0
        return float(self.alpha) / float(self.rank)

    @property
    def is_peft(self) -> bool:
        return self.kind in ("lora", "qlora")

    @property
    def base_is_frozen(self) -> bool:
        """基座权重是否**不**接收梯度。四种策略里只有 full 是 False。"""
        return self.kind != "full"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "rank": self.rank,
            "alpha": self.alpha,
            "target_modules": list(self.target_modules),
            "freeze_layers": self.freeze_layers,
            "quant_bits": self.quant_bits,
            "quant_group": self.quant_group,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FinetuneStrategy":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def fingerprint_payload(self) -> dict[str, Any]:
        """进指纹的形态：浮点先量化成整数。"""
        return {
            "kind": self.kind,
            "rank": self.rank,
            "alpha_micro": quantize_for_hash(self.alpha),
            "target_modules": sorted(self.target_modules),
            "freeze_layers": self.freeze_layers,
            "quant_bits": self.quant_bits,
            "quant_group": self.quant_group,
        }


# ------------------------------------------------------- 其余子 Spec（都是数据）


@dataclass(frozen=True)
class OptimizerSpec:
    name: str = "adamw"
    lr: float = 3e-3
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    eps: float = 1e-8

    def __post_init__(self) -> None:
        _require_choice("optimizer.name", self.name, OPTIMIZERS)
        _require_finite("optimizer.lr", self.lr, lo=1e-8, hi=1.0)
        _require_finite("optimizer.weight_decay", self.weight_decay, lo=0.0, hi=1.0)
        _require_finite("optimizer.eps", self.eps, lo=1e-30, hi=1e-2)
        object.__setattr__(self, "betas", tuple(float(b) for b in self.betas))
        _require(len(self.betas) == 2, "optimizer.betas 必须是两个数")
        for b in self.betas:
            _require_finite("optimizer.betas[*]", b, lo=0.0, hi=1.0)

    @property
    def state_multiplier(self) -> int:
        """每个可训练参数需要几份额外的优化器状态（**以参数为单位**）。

        AdamW 是 `m` 与 `v` 两份，所以是 2。这个数是显存公式里最大的一块 ——
        full FT 的优化器状态是参数本身的 2 倍，比梯度还大。
        """
        return 2 if self.name == "adamw" else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lr": self.lr,
            "betas": list(self.betas),
            "weight_decay": self.weight_decay,
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "OptimizerSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lr_micro": quantize_for_hash(self.lr, digits=12),
            "betas_micro": [quantize_for_hash(b, digits=12) for b in self.betas],
            "wd_micro": quantize_for_hash(self.weight_decay, digits=12),
            "eps_micro": quantize_for_hash(self.eps, digits=18),
        }


@dataclass(frozen=True)
class SchedulerSpec:
    name: str = "cosine"
    warmup_steps: int = 10
    total_steps: int = 120

    def __post_init__(self) -> None:
        _require_choice("scheduler.name", self.name, SCHEDULERS)
        _require_int("scheduler.warmup_steps", self.warmup_steps, lo=0, hi=1 << 20)
        _require_int("scheduler.total_steps", self.total_steps, lo=1, hi=1 << 20)
        _require(
            self.warmup_steps <= self.total_steps,
            f"scheduler.warmup_steps({self.warmup_steps}) 不能超过 total_steps({self.total_steps})",
        )

    def lr_at(self, step: int) -> float:
        """第 `step` 步的学习率倍率（相对 `optimizer.lr`）。

        写成**纯函数**而不是「跑的时候顺手更新一个变量」：调度是 IR 的一部分，
        它必须能被单独复算 —— 否则「同一份 Spec 同一个 Artifact」就多了一个
        只能靠「跑一遍看看」来验证的环节。
        """
        if step < 0:
            raise IRError(f"scheduler.lr_at 的 step 不能为负，得到 {step}")
        if self.total_steps <= 0:
            return 1.0
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        if self.name == "constant":
            return 1.0
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        if self.name == "linear":
            return 1.0 - progress
        # cosine：从 1 降到 0.1，不降到 0（降到 0 的最后几步对指标毫无贡献，
        # 却会让「最后一步的权重」变成噪声最大的一步）。
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "warmup_steps": self.warmup_steps, "total_steps": self.total_steps}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SchedulerSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def fingerprint_payload(self) -> dict[str, Any]:
        return dict(self.to_dict())


@dataclass(frozen=True)
class BatchSpec:
    size: int = 8
    grad_accum: int = 1

    def __post_init__(self) -> None:
        _require_int("batch.size", self.size, lo=1, hi=1 << 12)
        _require_int("batch.grad_accum", self.grad_accum, lo=1, hi=1 << 12)

    @property
    def effective(self) -> int:
        return self.size * self.grad_accum

    def to_dict(self) -> dict[str, Any]:
        return {"size": self.size, "grad_accum": self.grad_accum}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BatchSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def fingerprint_payload(self) -> dict[str, Any]:
        return dict(self.to_dict())


@dataclass(frozen=True)
class CheckpointSpec:
    every_n_steps: int = 40
    keep_last: int = 2
    save_optimizer: bool = False

    def __post_init__(self) -> None:
        _require_int("checkpoint.every_n_steps", self.every_n_steps, lo=1, hi=1 << 20)
        _require_int("checkpoint.keep_last", self.keep_last, lo=0, hi=1 << 10)

    def to_dict(self) -> dict[str, Any]:
        return {
            "every_n_steps": self.every_n_steps,
            "keep_last": self.keep_last,
            "save_optimizer": self.save_optimizer,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CheckpointSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def fingerprint_payload(self) -> dict[str, Any]:
        return dict(self.to_dict())


@dataclass(frozen=True)
class EvalSpec:
    every_n_steps: int = 40
    batch_size: int = 32

    def __post_init__(self) -> None:
        _require_int("evaluation.every_n_steps", self.every_n_steps, lo=1, hi=1 << 20)
        _require_int("evaluation.batch_size", self.batch_size, lo=1, hi=1 << 12)

    def to_dict(self) -> dict[str, Any]:
        return {"every_n_steps": self.every_n_steps, "batch_size": self.batch_size}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EvalSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def fingerprint_payload(self) -> dict[str, Any]:
        return dict(self.to_dict())


# ------------------------------------------------------------------ TrainingSpec


@dataclass(frozen=True)
class TrainingSpec:
    """统一 Training IR。

    字段顺序照抄 README 第五节，多出来的 `batch` / `seed` 是跑得起来所必需的。
    """

    model: ModelSpec
    dataset: DatasetSpec
    finetuning: FinetuneStrategy = FinetuneStrategy()
    objective: str = "sft"
    optimizer: OptimizerSpec = OptimizerSpec()
    scheduler: SchedulerSpec = SchedulerSpec()
    batch: BatchSpec = BatchSpec()
    precision: str = "fp32"
    distributed: str = "single"
    checkpoint: CheckpointSpec = CheckpointSpec()
    evaluation: EvalSpec = EvalSpec()
    seed: int = 0

    # ---- 构造期校验

    def __post_init__(self) -> None:
        _require_choice("objective", self.objective, OBJECTIVES)
        _require_choice("precision", self.precision, PRECISIONS)
        _require_choice("distributed", self.distributed, DISTRIBUTED_MODES)
        _require_int("seed", self.seed, lo=0, hi=1 << 31)
        if not self.finetuning.is_canonical():
            raise IRError(
                f"finetuning 不是规范形式：kind={self.finetuning.kind!r} 却带着"
                f" {self.finetuning._non_canonical_fields()}。"
                "请用 make_spec / with_kind / FinetuneStrategy.canonical 生成 Spec —— "
                "一个被**静默归一化**的字段会让「同一份 Spec」变成一句假话。"
            )
        self.finetuning.validate(self.model)
        _require(
            self.dataset.seq_len <= self.model.max_len,
            f"dataset.seq_len({self.dataset.seq_len}) 不能超过 model.max_len({self.model.max_len})",
        )
        _require(
            self.scheduler.total_steps >= 1,
            "scheduler.total_steps 必须 >= 1（0 步的训练不是一个实验）",
        )

    # ---- 归一化

    def normalized(self) -> "TrainingSpec":
        """恒等返回 —— **一份 TrainingSpec 在构造时就已经是规范形式了**。

        保留这个方法是为了让调用点可以无条件写 `spec.normalized()`（比如从
        `from_dict` 重建之后），而不用去关心「Spec 是不是已经规范」。
        这条不变量由 `__post_init__` 的严格校验维持：非规范的 `finetuning`
        根本构造不出 TrainingSpec。
        """
        return self

    def is_canonical(self) -> bool:
        return self.finetuning.is_canonical()

    # ---- 摘要

    def fingerprint_payload(self) -> dict[str, Any]:
        """进指纹的形态。**浮点已量化，容器已排序。**"""
        s = self.normalized()
        return {
            "model": s.model.to_dict(),
            "dataset": s.dataset.to_dict(),
            "finetuning": s.finetuning.fingerprint_payload(),
            "objective": s.objective,
            "optimizer": s.optimizer.fingerprint_payload(),
            "scheduler": s.scheduler.fingerprint_payload(),
            "batch": s.batch.fingerprint_payload(),
            "precision": s.precision,
            "distributed": s.distributed,
            "checkpoint": s.checkpoint.fingerprint_payload(),
            "evaluation": s.evaluation.fingerprint_payload(),
            "seed": s.seed,
        }

    def digest(self) -> str:
        """全长内容摘要。**比较一律用它，展示才用 `spec_id`。**"""
        return sha256_text(canonical_json(self.fingerprint_payload()))

    @property
    def spec_id(self) -> str:
        """短标识：`<objective>-<kind>-<12 位摘要>`。

        带上 `objective` 与 `kind` 是**为了可读**（`sft-lora-a1b2c3d4e5f6`
        比 `a1b2c3d4e5f6` 好认），而唯一性仍然完全由摘要保证。
        """
        return f"{self.objective}-{self.finetuning.kind}-{short_digest(self.digest(), 12)}"

    # ---- 「切换 finetuning 字段」这个操作本身

    def with_kind(self, kind: str) -> "TrainingSpec":
        """把 `finetuning.kind` 换成 `kind`，其余字段一律不动。

        **README 验收标准第一条就是这个方法**：同一份 TrainingSpec 通过切换
        `finetuning` 字段即可产出四种 Artifact。所以它必须是一个纯函数，
        而且换出来的 Spec 与原 Spec **只在 `finetuning` 上不同** ——
        这条性质由测试逐字段断言（`without_finetuning()` 四者相等）。

        实现上它**不是** `replace(kind=kind).normalized()`：归一化会把
        `lora` 的 `quant_bits` 清零，于是 `full → qlora` 会丢掉量化位数。
        正确做法是**只带目标策略认得的那几个字段**（`MEANINGFUL[kind]`），
        其余留空由 :meth:`FinetuneStrategy.canonical` 填默认值。于是：

        - `qlora → lora`：丢掉 `quant_bits`（lora 没有量化），保住 rank / targets
        - `full → qlora`：full 什么都不带，于是拿 qlora 的默认 4-bit / group=32
        - `lora → full`：什么都不带
        """
        _require_choice("kind", kind, STRATEGY_KINDS)
        f = self.finetuning
        carry = {name: getattr(f, name) for name in MEANINGFUL[kind]}
        new = FinetuneStrategy.canonical(kind, n_layers=self.model.n_layers, **carry)
        return replace(self, finetuning=new)

    def with_finetuning(self, **kwargs: Any) -> "TrainingSpec":
        """改 `finetuning` 的若干字段（用于扇出：rank / target_modules / freeze_layers）。

        改完之后**重新走一遍 canonical**，于是「把 rank 改成 0」这种越界输入
        会落回默认值而不是构造出一个非法 Spec —— 扇出网格里少一格比多一个
        无法解释的 Spec 好。但**给一个当前 kind 认不得的字段会报错**：
        `with_finetuning(quant_bits=4)` 落在一份 lora 的 Spec 上是调用方的 bug，
        不是「顺手帮它升级成 qlora」。
        """
        f = replace(self.finetuning, **kwargs)
        carry = {name: getattr(f, name) for name in MEANINGFUL[f.kind]}
        new = FinetuneStrategy.canonical(f.kind, n_layers=self.model.n_layers, **carry)
        return replace(self, finetuning=new)

    def with_lr(self, lr: float) -> "TrainingSpec":
        return replace(self, optimizer=replace(self.optimizer, lr=lr))

    def with_capability(self, capability: str) -> "TrainingSpec":
        return replace(self, dataset=replace(self.dataset, capability=capability))

    def with_objective(self, objective: str) -> "TrainingSpec":
        return replace(self, objective=objective)

    def with_template(self, template: str) -> "TrainingSpec":
        return replace(self, dataset=replace(self.dataset, template=template))

    def without_finetuning(self) -> dict[str, Any]:
        """去掉 `finetuning` 之后的指纹载荷。

        「四份 Spec 只在 finetuning 上不同」这条断言需要一个**能直接比较的东西**，
        而比较 dict 比比较 dataclass 更直白（也更容易在断言失败时看清差在哪）。
        """
        payload = self.fingerprint_payload()
        payload.pop("finetuning")
        return payload

    # ---- 序列化

    def to_dict(self) -> dict[str, Any]:
        """完整往返形态（**不量化浮点**）—— 用于落盘与重新载入。"""
        s = self.normalized()
        return {
            "spec_id": s.spec_id,
            "digest": s.digest(),
            "model": s.model.to_dict(),
            "dataset": s.dataset.to_dict(),
            "finetuning": s.finetuning.to_dict(),
            "objective": s.objective,
            "optimizer": s.optimizer.to_dict(),
            "scheduler": s.scheduler.to_dict(),
            "batch": s.batch.to_dict(),
            "precision": s.precision,
            "distributed": s.distributed,
            "checkpoint": s.checkpoint.to_dict(),
            "evaluation": s.evaluation.to_dict(),
            "seed": s.seed,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TrainingSpec":
        """从 `to_dict` 的产物重建，并**核对摘要**。

        重建之后摘要对不上，说明序列化丢了信息 —— 这必须当场炸，不能等到
        「重跑一遍结果不一样」才发现。`spec_id` 与 `digest` 是**派生的**，
        所以它们对不上是硬错误，不是可以忽略的冗余字段。
        """
        spec = cls(
            model=ModelSpec.from_dict(d["model"]),
            dataset=DatasetSpec.from_dict(d["dataset"]),
            finetuning=FinetuneStrategy.from_dict(d["finetuning"]),
            objective=d["objective"],
            optimizer=OptimizerSpec.from_dict(d["optimizer"]),
            scheduler=SchedulerSpec.from_dict(d["scheduler"]),
            batch=BatchSpec.from_dict(d["batch"]),
            precision=d["precision"],
            distributed=d["distributed"],
            checkpoint=CheckpointSpec.from_dict(d["checkpoint"]),
            evaluation=EvalSpec.from_dict(d["evaluation"]),
            seed=int(d["seed"]),
        )
        if "digest" in d and d["digest"] != spec.digest():
            raise IRError(
                "TrainingSpec 往返后摘要变了：序列化丢了信息。\n"
                f"  落盘 {d['digest']}\n  重建 {spec.digest()}"
            )
        if "spec_id" in d and d["spec_id"] != spec.spec_id:
            raise IRError(f"spec_id 对不上：落盘 {d['spec_id']}，重建 {spec.spec_id}")
        return spec


def make_spec(
    model: ModelSpec,
    dataset: DatasetSpec | Mapping[str, Any] | str,
    **overrides: Any,
) -> TrainingSpec:
    """基座模型 + 数据 + 默认口径 + 覆盖。**所有调用点都从这里出发**。

    `model` 与 `dataset` 是**必填**的，其余全部有默认值：

    - `model` 不接受省略 —— 一份没有基座的 TrainingSpec 不是一个不完整的 Spec，
      它根本不是一个 Spec。基座本身是**数据**（`data/models/<name>.json`），
      所以必须由调用方显式给出来自哪个文件；代码里不许有一个「默认模型」，
      否则报告里的「Base」指的是谁就说不清了。
    - `dataset` 可以直接给一个**能力名**（`"arith"`）—— 「微调哪个能力」是这份
      Spec 唯一必须回答的实验问题，把它放在签名里而不是藏进默认值里。

    其余子 Spec 给 `dict` 时是**在默认值上覆盖**（不是替换），于是
    `make_spec(m, "arith", optimizer={"lr": 1e-3})` 只会动 `lr`。
    """
    if not isinstance(model, ModelSpec):
        raise IRError(f"make_spec 的 model 必须是 ModelSpec，得到 {type(model).__name__}")
    if isinstance(dataset, str):
        dataset = DatasetSpec(capability=dataset)
    elif isinstance(dataset, Mapping):
        # 用 dict 给的时候 **capability 必须显式出现**。这里曾经写成
        # `replace(DatasetSpec(), **...)`，而 `DatasetSpec.capability` 是必填字段、
        # 没有默认值 —— 于是这条分支从来没被走到过：任何人用 dict 传 dataset
        # 都会在 `DatasetSpec()` 那一行拿到 TypeError。这个 bug 之所以能活着，
        # 正是因为「给一个 dict 就等于给一份不完整的 Spec」这件事看起来太合理了。
        allowed = {k: v for k, v in dataset.items() if k in DatasetSpec.__dataclass_fields__}
        if "capability" not in allowed:
            raise IRError(
                "dataset 用 dict 给的时候必须带 capability —— "
                "「微调哪个能力」是这份 Spec 唯一必须回答的实验问题，它不能有默认值"
            )
        dataset = DatasetSpec(**allowed)
    elif not isinstance(dataset, DatasetSpec):
        raise IRError(f"make_spec 的 dataset 必须是 DatasetSpec / dict / 能力名，得到 {type(dataset).__name__}")

    nested = {
        "finetuning": FinetuneStrategy,
        "optimizer": OptimizerSpec,
        "scheduler": SchedulerSpec,
        "batch": BatchSpec,
        "checkpoint": CheckpointSpec,
        "evaluation": EvalSpec,
    }
    flat: dict[str, Any] = {"model": model, "dataset": dataset}
    for key, cls_ in nested.items():
        sub = overrides.pop(key, None)
        if isinstance(sub, cls_):
            flat[key] = sub
        elif isinstance(sub, Mapping):
            if key == "finetuning":
                # `finetuning` 的默认值不是「字段默认值」而是**按 kind 展开的规范形式**，
                # 所以它走 canonical 而不是 `replace(cls_(), **sub)`。
                kind = sub.get("kind", FinetuneStrategy().kind)
                known = {k: v for k, v in sub.items() if k in cls_.__dataclass_fields__ and k != "kind"}
                flat[key] = FinetuneStrategy.canonical(kind, n_layers=model.n_layers, **known)
            else:
                base = cls_()
                flat[key] = replace(base, **{k: v for k, v in sub.items() if k in cls_.__dataclass_fields__})
        elif sub is None:
            flat[key] = FinetuneStrategy.canonical("lora", n_layers=model.n_layers) if key == "finetuning" else cls_()
        else:
            raise IRError(f"{key} 必须是 {cls_.__name__} 或 dict，得到 {type(sub).__name__}")
    return TrainingSpec(**flat, **overrides)
