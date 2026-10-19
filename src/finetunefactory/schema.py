"""运行期对象：一次训练跑出了什么、被评成了什么、导出了什么。

**这里不定义 Spec。** Spec 是「输入」，活在 :mod:`ir` 里；本模块是「输出」。
这条边界要守住，否则「同一份 Spec 得到同一个 Artifact」这句话会失去意义 ——
读代码的人分不清哪个字段是他配的、哪个是跑出来的。

四个对象的分工，正好对应血缘图里的四种节点：

| 对象 | 血缘节点 | 它回答的问题 |
| --- | --- | --- |
| :class:`RunRecord` | `run:<spec>` / `checkpoint:<spec>:<step>` | 跑了多少步、损失降到哪 |
| :class:`EvalResult` | `eval:<spec>` | 涨了多少、别的能力掉了多少 |
| :class:`Artifact` | `artifact:<spec>` | 导出了什么、能不能合并回去 |
| :class:`ParamPlan` | （不建节点） | 哪些参数可训练、显存花在哪 |

`ParamPlan` 在血缘图里没有节点是**故意的**：它是 Spec 的纯函数（同一个 Spec
必然给出同一张计划表），所以它不是「发生过什么」，而是「这份 Spec 意味着什么」。
把推导量画进图里会让图里出现一批「不携带新信息」的节点。

### 序列化纪律（见 skill 坑 42）

每个 `from_dict` **显式按字段取值，绝不 `(**d)`**。`to_dict` 里多写一个展示用的
字段就会让 `(**d)` 抛 `TypeError`，而这条路径平时走不到 —— 只有 round-trip
测试才抓得到，而它一旦漏掉，`load_result` 就会在某个只读子命令上炸。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "BASE_KEY",
    "base_key",
    "CheckpointRecord",
    "RunRecord",
    "EvalResult",
    "Artifact",
    "ModelEntry",
    "PipelineResult",
]


def base_key(model: str, capability: str) -> str:
    """基座评测的键：``<model>|<capability>``。

    用 ``|`` 而不是 ``:`` —— 模型名与能力名里都可能出现 ``:``（``tiny:2l``），
    而键的**可解析性**是这个项目里唯一一次「用字符串当复合键」的场合，
    撞了就再也拆不回来。空能力（``""``）表示「这个模型的总体基线」。
    """
    return f"{model}|{capability}"


BASE_KEY = base_key


# ----------------------------------------------------------------------
# 训练
# ----------------------------------------------------------------------
@dataclass
class CheckpointRecord:
    """一次落点的快照。

    `eval_acc` 是**那一步**在目标能力上的准确率（可能为 `None`：评测比训练贵，
    默认只在 ``evaluation.every_n_steps`` 的倍数上跑）。留 `None` 而不是填 0 ——
    「没测」与「测出来是 0」在报告里必须是两件事（见 skill 坑 10「不适用 ≠ 不满足」）。
    """

    step: int
    loss: float
    eval_acc: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "loss": self.loss, "eval_acc": self.eval_acc}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CheckpointRecord":
        return cls(step=int(d["step"]), loss=float(d["loss"]), eval_acc=d.get("eval_acc"))


@dataclass
class RunRecord:
    """一次训练运行。

    `macs` 存的是 :class:`autograd.MacCounter` 的产物 —— 它是**代价三路互校的
    第 ③ 路**：解析公式算「按 Spec 应该花多少」，计数器记「这次真的花了多少」。
    两者不等就说明有一层没被算进去（某个 matmul 被跳过、或者某个 bias 忘了加）。
    """

    spec_id: str
    steps: int
    batch_size: int
    effective_batch: int
    n_train_examples: int
    train_flops: int
    losses: list[float] = field(default_factory=list)
    checkpoints: list[CheckpointRecord] = field(default_factory=list)
    macs: dict[str, Any] = field(default_factory=dict)
    lr_trace: list[float] = field(default_factory=list)
    train_seconds: float = 0.0
    #: 实际参与更新的参数个数（来自参数计划）。**不是** Spec 里的数字，
    #: 而是「真的拿到梯度的张量」数出来的 —— 它是计划与引擎互校的一条副证。
    n_updated_params: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def first_loss(self) -> float:
        return self.losses[0] if self.losses else float("nan")

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "effective_batch": self.effective_batch,
            "n_train_examples": self.n_train_examples,
            "train_flops": self.train_flops,
            "losses": list(self.losses),
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "macs": dict(self.macs),
            "lr_trace": list(self.lr_trace),
            "train_seconds": self.train_seconds,
            "n_updated_params": self.n_updated_params,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RunRecord":
        return cls(
            spec_id=str(d["spec_id"]),
            steps=int(d["steps"]),
            batch_size=int(d.get("batch_size", 0)),
            effective_batch=int(d.get("effective_batch", 0)),
            n_train_examples=int(d.get("n_train_examples", 0)),
            train_flops=int(d.get("train_flops", 0)),
            losses=[float(v) for v in d.get("losses", [])],
            checkpoints=[CheckpointRecord.from_dict(c) for c in d.get("checkpoints", [])],
            macs=dict(d.get("macs", {})),
            lr_trace=[float(v) for v in d.get("lr_trace", [])],
            train_seconds=float(d.get("train_seconds", 0.0)),
            n_updated_params=int(d.get("n_updated_params", 0)),
            meta=dict(d.get("meta", {})),
        )


# ----------------------------------------------------------------------
# 评测
# ----------------------------------------------------------------------
@dataclass
class EvalResult:
    """一次评测：**目标能力的收益 + 其它能力的干扰**。

    两个数各有各的方向，**不能合成一个「总分」**：

    - `target_acc` 越高越好（`gain` 是它相对基座的增量）；
    - `interference` 越低越好（其它能力的平均跌幅，**可以为负** —— 负值表示
      微调顺带把别处也改善了，这比「没掉」更有信息量）。

    揉成一个「微调质量分」会让「涨了 5 个点但掉了 12 个点」和「只涨了 5 个点」
    看起来一样，而它们是完全相反的两个结论。
    """

    spec_id: str
    capability: str
    per_capability: dict[str, float] = field(default_factory=dict)
    n_eval: int = 0
    n_correct: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def target_acc(self) -> float:
        return float(self.per_capability.get(self.capability, 0.0))

    @property
    def mean_acc(self) -> float:
        if not self.per_capability:
            return 0.0
        return sum(self.per_capability.values()) / len(self.per_capability)

    def interference(self, base: Mapping[str, float]) -> float:
        """其它能力的平均跌幅。**只统计「两边都测了」的能力。**

        缺测的能力必须**跳过**而不是当 0 —— 把「不适用」当 0 算进均值，
        是这个工作区里出现次数最多的静默偏差（skill 坑 10）。
        """
        drops = [
            base[c] - self.per_capability[c]
            for c in self.per_capability
            if c != self.capability and c in base
        ]
        return sum(drops) / len(drops) if drops else 0.0

    def gain(self, base: Mapping[str, float]) -> float:
        if self.capability not in base:
            return float("nan")
        return self.target_acc - base[self.capability]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "capability": self.capability,
            "per_capability": dict(sorted(self.per_capability.items())),
            "target_acc": self.target_acc,
            "mean_acc": self.mean_acc,
            "n_eval": self.n_eval,
            "n_correct": self.n_correct,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EvalResult":
        return cls(
            spec_id=str(d["spec_id"]),
            capability=str(d["capability"]),
            per_capability={k: float(v) for k, v in (d.get("per_capability") or {}).items()},
            n_eval=int(d.get("n_eval", 0)),
            n_correct=int(d.get("n_correct", 0)),
            meta=dict(d.get("meta", {})),
        )


# ----------------------------------------------------------------------
# 产物
# ----------------------------------------------------------------------
@dataclass
class Artifact:
    """导出的「微调产物」。

    它要回答三个问题，缺一个这个项目就没跑完：

    1. **多大**（`n_bytes` / `payload_bytes`）—— 显存与存储的对比全靠它；
    2. **能不能合并回基座**（`merge_max_abs_diff`）—— LoRA 的核心承诺，
       而它「看起来总是对的」，所以必须有一个可断言的上界；
    3. **量化损失了多少**（`quant_error`）—— QLoRA 的核心代价，
       且必须随位数单调下降，否则「4-bit 够用」这个说法不成立。
    """

    spec_id: str
    kind: str
    n_trainable_params: int
    n_bytes: int
    digest: str = ""
    payload_bytes: int = 0
    #: LoRA 合并回基座后，与「适配器模型直接前向」的最大绝对差。`None` = 没测。
    merge_max_abs_diff: float | None = None
    #: 基座权重的量化相对误差（‖W − dequant(quant(W))‖ / ‖W‖）。非量化策略为 0。
    quant_error: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def merge_is_exact(self, tolerance: float = 1e-5) -> bool:
        """合并是否**足够准**。判据是 ``≤ tolerance``，不是 ``== 0.0``。

        ``== 0.0`` 是错的两个方向都错：

        - 它**要求两份不同的实现给出逐位相同的结果**。合并走 :func:`model.merge_weights`
          的手写三重求和，前向走 :func:`autograd.matmul`，浮点累加顺序不同，
          逐位相同的概率是零。那条断言会永远失败。
        - 更要紧的是：如果它**真的**成立了，说明其中一份被换成了另一份
          （skill 坑 35，oracle 复用）—— 那才是该报警的情形。

        所以容差是必须的，而「差值恰好为 0」被 :func:`compare.verify_merge` 单独
        报成 ``identical`` 一个可疑信号，由人来看。默认 1e-5 与
        ``acceptance.merge_tolerance`` 同值，调用方应当显式把配置里的那个数传进来。
        """
        return self.merge_max_abs_diff is not None and self.merge_max_abs_diff <= tolerance

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "kind": self.kind,
            "n_trainable_params": self.n_trainable_params,
            "n_bytes": self.n_bytes,
            "digest": self.digest,
            "payload_bytes": self.payload_bytes,
            "merge_max_abs_diff": self.merge_max_abs_diff,
            "quant_error": self.quant_error,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Artifact":
        return cls(
            spec_id=str(d["spec_id"]),
            kind=str(d["kind"]),
            n_trainable_params=int(d.get("n_trainable_params", 0)),
            n_bytes=int(d.get("n_bytes", 0)),
            digest=str(d.get("digest", "")),
            payload_bytes=int(d.get("payload_bytes", 0)),
            merge_max_abs_diff=d.get("merge_max_abs_diff"),
            quant_error=float(d.get("quant_error", 0.0)),
            meta=dict(d.get("meta", {})),
        )


# ----------------------------------------------------------------------
# 输入对象
# ----------------------------------------------------------------------
@dataclass
class ModelEntry:
    """模型注册表里的一项。`spec` 是结构事实，**权重不在这里** ——

    权重由 :mod:`model` 从 `ModelSpec` + 种子**确定性地初始化**出来。这不是取巧：
    「同一份 Spec 得到同一个 Artifact」要求初始化可复现，而落盘一份几万个浮点
    的权重只会让「同一份 Spec」多出一个可以漂移的自由度。
    """

    name: str
    spec: Any  # ir.ModelSpec —— 用 Any 是为了让本模块不 import ir（避免环）
    n_params: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "spec": self.spec.to_dict(),
            "n_params": self.n_params,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ModelEntry":
        from .ir import ModelSpec

        return cls(
            name=str(d["name"]),
            spec=ModelSpec.from_dict(d["spec"]),
            n_params=int(d.get("n_params", 0)),
            meta=dict(d.get("meta", {})),
        )


# ----------------------------------------------------------------------
# 一次完整流水线
# ----------------------------------------------------------------------
@dataclass
class PipelineResult:
    """一次完整扇出-收拢的结果。

    字段顺序照抄 :func:`lineage.build_lineage` 的访问顺序 —— 那张图是本类**唯一
    的消费者契约**，两者必须同步改。

    `costs` / `plans` 不进产物：它们是 Spec 的纯函数，重算比落盘更可靠
    （skill 坑 93：**推导量不进主地址空间，但必须可重算**）。
    """

    cfg: dict[str, Any] = field(default_factory=dict)
    models: list[ModelEntry] = field(default_factory=list)
    specs: dict[str, Any] = field(default_factory=dict)
    runs: dict[str, RunRecord] = field(default_factory=dict)
    evals: dict[str, EvalResult] = field(default_factory=dict)
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    frontier: list[Any] = field(default_factory=list)
    #: 微调前那个模型在各项能力上的**实测**准确率。它是 **gain** 的基准点。
    base_accs: dict[str, float] = field(default_factory=dict)
    #: 乱猜的准确率（1 / 输出字母表大小）。它是 **interference** 的基准点。
    #:
    #: 两个问题必须用两个基准，混用会让其中一个数字失去意义：
    #:
    #: - ``gain = acc − base_accs`` 问的是「训练有没有帮上忙」。
    #: - ``interference = mean(chance − acc)`` 问的是「训练有没有把不相干的能力
    #:   踹到乱猜以下」。
    #:
    #: 为什么 interference 不能用 ``base_accs`` 当基准：本项目是**从随机初始化的
    #: 基座**开始训的，而随机基座在四项能力上全都**低于 chance**（实测
    #: upper 0.042 / shift 0.052 / reverse 0.031 / sort 0.073）。拿它当基准，
    #: 「干扰」就恒为负数（一切都是提升），而且**参数越多的策略提升越大** ——
    #: 于是 `full` 会算出「干扰最小」，把结论恰好反过来，报告里却看不出任何异常。
    chance_accs: dict[str, float] = field(default_factory=dict)
    frontier_keys: dict[str, list[str]] = field(default_factory=dict)
    #: 配置里**声明**的四条设计期望。**单独存一份，不靠 `cfg`。**
    #:
    #: 理由：`cfg` 是「输入」，而输出目录里刻意不存整份配置（它属于仓库，
    #: 不属于一次运行）。但报告要能回答「配置里要求裁决哪几条、实际裁决出来哪几条」，
    #: 而**没被裁决的那一条正是最该被看见的** —— 期望被悄悄地漏掉，
    #: 表现是裁决表里少了一行，读者不会注意到少的那一行。
    #:
    #: 所以这里存的是**声明本身**（不是推导量），与 `verdicts` 一起落盘。
    expectations: dict[str, Any] = field(default_factory=dict)
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    counters: dict[str, Any] = field(default_factory=dict)
    protocol_digest: str = ""
    version_digest: str = ""
    dag: Any = None
    #: 产物的**字节内容**，spec_id → 裸 float32 小端。**刻意不进 `to_dict()`。**
    #:
    #: 它和 `plans` / `costs` 是两类东西，别混：那两个是 Spec 的纯函数（重算即可），
    #: 而产物内容是**产品本身** —— JSON 装不下 bytes，所以它落进
    #: `artifacts/<spec_id>.bin`，而这个字段只是把「已经算好的那份字节」
    #: 从 `run_pipeline` 递给 `write_outputs` 的通道。
    #:
    #: 它必须**在内存里算一次就定稿**，不能在写盘时重算：`Artifact.payload_bytes`
    #: 是算术值，落盘后要与文件大小**逐位相等**。写盘时重算就等于让两个来源
    #: 各自算一遍同一个数 —— 它们一致是运气，不一致才是发现。
    payloads: dict[str, bytes] = field(default_factory=dict)

    # ---- 查询

    def spec(self, spec_id: str) -> Any:
        return self.specs.get(spec_id)

    def known_spec_ids(self) -> list[str]:
        return sorted(self.specs)

    def is_frontier(self, spec_id: str) -> bool:
        return any(s.spec_id == spec_id for s in self.frontier)

    @property
    def kind_of(self) -> dict[str, str]:
        return {sid: s.finetuning.kind for sid, s in self.specs.items()}

    def gain_of(self, spec_id: str) -> float:
        ev = self.evals.get(spec_id)
        spec = self.specs.get(spec_id)
        if ev is None or spec is None:
            return float("nan")
        return ev.gain(self.base_accs)

    def interference_of(self, spec_id: str) -> float:
        """其他能力的平均跌幅。基准是 **chance**，不是 `base_accs` —— 理由见字段注释。

        缺基准时返回 NaN 而不是 0.0。0.0 是个**看起来完全正常的值**（「没有干扰」），
        它会让「基准没算出来」和「测出来确实没有干扰」变成同一个数字。
        """
        ev = self.evals.get(spec_id)
        if ev is None or not self.chance_accs:
            return float("nan")
        return ev.interference(self.chance_accs)

    def to_dict(self) -> dict[str, Any]:
        """摘要形态：**刻意不含 `plans` / `costs`**（见类文档）。"""
        return {
            "protocol_digest": self.protocol_digest,
            "version_digest": self.version_digest,
            "models": [m.to_dict() for m in self.models],
            "base_accs": dict(sorted(self.base_accs.items())),
            "chance_accs": dict(sorted(self.chance_accs.items())),
            "specs": {sid: s.to_dict() for sid, s in sorted(self.specs.items())},
            "frontier": [s.spec_id for s in self.frontier],
            "frontier_keys": {k: list(v) for k, v in sorted(self.frontier_keys.items())},
            "expectations": dict(sorted(self.expectations.items())),
            "verdicts": [dict(v) for v in self.verdicts],
            "counters": dict(self.counters),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PipelineResult":
        from .ir import TrainingSpec

        result = cls(
            protocol_digest=str(d.get("protocol_digest", "")),
            version_digest=str(d.get("version_digest", "")),
            base_accs={k: float(v) for k, v in (d.get("base_accs") or {}).items()},
            chance_accs={k: float(v) for k, v in (d.get("chance_accs") or {}).items()},
            frontier_keys={k: list(v) for k, v in (d.get("frontier_keys") or {}).items()},
            expectations=dict(d.get("expectations") or {}),
            verdicts=[dict(v) for v in d.get("verdicts", [])],
            counters=dict(d.get("counters", {})),
        )
        result.models = [ModelEntry.from_dict(m) for m in d.get("models", [])]
        result.specs = {sid: TrainingSpec.from_dict(s) for sid, s in (d.get("specs") or {}).items()}
        result.frontier = [result.specs[sid] for sid in d.get("frontier", []) if sid in result.specs]
        return result
