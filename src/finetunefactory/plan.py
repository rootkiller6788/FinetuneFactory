"""参数计划：把一份 :class:`~finetunefactory.ir.TrainingSpec` 变成一张表。

**这是整个项目里唯一认识「策略」的模块。**

:mod:`model` 不认识策略。它只接受一个 ``weights`` 映射（名字 → `Tensor` 或裸 list），
然后跑同一段前向代码。四种策略的区别**全部**落在本模块给出的那张表上：
「这个映射里哪些条目是可训练叶子、哪些是冻结常量、哪些是现算出来的」。

这条边界是「单调用路径」的形式化版本。它带来的可检验后果是：

1. `model.forward` 与 `train.step` 里**没有** ``if kind ==``。有的话，
   「同一份 Spec 得到同一个 Artifact」就不再能靠读代码核实 ——
   因为不同的 kind 会走到不同的代码路径上，而它们的漂移不会报错。
2. 换一个 kind 换掉的是**一张表**，不是一段控制流。于是
   :meth:`TrainingSpec.with_kind` 是纯粹的字段改写，四种产物必然共享
   同一段前向、同一段损失、同一个优化器步。

### 诚实性：两类数字，两种来源

本模块产出两类数字，报告里必须分开列：

- **产物字节数**（``artifact_bytes``）—— 可测量，对应真的写到磁盘上的那个文件。
- **显存记账**（``cost()``）—— 是一个**模型**，不是一次测量。它算的是
  「按这个计划，进程需要为哪些东西留出内存」。它的每一列都会被报告逐列打出来，
  因为一个说不清来源的显存表没人信。

QLoRA 的显存结论**全部**压在「基座以 4-bit 常驻」这一句上，所以本模块持有的是
:class:`~finetunefactory.model.QuantizedBase`（量化码），每次前向现反量化一份
fp32 视图然后丢掉。反过来写（存一份反量化后的 fp32 权重图省事）也能跑，
数字也好看，但它报的 4-bit 省显存就是假的 —— 而没有任何东西会报错。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from . import model as M
from .autograd import Tensor, add, constant, matmul, scale
from .ir import (
    FinetuneStrategy,
    ModelSpec,
    TrainingSpec,
)
from .rng import derive_seed

__all__ = [
    "ROLE_TRAINABLE",
    "ROLE_FROZEN",
    "ROLE_ADAPTER",
    "ADAPTER_SUFFIX_A",
    "ADAPTER_SUFFIX_B",
    "ParamEntry",
    "ParamPlan",
    "build_plan",
    "frozen_base_names",
    "adapter_names",
    "adapter_init",
]

ROLE_TRAINABLE = "trainable"
ROLE_FROZEN = "frozen"
ROLE_ADAPTER = "adapter"

ADAPTER_SUFFIX_A = ".lora_A"
ADAPTER_SUFFIX_B = ".lora_B"


# ----------------------------------------------------------------------
# 表项
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ParamEntry:
    """表里的一行：一个张量的身份、形状、角色。

    `role` 只有三种取值，而**这三种恰好就是「同一段前向代码能见到的三种东西」**：

    - `trainable`：可训练叶子。基座里被解冻的那些（full / freeze）。
    - `frozen`：冻结常量。同一段前向照样读它，只是它不接收梯度。
    - `adapter`：LoRA 的 A / B。它们**不是基座的一部分**，`base` 指向它们要改的那一行。

    `at_precision` 记的是**存储**精度而不是计算精度：量化的基座无论 4-bit 还是
    8-bit，算的时候都会先反量化成 fp32。把「存储」写成「计算」是量化类报告里
    最常见的一句错话（「4-bit 模型算得更快」），所以这个字段单独存在。
    """

    name: str
    role: str
    shape: tuple[int, ...]
    n_params: int
    base: str = ""
    at_precision: str = "fp32"

    @property
    def trainable(self) -> bool:
        return self.role in (ROLE_TRAINABLE, ROLE_ADAPTER)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "shape": list(self.shape),
            "n_params": self.n_params,
            "base": self.base,
            "at_precision": self.at_precision,
        }


def _numel(shape: Sequence[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def frozen_base_names(spec: ModelSpec, freeze_layers: int) -> frozenset[str]:
    """`freeze` 策略下被冻结的基座参数名。

    **冻结口径：emb 与第 ``[0, k)`` 层冻结，其余（第 ``[k, n)`` 层与 ``norm_f``）可训练。**

    为什么把 emb 也冻上：`tie_embeddings=True` 时 emb **同时是输出头**，
    于是「冻 emb」等于「冻结输出投影」。这让这个策略变成一个真实的约束
    （隐藏状态必须去对齐一组固定的行向量），而不是一次形式上的层数裁剪。
    代价是它更难学 —— 如果实测出来 freeze 的 gain 明显低，那是这个约束的真实后果，
    而不是实现出错。报告必须把这一条写在 freeze 的行旁边。
    """
    frozen = {M.EMB}
    for i in range(freeze_layers):
        frozen.update(f"layers.{i}.{s}" for s in M.PARAM_SUFFIXES)
    return frozenset(frozen)


def adapter_names(full_name: str) -> tuple[str, str]:
    """一个挂载点的 A / B 两个名字。**顺序即落盘顺序**，所以它是元组而不是集合。"""
    return full_name + ADAPTER_SUFFIX_A, full_name + ADAPTER_SUFFIX_B


def adapter_shapes(base_shape: Sequence[int], rank: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """LoRA 的 A / B 形状。**这里是最容易写反的地方，所以把推导写在注释里。**

    `model.param_shapes` 里的形状一律是 ``[in, out]``（因为前向算的是 ``x @ W``）。
    要让 ``W' = W + scale·(B@A)`` 与原矩阵同形，必须有::

        B: [in, rank]  @  A: [rank, out]  →  [in, out]

    于是 ``A`` 是 ``(rank, out)``、``B`` 是 ``(in, rank)``。与 PyTorch 的 LoRA
    实现（``A: [r, in]``、``B: [out, r]``）**恰好转置**，因为那边的 ``W`` 是
    ``[out, in]``。照着 PyTorch 的注释抄会得到形状不匹配 —— 这一处会报错，
    是这个设计里少数会当场报错而不是静默出错的地方。
    """
    if len(base_shape) != 2:
        raise ValueError(f"LoRA 只能挂在线性层上，得到形状 {tuple(base_shape)}")
    in_dim, out_dim = base_shape
    return (rank, out_dim), (in_dim, rank)


def adapter_init(a_shape: Sequence[int], b_shape: Sequence[int], seed: int, name: str) -> list[list[float]]:
    """A 随机、B 全零。

    **B 全零不是随手写的。** 它让 ``scale·B@A`` 在初始化时**逐位等于 0**，
    于是「第 0 步的适配器模型 = 基座模型」是一个可以断言的等式，而不是
    「近似相等」。这条等式撑起报告里的一个关键读法：**第 0 步的 gain 必须恰好为 0**。
    如果 B 也随机初始化，第 0 步的 gain 会是一个小的非零数，
    而它究竟是「随机扰动」还是「已经学到了东西」就分不清了。

    A 的尺度取 ``1/sqrt(out_dim)``：out_dim 是 A 的 fan-in（``B@A`` 里 A 是右乘子，
    它的每一行在 ``out`` 维上被求和）。取错 fan-in 的后果是初始化方差随
    ``d_ff/d_model`` 漂移，而这个漂移**不报错** —— 它表现为「gate/up 那两个
    挂载点的训练明显更差」，很容易被解读成「挂载点选择很重要」。
    """
    rnd = random.Random(derive_seed("lora_init", name, seed))
    rank, out_dim = a_shape
    std = 1.0 / math.sqrt(max(1, out_dim))
    a = [[rnd.gauss(0.0, std) for _ in range(out_dim)] for _ in range(rank)]
    in_dim, _ = b_shape
    b = [[0.0] * rank for _ in range(in_dim)]
    return a, b


# ----------------------------------------------------------------------
# 计划
# ----------------------------------------------------------------------
@dataclass
class ParamPlan:
    """一张表 + 一组常驻张量。

    **常驻的东西只有两类**：

    - `tensors`：每个基座参数一个 `Tensor`。可训练的（full / freeze 里解冻的那些）
      是叶子，梯度与优化器状态挂在它们身上、跨步存活；冻结的是 ``.freeze()`` 过的
      常量。**它们必须是 Tensor 而不是裸 list** —— 裸 list 每次进前向都会被
      `_as_tensor` 包成一个**未冻结**的新张量，于是前向会为一批永远不会被用到的
      权重建图、算梯度、占内存。症状是「LoRA 的显存并没有比 full 少多少」，
      而这恰好把本项目要测的那个结论抹掉了。
    - `adapters`：每个挂载点的 A / B 叶子。它们跨步存活，优化器状态才有效。

    **每次 `build()` 现算的东西**：LoRA 的 ``W + scale·(B@A)``，以及 qlora 的
    反量化视图。两者都是「常驻表示 → 本次前向需要的表示」的纯函数，算完即弃。
    """

    spec: TrainingSpec
    base_names: tuple[str, ...]
    shapes: dict[str, tuple[int, ...]]
    entries: tuple[ParamEntry, ...]
    tensors: dict[str, Tensor]
    adapters: dict[str, dict[str, Tensor]]
    target_names: tuple[str, ...]
    quant: M.QuantizedBase | None = None
    _frozen_base: frozenset[str] = frozenset()
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- 派生量

    @property
    def strategy(self) -> FinetuneStrategy:
        return self.spec.finetuning

    @property
    def kind(self) -> str:
        return self.spec.finetuning.kind

    @property
    def model_spec(self) -> ModelSpec:
        return self.spec.model

    @property
    def n_base_params(self) -> int:
        """基座参数个数。**由形状表算出，与 ``ModelSpec.param_count()`` 逐位相等**
        （见 :meth:`self_check`）—— 解析式与实际实例化必须对得上，
        否则所有代价数字都会整体偏移一个固定比例，而它们看起来都很正常。"""
        return sum(_numel(self.shapes[n]) for n in self.base_names)

    def trainable_entries(self) -> list[ParamEntry]:
        return [e for e in self.entries if e.trainable]

    def trainable_leaves(self) -> list[tuple[str, Tensor]]:
        """全部可训练叶子，顺序确定（基座按参数序，然后适配器按参数序）。

        **顺序确定**不是风格问题：优化器状态按这个顺序建立，训练循环按这个顺序
        读梯度。顺序一变，同一份 Spec 的两次运行就会更新到不同的张量上。
        """
        out: list[tuple[str, Tensor]] = []
        for n in self.base_names:
            t = self.tensors.get(n)
            if t is not None and not t.frozen:
                out.append((n, t))
        for full in self.target_names:
            for suffix in (ADAPTER_SUFFIX_A, ADAPTER_SUFFIX_B):
                out.append((full + suffix, self.adapters[full][suffix]))
        return out

    @property
    def n_trainable_params(self) -> int:
        return sum(e.n_params for e in self.trainable_entries())

    def is_trainable(self, name: str) -> bool:
        t = self.tensors.get(name)
        return t is not None and not t.frozen

    # ---- 前向用的权重映射

    def build(self) -> dict[str, Any]:
        """本次前向要的 ``weights`` 映射。

        **它是纯函数**（除了 qlora 每次新造一批常量之外没有状态变化），
        于是「同一份 Spec 的第 n 次前向」与「第 m 次前向」在数值上必然一致，
        前提是叶子没被更新过。这条性质让「第 0 步的 gain 恰好为 0」可断言。
        """
        if self.quant is not None:
            base: dict[str, Any] = {
                n: constant(v).freeze() for n, v in self.quant.dequant().items()
            }
        else:
            base = dict(self.tensors)
        if not self.adapters:
            return base
        scale_factor = self.strategy.scale
        for full in self.target_names:
            pair = self.adapters[full]
            delta = matmul(pair[ADAPTER_SUFFIX_B], pair[ADAPTER_SUFFIX_A])
            base[full] = add(base[full], scale( delta, scale_factor))
        return base

    def merge_base(self) -> dict[str, list[Any]]:
        """合并检验用的基座快照 —— 与 :meth:`build` **在当前时刻看到的是同一批数**。

        对量化策略**必须**返回反量化后的 fp32，而不是 ``self.tensors``：qlora 的
        ``self.tensors`` 是空的（这正是它的显存声明成立的原因，见 :meth:`self_check`），
        而合并要贴着模型实际用的那份权重做。直接取 ``self.tensors`` 会在 qlora 上
        得到一张**缺了所有基座参数**的表，然后 ``merge_weights`` 会报「基座里没有 X」
        —— 报错算是好的结果；如果恰好没报错，比的就是两份不同的东西。
        """
        if self.quant is not None:
            return {n: [list(r) for r in v] if v and isinstance(v[0], list) else list(v)
                    for n, v in self.quant.dequant().items()}
        out: dict[str, list[Any]] = {}
        for name, t in self.tensors.items():
            d = t.data
            out[name] = [list(r) for r in d] if d and isinstance(d[0], list) else list(d)
        return out

    def merge_recipe(self) -> dict[str, tuple[list[list[float]], list[list[float]]]]:
        """给 :func:`model.merge_weights` 的适配器映射：名字 → ``(A, B)``，取**当前值**。

        它返回的是叶子里的**数值快照**（裸 list），不是 Tensor —— 合并是一次
        导出动作，不是一次前向，所以它不该带图。这一点很重要：如果这里返回
        带图的张量，合并检验就会顺带把梯度也算一遍，而那个梯度没有任何用处。
        """
        out: dict[str, tuple[list[list[float]], list[list[float]]]] = {}
        for full in self.target_names:
            pair = self.adapters[full]
            out[full] = (
                [list(row) for row in pair[ADAPTER_SUFFIX_A].data],
                [list(row) for row in pair[ADAPTER_SUFFIX_B].data],
            )
        return out

    # ---- 产物
    def artifact_entries(self) -> list[ParamEntry]:
        """要写进产物文件的东西。

        **产物是「增量」，不是「整模型」** —— 四种策略一律如此：

        - `full`：全部基座参数（它的增量就是整模型，这不是特例，是同一口径下的结果）；
        - `freeze`：只写被解冻的那些层；
        - `lora` / `qlora`：只写 A / B。

        统一口径的价值在于「产物字节数」这一列是**可比的**。如果 full 写整模型而
        lora 只写适配器，两列的差里就混进了一个口径差，读者没法判断那个差有多少
        来自策略、多少来自记账方式。
        """
        if self.adapters:
            return [e for e in self.entries if e.role == ROLE_ADAPTER]
        return [e for e in self.entries if e.role == ROLE_TRAINABLE]

    def artifact_bytes(self) -> int:
        """产物字节数：**算术值**，由表推出。落盘之后必须与实测文件大小**逐位相等**。

        这条等式是本项目里唯一一处「声明值 vs 实测值」的强核对，而且它核的是
        最贵的那一列（产物大小）。它成立的前提是产物文件里**只**放 float32 数，
        没有任何对齐 / 头部 / 格式化开销 —— 所以落盘用的是裸 float32 小端，
        元数据全部放进旁边的 JSON。
        """
        return sum(e.n_params * self._bytes_of(e.at_precision) for e in self.artifact_entries())

    def _bytes_of(self, precision: str) -> int:
        """一个参数在给定精度下占几字节。**清单里的 float32 一律 4 字节**，
        即使它是被量化过的基座：产物里写的永远是反量化之后的浮点增量，
        量化基座不属于产物（它是基座的一个存储选择，而基座不是我们的产物）。"""
        return 4 if precision == "fp32" else 4

    def payload_tensors(self) -> list[tuple[str, list[Any]]]:
        """产物里的 (名字, 数值) 序列，顺序即落盘顺序。

        名字是**全名**（`layers.0.q_proj.lora_A`），于是产物文件本身携带了
        「这些数字要贴回哪里」的信息。只写数值不写名字的话，产物就只是一堆浮点，
        而「合并回基座」这件事会变成需要从旁处推断的魔法。
        """
        out: list[tuple[str, list[Any]]] = []
        for e in self.artifact_entries():
            if e.role == ROLE_ADAPTER:
                suffix = ADAPTER_SUFFIX_B if e.name.endswith(ADAPTER_SUFFIX_B) else ADAPTER_SUFFIX_A
                base = e.name[: -len(suffix)]
                data = self.adapters[base][suffix].data
            else:
                t = self.tensors.get(e.name)
                if t is None:
                    raise ValueError(
                        f"产物条目 {e.name!r} 没有对应的常驻张量："
                        "量化策略的产物只该包含适配器，出现基座条目说明计划自相矛盾"
                    )
                data = t.data
            out.append((e.name, [list(r) for r in data] if data and isinstance(data[0], list) else list(data)))
        return out

    # ---- 代价

    def cost(self, *, seq_len: int, batch: int) -> dict[str, Any]:
        """显存记账。**逐列可指认**，报告会把每一列打出来。

        ``vram_bytes = 权重 + 梯度 + 优化器状态 + 激活 + 固定开销``

        其中「权重」= 基座按它**实际存储**的精度 + 适配器。QLoRA 的省显存就来自
        基座那一项从 ``4 B/参数`` 变成 ``quant_bytes``。

        ⚠️ **激活那一项与策略无关**。它只依赖 (d_model, n_layers, seq_len, batch)。
        这不是实现细节，而是一个结论：在本项目的规模下激活是四种策略共同的底，
        于是策略之间的差异**只体现在参数与优化器状态上**。
        报告必须把这一列单独打出来，否则「qlora 显存只有 full 的 1/8」这种话
        会在读者脑子里自动成立，而实测的总显存差距远小于 8 倍。
        """
        cost_cfg = dict(self.meta.get("cost") or {})
        base_b = int(cost_cfg.get("base_bytes", 4))
        grad_b = int(cost_cfg.get("grad_bytes", 4))
        optim_b = int(cost_cfg.get("optimizer_bytes", 4))
        adapter_b = int(cost_cfg.get("adapter_bytes", 4))
        act_b = int(cost_cfg.get("activation_bytes", 4))
        act_f = float(cost_cfg.get("activation_factor", 8.0))
        overhead = int(cost_cfg.get("overhead_bytes", 0))

        n_base = self.n_base_params
        n_train = self.n_trainable_params
        n_adapter = sum(e.n_params for e in self.entries if e.role == ROLE_ADAPTER)
        base_storage = self.quant.n_bytes if self.quant is not None else n_base * base_b
        # 可训练的那部分基座**既是权重也是梯度也是优化器状态**，不能重复计。
        # full 的 n_train == n_base，于是它的梯度/优化器状态项就是「整模型的两倍」，
        # 这正是全参微调显存大的全部原因，而它比「权重本身」大得多。
        states = self.spec.optimizer.state_multiplier
        weights = base_storage + n_adapter * adapter_b
        grads = n_train * grad_b
        optim = n_train * optim_b * states
        # 激活那一项含一个浮点系数，所以取整。**四舍五入到整字节是这一列唯一的
        # 近似**，其余各列都是整数算术 —— 报告里会把这一列标成「估算」。
        act = int(round(act_b * act_f * self.model_spec.d_model * self.model_spec.n_layers * int(seq_len) * int(batch)))
        return {
            "base_storage_bytes": base_storage,
            "adapter_bytes": n_adapter * adapter_b,
            "weights_bytes": weights,
            "grad_bytes": grads,
            "optimizer_bytes": optim,
            "activation_bytes": act,
            "overhead_bytes": overhead,
            "vram_bytes": weights + grads + optim + act + overhead,
            "n_base_params": n_base,
            "n_adapter_params": n_adapter,
            "n_trainable_params": n_train,
            "optimizer_states": states,
            "quant_bytes_per_param": (self.quant.n_bytes / n_base) if self.quant is not None else float(base_b),
        }

    # ---- 三路互校里的第 ① 路：解析 MAC

    def forward_macs(self, *, seq_len: int) -> dict[str, int]:
        """**一次单样本「build + 前向」**的解析算子表，键与
        :class:`autograd.MacCounter` 的算子名一一对应。

        每一项都是从**结构参数**（T / d / ff / nl / H / vocab）与计划本身
        （挂了哪几个模块）推出来的，不读任何实测数字。它的用处是当**结构的地契**：
        谁往里加一个算子、或者不小心把最后那个 ``rmsnorm`` 拿掉了，
        这张表立刻与实测对不上。而那种改动**不会让任何别的东西报错** ——
        损失照降、准确率照动、报告照出。

        ⚠️ 别把它当成正确性证明。它证明的是「跑出来的算子集合与设计的一致」，
        不是「每个算子的数学是对的」。梯度对不对是另一件事，由
        :func:`~finetunefactory.compare.check_gradients` 用数值差分管。

        几个容易算错的项，都在下面标了它对应前向里的哪一行：

        - ``embed`` 是 ``T×d``：查表会把 ``T`` 行各拷 ``d`` 个数，**不是乘加**，
          但计数器照样记，所以这里也照样写 —— 不写就会让两路差出一整项。
        - ``rmsnorm`` 含**最后那一个**：``nl×(2·T·d) + T·d``。漏掉末尾的 norm
          是最常见的一处，因为它在所有循环之外。
        - ``add`` 含**残差**的两处：``nl×(H·T·T + 2·T·d)``。注意力掩码一处、
          残差两处，只写掩码会少一多半。
        - ``scale`` 是注意力分数那一次（``nl·H·T·T``），如果挂了 LoRA，
          再加上 ``build()`` 里对 ``Δ`` 的那一次（每个挂载点 ``in×out``）。
        """
        s = self.model_spec
        T, d, ff, nl, v, H = int(seq_len), s.d_model, s.d_ff, s.n_layers, s.vocab, s.n_heads
        per_layer_linear = T * d * d * 4 + T * d * ff * 3  # qkvo + gate/up/down
        attn_scores = T * T * d  # QK^T
        attn_apply = T * T * d  # P·V
        table = {
            "embed": T * d,
            "rmsnorm": nl * (2 * T * d) + T * d,
            "rope": 2 * nl * T * d,
            "matmul": nl * (per_layer_linear + attn_scores + attn_apply) + T * d * v,
            "scale": nl * H * T * T,
            "add": nl * (H * T * T + 2 * T * d),
            "softmax": nl * H * T * T,
            "silu": nl * T * ff,
            "mul": nl * T * ff,
        }
        # LoRA 的 Δ = scale·(B@A) 在 build() 里造，也走同样的算子。
        # 它每个挂载点要一次 matmul（in×rank×out）、一次 scale、一次 add（都按 in×out 记）。
        for full in self.target_names:
            in_dim, out_dim = self.shapes[full]
            table["matmul"] += in_dim * self.strategy.rank * out_dim
            table["scale"] += in_dim * out_dim
            table["add"] += in_dim * out_dim
        return table

    def forward_macs_total(self, *, seq_len: int) -> int:
        """解析表的合计。**报告里的「每次前向多少 MAC」就是它。**"""
        return sum(self.forward_macs(seq_len=seq_len).values())

    # ---- 自检

    def self_check(self) -> dict[str, Any]:
        """结构自检。**在跑之前**就该全部成立，所以它由 :mod:`pipeline` 在开场调用。

        三件事：
        ① 表里的基座参数量 == ``ModelSpec.param_count()``（解析式与实际实例化逐位相等，
           skill 坑 53/59）；
        ② 常驻张量的元素个数 == 表里声明的个数（表说一套、张量是另一套是可能的，
           而它不会报错，只会让所有代价数字偏移一个固定比例）；
        ③ 产物条目集合与策略相符（lora 不许有基座条目，full 不许有适配器条目）。
        """
        problems: list[str] = []
        analytic = self.model_spec.param_count()
        table_base = self.n_base_params
        if table_base != analytic:
            problems.append(f"表里基座参数量 {table_base} != ModelSpec.param_count() {analytic}")
        if self.quant is not None:
            # 量化策略**没有**常驻的 fp32 基座 —— 这正是它省显存的全部来源。
            # 所以要核的是「量化码的个数」而不是「常驻张量的元素数」。
            if self.quant.n_params != table_base:
                problems.append(f"量化基座参数量 {self.quant.n_params} != 表里 {table_base}")
            if self.tensors:
                problems.append(
                    f"量化策略不该常驻 fp32 基座张量，但 tensors 里有 {len(self.tensors)} 项"
                    "—— 留着一份 fp32 副本会让「4-bit 省显存」这句话变成假的"
                )
        else:
            alive = M.count_params({n: self.tensors[n] for n in self.base_names})
            if alive != table_base:
                problems.append(f"常驻基座张量元素数 {alive} != 表里 {table_base}")
        art = self.artifact_entries()
        if self.adapters:
            if any(e.role == ROLE_TRAINABLE for e in art):
                problems.append("peft 策略的产物里不该有基座参数")
            if not art:
                problems.append("peft 策略的产物是空的")
        else:
            if any(e.role == ROLE_ADAPTER for e in art):
                problems.append("非 peft 策略的产物里不该有适配器")
        train_names = {e.name for e in self.trainable_entries()}
        leaf_names = {n for n, _ in self.trainable_leaves()}
        if train_names != leaf_names:
            problems.append(f"表里声明的可训练集合与真实叶子不一致：{sorted(train_names ^ leaf_names)}")
        for name, t in self.trainable_leaves():
            if t.grad is not None:
                problems.append(f"叶子 {name!r} 带着上一次的梯度进计划：状态会跨 spec 泄漏")
        return {
            "ok": not problems,
            "problems": problems,
            "n_base_params": table_base,
            "n_trainable_params": self.n_trainable_params,
            "n_artifact_params": sum(e.n_params for e in art),
        }

    def to_dict(self) -> dict[str, Any]:
        """计划的可序列化形态。**它不进产物**（见 :class:`schema.PipelineResult`），
        只用于 `explain` / `plan` 这类只读子命令现场重算后打印。"""
        return {
            "spec_id": self.spec.spec_id,
            "kind": self.kind,
            "strategy": self.strategy.to_dict(),
            "base_names": list(self.base_names),
            "target_names": list(self.target_names),
            "n_base_params": self.n_base_params,
            "n_trainable_params": self.n_trainable_params,
            "n_adapter_params": sum(e.n_params for e in self.entries if e.role == ROLE_ADAPTER),
            "entries": [e.to_dict() for e in self.entries],
            "artifact_names": [e.name for e in self.artifact_entries()],
            "artifact_bytes": self.artifact_bytes(),
            "quant": self.quant.storage_report() if self.quant is not None else None,
            "meta": dict(self.meta),
        }


# ----------------------------------------------------------------------
# 构造
# ----------------------------------------------------------------------
def _copy_nested(data: list[Any]) -> list[Any]:
    """深拷一层嵌套。只拷到 list 的层级 —— 叶子一定是 float。

    为什么必须拷：:func:`autograd.constant` 把传入的 list **直接当成** ``Tensor.data``。
    于是 ``build_plan(spec, bw)`` 造出来的可训练叶子的 data **就是** ``bw[name]`` 那个
    list 本身，而训练是**原地**改 ``leaf.data`` 的 —— 一次训练跑完，``bw`` 已经被改成
    微调后的权重了。

    后果非常隐蔽：同一个进程里用同一份 ``bw`` 建第二个计划，它是从**第一个计划训练
    结束的地方**接着训的。每条曲线单独看都很正常（损失在降、准确率在涨），
    但 48 个 spec 里只有第一个是从真基座出发的 —— 整个扫描的对照关系已经没了，
    而报告里没有任何一处会显得可疑。扇出型项目一定会多次复用同一份基座权重，
    所以这个拷贝放在库里，而不是指望每个调用方记得自己拷。
    """
    return [list(r) for r in data] if data and isinstance(data[0], list) else list(data)


def _make_tensor(data: list[Any], *, trainable: bool) -> Tensor:
    t = constant(_copy_nested(data))
    return t if trainable else t.freeze()


def build_plan(
    spec: TrainingSpec,
    base_weights: Mapping[str, Any],
    *,
    cost: Mapping[str, Any] | None = None,
) -> ParamPlan:
    """从 Spec + 基座权重造计划。**这是本模块唯一的入口。**

    `cost` 是 :mod:`config` 里的显存口径。它**不参与指纹**（口径变了报告里的数字会变，
    但 Spec 没变），所以它挂在 `meta` 上而不进任何摘要。
    """
    strategy = spec.finetuning
    strategy.validate(spec.model)
    if not strategy.is_canonical():
        raise ValueError(
            f"计划只接受规范形式的策略，得到 {strategy.to_dict()}（与 kind 不符的字段非零）"
        )
    m = spec.model
    shapes = M.param_shapes(m)
    names = tuple(M.layer_param_names(m))
    missing = [n for n in names if n not in base_weights]
    if missing:
        raise ValueError(f"基座权重里缺参数 {missing}")

    frozen = frozen_base_names(m, strategy.freeze_layers) if strategy.kind == "freeze" else frozenset()
    quant: M.QuantizedBase | None = None
    precision = "fp32"
    if strategy.kind == "qlora":
        quant = M.quantize_base(base_weights, m, bits=strategy.quant_bits, group=strategy.quant_group)
        precision = "int%d" % strategy.quant_bits

    tensors: dict[str, Tensor] = {}
    entries: list[ParamEntry] = []
    for n in names:
        shape = shapes[n]
        if strategy.kind == "full":
            role = ROLE_TRAINABLE
        elif strategy.kind == "freeze":
            role = ROLE_FROZEN if n in frozen else ROLE_TRAINABLE
        else:
            role = ROLE_FROZEN
        if quant is None:
            # 非量化策略的基座**常驻**：可训练的是叶子（跨步带梯度与优化器状态），
            # 冻结的是 `.freeze()` 过的常量。两者都必须是 Tensor 而不是裸 list，
            # 否则每次进前向都会被 `_as_tensor` 包成一个未冻结的新张量，
            # 于是前向会为一批永远不会被用到的权重建图、算梯度、占内存。
            tensors[n] = _make_tensor(base_weights[n], trainable=(role == ROLE_TRAINABLE))
        entries.append(
            ParamEntry(
                name=n,
                role=role,
                shape=tuple(shape),
                n_params=_numel(shape),
                at_precision=precision,
            )
        )

    target_names: tuple[str, ...] = ()
    adapters: dict[str, dict[str, Tensor]] = {}
    if strategy.is_peft:
        target_names = tuple(M.lora_param_names(m, strategy.target_modules))
        for full in target_names:
            a_shape, b_shape = adapter_shapes(shapes[full], strategy.rank)
            a_data, b_data = adapter_init(a_shape, b_shape, spec.seed, full)
            a_t = constant(a_data)
            b_t = constant(b_data)
            adapters[full] = {ADAPTER_SUFFIX_A: a_t, ADAPTER_SUFFIX_B: b_t}
            entries.append(
                ParamEntry(
                    name=full + ADAPTER_SUFFIX_A,
                    role=ROLE_ADAPTER,
                    shape=tuple(a_shape),
                    n_params=_numel(a_shape),
                    base=full,
                )
            )
            entries.append(
                ParamEntry(
                    name=full + ADAPTER_SUFFIX_B,
                    role=ROLE_ADAPTER,
                    shape=tuple(b_shape),
                    n_params=_numel(b_shape),
                    base=full,
                )
            )

    plan = ParamPlan(
        spec=spec,
        base_names=names,
        shapes=shapes,
        entries=tuple(entries),
        tensors=tensors,
        adapters=adapters,
        target_names=target_names,
        quant=quant,
        _frozen_base=frozen,
        meta={"cost": dict(cost or {})},
    )
    # 别名检查：计划**不许**与传进来的基座共享 float 存储。共享的话，训练会在
    # 原地改 `base_weights`，于是同一份 `bw` 建的第二个计划从第一个计划训完的
    # 地方出发（详见 `_copy_nested` 的说明）。这一条是 O(1) 的，而且它核的是
    # 「这个模块的契约」，不是「某一次调用的结果」。
    # `plan.tensors` 对量化策略**按设计是空的**（常驻的是量化码，不是 fp32），
    # 所以这里用 .get —— KeyError 会让「qlora 不能建计划」，那是个假故障。
    aliased = [
        n for n in plan.base_names
        if n in base_weights
        and (t := plan.tensors.get(n)) is not None
        and t.data is base_weights[n]
    ]
    if aliased:
        raise ValueError(
            f"参数计划与传入的基座权重共享存储：{aliased[:4]}（共 {len(aliased)} 个）。"
            "训练会原地改这些 list，于是第二份计划会从第一份训完的地方出发。"
        )

    check = plan.self_check()
    if not check["ok"]:
        raise ValueError(f"参数计划自检失败（spec={spec.spec_id}）：\n  - " + "\n  - ".join(check["problems"]))
    return plan
