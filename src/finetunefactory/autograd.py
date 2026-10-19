"""微型张量 autograd 引擎（零第三方依赖）。

**为什么手写。** 本项目的核心命题是「训练策略是数据，不是代码分支」。要证明它，
就必须能逐行指出：**同一个 Train Engine，只因为参数计划不同，就长出了四种行为。**
用框架会把这条链路埋进几十层调用里（`Trainer` → `accelerate` → `peft` → …），
而这里它是一张表（:mod:`plan`）。生产当然该用 PyTorch，故 `pyproject.toml`
里留了 torch / peft / bitsandbytes 扩展点。

**与 L2 那几个项目的关系。** 共用同一套数值约定（嵌套 list 张量、每算子自带
backward 闭包、内积一律 ``math.sumprod``）。但**算子集不一样**：这里第一次需要
注意力，所以有 `split_cols` / `concat_cols` / `softmax_rows`；同时**不需要**
L2 里那些为结构搜索准备的算子。

**设计取舍。**

- 张量 = 嵌套 ``list``（1D: ``list[float]``；2D: ``list[list[float]]``）。
- 反向传播 = 每个算子自带 backward 闭包 + 反向拓扑遍历。
- **内积一律走 :func:`math.sumprod`**（CPython 3.12+ 标准库，C 实现）。
  这是「60 格网格能在两分钟内跑完」的唯一原因 —— 手写三重循环会让它变成 40 分钟。
- **算子接受「Tensor 或裸 list」**。裸 list 会被自动包成常量，于是
  「冻结的基座权重」就是图里的一个常量，而不是一个「梯度被丢掉的叶子」。
  这个区分不是修辞：它决定了 QLoRA 能不能被表达成「同一张图、不同的常量来源」。
- 每个算子都配「解析梯度 == 数值梯度」的测试（``tests/test_autograd.py``）——
  梯度写错**不会报错**，只会让训练静默变差，而「训练没用」在本项目里
  是一个**合法结论**（比如 full 和 lora 都不涨），两者输出上一模一样。

**算子计数。** :func:`mac_counter` 让引擎把自己跑的 MACs 记下来。这是代价
三路互校里的第 ② 路 —— 它与参数计划的解析公式**互相独立**：
解析公式只看 IR，计数看实际执行的算子。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterator, Sequence

#: C 速度的内积。``math.sumprod`` 自 CPython 3.12 起提供（见模块 docstring）。
_dot = math.sumprod

__all__ = [
    "Tensor",
    "constant",
    "ones",
    "zeros",
    "mac_counter",
    "MacCounter",
    "matmul",
    "add",
    "add_bias",
    "mul",
    "scale",
    "split_cols",
    "concat_cols",
    "transpose",
    "softmax_rows",
    "softmax_cross_entropy",
    "batch_cross_entropy",
    "rmsnorm",
    "layernorm",
    "silu",
    "embed",
    "sum_rows",
    "mean_rows",
    "rope",
]


# ----------------------------------------------------------------------
# 形状工具
# ----------------------------------------------------------------------
def _is2d(x: Sequence[Any]) -> bool:
    return isinstance(x, list) and bool(x) and isinstance(x[0], list)


def _is1d(x: Any) -> bool:
    return isinstance(x, list)


def _zeros_like(x: Any) -> Any:
    if not _is1d(x):
        return 0.0
    if _is2d(x):
        return [[0.0] * len(row) for row in x]
    return [0.0] * len(x)


def _ones_like(x: Any) -> Any:
    """标量（float）也要能走这里 —— 损失就是一个 float。

    早先的实现只处理 list，于是 `loss.backward()` 在**标量**上报
    `TypeError: 'float' object is not subscriptable`。这条路径平时走不到
    （大多数算子的输出是张量），只有「直接对损失求导」时才暴露 ——
    而「直接对损失求导」恰恰是所有梯度校验的第一步。
    """
    if not _is1d(x):
        return 1.0
    if _is2d(x):
        return [[1.0] * len(row) for row in x]
    return [1.0] * len(x)


def _rows(x: Sequence[Any]) -> int:
    return len(x)


def _cols(x: Sequence[Any]) -> int:
    if not x:
        return 0
    return len(x[0]) if _is2d(x) else 1  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# 算子计数
# ----------------------------------------------------------------------
class MacCounter:
    """累计引擎实际执行的乘加次数。

    **它数的是「实际跑的」，不是「应该跑的」。** 所以它与 :mod:`plan` 里的
    解析公式是两路独立证据：公式算的是「按 Spec 应该花多少」，计数记的是
    「这次真的花了多少」。两者不等就说明有一层没被算进去（比如某处
    `matmul` 被跳过了，或者某个 bias 忘了加）。
    """

    def __init__(self) -> None:
        self.by_op: dict[str, int] = {}
        self.total: int = 0

    def add(self, op: str, macs: int) -> None:
        self.by_op[op] = self.by_op.get(op, 0) + int(macs)
        self.total += int(macs)

    def to_dict(self) -> dict[str, Any]:
        return {"total": self.total, "by_op": dict(sorted(self.by_op.items()))}


_ACTIVE_COUNTER: MacCounter | None = None


def mac_counter() -> "MacCounterContext":
    """``with mac_counter() as c: ...`` —— 只统计这个块里跑的算子。"""
    return MacCounterContext()


class MacCounterContext:
    def __enter__(self) -> MacCounter:
        global _ACTIVE_COUNTER
        self._saved = _ACTIVE_COUNTER
        _ACTIVE_COUNTER = MacCounter()
        return _ACTIVE_COUNTER

    def __exit__(self, *exc: object) -> bool:
        global _ACTIVE_COUNTER
        _ACTIVE_COUNTER = self._saved
        return False


def _count(op: str, macs: int) -> None:
    if _ACTIVE_COUNTER is not None:
        _ACTIVE_COUNTER.add(op, macs)


# ----------------------------------------------------------------------
# Tensor
# ----------------------------------------------------------------------
class Tensor:
    """一个带梯度的值。

    `data` 与 `grad` 同形。叶子用 :func:`constant` 造。
    **`name` 不是装饰**：参数计划按名字决定谁可训练，而图里必须能追回名字。
    """

    __slots__ = ("data", "grad", "name", "_backward", "_prev", "_frozen")

    def __init__(self, data: Any, *, name: str | None = None, prev: Sequence["Tensor"] = ()) -> None:
        self.data = data
        self.grad: Any = None
        self.name = name
        self._backward: Callable[[], None] = lambda: None
        self._prev = tuple(prev)
        self._frozen = False

    # ---- 构造

    @property
    def shape(self) -> tuple[int, ...]:
        if _is2d(self.data):
            return (len(self.data), len(self.data[0]))
        return (len(self.data),)

    @property
    def numel(self) -> int:
        if _is2d(self.data):
            return len(self.data) * (len(self.data[0]) if self.data else 0)
        return len(self.data)

    def zero_grad(self) -> None:
        self.grad = None

    def detach(self) -> "Tensor":
        return Tensor([list(r) for r in self.data] if _is2d(self.data) else list(self.data), name=self.name)

    def freeze(self) -> "Tensor":
        """标记为「不接收梯度」。用于参数计划里被冻结的叶子。

        与「在图里当常量」不是一回事：常量**根本不进图**（没有 `_prev`），
        而冻结的叶子进了图、只是梯度被丢掉。本项目两种都用：
        LoRA 把基座当**常量**（省显存），freeze 把前几层当**冻结叶子**
        （因为它们和可训练层共享同一个 `matmul` 调用点，拆不开）。
        """
        self._frozen = True
        return self

    @property
    def frozen(self) -> bool:
        return self._frozen

    # ---- 反向

    def backward(self, seed: Any | None = None) -> None:
        topo: list[Tensor] = []
        seen: set[int] = set()

        def build(t: Tensor) -> None:
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._prev:
                build(p)
            topo.append(t)

        build(self)
        if seed is None:
            self.grad = _ones_like(self.data)
        else:
            self.grad = seed
        for t in reversed(topo):
            t._backward()

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        tag = f" name={self.name!r}" if self.name else ""
        return f"Tensor(shape={self.shape}{tag})"


def _copy_grad(g: Any) -> Any:
    """按梯度形状拷一层。**这个拷贝不是性能优化，是正确性。**

    梯度是嵌套 list，而 list 是可变的：`t.grad = g` 挂上去的是**消费者的那个对象**。
    同一个 `_bw` 常常把自己收到的那一个 `t.grad` 分给多个输入
    （`add` 就是最典型的：`_acc(ta, t.grad)` 与 `_acc(tb, t.grad)`），
    于是两个**不同的节点**从此共享同一个 list。之后任何一次对其中之一的
    原地累加（下面那两个 `+=` 分支）都会**同时改掉另一个**。

    实测过的最小复现：`y = x@W; s = y + 0; z = s + b; z2 = z + x`，反向后
    `y.grad is x.grad` 为 True，而 `y.grad` 显示 `[[2,2]]` —— 它真实的梯度是
    `[[1,1]]`，多出来的那一份是从 `x` 那条路上**改过来**的。

    这个错误不报错、不产生 NaN、形状也一直是对的。它只在「一个节点有多个消费者、
    且它的兄弟与它共享了同一份梯度对象」时出现，于是模型越大越容易碰，
    而单看某一条曲线的数字又完全正常。梯度数值差分是唯一能把它照出来的东西 ——
    `compare.check_gradients` 正是为此存在。
    """
    if _is2d(g):
        return [list(row) for row in g]
    return list(g)


def _acc(t: Tensor, g: Any) -> None:
    """梯度累加。**必须累加不能覆盖** —— 一个张量被用两次时覆盖会静默少一半梯度。

    ⚠️ 首次赋值也必须**拷贝**（见 :func:`_copy_grad`）：挂上消费者的对象会让
    两个节点共享同一份梯度，后续的原地累加就会互相污染。
    """
    if t._frozen:
        return
    if t.grad is None:
        t.grad = _copy_grad(g)
    elif _is2d(g):
        for i, row in enumerate(g):
            for j, v in enumerate(row):
                t.grad[i][j] += v
    elif isinstance(g, list):
        for i, v in enumerate(g):
            t.grad[i] += v
    else:
        # 0 维（标量）梯度。`batch_cross_entropy` 之类聚合成一个数时会出现。
        t.grad += g


def _as_tensor(x: Any, *, name: str | None = None) -> Tensor:
    """把裸 list 包成常量。**所有算子都从这里进来。**

    这是「冻结基座 = 图里的常量」这条设计的落点：调用点不需要知道
    「这个权重是可训练的还是冻结的」，它只需要知道「这是个值」。
    """
    if isinstance(x, Tensor):
        return x
    return constant(x, name=name)


def constant(data: Any, *, name: str | None = None) -> Tensor:
    return Tensor(data, name=name)


def ones(shape: tuple[int, ...], *, name: str | None = None) -> Tensor:
    if len(shape) == 1:
        return Tensor([1.0] * shape[0], name=name)
    return Tensor([[1.0] * shape[1] for _ in range(shape[0])], name=name)


def zeros(shape: tuple[int, ...], *, name: str | None = None) -> Tensor:
    if len(shape) == 1:
        return Tensor([0.0] * shape[0], name=name)
    return Tensor([[0.0] * shape[1] for _ in range(shape[0])], name=name)


# ----------------------------------------------------------------------
# 线性代数
# ----------------------------------------------------------------------
def matmul(a: Any, b: Any) -> Tensor:
    """2D 矩阵乘。``a`` 可以是 Tensor 或裸 list。

    MACs = ``rows(a) × cols(a) × cols(b)``。这是全项目唯一的热点，
    所以它也是 :func:`mac_counter` 里占比最大的那一项。
    """
    ta, tb = _as_tensor(a), _as_tensor(b)
    A, B = ta.data, tb.data
    if not _is2d(A) or not _is2d(B):
        raise ValueError(f"matmul 只支持 2D，得到 {ta.shape} × {tb.shape}")
    m, k, n = len(A), len(B), len(B[0]) if B else 0
    if len(A[0]) != k:
        raise ValueError(f"matmul 形状不匹配：{ta.shape} × {tb.shape}")
    Bt = list(zip(*B))  # 转置一次，内层循环就只用走 sumprod
    out = [[_dot(row, col) for col in Bt] for row in A]
    _count("matmul", m * k * n)
    t = Tensor(out, prev=(ta, tb))

    def _bw() -> None:
        if t.grad is None:
            return
        G = t.grad
        # dA = G @ Bᵀ ; dB = Aᵀ @ G
        if not ta._frozen:
            gA = [[_dot(row, col) for col in B] for row in G]
            _acc(ta, gA)
        if not tb._frozen:
            At = list(zip(*A))
            Gt = list(zip(*G))
            gB = [[_dot(row, col) for col in Gt] for row in At]
            _acc(tb, gB)

    t._backward = _bw
    return t


def transpose(a: Any) -> Tensor:
    ta = _as_tensor(a)
    out = [list(col) for col in zip(*ta.data)]
    t = Tensor(out, prev=(ta,))

    def _bw() -> None:
        if t.grad is None:
            return
        _acc(ta, [list(col) for col in zip(*t.grad)])

    t._backward = _bw
    return t


def split_cols(x: Any, sizes: Sequence[int]) -> tuple[Tensor, ...]:
    """沿最后一维切成若干块。**注意力按头切分就靠它。**

    反向是 `concat_cols` —— 于是「切了再拼回去」是一个恒等变换，
    而这条性质可以直接用数值差分验（见 :func:`~finetunefactory.compare.check_gradients`）。

    ⚠️ **每个输出各自负责自己那一段**，不要图省事让所有输出共用一个「把整条
    梯度拼回来」的闭包。引擎对**每一个节点**各调一次 `_backward`
    （`Tensor.backward` 是从后往前遍历 `topo`，而 N 个输出就是 N 个节点），
    共用闭包会让同一条合并梯度被 `_acc` 累加 N 次 —— `tx.grad` 直接变成 **N 倍**。

    这个错误曾经真的在这里：按头切分时 `nh=4`，于是 `q` 的梯度是 4 倍。
    它让 `norm1` 的梯度在**每一层**都不对，而 `w_q`/`w_k`/`w_v` 看起来是对的
    （它们由 `h^T @ G_q` 算出，乘数与真值同倍放大后被同一个差分口径吃掉一部分），
    报告里的损失曲线、准确率、显存全都正常 —— 只有梯度数值差分能照出它。
    这就是 `compare.check_gradients` 存在的理由。
    """
    tx = _as_tensor(x)
    width = _cols(tx.data)
    if sum(sizes) != width:
        raise ValueError(f"split_cols 宽度对不上：{width} vs sum({list(sizes)})")
    parts: list[list[Any]] = []
    offsets: list[int] = []
    off = 0
    for s in sizes:
        offsets.append(off)
        parts.append([row[off : off + s] for row in tx.data])
        off += s
    outs = tuple(Tensor(p, prev=(tx,)) for p in parts)

    def _make_bw(o: Tensor, start: int, s: int) -> Any:
        def _bw() -> None:
            if o.grad is None:
                return
            # 只填自己那一段，其余补 0：这样 N 个输出各累加一次，合起来正好是一次拼接。
            # 补 0 而不是「只写自己那一段的窄表」，是因为 `_acc` 是按位置累加的，
            # 而 `tx.grad` 必须始终是全宽 —— 直接交一张半宽的表会静默改动全宽的定义。
            rows = len(tx.data)
            merged = [[0.0] * width for _ in range(rows)]
            for i in range(rows):
                for j in range(s):
                    merged[i][start + j] = o.grad[i][j]
            _acc(tx, merged)

        return _bw

    for o, start, s in zip(outs, offsets, sizes):
        o._backward = _make_bw(o, start, s)
    return outs


def concat_cols(parts: Sequence[Any]) -> Tensor:
    ts = tuple(_as_tensor(p) for p in parts)
    rows = len(ts[0].data)
    sizes = [len(t.data[0]) for t in ts]
    out = [[v for t in ts for v in t.data[i]] for i in range(rows)]
    t = Tensor(out, prev=ts)

    def _bw() -> None:
        if t.grad is None:
            return
        off = 0
        for tt, s in zip(ts, sizes):
            _acc(tt, [row[off : off + s] for row in t.grad])
            off += s

    t._backward = _bw
    return t


# ----------------------------------------------------------------------
# 逐元素
# ----------------------------------------------------------------------
def add(a: Any, b: Any) -> Tensor:
    ta, tb = _as_tensor(a), _as_tensor(b)
    if ta.shape != tb.shape:
        raise ValueError(f"add 形状不匹配：{ta.shape} vs {tb.shape}")
    if _is2d(ta.data):
        out = [[x + y for x, y in zip(r1, r2)] for r1, r2 in zip(ta.data, tb.data)]
    else:
        out = [x + y for x, y in zip(ta.data, tb.data)]
    _count("add", ta.numel)
    t = Tensor(out, prev=(ta, tb))

    def _bw() -> None:
        if t.grad is None:
            return
        _acc(ta, t.grad)
        _acc(tb, t.grad)

    t._backward = _bw
    return t


def add_bias(x: Any, b: Any) -> Tensor:
    """给 2D 张量的每一行加上同一个 1D 偏置。

    **偏置是参数计划里最容易被漏掉的一项**：LoRA 只挂在线性层的权重上，
    偏置要么全训要么全冻。漏掉它，参数量与显存都会对不上。
    """
    tx, tb = _as_tensor(x), _as_tensor(b)
    if not _is2d(tx.data):
        raise ValueError("add_bias 只支持 2D 输入")
    if len(tb.data) != _cols(tx.data):
        raise ValueError(f"add_bias 宽度不匹配：{tb.shape} vs {tx.shape}")
    out = [[v + bv for v, bv in zip(row, tb.data)] for row in tx.data]
    _count("add_bias", tx.numel)
    t = Tensor(out, prev=(tx, tb))

    def _bw() -> None:
        if t.grad is None:
            return
        _acc(tx, t.grad)
        acc = [0.0] * len(tb.data)
        for row in t.grad:
            for j, v in enumerate(row):
                acc[j] += v
        _acc(tb, acc)

    t._backward = _bw
    return t


def mul(a: Any, b: Any) -> Tensor:
    """逐元素相乘（同形，或 1D 沿最后一维广播）。"""
    ta, tb = _as_tensor(a), _as_tensor(b)
    if ta.shape == tb.shape:
        if _is2d(ta.data):
            out = [[x * y for x, y in zip(r1, r2)] for r1, r2 in zip(ta.data, tb.data)]
        else:
            out = [x * y for x, y in zip(ta.data, tb.data)]
        bc = False
    elif _is2d(ta.data) and len(tb.data) == _cols(ta.data):
        out = [[x * y for x, y in zip(row, tb.data)] for row in ta.data]
        bc = True
    else:
        raise ValueError(f"mul 形状不匹配：{ta.shape} vs {tb.shape}")
    _count("mul", ta.numel)
    t = Tensor(out, prev=(ta, tb))

    def _bw() -> None:
        if t.grad is None:
            return
        if bc:
            gb = [0.0] * len(tb.data)
            for i, row in enumerate(t.grad):
                for j, g in enumerate(row):
                    gb[j] += g * ta.data[i][j]
            _acc(tb, gb)
            _acc(ta, [[g * y for g, y in zip(row, tb.data)] for row in t.grad])
        else:
            _acc(ta, _ewise(t.grad, tb.data, "*"))
            _acc(tb, _ewise(t.grad, ta.data, "*"))

    t._backward = _bw
    return t


def _ewise(g: Any, other: Any, op: str) -> Any:
    if _is2d(g):
        if _is2d(other):
            return [[(x * y if op == "*" else x / y) for x, y in zip(r1, r2)] for r1, r2 in zip(g, other)]
        return [[(x * y if op == "*" else x / y) for x, y in zip(row, other)] for row in g]
    return [(x * y if op == "*" else x / y) for x, y in zip(g, other)]


def scale(a: Any, s: float) -> Tensor:
    ta = _as_tensor(a)
    out = [[v * s for v in row] for row in ta.data] if _is2d(ta.data) else [v * s for v in ta.data]
    _count("scale", ta.numel)
    t = Tensor(out, prev=(ta,))

    def _bw() -> None:
        if t.grad is None:
            return
        _acc(ta, [[v * s for v in row] for row in t.grad] if _is2d(t.grad) else [v * s for v in t.grad])

    t._backward = _bw
    return t


def silu(a: Any) -> Tensor:
    ta = _as_tensor(a)

    def f(v: float) -> float:
        return v / (1.0 + math.exp(-v)) if v > -30 else 0.0

    def df(v: float) -> float:
        if v < -30:
            return 0.0
        s = 1.0 / (1.0 + math.exp(-v))
        return s * (1.0 + v * (1.0 - s))

    out = [[f(v) for v in row] for row in ta.data] if _is2d(ta.data) else [f(v) for v in ta.data]
    _count("silu", ta.numel)
    t = Tensor(out, prev=(ta,))

    def _bw() -> None:
        if t.grad is None:
            return
        if _is2d(ta.data):
            _acc(ta, [[g * df(v) for g, v in zip(gr, vr)] for gr, vr in zip(t.grad, ta.data)])
        else:
            _acc(ta, [g * df(v) for g, v in zip(t.grad, ta.data)])

    t._backward = _bw
    return t


# ----------------------------------------------------------------------
# 归一化与 softmax
# ----------------------------------------------------------------------
def rmsnorm(x: Any, w: Any, *, eps: float = 1e-6) -> Tensor:
    """RMSNorm：``w * x / sqrt(mean(x²) + eps)``。

    注意分母是 **mean** 不是 sum：用 sum 会让输出尺度随 d_model 变化，
    于是「同一个学习率在不同宽度下表现不同」—— 而这个项目要横向比不同配置，
    那种差异会被误读成策略差异。
    """
    tx, tw = _as_tensor(x), _as_tensor(w)
    if not _is2d(tx.data):
        raise ValueError("rmsnorm 只支持 2D 输入")
    d = _cols(tx.data)
    if len(tw.data) != d:
        raise ValueError(f"rmsnorm 宽度不匹配：{tw.shape} vs {tx.shape}")
    out: list[list[float]] = []
    inv: list[float] = []
    for row in tx.data:
        ms = _dot(row, row) / d
        r = 1.0 / math.sqrt(ms + eps)
        inv.append(r)
        out.append([v * r * wi for v, wi in zip(row, tw.data)])
    _count("rmsnorm", tx.numel)
    t = Tensor(out, prev=(tx, tw))

    def _bw() -> None:
        if t.grad is None:
            return
        G = t.grad
        gw = [0.0] * d
        for i, row in enumerate(tx.data):
            for j, v in enumerate(row):
                gw[j] += G[i][j] * v * inv[i]
        _acc(tw, gw)
        gx: list[list[float]] = []
        for i, row in enumerate(tx.data):
            r = inv[i]
            s = 0.0
            for j in range(d):
                s += G[i][j] * tw.data[j] * row[j]
            s = s * (r ** 3) / d
            gx.append([G[i][j] * tw.data[j] * r - row[j] * s for j in range(d)])
        _acc(tx, gx)

    t._backward = _bw
    return t


def layernorm(x: Any, w: Any, b: Any, *, eps: float = 1e-5) -> Tensor:
    """LayerNorm。**留着它是因为「同一个 IR 换一个 norm 实现」也是一次实验**，
    而两套实现的 backward 是独立的，不能靠「看起来差不多」互证。
    """
    tx, tw, tb = _as_tensor(x), _as_tensor(w), _as_tensor(b)
    if not _is2d(tx.data):
        raise ValueError("layernorm 只支持 2D 输入")
    d = _cols(tx.data)
    out: list[list[float]] = []
    stats: list[tuple[float, float]] = []
    for row in tx.data:
        mu = sum(row) / d
        var = sum((v - mu) ** 2 for v in row) / d
        r = 1.0 / math.sqrt(var + eps)
        stats.append((mu, r))
        out.append([(v - mu) * r * wi + bi for v, wi, bi in zip(row, tw.data, tb.data)])
    _count("layernorm", tx.numel)
    t = Tensor(out, prev=(tx, tw, tb))

    def _bw() -> None:
        if t.grad is None:
            return
        G = t.grad
        gw = [0.0] * d
        gb = [0.0] * d
        gx = [[0.0] * d for _ in range(len(tx.data))]
        for i, row in enumerate(tx.data):
            mu, r = stats[i]
            xh = [(v - mu) * r for v in row]
            # ⚠️ 均值必须对 **g·w** 求，不是对 g 求。
            # `y = xh*w + b`，所以 dL/dxh = g*w；早先的实现漏了那个 w，
            # 于是 layernorm 的解析梯度错了 1.1（数值梯度 1e-10 量级，
            # 而解析给的是另一个数）—— 而它**不报错**，只让训练静默变差。
            gwh = [G[i][j] * tw.data[j] for j in range(d)]
            gm = sum(gwh) / d
            gxh = sum(gwh[j] * xh[j] for j in range(d)) / d
            for j in range(d):
                gw[j] += G[i][j] * xh[j]
                gb[j] += G[i][j]
                gx[i][j] = r * (gwh[j] - gm - xh[j] * gxh)
        _acc(tw, gw)
        _acc(tb, gb)
        _acc(tx, gx)

    t._backward = _bw
    return t


def softmax_rows(x: Any) -> Tensor:
    tx = _as_tensor(x)
    if not _is2d(tx.data):
        raise ValueError("softmax_rows 只支持 2D 输入")
    out: list[list[float]] = []
    for row in tx.data:
        m = max(row)
        ex = [math.exp(v - m) for v in row]
        s = sum(ex)
        out.append([v / s for v in ex])
    _count("softmax", tx.numel)
    t = Tensor(out, prev=(tx,))

    def _bw() -> None:
        if t.grad is None:
            return
        gx: list[list[float]] = []
        for p, g in zip(out, t.grad):
            dot = _dot(p, g)
            gx.append([pi * (gi - dot) for pi, gi in zip(p, g)])
        _acc(tx, gx)

    t._backward = _bw
    return t


def _ce_parts(
    logits_data: Sequence[Sequence[float]],
    targets: Sequence[int],
    mask: Sequence[float] | None,
) -> tuple[list[list[float]], float, float]:
    """交叉熵的**共享核心**：返回（逐行 softmax 概率、掩码平均损失、分母）。

    抽出来是为了让标量版与批量版**共用同一份数学**。两份实现漂移的后果是
    「单条样本的损失」与「一个 batch 的损失」口径不一致 —— 报告里的
    loss 曲线与逐样本诊断会对不上，而两者都不报错。
    """
    T = len(logits_data)
    if len(targets) != T:
        raise ValueError(f"targets 长度 {len(targets)} != logits 行数 {T}")
    if mask is None:
        mask = [1.0 if t >= 0 else 0.0 for t in targets]
    if len(mask) != T:
        raise ValueError(f"mask 长度 {len(mask)} != logits 行数 {T}")
    probs: list[list[float]] = []
    acc = 0.0
    for row, tgt, mi in zip(logits_data, targets, mask):
        if tgt < 0:
            probs.append([0.0] * len(row))
            continue
        m = max(row)
        ex = [math.exp(v - m) for v in row]
        s = sum(ex)
        p = [v / s for v in ex]
        probs.append(p)
        if mi != 0.0:
            acc += mi * -math.log(max(p[tgt], 1e-30))
    denom = sum(mask)
    if denom <= 0:
        raise ValueError("mask 全为 0：这条样本没有任何监督信号，不该进 batch")
    return probs, acc / denom, denom


def softmax_cross_entropy(logits: Any, targets: Sequence[int], mask: Sequence[float] | None = None) -> Tensor:
    """单条样本的掩码交叉熵。**`mask` 是 IR 的一部分，不是实现细节。**

    `objective="sft"` 与 `"sft_response_only"` 的**唯一**差别就是这个 mask：
    前者全 1，后者只在回答那段 token 上是 1。把它做成一个参数（而不是两段代码），
    是「策略是数据」这条命题在损失函数上的落点。

    `targets[t] == -1` 表示「这个位置没有监督信号」（padding）。掩码为 0 的位置
    既不进分子也不进分母 —— **除以 `sum(mask)` 而不是 `T`**，否则回答短的那条
    样本会被系统性低估。

    ⚠️ 它的输出是一个**标量** `Tensor`（`data` 是 float），而本引擎的算子只认
    嵌套 list。所以标量损失**不能**用 :func:`add` / :func:`scale` 相加 ——
    要合并一个 batch 请用 :func:`batch_cross_entropy`。
    """
    tl = _as_tensor(logits)
    if not _is2d(tl.data):
        raise ValueError("softmax_cross_entropy 只支持 2D logits")
    probs, loss, denom = _ce_parts(tl.data, targets, mask)
    if mask is None:
        mask = [1.0 if t >= 0 else 0.0 for t in targets]
    _count("ce", sum(len(row) for row in tl.data))
    t = Tensor(loss, prev=(tl,))

    def _bw() -> None:
        if t.grad is None:
            return
        g = t.grad * (1.0 / denom)
        gx: list[list[float]] = []
        for i, (p, tgt) in enumerate(zip(probs, targets)):
            mi = mask[i]
            if mi == 0 or tgt < 0:
                gx.append([0.0] * len(p))
                continue
            row_g = [mi * g * pi for pi in p]
            row_g[tgt] -= mi * g
            gx.append(row_g)
        _acc(tl, gx)

    t._backward = _bw
    return t


def batch_cross_entropy(
    pairs: Sequence[tuple[Any, Sequence[int], Sequence[float] | None]],
    *,
    weight: float = 1.0,
) -> Tensor:
    """一个 batch 的平均损失，返回 ``[loss]``（**一个只有一个元素的 list**）。

    `pairs` 是 ``[(logits, targets, mask), ...]``，每条样本一个前向结果。

    `weight` 是**留给梯度累积的**：把整个损失乘一个常数。梯度累积的每个微批
    各调一次 ``backward``，梯度是**累加**进叶子的，所以 ``grad_accum`` 个微批之后
    叶子上是「各微批均值的和」，要得到「等效大批次的均值」就得乘 ``1/grad_accum``。
    用「损失乘常数」还是「学习率除常数」在数学上等价，但损失乘更安全 ——
    学习率还要被调度器乘一次，两处都改会很容易漏掉一处。

    **为什么需要它。** 本引擎的算子是「嵌套 list 进、嵌套 list 出」，而
    :func:`softmax_cross_entropy` 的产物是**标量**（`data` 是 float）。
    标量之间没有任何可用的加法算子（`add` 会去读 `.shape` 然后炸），
    于是「一个 batch 的损失」根本无法用现有算子拼出来。把它做成一个算子，
    数学与图连接性就都在一处。

    **`batch` 的损失一律取平均，绝不取和**（skill 坑 48）。取和会让梯度范数
    正比于 batch 大小，于是 `grad_clip` 几乎总是把梯度整体裁掉 ——
    症状极隐蔽：loss 曲线在降（在降和），数值「合理」，但参数几乎没更新，
    而且「改 batch 大小」会等效于「改学习率」。

    每条样本**先在自己的掩码分母上归一化，再对样本取平均** —— 顺序反了
    会让回答短的那条样本被系统性低估。
    """
    if not pairs:
        raise ValueError("batch_cross_entropy 收到空的 batch")
    tensors = [_as_tensor(p[0]) for p in pairs]
    parts: list[tuple[list[list[float]], float, float]] = []
    total_rows = 0
    for tl, (_, targets, mask) in zip(tensors, pairs):
        if not _is2d(tl.data):
            raise ValueError("batch_cross_entropy 的 logits 必须是 2D")
        parts.append(_ce_parts(tl.data, targets, mask))
        total_rows += len(tl.data)
    n = len(parts)
    loss = weight * sum(p[1] for p in parts) / n
    _count("ce", total_rows * (len(tensors[0].data[0]) if tensors[0].data else 0))
    t = Tensor([loss], prev=tuple(tensors))

    def _bw() -> None:
        if t.grad is None:
            return
        g = weight * t.grad[0] / n
        for tl, (_, targets, mask), (probs, _loss, denom) in zip(tensors, pairs, parts):
            if mask is None:
                mask = [1.0 if tg >= 0 else 0.0 for tg in targets]
            scale_i = g / denom
            gx: list[list[float]] = []
            for i, (p, tgt) in enumerate(zip(probs, targets)):
                mi = mask[i]
                if mi == 0 or tgt < 0:
                    gx.append([0.0] * len(p))
                    continue
                row_g = [mi * scale_i * pi for pi in p]
                row_g[tgt] -= mi * scale_i
                gx.append(row_g)
            _acc(tl, gx)

    t._backward = _bw
    return t


# ----------------------------------------------------------------------
# 嵌入与归约
# ----------------------------------------------------------------------
def embed(table: Any, ids: Sequence[int]) -> Tensor:
    """按行取嵌入。反向是「把梯度散加回去」—— **同一个 id 出现两次必须加两次**。"""
    tt = _as_tensor(table)
    if not _is2d(tt.data):
        raise ValueError("embed 只支持 2D 表")
    n, d = len(tt.data), len(tt.data[0])
    for i in ids:
        if not (0 <= i < n):
            raise IndexError(f"embed 下标越界：{i} 不在 [0, {n})")
    out = [list(tt.data[i]) for i in ids]
    _count("embed", len(ids) * d)
    t = Tensor(out, prev=(tt,))

    def _bw() -> None:
        if t.grad is None:
            return
        g = [[0.0] * d for _ in range(n)]
        for row_g, i in zip(t.grad, ids):
            for j, v in enumerate(row_g):
                g[i][j] += v
        _acc(tt, g)

    t._backward = _bw
    return t


def sum_rows(x: Any) -> Tensor:
    tx = _as_tensor(x)
    if not _is2d(tx.data):
        raise ValueError("sum_rows 只支持 2D 输入")
    out = [sum(col) for col in zip(*tx.data)]
    _count("sum_rows", tx.numel)
    t = Tensor(out, prev=(tx,))

    def _bw() -> None:
        if t.grad is None:
            return
        _acc(tx, [list(t.grad) for _ in tx.data])

    t._backward = _bw
    return t


def mean_rows(x: Any) -> Tensor:
    tx = _as_tensor(x)
    if not _is2d(tx.data):
        raise ValueError("mean_rows 只支持 2D 输入")
    n = len(tx.data)
    out = [s / n for s in map(sum, zip(*tx.data))]
    _count("mean_rows", tx.numel)
    t = Tensor(out, prev=(tx,))

    def _bw() -> None:
        if t.grad is None:
            return
        g = [v / n for v in t.grad]
        _acc(tx, [list(g) for _ in tx.data])

    t._backward = _bw
    return t


# ----------------------------------------------------------------------
# 旋转位置编码（RoPE）
# ----------------------------------------------------------------------
#: RoPE 的底。改它等于改「模型能分辨多远的位置」，所以它进 IR 的指纹没有意义
#: （它不进指纹），但它必须是一个**具名常量**而不是散落在算式里的 10000.0。
ROPE_BASE = 10000.0


def rope(x: Any, positions: Sequence[int], *, base: float = ROPE_BASE) -> Tensor:
    """把每个头内的通道两两配对做旋转。

    **它存在的原因是 IR 的参数量公式里没有位置嵌入。** :meth:`ir.ModelSpec.param_count`
    只数了 `emb + 各层 + 最终 norm`，所以任何「可学习的位置表」都会让
    「解析公式 == 实测参数量」这条互校（skill 坑 53/59）当场破掉。
    RoPE 不带参数，于是它既给了模型位置信息（没有它「按位置寻址」类任务在数学上
    不可解，见坑 46），又不破坏那条互校。

    形状：`x` 是 ``[T, head_dim]``（**单个头**），`positions` 长度必须等于 `T`。
    配对方式是相邻两个通道一组（``2j`` 与 ``2j+1``），所以 `head_dim` 必须是偶数 ——
    `ir.ModelSpec` 已经在构造期拒绝奇数 `head_dim` 了，这里再断言一次。

    反向是**同一个旋转的逆**（旋转矩阵正交，转置即逆）：
    ``gx = R(-θ)·g``。写出这一步时最容易犯的错是只对 ``g`` 用 ``cos`` 而忘了
    ``sin`` 项要变号 —— 它**不报错**，只让梯度比真值小一点，表现为「训练慢」，
    而「训练慢」在这个项目里是一个合法结论，于是这种 bug 会被当成实验现象。
    """
    tx = _as_tensor(x)
    if not _is2d(tx.data):
        raise ValueError("rope 只支持 2D 输入（单头：[T, head_dim]）")
    T = len(tx.data)
    hd = _cols(tx.data)
    if hd % 2 != 0:
        raise ValueError(f"rope 要求 head_dim 为偶数，得到 {hd}")
    if len(positions) != T:
        raise ValueError(f"rope 的 positions 长度 {len(positions)} != 行数 {T}")

    half = hd // 2
    # 每个通道对的角频率：θ_j = base^(-2j/hd)
    theta = [base ** (-2.0 * j / hd) for j in range(half)]
    # 一次算好 (cos, sin) 表 —— 它是位置的纯函数，与 x 无关
    table: list[list[tuple[float, float]]] = []
    for p in positions:
        row: list[tuple[float, float]] = []
        for j in range(half):
            ang = float(p) * theta[j]
            row.append((math.cos(ang), math.sin(ang)))
        table.append(row)

    out: list[list[float]] = []
    for i, row in enumerate(tx.data):
        new_row = [0.0] * hd
        for j in range(half):
            c, s = table[i][j]
            a, b = row[2 * j], row[2 * j + 1]
            new_row[2 * j] = a * c - b * s
            new_row[2 * j + 1] = a * s + b * c
        out.append(new_row)
    _count("rope", tx.numel)
    t = Tensor(out, prev=(tx,))

    def _bw() -> None:
        if t.grad is None:
            return
        G = t.grad
        gx: list[list[float]] = []
        for i, grow in enumerate(G):
            new_row = [0.0] * hd
            for j in range(half):
                c, s = table[i][j]
                ga, gb = grow[2 * j], grow[2 * j + 1]
                # R(-θ) 作用在梯度上（= R(θ)ᵀ）
                new_row[2 * j] = ga * c + gb * s
                new_row[2 * j + 1] = -ga * s + gb * c
            gx.append(new_row)
        _acc(tx, gx)

    t._backward = _bw
    return t
