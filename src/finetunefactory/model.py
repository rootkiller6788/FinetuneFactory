"""微型 Transformer：结构、初始化、前向、合并、量化。

**这个模块不认识策略。** 它只接受一个 `weights` 映射（名字 → `Tensor` 或裸 list），
然后跑同一段前向代码。四种策略的区别**全部**由 :mod:`plan` 决定「这个映射里
哪些条目是可训练的叶子、哪些是冻结常量」—— 于是「策略是数据，不是代码分支」
这条命题在代码层面是可逐行指认的：本文件里搜不到 `if kind ==`。

### 结构（与 `ir.ModelSpec.param_count` 逐项对齐）

```
emb [vocab, d]
每层：
  norm1 [d]                       rmsnorm
  q_proj / k_proj / v_proj / o_proj   [d, d]        ×4
  norm2 [d]                       rmsnorm
  gate_proj / up_proj [d_ff, d]，down_proj [d, d_ff]  ×3   （SwiGLU）
norm_f [d]
```

合计 `vocab*d + n_layers*(4d² + 3d·d_ff + 2d) + d` —— 与
:meth:`ir.ModelSpec.param_count` **逐位相等**。这条等式由
:func:`count_params` 与 `ModelSpec.param_count` 当场互校（见 :mod:`plan`
的三路互校第 ① 路）：参数量公式与实例化漂移是静默的，校验说没超预算、
装配出来却超了（skill 坑 53/59）。

### 为什么位置信息走 RoPE 而不是可学习的位置表

`param_count` 的公式里**没有**位置嵌入这一项。任何可学习的位置表都会让上面
那条互校当场破掉。RoPE 不带参数，于是它两件事同时成立：模型拿得到位置信息
（没有它，「按位置寻址」类任务在无位置嵌入的模型上**数学不可解**，见坑 46），
参数量公式也不用动。代价是 `head_dim` 必须是偶数 —— `ModelSpec` 已经在
构造期拒了奇数。

### 初始化必须可复现，且 LoRA 的 B 必须是零

每个参数的初始化种子来自 `derive_seed("init", name, seed)`，**只依赖参数名与
总种子**，不依赖遍历顺序 —— 否则「同一份 Spec 得到同一个 Artifact」会多出一个
可以漂移的自由度。

LoRA / QLoRA 的 `B` 初始化为全 0（`A` 走正态）。于是 `scale·B@A = 0`，
**适配器模型在第 0 步与基座逐位相同**。这不是为了好看：它让「微调的起点就是基座」
成为一条可断言的性质（`gain` 在 step 0 必须为 0），否则「涨了多少」这个数字
里会混进「初始化的运气」。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .autograd import (
    Tensor,
    add,
    concat_cols,
    constant,
    embed,
    matmul,
    mul,
    rmsnorm,
    rope,
    scale,
    silu,
    softmax_rows,
    split_cols,
    transpose,
)
from .ir import LORA_ABLE_MODULES, ModelSpec
from .rng import derive_seed

__all__ = [
    "EMB",
    "NORM_F",
    "PARAM_SUFFIXES",
    "MASK_NEG",
    "QuantizedBase",
    "param_shapes",
    "layer_param_names",
    "lora_param_names",
    "count_params",
    "init_base_weights",
    "causal_mask",
    "forward",
    "merge_weights",
    "quantize_base",
    "flatten_by_name",
    "inflate_by_name",
    "logits_of",
]

EMB = "emb"
NORM_F = "norm_f"

#: 每层里的参数后缀。**顺序即落盘顺序、即量化分组的顺序**，
#: 所以它必须是一个元组而不是集合 —— 集合的迭代顺序会让分组悄悄改变。
PARAM_SUFFIXES: tuple[str, ...] = (
    "norm1",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "norm2",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def layer_param_names(spec: ModelSpec) -> list[str]:
    """全部参数名，按「量化分组顺序」排列。"""
    names = [EMB]
    for i in range(spec.n_layers):
        names.extend(f"layers.{i}.{s}" for s in PARAM_SUFFIXES)
    names.append(NORM_F)
    return names


def lora_param_names(spec: ModelSpec, targets: Sequence[str]) -> list[str]:
    """LoRA 挂载点对应的**全名**列表。`targets` 给的是短名（`q_proj`）。

    短名 → 全名只在这一处发生。「模板与实例的边界必须只有一个地方跨越」
    （skill 坑 79/80）—— 跨越两次就一定会有一处忘了替换。
    """
    unknown = [t for t in targets if t not in LORA_ABLE_MODULES]
    if unknown:
        raise ValueError(f"LoRA 目标里有不存在的模块：{unknown}")
    out: list[str] = []
    for i in range(spec.n_layers):
        for t in targets:
            out.append(f"layers.{i}.{t}")
    return out


def param_shapes(spec: ModelSpec) -> dict[str, tuple[int, ...]]:
    """每个参数的名字与形状。**它不读权重，只读 Spec** ——
    所以它可以在「还没有权重」的时候就算出来（预算校验就靠这一点）。

    形状一律是 **`[in, out]`**，因为 :func:`autograd.matmul` 算的是 `x @ W`。
    写反（`[out, in]`）的后果是 `matmul` 报形状不匹配 —— 这一处侥幸会报错，
    而同样写反的初始化尺度（下面 `_std_for` 的 fan-in）**不会报错**，
    它只让残差流逐层放大，表现为「full 这种全参数策略最先发散」。
    两处必须一起看。
    """
    d, ff = spec.d_model, spec.d_ff
    shapes: dict[str, tuple[int, ...]] = {EMB: (spec.vocab, d)}
    for i in range(spec.n_layers):
        p = f"layers.{i}."
        shapes[p + "norm1"] = (d,)
        shapes[p + "q_proj"] = (d, d)
        shapes[p + "k_proj"] = (d, d)
        shapes[p + "v_proj"] = (d, d)
        shapes[p + "o_proj"] = (d, d)
        shapes[p + "norm2"] = (d,)
        shapes[p + "gate_proj"] = (d, ff)
        shapes[p + "up_proj"] = (d, ff)
        shapes[p + "down_proj"] = (ff, d)
    shapes[NORM_F] = (d,)
    return shapes


def count_params(weights: Mapping[str, Any]) -> int:
    """数一个 weights 映射里的**元素个数**。

    `to_dict` 之类的展示形态不算参数量；这个函数只看嵌套 list 的形状，
    于是它也能数一个「常量还没变成 Tensor」的映射。
    """
    total = 0
    for value in weights.values():
        total += _numel(value)
    return total


def _numel(value: Any) -> int:
    if isinstance(value, Tensor):
        return value.numel
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return len(value) * _numel(value[0])
        return len(value)
    return 1


# ----------------------------------------------------------------------
# 初始化
# ----------------------------------------------------------------------
def _std_for(name: str, shape: tuple[int, ...]) -> float:
    """按 fan-in 定初始化尺度。返回 0.0 表示「这个参数初始化为全 1」（norm）。

    fan-in 是 ``shape[0]``（形状是 `[in, out]`），**两个例外**：
    `emb` 是查表不是矩阵乘，它的尺度由 `d_model`（``shape[1]``）决定；
    norm 的缩放初始化为 1。

    `down_proj` 的 fan-in 是 `d_ff` 而不是 `d` —— 这一项写错的后果是**静默的**：
    它的输出会大 `sqrt(d_ff/d)` 倍，残差流一层层放大，于是 **full 这种
    「所有层都可训练」的策略最先发散**，报告里看起来像「full 不适合这个模型」，
    而其实是初始化写错了。所以 fan-in 必须看 ``shape[0]`` 而不是想当然的 `d`。
    """
    if name.endswith("norm1") or name.endswith("norm2") or name == NORM_F:
        return 0.0  # → 初始化为全 1
    if name == EMB:
        return 1.0 / math.sqrt(max(1, shape[1]))
    return 1.0 / math.sqrt(max(1, shape[0]))


def init_base_weights(spec: ModelSpec, seed: int) -> dict[str, list[Any]]:
    """基座权重的确定性初始化。**只依赖 `(参数名, seed)`，不依赖顺序。**"""
    out: dict[str, list[Any]] = {}
    for name, shape in param_shapes(spec).items():
        rng = random.Random(derive_seed("init", name, spec.name, int(seed)))
        std = _std_for(name, shape)
        if std == 0.0:
            out[name] = [1.0] * shape[0]
            continue
        if len(shape) == 1:
            out[name] = [rng.gauss(0.0, std) for _ in range(shape[0])]
        else:
            out[name] = [[rng.gauss(0.0, std) for _ in range(shape[1])] for _ in range(shape[0])]
    return out


# ----------------------------------------------------------------------
# 因果掩码
# ----------------------------------------------------------------------
_MASKS: dict[int, Tensor] = {}

#: 掩码用的负无穷替身。取 `-1e9` 而不是 `float("-inf")`：
#: `-inf` 进 softmax 会得到 `nan - nan`（`-inf - (-inf)`），而 `nan` 会
#: **绕过所有比较**（skill 坑 9：`abs(x - nan) > tol` 恒为 False）。
#: `-1e9` 在 fp32 下 `exp(-1e9)` 也是 0，效果一样而没有 NaN 风险。
MASK_NEG = -1e9


def causal_mask(T: int) -> Tensor:
    """上三角为 `MASK_NEG` 的常量。**冻结** —— 否则 `add` 的反向会给它分配梯度。"""
    m = _MASKS.get(T)
    if m is None:
        m = constant(
            [[0.0 if j <= i else MASK_NEG for j in range(T)] for i in range(T)],
            name=f"causal_mask[{T}]",
        ).freeze()
        _MASKS[T] = m
    return m


# ----------------------------------------------------------------------
# 前向
# ----------------------------------------------------------------------
def forward(weights: Mapping[str, Any], ids: Sequence[int], spec: ModelSpec) -> Tensor:
    """一次前向。**返回整段序列的 logits（`[T, vocab]`）。**

    `weights` 的每一项可以是 `Tensor`（可训练叶子，或 LoRA 合成的表达式）
    或裸 list（会被 :func:`autograd._as_tensor` 包成常量）。**本函数不区分这两者**
    —— 这正是「策略是数据」的落点。

    ⚠️ 传给这里的冻结权重**必须**已经是 `.freeze()` 过的 `Tensor`，而不是裸 list。
    裸 list 会被 `_as_tensor` 包成一个**未冻结**的常量，于是 `matmul` 的反向会
    认认真真地把它的梯度算出来、再被 `_acc` 存下来 —— 白算一遍还白占一份显存，
    而报告里的显存表是按「冻结 = 不花梯度钱」算的。这个 bug 不报错，
    只让「LoRA 省显存」这句话变成假的。
    """
    T = len(ids)
    if T == 0:
        raise ValueError("forward 收到空序列")
    x = embed(weights[EMB], ids)
    nh, hd = spec.n_heads, spec.head_dim
    head_sizes = [hd] * nh
    mask = causal_mask(T)
    positions = range(T)
    attn_scale = 1.0 / math.sqrt(hd)

    for i in range(spec.n_layers):
        p = f"layers.{i}."
        h = rmsnorm(x, weights[p + "norm1"])
        q = matmul(h, weights[p + "q_proj"])
        k = matmul(h, weights[p + "k_proj"])
        v = matmul(h, weights[p + "v_proj"])
        qh = split_cols(q, head_sizes)
        kh = split_cols(k, head_sizes)
        vh = split_cols(v, head_sizes)
        heads = []
        for j in range(nh):
            qj = rope(qh[j], positions)
            kj = rope(kh[j], positions)
            scores = scale(matmul(qj, transpose(kj)), attn_scale)
            probs = softmax_rows(add(scores, mask))
            heads.append(matmul(probs, vh[j]))
        x = add(x, matmul(concat_cols(heads), weights[p + "o_proj"]))

        h2 = rmsnorm(x, weights[p + "norm2"])
        gate = silu(matmul(h2, weights[p + "gate_proj"]))
        up = matmul(h2, weights[p + "up_proj"])
        x = add(x, matmul(mul(gate, up), weights[p + "down_proj"]))

    x = rmsnorm(x, weights[NORM_F])
    if spec.tie_embeddings:
        # 权重绑定：`emb` 兼任输出投影。它让「解析参数量」与「实测参数量」
        # 的等式成立（`param_count` 里 `head = 0`），所以这里必须真的复用同一张表 ——
        # 另开一个 `lm_head` 会让实测比公式多出 `vocab*d` 个参数。
        return matmul(x, transpose(weights[EMB]))
    return matmul(x, weights["lm_head"])


def logits_of(weights: Mapping[str, Any], ids: Sequence[int], spec: ModelSpec) -> list[list[float]]:
    """取前向的数值结果（`[T, vocab]`）。

    **它故意仍然走 :func:`forward`，不另写一条「纯数值前向」的捷径。**
    多写一条评测路径看起来能省掉构图开销，代价却是这个项目最贵的东西：
    「训练与评测跑的是同一个模型」这句话会退化成一句口头承诺 ——
    两条路径一旦漂移，报告里的准确率就不再是那个被训练出来的模型的表现，
    而**没有任何东西会报错**。构图的额外开销远小于这个风险。
    """
    return forward(weights, ids, spec).data


# ----------------------------------------------------------------------
# 合并：LoRA 的核心承诺
# ----------------------------------------------------------------------
def merge_weights(
    base: Mapping[str, Any],
    adapters: Mapping[str, tuple[list[list[float]], list[list[float]]]],
    scale_factor: float,
) -> dict[str, list[Any]]:
    """把适配器**并回**基座：``W' = W + scale·(B@A)``。

    它必须是一份**独立实现**：这里手写三重求和，而适配器模型的前向走
    :func:`autograd.matmul`。两条路算的是同一件事，但浮点累加顺序不同 ——
    于是「合并前后逐位相同」会得到 `~1e-16` 而不是恒等于 0，
    `acceptance.merge_tolerance` 因此是一个**有内容的**阈值。

    如果这里改成复用 `autograd.matmul`，那条断言就变成了同义反复
    （skill 坑 35：**不要复用被试对象当 oracle**）。
    """
    out: dict[str, list[Any]] = {}
    for name, value in base.items():
        out[name] = [list(row) for row in value] if value and isinstance(value[0], list) else list(value)
    for name, (a_mat, b_mat) in adapters.items():
        if name not in out:
            raise KeyError(f"合并时基座里没有 {name!r}")
        w = out[name]
        if not w or not isinstance(w[0], list):
            raise ValueError(f"合并只支持 2D 权重，{name!r} 不是矩阵")
        rows, cols = len(w), len(w[0])
        r = len(a_mat)
        if len(b_mat) != rows or (a_mat and len(a_mat[0]) != cols):
            raise ValueError(
                f"适配器形状与基座对不上：{name!r} 基座 {rows}×{cols}，"
                f"B {len(b_mat)}×{len(b_mat[0]) if b_mat else 0}，A {r}×{len(a_mat[0]) if a_mat else 0}"
            )
        for i in range(rows):
            bi = b_mat[i]
            wi = w[i]
            for j in range(cols):
                acc = 0.0
                for kk in range(r):
                    acc += bi[kk] * a_mat[kk][j]
                wi[j] = wi[j] + scale_factor * acc
    return out


# ----------------------------------------------------------------------
# 量化：QLoRA 的核心代价
# ----------------------------------------------------------------------
def flatten_by_name(weights: Mapping[str, Any], names: Sequence[str]) -> list[float]:
    """按给定名字顺序把权重压成一维。**顺序必须与 `quant_bytes` 的分组口径一致。**

    `ir.ModelSpec.quant_bytes` 用的分组数是 ``ceil(总参数量 / group)`` ——
    也就是把**全部**参数当成一条扁平向量来分组。所以这里也必须先压成一条
    再切组，而不是「每个张量各自分组」：后者在张量长度不是 `group` 的整数倍时
    会多出几组，于是计划表算出的字节数**永远比公式大一点**，
    而三路互校会因此静默失败（它只是报「对不上」，说不清对不上在哪）。
    """
    flat: list[float] = []
    for name in names:
        value = weights[name]
        if value and isinstance(value[0], list):
            for row in value:
                flat.extend(map(float, row))
        else:
            flat.extend(map(float, value))
    return flat


def inflate_by_name(
    flat: Sequence[float], shapes: Mapping[str, tuple[int, ...]], names: Sequence[str]
) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    off = 0
    for name in names:
        shape = shapes[name]
        if len(shape) == 1:
            out[name] = [float(v) for v in flat[off : off + shape[0]]]
            off += shape[0]
        else:
            rows, cols = shape
            out[name] = [
                [float(v) for v in flat[off + i * cols : off + (i + 1) * cols]] for i in range(rows)
            ]
            off += rows * cols
    if off != len(flat):
        raise ValueError(f"inflate 消耗了 {off} 个元素，但收到 {len(flat)} 个")
    return out


@dataclass(frozen=True)
class QuantizedBase:
    """量化**存储**（不是量化后的浮点）＋ 一个按需反量化的视图。

    **为什么持有的是码而不是浮点。** QLoRA 省显存这件事，全部重量压在
    「基座权重以 4-bit 存在内存里」这一句上。如果实现里存的是一份反量化后的
    fp32 权重（那是最省事的写法），那么无论报告里写多少「省了 8 倍」，
    实际占着的仍是 fp32 —— 而**没有任何东西会报错**，数字全都长得很好看。

    所以：常驻的是 `codes` / `scales` / `zeros`，`dequant()` 在每次前向之前
    现算出一份 fp32 视图，算完丢掉。这与真实 QLoRA 的做法一致
    （分块反量化到 bf16 再进 matmul）。顺带把一句容易说反的话钉死：
    **省下的是存储，不是计算** —— 反量化之后照样是浮点乘法。

    ⚠️ 记账口径：`n_bytes` 来自 :meth:`ir.ModelSpec.quant_bytes`，也就是
    「比特打包后」的大小。本引擎把码存成 Python 的 `int` 列表，解释器开销
    远大于 4 bit/参数，但那部分**不进这个账**（它是宿主语言的开销，不是
    这个格式的开销）。账本里可测量的那一列是 **artifact 的落盘字节数**，
    它对着真的写出来的文件量，而量化基座不属于 artifact。
    """

    names: tuple[str, ...]
    shapes: dict[str, tuple[int, ...]]
    codes: tuple[int, ...]
    scales: tuple[float, ...]
    zeros: tuple[float, ...]
    bits: int
    group: int
    n_bytes: int
    rel_error: float
    n_params: int

    @property
    def n_groups(self) -> int:
        return len(self.scales)

    def dequant_flat(self) -> list[float]:
        out = [0.0] * self.n_params
        codes = self.codes
        n, group = self.n_params, self.group
        for gi, (s, z) in enumerate(zip(self.scales, self.zeros)):
            start = gi * group
            end = min(start + group, n)
            for i in range(start, end):
                out[i] = codes[i] * s + z
        return out

    def dequant(self) -> dict[str, list[Any]]:
        return inflate_by_name(self.dequant_flat(), self.shapes, self.names)

    def storage_report(self) -> dict[str, Any]:
        """账本。**每一列都要能被单独指认** —— 一个说不清的显存表没人信。"""
        return {
            "bits": self.bits,
            "group": self.group,
            "n_params": self.n_params,
            "n_groups": self.n_groups,
            "code_bytes": math.ceil(self.n_params * self.bits / 8),
            "scale_bytes": self.n_groups * 8,
            "n_bytes": self.n_bytes,
            "bytes_per_param": self.n_bytes / self.n_params if self.n_params else 0.0,
            "rel_error": self.rel_error,
        }


def quantize_base(
    weights: Mapping[str, Any],
    spec: ModelSpec,
    *,
    bits: int,
    group: int,
) -> QuantizedBase:
    """分组非对称量化 → :class:`QuantizedBase`。

    每 `group` 个元素共享一个 fp32 `scale` 与一个 fp32 `zero`，这正是
    `ir.ModelSpec.quant_bytes` 里 `n_groups * 8` 那一项的来源。分组是有代价的：
    group=32 时每 32 个参数多出 8 字节，于是实际是 ``4/8 + 8/32 = 0.75`` 字节/参数，
    而不是 ``0.5`` —— 差 50%。报告里如果写 0.5，显存表就跟 `quant_bytes` 对不上了。

    分组的边界按**整个模型拉平后**的线性下标切，而不是按张量切。这一条不是风格问题：
    `ir.ModelSpec.quant_bytes` 用的组数就是 ``ceil(总参数量 / group)``，两者必须
    逐位相等，否则「计划里算的字节数」与「Spec 里算的字节数」会在报告里同时出现
    且不一样（见 skill 坑 53/59）。

    `bits=8` 的相对误差必须 **小于** `bits=4`（`acceptance.quant_error_decreasing`）。
    这条看起来显然，但它守的是「量化确实按位数在丢信息」——
    一个把 `scale` 算错（比如漏了零点）的实现会让两种位数给出同样的误差，
    而报告里两个数字看起来都很小、很正常。
    """
    names = list(layer_param_names(spec))
    shapes = param_shapes(spec)
    flat = flatten_by_name(weights, names)
    n = len(flat)
    if bits <= 0:
        raise ValueError("quantize_base 的 bits 必须 > 0；不量化时不要调用它")
    if group <= 0:
        raise ValueError("quantize_base 的 group 必须 > 0")
    levels = (1 << bits) - 1
    codes = [0] * n
    scales: list[float] = []
    zeros: list[float] = []
    for start in range(0, n, group):
        chunk = flat[start : start + group]
        lo, hi = min(chunk), max(chunk)
        span = hi - lo
        if span <= 0.0:
            # 常量块：step 取 1、code 取 0、zero 取常量本身，反量化后逐位还原。
            scales.append(1.0)
            zeros.append(lo)
            continue
        step = span / levels
        scales.append(step)
        zeros.append(lo)
        for idx, v in enumerate(chunk):
            q = int(round((v - lo) / step))
            codes[start + idx] = 0 if q < 0 else (levels if q > levels else q)

    deq = [0.0] * n
    for gi, (s, z) in enumerate(zip(scales, zeros)):
        start = gi * group
        for i in range(start, min(start + group, n)):
            deq[i] = codes[i] * s + z

    sq_err = 0.0
    sq_ref = 0.0
    for a, b in zip(flat, deq):
        sq_err += (a - b) * (a - b)
        sq_ref += a * a
    rel = math.sqrt(sq_err / sq_ref) if sq_ref > 0 else 0.0
    return QuantizedBase(
        names=tuple(names),
        shapes=shapes,
        codes=tuple(codes),
        scales=tuple(scales),
        zeros=tuple(zeros),
        bits=int(bits),
        group=int(group),
        n_bytes=spec.quant_bytes(bits=bits, group=group),
        rel_error=rel,
        n_params=n,
    )
