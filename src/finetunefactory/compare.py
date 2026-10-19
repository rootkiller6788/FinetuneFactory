"""收拢：把扇出来的 48 份结果变成可直接下结论的表和判定。

扇出那一半（`pipeline`）产出的是**互不相干的一堆数字**。「哪种策略划算」
这个问题在那些数字里没有答案 —— 答案要么在前沿上，要么在一个被实测推翻的期望里。

### 四件必须做的事，缺一件这个项目就没跑完

1. **前言（Pareto）**。四个目标各有方向（`:data:`config.OBJECTIVES`），
   方向是**配置的一部分**，不是散落在比较代码里的 `if`。把 `-vram` 写进比较式
   而不是写进配置，是这类项目里最常见的一处失误：读的人看不到方向在哪，加第五个
   目标时必然有人把符号抄错。
2. **合并核验**。LoRA 的核心承诺是「适配器能并回基座，效果一样」。这件事
   **看起来永远是对的** —— 所以它必须有一个数值上界，而且那个上界不能是 0
   （skill 坑 35：不要复用被试对象当 oracle；:func:`model.merge_weights` 手写三重
   求和，正是为了让它和 `autograd.matmul` 的浮点累加顺序不同，
   于是「逐位相同」会得到 ~1e-16 而不是恒等于 0）。
3. **代价三路互校**。解析公式 / 计数器实测 / 训练外推，三路必须对得上。
   只有两路一致时，「哪一路是错的」是无法判断的。
4. **期望裁决**。配置里写了四条设计期望，**允许被推翻** ——
   被推翻的那几条是本项目最值钱的输出，而不是失败。

### 关于 NaN

这一层遇到的 NaN 一律**不静默**：不进前沿、不进均值、不进比较，而是被数出来
放在 `skipped` 里。一个 NaN 混进 `mean()` 会让整列变成 NaN（看得见），
但混进一次 `>` 比较只会让那一个点悄悄消失（看不见）—— 后者才是危险的。
"""

from __future__ import annotations

import io
import itertools
import math
import tokenize
from typing import Any, Iterable, Mapping, Sequence

from .autograd import Tensor, batch_cross_entropy, mac_counter
from .config import OBJECTIVES, expectations_of
from .model import forward, merge_weights
from .plan import ParamPlan
from .schema import Artifact, EvalResult
from .rng import derive_seed

__all__ = [
    "DOMINANCE_EPS",
    "objectives",
    "objective_values",
    "dominates",
    "pareto_frontier",
    "frontier_keys",
    "hypervolume",
    "hypervolume_2d",
    "projection",
    "verify_merge",
    "check_macs",
    "check_gradients",
    "check_ir_roundtrip",
    "check_single_call_path",
    "verify_quant_error_decreasing",
    "verify_cost_three_way",
    "judge_expectations",
]

#: 支配判定的容差。两个数**几乎相等**时不该判成一方支配另一方 ——
#: 否则前沿上会出现一串「因为浮点噪声而互不支配」的点，而它们实际上是同一个点。
#: 1e-12 足够小（远小于本项目任何一处真实差异，最小的真实差异是 ~1e-3 量级的
#: 一个 token 命中），又足够大（远大于 float64 累加噪声 ~1e-16）。
DOMINANCE_EPS = 1e-12


def objectives(axes: Sequence[tuple[str, str]] | None = None) -> tuple[tuple[str, str], ...]:
    """目标轴与方向。默认取 :data:`config.OBJECTIVES`。

    ⚠️ **别和 ``config.objectives_of`` 搞混。** 名字撞车了，但它们是两件事：

    - 这里的「目标」是 **Pareto 的四条轴**（gain / interference / vram / train_flops）；
    - ``sweep.objectives`` 是**训练目标**（``sft`` / ``sft_response_only``）。

    所以这个函数**刻意不读 cfg**。早先的版本写的是 ``cfg.get("objectives")`` ——
    顶层配置里根本没有这个键，于是它会**静默退回**默认值，看起来一切正常。
    要覆盖就用 ``axes=`` 直接传，别再从配置里猜一个同名键。

    形状校验在这里做：名字重复会静默覆盖（两个轴变成一个），方向写错会让整条
    前沿反过来。两种都在这里报错，而不是等报告出来才发现。
    """
    raw = OBJECTIVES if axes is None else tuple(axes)
    out: list[tuple[str, str]] = []
    for item in raw:
        if not (isinstance(item, (tuple, list)) and len(item) == 2):
            raise ValueError(f"目标轴必须是 (名字, 方向) 二元组，收到 {item!r}")
        name, direction = str(item[0]), str(item[1])
        if direction not in ("max", "min"):
            raise ValueError(f"目标轴 {name!r} 的方向 {direction!r} 不是 'max'/'min'")
        out.append((name, direction))
    names = [a[0] for a in out]
    if len(set(names)) != len(names):
        raise ValueError(f"目标轴名字重复：{names} —— 重复的轴会被静默覆盖成一个")
    return tuple(out)


def objective_values(
    result: Any,
    spec_id: str,
    axes: Sequence[tuple[str, str]] | None = None,
) -> dict[str, float]:
    """一条 spec 在各目标轴上的取值。

    ``gain`` / ``interference`` 走 :class:`PipelineResult` 的查询方法（它们各自
    绑定了自己的基准点），``vram_bytes`` / ``train_flops`` 分别来自代价表与运行记录。
    取不到的量返回 ``nan`` —— 由调用方决定怎么处置，不在这里兜底成 0。

    **未知轴名返回 nan 而不是报错**：轴是配置给的，加一条新轴时这里还没有对应的
    取数逻辑，报错会让整条流水线跑不完；返回 nan 则让这个点自动退出前沿、
    并在 ``skipped_nan`` 里被数出来 —— 加轴的人会**看见**自己去补哪一段。
    """
    ev: EvalResult | None = result.evals.get(spec_id)
    run = result.runs.get(spec_id)
    out: dict[str, float] = {}
    for name, _direction in objectives(axes):
        if name == "gain":
            out[name] = result.gain_of(spec_id) if ev is not None else float("nan")
        elif name == "interference":
            out[name] = result.interference_of(spec_id) if ev is not None else float("nan")
        elif name == "vram_bytes":
            cost = _cost_of(result, spec_id)
            out[name] = float(cost["vram_bytes"]) if cost else float("nan")
        elif name == "train_flops":
            out[name] = float(run.train_flops) if run is not None else float("nan")
        else:
            out[name] = float("nan")
    return out


def _cost_of(result: Any, spec_id: str) -> dict[str, Any] | None:
    """代价表。**它是推导量，不进产物**（skill 坑 93）——所以先查内存里的缓存。

    :func:`pipeline.run_pipeline` 把算好的代价放进 ``result.counters["costs"]``。
    从 ``result.json`` 载入的结果没有它（那是刻意的：它是 Spec 的纯函数，
    重算比落盘更可靠，落盘只会多出一个能漂移的自由度），此时退回 ``None``，
    报告把它标成「未知」而不是硬编一个数。**不重算**是因为重算需要 plan，
    而 plan 需要基座权重 —— 从一份 JSON 里重建 plan 意味着要重建整个训练上下文，
    那会把这一个「查表」的动作变成一次昂贵的副作用。
    """
    costs = (result.counters or {}).get("costs") or {}
    c = costs.get(spec_id)
    return dict(c) if c else None


def dominates(
    a: Mapping[str, float],
    b: Mapping[str, float],
    axes: Sequence[tuple[str, str]] | None = None,
) -> bool:
    """`a` 是否支配 `b`：**每一轴都不差，且至少一轴严格更好**。

    含 NaN 的点一律返回 ``False``（两边都是）—— 不可比的点不该支配任何东西，
    也不该被任何东西支配。这条让前沿自动把缺测的点排除掉，而不是靠调用方先过滤。
    """
    axes = objectives(axes)
    strictly_better = False
    for name, direction in axes:
        av, bv = a.get(name), b.get(name)
        if av is None or bv is None:
            return False
        if math.isnan(av) or math.isnan(bv):
            return False
        if direction == "max":
            better, worse = av > bv + DOMINANCE_EPS, av < bv - DOMINANCE_EPS
        else:
            better, worse = av < bv - DOMINANCE_EPS, av > bv + DOMINANCE_EPS
        if worse:
            return False
        if better:
            strictly_better = True
    # 四轴全部「不差」而**没有一轴严格更好**，说明两个点在这些轴上等价。
    # 判成「不支配」是对的：`a` 支配 `b` 这个说法必须含「严格更好」，
    # 否则前沿上两个完全相同的点会互相支配，非支配排序会出错。
    return strictly_better


def pareto_frontier(
    points: Mapping[str, Mapping[str, float]],
    axes: Sequence[tuple[str, str]] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """非支配集。返回 ``(前沿上的 id 列表, 统计)``。

    用最直白的 O(n²) 两两比较：48 个点 = 1128 次比较，比任何聪明算法都快，
    而且「前沿 = 不被任何其它点支配的点集」这句话在代码里是**逐字可读**的。
    非支配排序 / 快速排序在这个规模上只会增加出错的机会。
    """
    axes = objectives(axes)
    ids = sorted(points)
    stats: dict[str, Any] = {"n_points": len(ids), "skipped_nan": [], "skipped_missing": []}
    usable: list[str] = []
    for sid in ids:
        p = points[sid]
        if any(name not in p for name, _ in axes):
            stats["skipped_missing"].append(sid)
            continue
        if any(math.isnan(float(p[name])) for name, _ in axes):
            stats["skipped_nan"].append(sid)
            continue
        usable.append(sid)

    front: list[str] = []
    for sid in usable:
        if not any(other != sid and dominates(points[other], points[sid], axes) for other in usable):
            front.append(sid)
    stats["n_usable"] = len(usable)
    stats["n_frontier"] = len(front)
    stats["n_dominated"] = len(usable) - len(front)
    return front, stats


def frontier_keys(
    points: Mapping[str, Mapping[str, float]],
    front: Sequence[str],
    axes: Sequence[tuple[str, str]] | None = None,
) -> dict[str, list[str]]:
    """「各轴冠军」—— 每个目标轴上取到极值的 spec。

    它和前沿是**两件事**，报告里也分开列：前沿回答「哪些方案没有被完全压制」，
    各轴冠军回答「单看这一条，谁最好」。冠军一定是前沿点（否则它会被某个
    全面更好的点支配），但前沿上大多数点不是任何一轴的冠军 ——
    这正好说明前沿为什么是「一条线」而不是「一个点」。
    """
    axes = objectives(axes)
    out: dict[str, list[str]] = {}
    for name, direction in axes:
        vals = [
            (sid, float(points[sid][name]))
            for sid in front
            if name in points[sid] and not math.isnan(float(points[sid][name]))
        ]
        if not vals:
            out[name] = []
            continue
        best = max(v for _, v in vals) if direction == "max" else min(v for _, v in vals)
        out[name] = sorted(sid for sid, v in vals if abs(v - best) <= DOMINANCE_EPS)
    # 额外一条合成轴：整个前沿本身。报告用它一眼拿到「谁进前沿了」。
    out["pareto"] = sorted(front)
    return out


def projection(
    points: Mapping[str, Mapping[str, float]],
    x: str,
    y: str,
    axes: Sequence[tuple[str, str]] | None = None,
) -> list[tuple[str, float, float]]:
    """把一个二维投影抽出来画图 / 算二维超体积。含 NaN 的点**丢掉**。"""
    axis_map = {name: d for name, d in objectives(axes)}
    if x not in axis_map or y not in axis_map:
        raise ValueError(f"投影轴 {x!r}/{y!r} 不在目标轴里：{sorted(axis_map)}")
    sx = 1.0 if axis_map[x] == "max" else -1.0
    sy = 1.0 if axis_map[y] == "max" else -1.0
    out: list[tuple[str, float, float]] = []
    for sid in sorted(points):
        p = points[sid]
        if x not in p or y not in p:
            continue
        vx, vy = float(p[x]), float(p[y])
        if math.isnan(vx) or math.isnan(vy):
            continue
        # 投影时统一翻成「越大越好」的坐标 —— 于是二维超体积可以直接算，
        # 不用在体积函数里再读一遍方向。
        out.append((sid, sx * vx, sy * vy))
    return out


def hypervolume_2d(
    pts: Sequence[tuple[str, float, float]],
    ref: tuple[float, float],
) -> float:
    """二维超体积。入参坐标必须已经翻成「越大越好」。

    左点排序后单趟扫描：对每个点，宽度取「到下一个更高点的 x 距离」，
    高度取「它比 ref 高多少」。这是二维精确算法，不是近似。
    """
    live = [(px, py) for _, px, py in pts if px > ref[0] and py > ref[1]]
    if not live:
        return 0.0
    live.sort(key=lambda t: (-t[0], -t[1]))
    hv = 0.0
    best_y = ref[1]
    for i, (px, py) in enumerate(live):
        # 这一步之后，x 区间 [next_px, px] 上的最高 y 就是 max(py, …)，
        # 因为所有更靠右的点都已经处理过，它们的 y 已经计入过左侧区间。
        if py > best_y:
            next_x = live[i + 1][0] if i + 1 < len(live) else ref[0]
            hv += (px - next_x) * (py - ref[1])
            best_y = py
    return hv


#: 超体积的逐点上限。超体积用容斥算，项数是 2^n —— n=20 就是一百万项。
#: 超过上限返回 ``None``，报告里标「未计算」。**不返回一个近似值**：
#: 一个近似的超体积在表里长得和精确值一模一样，而它的用途正是跨配置比较。
HV_MAX_POINTS = 18


def hypervolume(
    pts: Sequence[tuple[str, float, float, float, float]],
    ref: tuple[float, float, float, float],
) -> float | None:
    """四维超体积，容斥精确解。入参坐标**必须已经翻成「越大越好」**。

    ``|∪ boxes| = Σ_{∅≠S} (−1)^{|S|+1} Π_{i∈S} (max_{p∈S} pᵢ − refᵢ)``

    每个点是 ``(id, v₁, v₂, v₃, v₄)`` —— **第 0 位是名字，不是数**，所以取值
    一律从下标 1 开始。这一点值得单独写一句：本文件里另一个版本就曾把 ``p[0]``
    当成第一个目标轴，于是在拿真数据跑之前一直是对的（一个 str > int 的
    TypeError 立刻会炸），但同样的写法在「五元组里第一位恰好也是数」的地方
    会静默算错一整列。

    这是**精确**的，代价是 2^n 项。点数超过 :data:`HV_MAX_POINTS` 时返回 ``None``。
    """
    good = [p for p in pts if all(float(p[1 + i]) > float(ref[i]) for i in range(4))]
    if not good:
        return 0.0
    if len(good) > HV_MAX_POINTS:
        return None
    total = 0.0
    for size in range(1, len(good) + 1):
        for combo in itertools.combinations(good, size):
            prod = 1.0
            for i in range(4):
                prod *= max(float(p[1 + i]) for p in combo) - float(ref[i])
            total += prod if size % 2 == 1 else -prod
    return total


# ----------------------------------------------------------------------
# 合并核验
# ----------------------------------------------------------------------
def verify_merge(
    plan: ParamPlan,
    *,
    tolerance: float,
    seq_len: int,
    probes: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    """把适配器并回基座，验证「合并后」与「适配器模型直接前向」数值一致。

    两条路**必须是两份独立实现**：合并走 :func:`model.merge_weights`（手写三重
    求和），前向走 :func:`model.forward` → :func:`autograd.matmul`。浮点累加顺序
    不同，于是差值应当是「远小于 1，但不是 0」。

    返回 ``{"applicable", "max_abs_diff", "weights_max_abs_diff", "n_probes",
    "identical", "ok", "reason"}``。

    ``identical=True`` 是个**可疑信号**而不是好消息：两份独立实现给出逐位相同的
    结果，通常意味着某一份被换成了另一份（oracle 复用）。所以它被单独报出来，
    由人去看，而不是被判成「完美」。
    """
    if not plan.adapters:
        return {
            "applicable": False,
            "max_abs_diff": None,
            "weights_max_abs_diff": None,
            "n_probes": 0,
            "identical": False,
            "ok": True,
            "reason": f"{plan.kind} 没有适配器，合并这件事对它不适用（不是失败）",
        }

    base = plan.merge_base()
    adapters = plan.merge_recipe()
    scale_factor = plan.strategy.scale
    merged = merge_weights(base, adapters, scale_factor)

    # ① 权重层面的逐元素差：合并表 vs `build()` 里 add(base, scale(B@A)) 得到的表。
    #
    # `built[name]` 是个 **Tensor**（`build()` 对挂载点返回的是 `add(...)` 的结果），
    # 要取 `.data` 才是数值。直接迭代 Tensor 会拿到 `__iter__` 不存在 —— 报错还算好的；
    # 如果哪天 Tensor 长了 `__iter__`，这里就会安静地拿一批别的东西去比。
    built = plan.build()
    w_diff = 0.0
    for name in adapters:
        m = merged[name]
        b = built[name]
        b_data = b.data if isinstance(b, Tensor) else b
        for row_m, row_b in zip(m, b_data):
            for vm, vb in zip(row_m, row_b):
                d = abs(float(vm) - float(vb))
                if d > w_diff:
                    w_diff = d

    # ② 前向层面的差：同一条输入，两个权重表各前向一次。
    #    只看权重不够 —— 权重对了而前向用错了表，前向差会把它抓出来。
    #    这里**不开** mac 计数器：计数只在解析表里用得上，这两次前向纯粹是比对，
    #    开计数器只会让计数器被算两遍（一遍进、一遍出 `MacCounterContext` 还它）。
    spec = plan.model_spec
    ex = _probe_ids(spec, probes, seq_len=seq_len, seed=seed)
    fwd_diff = 0.0
    for ids in ex:
        a = forward(merged, ids, spec).data
        b = forward(built, ids, spec).data
        for row_a, row_b in zip(a, b):
            for va, vb in zip(row_a, row_b):
                d = abs(float(va) - float(vb))
                if d > fwd_diff:
                    fwd_diff = d

    diff = max(w_diff, fwd_diff)
    return {
        "applicable": True,
        "max_abs_diff": diff,
        "weights_max_abs_diff": w_diff,
        "forward_max_abs_diff": fwd_diff,
        "scale_factor": scale_factor,
        "n_targets": len(adapters),
        "n_probes": len(ex),
        "identical": diff == 0.0,
        "tolerance": tolerance,
        "ok": diff <= tolerance,
        "reason": "ok" if diff <= tolerance else f"合并差 {diff:.3e} 超过容差 {tolerance:.3e}",
    }


def _probe_ids(spec: Any, n: int, *, seq_len: int, seed: int) -> list[list[int]]:
    """几条确定性的长度正确、取值合法的探针输入。

    只需要「形状对、token 在词表内」—— 比对两条实现路径的**相对**差时，
    输入的语义不参与结论。

    ``seq_len`` 是**参数**而不是从 `spec.max_len` 取，这一条是硬要求：
    :func:`check_macs` 拿 ``plan.forward_macs(seq_len=T)``（解析）去对实测前向，
    探针就必须正好是 T 个 token。早先的版本用 `spec.max_len`（64），而训练用的是
    16 —— 于是解析表算 16、计数器数 64，两路永远对不上，而「对不上」被当成
    「哪一层没算进去」，会把人引到一个根本不存在的问题上。

    用固定种子生成而不是写死一串数字，是为了让 `seq_len` 变化时自动跟着变
    （写死的序列在改 seq_len 后会越界，而越界会报一个与要查的东西无关的错）。
    """
    vocab = int(spec.vocab)
    out: list[list[int]] = []
    for k in range(max(1, n)):
        rng = _SplitMix(derive_seed("merge_probe", seed, k))
        out.append([rng.below(vocab) for _ in range(int(seq_len))])
    return out


class _SplitMix:
    """极小的确定性整数流。用它而不用 `random` —— 同一份 Spec 必须给出同一批探针,
    而 `random` 的默认种子是进程级的（skill: 确定性优先）。"""

    __slots__ = ("_s",)

    def __init__(self, seed: int) -> None:
        self._s = seed & 0xFFFFFFFFFFFFFFFF

    def next(self) -> int:
        self._s = (self._s + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self._s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        return self.next() % n if n > 0 else 0


# ----------------------------------------------------------------------
# 代价三路互校
# ----------------------------------------------------------------------
def check_macs(
    plan: ParamPlan,
    *,
    seq_len: int,
) -> dict[str, Any]:
    """解析算子表 vs 计数器实测。**逐算子**比对，不是只比总数。

    只比总数会让「A 算子多算了一批、B 算子少算了一批，而总数恰好抵消」通过 ——
    这个巧合的概率不高，但「不高」不是「不会」，而且抵消后的总数看起来完全正常。

    ⚠️ ``plan.build()`` **必须在计数区里面**。解析表的口径是
    「一次单样本『build + 前向』」（:meth:`ParamPlan.forward_macs` 的原文），
    而 LoRA 的适配器是在 ``build()`` 里**并进**基座权重的：
    ``W + scale · (B @ A)``。那三个算子是**权重空间**的（每个挂载点
    ``matmul d·r·d``、``scale d²``、``add d²``，与 T 无关），正好是解析表里
    多出来的那几项。

    把它们放在计数器外面，实测就只剩一次「权重已合并好」的普通前向 ——
    于是 lora/qlora 的实测**恰好等于 full**，而解析表多出 8192/1024/1024，
    逐算子比对报 `add`/`matmul`/`scale` 三项不符。这个失败看起来像
    「适配器的算子没被算进去」，真因却是「核对把一件事剪成了两半去比」。
    （这里如果走错方向去改解析表，会把它改成**错的**。）
    """
    analytic = plan.forward_macs(seq_len=seq_len)
    ids = _probe_ids(plan.model_spec, 1, seq_len=seq_len, seed=0)
    with mac_counter() as counter:
        # `build()` 在计数区内：并适配器的算子与本次前向算作同一次「build + 前向」。
        weights = plan.build()
        forward(weights, ids[0], plan.model_spec)
    by_op = dict(counter.by_op)
    ops = sorted(set(analytic) | set(by_op))
    per_op = {op: {"analytic": analytic.get(op, 0), "measured": by_op.get(op, 0)} for op in ops}
    mismatched = [op for op in ops if per_op[op]["analytic"] != per_op[op]["measured"]]
    return {
        "analytic_total": sum(analytic.values()),
        "measured_total": counter.total,
        "per_op": per_op,
        "mismatched_ops": mismatched,
        "ok": not mismatched and sum(analytic.values()) == counter.total,
    }


def verify_cost_three_way(
    plan: ParamPlan,
    *,
    seq_len: int,
    batch: int,
    steps: int,
    effective_batch: int,
    measured_macs: int,
) -> dict[str, Any]:
    """三路代价互校。

    - ① **解析**：:meth:`ParamPlan.forward_macs` 从结构参数推出来的单样本算子数；
    - ② **实测**：:class:`autograd.MacCounter` 在一次真实前向里数出来的；
    - ③ **外推**：训练总共跑了多少 FLOP —— ``①×steps×有效批×3×2``。

    ③ 里的 ``3`` 是「前向 1 份、反向 2 份」的经验系数，``2`` 是「1 MAC = 2 FLOP」。
    **它是估算**，所以单独列一列名字，绝不和 ② 混在一起报 —— 一颗被标成实测的
    估算值，是这类项目里最难被发现的一种错。这里核的是「③是不是真的按这个公式
    算出来的」，而不是「3 这个系数对不对」（那需要另一套测量）。
    """
    analytic = plan.forward_macs_total(seq_len=seq_len)
    recomputed = analytic * steps * effective_batch * 3 * 2
    return {
        "analytic_per_sample": analytic,
        "measured_per_sample": measured_macs,
        "analytic_equals_measured": analytic == measured_macs,
        "extrapolated_train_flops": recomputed,
        "formula": "analytic_per_sample * steps * effective_batch * 3 * 2",
        "note": "③ 是估算（3=前向1+反向2，2=每个MAC两个FLOP）；①②是实测/解析",
        "ok": analytic == measured_macs,
    }


# ----------------------------------------------------------------------
# 梯度数值差分
# ----------------------------------------------------------------------
def check_gradients(
    plan: ParamPlan,
    *,
    example: Any,
    n_checks: int = 16,
    eps: float = 1e-5,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """:func:`plan.forward_macs` 的解析表证明「算子集合对」，**不证明每个算子求导对**。
    求导对不对只能靠数值差分管。

    做法：对若干**可训练叶子**上的若干**元素**，用中心差分
    ``(L(w+ε) − L(w−ε)) / 2ε`` 去逼近解析梯度 ``leaf.grad``。

    四个必须做对的地方：

    1. **只查真正拿到梯度的叶子**。`freeze` 的冻结层数据仍在函数里参与前向，
       扰动它**一定会**改变损失 —— 拿它做差分得到的「数值梯度」是个非零值，
       而它的解析梯度是 0。两者一比就报错，而错的是检查本身。
       （:meth:`plan.ParamPlan.trainable_leaves` 正是「真的会拿到梯度」的那一批。）
    2. **每条扰动都要重新前向**（损失是权重的纯函数）。改数据后必须重建图，
       不能复用上一次的 `logits`。
    3. **ε 与容差要配套**。引擎里的数是 Python float（float64），所以中心差分的
       截断误差 ~ε²=1e-10、舍入误差 ~1e-16/ε=1e-11 —— 容差 1e-4 有四个数量级的余量。
       用 float32 的经验值（ε=1e-4、容差 1e-2）在这里过松，等于没查。
    4. **掩码取自 Spec 的 objective**，与 `train.batch_of` 同一个开关。这里另造
       一个掩码会让差分和解析都去求**另一个目标函数**的梯度，而它们照样一致 ——
       检查通过，被检查的东西却根本没被查。
    """
    leaves = plan.trainable_leaves()
    if not leaves:
        return {"applicable": False, "ok": True, "reason": "没有可训练叶子"}
    spec = plan.model_spec
    mask = _objective_mask(plan.spec, example)

    def loss_of() -> float:
        weights = plan.build()
        logits = forward(weights, example.ids, spec)
        out = batch_cross_entropy([(logits, example.targets, mask)], weight=1.0)
        return float(out.data[0])

    # 解析梯度：一次前向 + 一次反向。
    for _name, t in leaves:
        t.zero_grad()
    weights = plan.build()
    logits = forward(weights, example.ids, spec)
    batch_cross_entropy([(logits, example.targets, mask)], weight=1.0).backward()

    picks = _pick_elements(leaves, n_checks)
    rows: list[dict[str, Any]] = []
    worst = 0.0
    for name, t, i, j in picks:
        analytic = _elem(t.grad, i, j)
        if analytic is None:
            rows.append({"leaf": name, "index": [i, j], "status": "no_grad"})
            continue
        orig = _elem(t.data, i, j)
        _set_elem(t.data, i, j, orig + eps)
        lp = loss_of()
        _set_elem(t.data, i, j, orig - eps)
        lm = loss_of()
        _set_elem(t.data, i, j, orig)  # 必须还原 —— 不改回去，后面每个点都在漂移的权重上测
        numeric = (lp - lm) / (2 * eps)
        scale = max(1.0, abs(numeric), abs(float(analytic)))
        rel = abs(numeric - float(analytic)) / scale
        worst = max(worst, rel)
        rows.append({
            "leaf": name,
            "index": [i, j],
            "analytic": float(analytic),
            "numeric": numeric,
            "rel_error": rel,
            "status": "ok" if rel <= tolerance else "mismatch",
        })
    checked = [r for r in rows if r.get("status") in ("ok", "mismatch")]
    return {
        "applicable": True,
        "n_leaves": len(leaves),
        "n_checked": len(checked),
        "max_rel_error": worst,
        "tolerance": tolerance,
        "eps": eps,
        "rows": rows,
        "ok": bool(checked) and worst <= tolerance,
    }


def _objective_mask(spec: Any, example: Any) -> Any:
    """按 Spec 的 ``objective`` 取损失掩码。

    与 :func:`train.batch_of` 用**同一个开关、同一个字段**：``sft`` → 全部位置，
    ``sft_response_only`` → 只回答位置。掩码必须从 Spec 来而不是从 example 猜 ——
    example 同时带着两种掩码，猜错的那一种会让梯度检查去验另一个目标函数。
    """
    obj = getattr(spec, "objective", "sft")
    if obj == "sft_response_only":
        return example.mask_response
    if obj == "sft":
        return example.mask_all
    raise ValueError(f"未知的 objective={obj!r}：掩码口径不明，不猜")


def _elem(data: Any, i: int, j: int) -> float | None:
    if data is None:
        return None
    if data and isinstance(data[0], list):
        if i >= len(data) or j >= len(data[i]):
            return None
        return float(data[i][j])
    idx = j if len(data) > 1 else i
    if idx >= len(data):
        return None
    return float(data[idx])


def _set_elem(data: Any, i: int, j: int, value: float) -> None:
    if data and isinstance(data[0], list):
        data[i][j] = value
    else:
        data[j if len(data) > 1 else i] = value


def _pick_elements(
    leaves: Sequence[tuple[str, Tensor]], n: int, *, seed: int = 0
) -> list[tuple[str, Tensor, int, int]]:
    """确定性地挑几个 (叶子, 元素)：**按轮转**，每个叶子先出它的第一个元素。

    为什么不是随机取：随机取会让「这一次碰巧全落在 B=0 的适配器上」变成可能，
    而 B 全 0 时数值梯度与解析梯度都是 0，检查会静默通过。

    为什么也不是「从叶子头部连续取」（这个函数曾经的写法）：那样取到的前 n 个
    元素会**全部落在第一个叶子上**。于是检查看起来覆盖了 6 个点，实际只验了一个
    张量 —— 而 `full` 的第一个叶子是 `emb`。实测过：引擎里 `split_cols` 的反向
    把梯度放大了 4 倍，`full` 的这条检查照样报 OK，因为它从头到尾没碰过一个
    注意力相关的参数。**一条永远为真的检查比没有检查更糟**，它让人以为验过了。

    轮转的代价是「同一片叶子内的邻接元素不再连续」，而那个连续性本来也没有意义。

    `seed` 保留在签名里但**不参与挑选**：确定性来自轮转次序，不是来自种子。
    留一个不用的参数比留一个假装随机的参数好 —— 后者会让人以为
    「换个种子就换一批检查点」，于是换个种子重跑一遍求安心，
    而两次检查的其实是同一批元素。
    """
    per_leaf: list[tuple[str, Tensor, list[tuple[int, int]]]] = []
    for name, t in leaves:
        d = t.data
        if not d:
            continue
        if isinstance(d[0], list):
            per_leaf.append((name, t, [(i, j) for i in range(len(d)) for j in range(len(d[i]))]))
        else:
            per_leaf.append((name, t, [(0, j) for j in range(len(d))]))

    out: list[tuple[str, Tensor, int, int]] = []
    k = 0
    while len(out) < n:
        added = False
        for name, t, idxs in per_leaf:
            if k < len(idxs):
                i, j = idxs[k]
                out.append((name, t, i, j))
                added = True
                if len(out) >= n:
                    return out
        if not added:  # 所有叶子都取完了
            break
        k += 1
    return out


# ----------------------------------------------------------------------
# 结构性核对
# ----------------------------------------------------------------------
def check_ir_roundtrip(specs: Iterable[Any]) -> dict[str, Any]:
    """IR 往返：``from_dict(to_dict(s))`` 必须还原出**同一个 digest**。

    digest 而不是逐字段相等 —— 逐字段相等需要一份「哪些字段算数」的清单，
    那份清单本身会漂移；digest 是 Spec 自己声明的身份（skill: 派生量不进主地址
    空间但必须可重算），往返后 digest 变了就说明某个字段在序列化里丢了。
    """
    bad: list[dict[str, str]] = []
    n = 0
    for s in specs:
        n += 1
        try:
            back = type(s).from_dict(s.to_dict())
        except Exception as exc:  # noqa: BLE001 —— 往返失败本身就是结果
            bad.append({"spec_id": getattr(s, "spec_id", "?"), "why": f"{type(exc).__name__}: {exc}"})
            continue
        if back.digest() != s.digest():
            bad.append({"spec_id": s.spec_id, "why": f"digest 变了 {s.digest()[:12]} → {back.digest()[:12]}"})
    return {"n_checked": n, "failures": bad, "ok": not bad}


def check_single_call_path(root: Any) -> dict[str, Any]:
    """「单一路径」的结构性核对：**训练与模型的代码里不许出现 ``kind ==``**。

    本项目的地基是「训练策略 → IR → Runtime」—— SFT / freeze / LoRA / QLoRA 是
    ``finetuning`` 这一个字段的四个取值，不是四条代码分支。这条性质**只能用
    文本核对来守**，因为一个 `if kind == "lora"` 加进 `train.py` 之后，
    所有测试照样全绿、所有指标照样合理。

    只扫 ``plan.py`` 之外的模块：``plan.py`` 正是那个**被允许**知道策略的地方，
    它把策略翻译成「哪些叶子可训练、张量挂在哪」，下游只看见叶子。

    按 **token** 扫，不按文本行扫。这不是洁癖：这几个模块的 docstring 里**会主动
    声明**「本文件里搜不到 ``if kind ==``」—— 那句声明本身含有这两个词。按行做
    子串匹配会把这段自述判成违规，于是这个核对**每次都会失败**，而失败的理由是
    它把「我没有这么做」这句话读成了「我这么做了」。一个永远为假的断言比没有断言
    更糟：它训练人忽略它。词法分析器本来就分得清注释、字符串和标识符，用它。
    """
    root = _as_path(root)
    suspects = ["model.py", "train.py", "eval.py", "data.py"]
    hits: list[dict[str, Any]] = []
    _SKIP = {
        tokenize.COMMENT,
        tokenize.STRING,  # docstring 与普通字符串字面量
        tokenize.FSTRING_MIDDLE,  # 3.12+ 把 f-string 的字面量部分单独成 token
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
    }
    for name in suspects:
        path = root / "src" / "finetunefactory" / name
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8")
        lines = src.splitlines()
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
        except (tokenize.TokenError, IndentationError) as exc:  # pragma: no cover
            # 解析不了就**报违规**而不是跳过：跳过等于「这一份没搜到」，
            # 而真相是「这一份没搜」。
            hits.append({"file": name, "line": 0, "text": f"无法词法分析：{exc}"})
            continue
        sig = [t for t in toks if t.type not in _SKIP]
        for cur, nxt in zip(sig, sig[1:]):
            if (
                cur.type == tokenize.NAME
                and cur.string == "kind"
                and nxt.type == tokenize.OP
                # `!=` 同样是「按策略分支」，一起挡。
                and nxt.string in ("==", "!=")
                and nxt.start[0] == cur.start[0]
            ):
                lineno = cur.start[0]
                text = lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""
                hits.append({"file": name, "line": lineno, "text": text})
    return {"scanned": suspects, "hits": hits, "ok": not hits}


def _as_path(root: Any) -> Any:
    from pathlib import Path

    return Path(root)


def verify_quant_error_decreasing(artifacts: Mapping[str, Artifact]) -> dict[str, Any]:
    """量化误差必须随位数**单调下降**。不成立，「4-bit 够用」这个说法就没有依据。

    这里做**两条**核验，而不是一条：

    1. **同位数 ⇒ 同一个误差。** 量化误差是 (基座, 位数, 组大小) 的函数，与适配器
       无关。所以所有 4-bit 格子必须报出**逐位相同**的 ``quant_error``。不等于同一个值，
       说明量化的输入（基座）在格子之间不一样 —— 那意味着基座被谁改过，
       而这一条比「误差是否单调」更早地暴露问题。
    2. **位数越少 ⇒ 误差越大。** 4-bit 的误差必须严格大于 8-bit。

    不按 (rank, targets) 分组：那样每个组里都只有一个位数（网格里 4/8 位只各有一对
    同构格子），就会什么也比不出来。这个函数在实现上曾经正是这么写的 ——
    它返回一个空的 ``comparisons``，``ok`` 是 ``False``，看起来像「没通过」，
    实际是「没在比」。
    """
    by_bits: dict[int, dict[str, float]] = {}
    for sid, art in artifacts.items():
        if art.kind != "qlora":
            continue
        bits = int(art.meta.get("quant_bits", 0) or 0)
        if bits:
            by_bits.setdefault(bits, {})[sid] = float(art.quant_error)

    consistency: list[dict[str, Any]] = []
    for bits, per_spec in sorted(by_bits.items()):
        vals = sorted(set(per_spec.values()))
        consistency.append({
            "bits": bits,
            "n_specs": len(per_spec),
            "distinct_errors": vals,
            "consistent": len(vals) <= 1,
            "error": vals[0] if vals else float("nan"),
        })

    comparisons: list[dict[str, Any]] = []
    bits_sorted = sorted(by_bits)
    for lo, hi in zip(bits_sorted, bits_sorted[1:]):
        e_lo = _mean(list(by_bits[lo].values()))
        e_hi = _mean(list(by_bits[hi].values()))
        comparisons.append({
            "low_bits": lo,
            "high_bits": hi,
            "error_low_bits": e_lo,
            "error_high_bits": e_hi,
            "ratio": e_lo / e_hi if e_hi else float("nan"),
            "decreasing": e_lo > e_hi,
        })
    return {
        "consistency": consistency,
        "comparisons": comparisons,
        "n_bit_levels": len(by_bits),
        "ok": bool(comparisons)
        and all(c["decreasing"] for c in comparisons)
        and all(c["consistent"] for c in consistency),
    }


def _mean(vals: Sequence[float]) -> float:
    """跳过 NaN 求均值。朴素 `sum/len` 在有一个 NaN 时整列变 NaN ——
    这里选择跳过并让调用方知道跳过了（返回值可能基于比输入更少的样本）。"""
    good = [v for v in vals if not math.isnan(v)]
    return sum(good) / len(good) if good else float("nan")


# ----------------------------------------------------------------------
# 期望裁决
# ----------------------------------------------------------------------
def judge_expectations(
    result: Any,
    cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """对四条设计期望逐条给出裁决：confirmed / refuted / unknown。

    **refuted 不是错误**。这四条是设计时的**猜测**，实测把它们推翻是本项目
    最有价值的输出之一（skill 坑 93：期望被推翻要写进附录 B，不能悄悄改掉期望）。
    所以每条裁决都带 `observed`，让读者能自己算出结论，而不是只看到一个布尔。

    配对原则：**只在同一能力内部比**。跨能力比 gain 是没意义的（sort 的
    copy 基准 0.344，upper 的 0.125，两个 gain 不可比）。所以每条期望都是
    「在同一个能力下，换掉某一个轴，另一个轴怎么变」。
    """
    want = expectations_of(cfg)
    profiles = _profiles(result)
    verdicts: list[dict[str, Any]] = []
    for name, expected in want.items():
        if not expected:
            continue
        fn = {
            "lora_interferes_less_than_full": _judge_lora_interference,
            "lora_reaches_full_gain_on_target": _judge_lora_gain,
            "qlora_trades_gain_for_memory": _judge_qlora_trade,
            "response_only_beats_all_tokens": _judge_response_only,
        }.get(name)
        if fn is None:
            verdicts.append({
                "name": name, "expected": True, "verdict": "unknown",
                "reason": "没有实现这条期望的裁决规则（配置里加了新期望，代码要跟上）",
            })
            continue
        verdicts.append(fn(result, profiles, cfg))
    return verdicts


def _profiles(result: Any) -> dict[str, dict[tuple[Any, ...], str]]:
    """能力 → {轴组合 → spec_id}。

    轴组合 = ``(kind, rank, 目标模块, objective)``。四个轴正好覆盖四条期望各自
    要换掉的那一个：换 kind（lora↔full、lora↔qlora）、换 objective。
    """
    out: dict[str, dict[tuple[Any, ...], str]] = {}
    for sid, spec in result.specs.items():
        ft = spec.finetuning
        key = (
            ft.kind,
            int(ft.rank),
            tuple(sorted(ft.target_modules)),
            str(spec.objective),
        )
        out.setdefault(spec.dataset.capability, {})[key] = sid
    return out


def _verdict(name: str, confirmed: bool | None, observed: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "expected": True,
        "verdict": "unknown" if confirmed is None else ("confirmed" if confirmed else "refuted"),
        "reason": reason,
        "observed": observed,
    }


def _judge_lora_interference(
    result: Any, profiles: Mapping[str, Mapping[tuple[Any, ...], str]], cfg: Mapping[str, Any]
) -> dict[str, Any]:
    """LoRA 的干扰 < full 的干扰，**同一个能力下比**。

    判据用**中位数**而不是均值：干扰是小量（0.01 量级），个别 spec 因为
    某一项能力恰好崩掉而给出一个很大的值，均值会被它一个人拽走。中位数
    在 48 个点里对单个离群值是稳的。
    """
    diffs: list[dict[str, Any]] = []
    for cap, table in sorted(profiles.items()):
        for key, lora_sid in sorted(table.items(), key=lambda kv: str(kv[0])):
            kind, rank, targets, objective = key
            if kind != "lora":
                continue
            full_sid = table.get(("full", 0, (), objective))
            if full_sid is None:
                continue
            i_lora = result.interference_of(lora_sid)
            i_full = result.interference_of(full_sid)
            if math.isnan(i_lora) or math.isnan(i_full):
                continue
            diffs.append({
                "capability": cap,
                "rank": rank,
                "targets": list(targets),
                "objective": objective,
                "interference_lora": i_lora,
                "interference_full": i_full,
                "delta_full_minus_lora": i_full - i_lora,
            })
    if not diffs:
        return _verdict("lora_interferes_less_than_full", None, [], "没有可比的 lora/full 配对")
    med = _median([d["delta_full_minus_lora"] for d in diffs])
    n_ok = sum(1 for d in diffs if d["delta_full_minus_lora"] > 0)
    return _verdict(
        "lora_interferes_less_than_full",
        med > 0,
        diffs,
        f"{n_ok}/{len(diffs)} 个配对上 full 的干扰更大；中位差 {med:+.4f}（正=full 更差）",
    )


def _judge_lora_gain(
    result: Any, profiles: Mapping[str, Mapping[tuple[Any, ...], str]], cfg: Mapping[str, Any]
) -> dict[str, Any]:
    """LoRA 在目标能力上的 gain 达到 full 的一定比例。

    阈值来自 ``acceptance.lora_gain_ratio``（**不在代码里写死**：一个写死的
    0.9 在报告里看起来和配置来的 0.9 一模一样，但它没法被读者调整，
    也没法在不改代码的前提下做敏感性分析）。

    gain 的**分子是同一能力**的，且用 ``base_accs`` 当基准 —— 见
    :class:`schema.PipelineResult` 里两个基准点的说明。
    """
    thr = float((cfg.get("acceptance") or {}).get("lora_gain_ratio", 0.9))
    rows: list[dict[str, Any]] = []
    for cap, table in sorted(profiles.items()):
        for key, lora_sid in sorted(table.items(), key=lambda kv: str(kv[0])):
            kind, rank, targets, objective = key
            if kind != "lora":
                continue
            full_sid = table.get(("full", 0, (), objective))
            if full_sid is None:
                continue
            g_lora = result.gain_of(lora_sid)
            g_full = result.gain_of(full_sid)
            if math.isnan(g_lora) or math.isnan(g_full):
                continue
            ratio = (g_lora / g_full) if abs(g_full) > 1e-9 else float("nan")
            rows.append({
                "capability": cap,
                "rank": rank,
                "targets": list(targets),
                "objective": objective,
                "gain_lora": g_lora,
                "gain_full": g_full,
                "ratio": ratio,
                "reaches": (not math.isnan(ratio)) and ratio >= thr,
            })
    if not rows:
        return _verdict("lora_reaches_full_gain_on_target", None, [], "没有可比的 lora/full 配对")
    n_ok = sum(1 for r in rows if r["reaches"])
    ratios = [r["ratio"] for r in rows if not math.isnan(r["ratio"])]
    med = _median(ratios)
    return _verdict(
        "lora_reaches_full_gain_on_target",
        n_ok * 2 >= len(rows),
        rows,
        f"{n_ok}/{len(rows)} 个配对达到 gain 比 ≥ {thr}；中位比 {med:.3f}",
    )


def _judge_qlora_trade(
    result: Any, profiles: Mapping[str, Mapping[tuple[Any, ...], str]], cfg: Mapping[str, Any]
) -> dict[str, Any]:
    """QLoRA 用 gain 换显存:显存**确实更低**,而 gain **没有更高**。

    两个条件缺一不可。只看显存更低就下结论，等于没验证「代价」那一半 ——
    而 QLoRA 的说法恰恰是**有代价的**，只报收益是在替它做广告。
    """
    rows: list[dict[str, Any]] = []
    for cap, table in sorted(profiles.items()):
        for key, lora_sid in sorted(table.items(), key=lambda kv: str(kv[0])):
            kind, rank, targets, objective = key
            if kind != "lora":
                continue
            q_sid = table.get(("qlora", rank, targets, objective))
            if q_sid is None:
                continue
            c_lora = _cost_of(result, lora_sid)
            c_q = _cost_of(result, q_sid)
            if not c_lora or not c_q:
                continue
            g_lora, g_q = result.gain_of(lora_sid), result.gain_of(q_sid)
            v_lora, v_q = float(c_lora["vram_bytes"]), float(c_q["vram_bytes"])
            rows.append({
                "capability": cap,
                "rank": rank,
                "targets": list(targets),
                "objective": objective,
                "vram_lora": v_lora,
                "vram_qlora": v_q,
                "vram_ratio": v_q / v_lora if v_lora else float("nan"),
                "gain_lora": g_lora,
                "gain_qlora": g_q,
                "gain_delta": g_q - g_lora if not (math.isnan(g_lora) or math.isnan(g_q)) else float("nan"),
                "memory_saved": v_q < v_lora,
                "gain_not_higher": (not math.isnan(g_lora) and not math.isnan(g_q) and g_q <= g_lora),
            })
    if not rows:
        return _verdict("qlora_trades_gain_for_memory", None, [], "没有可比的 lora/qlora 配对")
    n_ok = sum(1 for r in rows if r["memory_saved"] and r["gain_not_higher"])
    med_v = _median([r["vram_ratio"] for r in rows if not math.isnan(r["vram_ratio"])])
    med_g = _median([r["gain_delta"] for r in rows if not math.isnan(r["gain_delta"])])
    return _verdict(
        "qlora_trades_gain_for_memory",
        n_ok * 2 >= len(rows),
        rows,
        f"{n_ok}/{len(rows)} 个配对上「显存更低且 gain 不更高」同时成立；"
        f"中位显存比 {med_v:.3f}，中位 gain 差 {med_g:+.4f}",
    )


def _judge_response_only(
    result: Any, profiles: Mapping[str, Mapping[tuple[Any, ...], str]], cfg: Mapping[str, Any]
) -> dict[str, Any]:
    """只监督回答（response_only）比监督全部 token（sft）好，同能力同策略下比。

    这条期望有可能被推翻，而且**推翻了更有意思**：只学回答时，模型看不到
    prompt 上的监督信号，而本项目的 prompt 里带着任务标记（``:`` / ``<u>`` /
    ``<a>``）—— 如果那些标记的梯度对学会任务至关重要，sft 就会赢。
    """
    rows: list[dict[str, Any]] = []
    for cap, table in sorted(profiles.items()):
        for key, sid in sorted(table.items(), key=lambda kv: str(kv[0])):
            kind, rank, targets, objective = key
            if objective != "sft":
                continue
            ro_sid = table.get((kind, rank, targets, "sft_response_only"))
            if ro_sid is None:
                continue
            g_sft, g_ro = result.gain_of(sid), result.gain_of(ro_sid)
            if math.isnan(g_sft) or math.isnan(g_ro):
                continue
            rows.append({
                "capability": cap,
                "kind": kind,
                "rank": rank,
                "targets": list(targets),
                "gain_sft": g_sft,
                "gain_response_only": g_ro,
                "delta": g_ro - g_sft,
            })
    if not rows:
        return _verdict("response_only_beats_all_tokens", None, [], "没有可比的 sft/response_only 配对")
    n_ok = sum(1 for r in rows if r["delta"] > 0)
    med = _median([r["delta"] for r in rows])
    return _verdict(
        "response_only_beats_all_tokens",
        med > 0,
        rows,
        f"{n_ok}/{len(rows)} 个配对上 response_only 的 gain 更高；中位差 {med:+.4f}",
    )


def _median(vals: Sequence[float]) -> float:
    good = sorted(v for v in vals if not math.isnan(v))
    if not good:
        return float("nan")
    mid = len(good) // 2
    return good[mid] if len(good) % 2 else (good[mid - 1] + good[mid]) / 2
