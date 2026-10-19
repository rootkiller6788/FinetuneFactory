"""报告：把一次扇出-收拢的结果渲染成 ``report.md``。

### 这个文件里唯一的硬纪律：**「未知」不是 0**

报告可以从两个地方来：

1. **内存里的** :class:`~finetunefactory.schema.PipelineResult`（``run`` 刚跑完）；
2. **落盘后再读回来的**（``out/result.json`` + ``runs.jsonl`` + ``evals.jsonl``
   + ``artifacts/*.json``，由 :func:`pipeline.load_result` 那条链拼起来）。

两条路径能拿到的字段**不一样**。载入路径没有 ``payloads``（JSON 装不下 bytes），
如果产物 JSON 丢了还会少 ``artifacts``。所有取数都走 :func:`_get` 这类**返回
``None`` 而不是 0** 的辅助函数，于是「没测 / 没存」在表里是 ``—`` 或 ``未知``，
而 ``0.0`` 永远是**测出来真的是 0**。

这一条在本项目里尤其要紧：``gain`` 与 ``interference`` 各有一个基准点，
基准缺失时它们会退化成 0 或 NaN —— 而 ``0`` 恰好是「训练毫无作用」这个
看起来完全正常的结论。报告里如果出现一列全 0 的 gain，读者没法分辨
「训练真的没用」和「评测结果没被载入」。

### 两个基准点必须在表里**看得见**

``gain`` 与 ``interference`` 用的是**两个不同的基准**（见
:class:`schema.PipelineResult` 的字段说明）。所以第一节不是摘要，是
**基准表**：读者要先看到「随机初始化的基座在四项能力上都低于 chance」，
才能接受为什么干扰要用 chance 当分母、而收益要用实测基座当分母。
把这一节删掉的报告是自相矛盾的 —— 它会同时给出两个用不同尺子量出来的数，
而没有一个字解释为什么。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from . import compare
from .config import OBJECTIVES, expectations_of

__all__ = [
    "UNKNOWN",
    "NA",
    "fmt",
    "fmt_signed",
    "fmt_int",
    "fmt_bytes",
    "markdown_table",
    "baseline_rows",
    "matrix_header",
    "matrix_rows",
    "cost_header",
    "cost_rows",
    "frontier_header",
    "frontier_rows",
    "champion_rows",
    "constant_axes",
    "verdict_rows",
    "check_rows",
    "expectation_status_rows",
    "explain",
    "summary_line",
    "meta_payload",
    "render_markdown",
]

#: 「这个量没测 / 没存」。与 ``0`` 严格区分：0 是一个测量结果。
UNKNOWN = "未知"
#: 「这件事对这一行不适用（不是失败）」—— 见 skill 坑 10。
NA = "—"

#: 每个目标轴的中文名与单位。表头用它，而不是直接用配置里的键名：
#: 报告是给人读的，``train_flops`` 和「训练 FLOPs」在读者脑中的重量不一样。
AXIS_LABEL: dict[str, str] = {
    "gain": "目标能力收益 gain",
    "interference": "干扰 interference",
    "vram_bytes": "显存 B",
    "train_flops": "训练 FLOPs",
}

#: 显存分解的列名与它在 ``plan.cost()`` 里的键。**顺序就是记账顺序**，
#: 合计是它们按顺序相加 —— 报告里的加法必须真的是加法，不能是「看起来像总计」。
COST_COLUMNS: tuple[tuple[str, str], ...] = (
    ("base_storage_bytes", "基座存储"),
    ("adapter_bytes", "适配器"),
    ("grad_bytes", "梯度"),
    ("optimizer_bytes", "优化器状态"),
    ("activation_bytes", "激活（估算）"),
    ("overhead_bytes", "固定开销"),
)


# ----------------------------------------------------------------------
# 格式化
# ----------------------------------------------------------------------
def fmt(value: Any, digits: int = 4) -> str:
    """把一个数渲染成表里的一格。**``None`` / ``NaN`` 有各自的写法。**

    这里刻意不写 ``or 0`` 之类的兜底：兜底会让「没测」以一个合法的数字出现在
    报告里，而它恰好是**最容易被当作结论**的那种数字。
    """
    if value is None:
        return NA
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        if math.isnan(value):
            return UNKNOWN
        if math.isinf(value):
            return "∞" if value > 0 else "−∞"
        if value == 0.0:
            # ``-0.0`` 与 ``0.0`` 在表里长得不一样，但它们相等。统一成正零 ——
            # 否则同一列里会出现「-0.0000」和「0.0000」两种写法。
            return f"{0.0:.{digits}f}"
        if abs(value) < 1e-4 or abs(value) >= 1e6:
            return f"{value:.{digits}g}"
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return fmt_int(value)
    return str(value)


def fmt_signed(value: Any, digits: int = 4) -> str:
    """带正负号的浮点。**差值列一律用它** ——

    差值的正负是有方向的（``interference`` 是「越低越好」，``vram`` 也一样），
    而 ``0.0125`` 与 ``-0.0125`` 在密集的表里只差一个字符。带上 ``+`` 之后，
    「哪一边更差」这件事在扫一眼时就能读出来。
    """
    if value is None:
        return NA
    if isinstance(value, float) and math.isnan(value):
        return UNKNOWN
    if isinstance(value, (int, float)) and float(value) > 0:
        return "+" + fmt(float(value), digits)
    return fmt(value, digits)


def fmt_int(value: Any) -> str:
    if value is None:
        return NA
    if isinstance(value, float) and math.isnan(value):
        return UNKNOWN
    return f"{int(value):,}"


def fmt_bytes(value: Any) -> str:
    """字节数。**精确值是主，人类可读是附。** 例如 ``16,384 B（16.0 KiB）``。

    反过来写（只给 ``16.0 KiB``）会让「产物字节数 = 声明值」这条核对没法在
    报告里被读者**自己算一遍** —— 而那条等式正是本项目唯一一处「声明 vs 实测」
    的强核对，它不该只在代码里成立。
    """
    if value is None:
        return NA
    if isinstance(value, float) and math.isnan(value):
        return UNKNOWN
    n = int(value)
    if n < 1024:
        return f"{n:,} B"
    for unit, div in (("KiB", 1024), ("MiB", 1024**2), ("GiB", 1024**3)):
        if n < div * 1024 or unit == "GiB":
            return f"{n:,} B（{n / div:.1f} {unit}）"
    return f"{n:,} B"


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """GFM 表格。空表返回一句 ``（空）`` 而不是一个只有表头的空壳 ——
    一个只有表头的表看起来像「这一节没问题」，而它真正的意思是「这一节没有数据」。"""
    if not rows:
        return "（空）"
    out = [
        "| " + " | ".join(str(h) for h in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


# ----------------------------------------------------------------------
# 取数
# ----------------------------------------------------------------------
def _label(result: Any, spec_id: str) -> str:
    """行名。**优先用配置里那个 label**，它比 ``spec_id`` 可读得多。

    ``label`` 住在 ``artifacts[sid].meta`` 里（它是配置给的，而 Spec 里没有这个字段 ——
    一个纯展示用的名字不该进 Spec，进了就会进指纹，改一句注释会让版本摘要变）。
    产物元数据不在时退回 ``spec_id``：**不发明一个名字**。
    """
    art = (getattr(result, "artifacts", None) or {}).get(spec_id)
    if art is not None:
        lab = (art.meta or {}).get("label")
        if lab:
            return str(lab)
    return spec_id


def _ordered_specs(result: Any) -> list[tuple[str, Any]]:
    """全部 spec，按行名排序。

    **不按 ``spec_id`` 排、也不按 ``specs`` 字典的插入序**：载入路径上，
    ``result.json`` 是 ``sort_keys=True`` 写出来的，插入序已经变成「按 id 字典序」，
    而内存路径上是「跑到谁是谁」—— 同一份结果在两条路径上会渲染成行序不同的两张表。
    按行名排则两条路径**逐行相同**，报告才可以被 diff。
    """
    specs = getattr(result, "specs", None) or {}
    return sorted(specs.items(), key=lambda kv: (_label(result, kv[0]), kv[0]))


def axis_fmt(name: str, value: Any) -> str:
    """按**轴自己的量纲**渲染一格。

    四条轴里两条是「比值」（gain / interference），另两条是**计数**
    （字节 / FLOP）。一律走 ``fmt`` 的话，字节数会渲染成 ``104512.0000`` ——
    一个不带千分位、还带四位小数的字节数，读者要自己数位数才知道它是 100 KB。

    这里不是审美问题：``vram_bytes`` 与 ``train_flops`` 都要和别的表对账
    （显存分解表的合计、三路互校里的外推值），对不上账的最常见原因就是
    两边用了不同的量纲或不同的取整。
    """
    if value is None:
        return NA
    if isinstance(value, float) and math.isnan(value):
        return UNKNOWN
    if name == "vram_bytes":
        return fmt_bytes(value)
    if name == "train_flops":
        return fmt_int(value)
    return fmt(value)


def _reading(delta: Any) -> str:
    """把「基座 − chance」读成一句话。**``0`` 有自己的说法。**

    写成「非负即高于」的话，「正好等于乱猜」会被读成「高于乱猜」—— 而这两个
    结论的方向完全不同：等于乱猜意味着这个能力上模型什么都没学出来，
    「高于」则意味着它至少捕到了一点信号。这一列是本节唯一的判据，
    它把 ``0.0000`` 归到哪一边，读者就会得到哪个结论。

    ``sort`` 这一项在本项目的默认口径下**恰好**是 ``0.0000``（0.1250 vs 0.1250），
    所以这不是一个假想的边界。
    """
    if delta is None:
        return UNKNOWN
    if isinstance(delta, float) and math.isnan(delta):
        return UNKNOWN
    if delta < 0:
        return "低于乱猜"
    if delta > 0:
        return "高于乱猜"
    return "正好等于乱猜"


def _get(mapping: Any, key: str) -> Any:
    """``mapping.get(key)`` 的显式版：**缺失返回 None，不返回 0**。"""
    if not isinstance(mapping, Mapping):
        return None
    return mapping.get(key)


def _cost(result: Any, spec_id: str) -> Mapping[str, Any]:
    costs = (getattr(result, "counters", None) or {}).get("costs") or {}
    c = costs.get(spec_id)
    return c if isinstance(c, Mapping) else {}


def _run(result: Any, spec_id: str) -> Any:
    return (getattr(result, "runs", None) or {}).get(spec_id)


def _eval(result: Any, spec_id: str) -> Any:
    return (getattr(result, "evals", None) or {}).get(spec_id)


def _check(result: Any, name: str) -> Mapping[str, Any]:
    checks = (getattr(result, "counters", None) or {}).get("checks") or {}
    c = checks.get(name)
    return c if isinstance(c, Mapping) else {}


# ----------------------------------------------------------------------
# 1. 基准表
# ----------------------------------------------------------------------
def baseline_rows(result: Any) -> list[tuple[str, ...]]:
    """四项能力的三个基准：chance、随机基座实测、差值。

    **差值那一列是这一节存在的理由。** 随机初始化的基座在每项能力上的实测值
    全都**低于 chance**，这不是噪声也不是 bug：未训练模型的 argmax 由
    「哪个 token 的 unembedding 范数最大」主导，而那个 token 与正确答案无关，
    所以它比均匀乱猜还差。看见这一列，读者才会接受
    「``interference`` 不能用实测基座当基准」这件事 —— 用了的话，
    干扰恒为负，而且**参数越多的策略越像「干扰最小」**。
    """
    base = dict(getattr(result, "base_accs", None) or {})
    chance = dict(getattr(result, "chance_accs", None) or {})
    rows: list[tuple[str, ...]] = []
    for cap in sorted(set(base) | set(chance)):
        b, c = base.get(cap), chance.get(cap)
        delta = (b - c) if (b is not None and c is not None) else None
        rows.append((
            cap,
            fmt(c, 4),
            fmt(b, 4),
            fmt_signed(delta, 4) if delta is not None else NA,
            _reading(delta),
        ))
    return rows


# ----------------------------------------------------------------------
# 2. 对比矩阵
# ----------------------------------------------------------------------
def matrix_header() -> tuple[str, ...]:
    return (
        "spec", "策略", "目标能力", "目标 acc", "基座 acc", "gain",
        "干扰", "可训练参数", "产物", "显存", "训练 FLOPs", "前沿",
    )


def matrix_rows(result: Any) -> list[tuple[str, ...]]:
    """主对比矩阵：效果四列 + 代价四列。

    四列代价全部来自**可重算或已落盘**的来源（参数计划 / 产物元数据 /
    ``counters["costs"]`` / ``runs.jsonl``），没有一列是这里现算的 ——
    报告不做算术，它只排版已经算好并被核对过的数。
    """
    rows: list[tuple[str, ...]] = []
    for spec_id, spec in _ordered_specs(result):
        ev = _eval(result, spec_id)
        run = _run(result, spec_id)
        art = (getattr(result, "artifacts", None) or {}).get(spec_id)
        cost = _cost(result, spec_id)
        cap = spec.dataset.capability
        base = (getattr(result, "base_accs", None) or {}).get(cap)
        target_acc = ev.target_acc if ev is not None else None
        rows.append((
            _label(result, spec_id),
            spec.finetuning.kind,
            cap,
            fmt(target_acc, 4),
            fmt(base, 4),
            fmt_signed(result.gain_of(spec_id), 4) if ev is not None else NA,
            fmt_signed(result.interference_of(spec_id), 4) if ev is not None else NA,
            fmt_int(art.n_trainable_params if art is not None else None),
            fmt_bytes(art.payload_bytes) if art is not None else NA,
            fmt_bytes(_get(cost, "vram_bytes")),
            fmt_int(run.train_flops if run is not None else None),
            "★" if result.is_frontier(spec_id) else "",
        ))
    return rows


def cost_header() -> tuple[str, ...]:
    return ("spec",) + tuple(name for _, name in COST_COLUMNS) + ("合计", "可训/总参数")


def cost_rows(result: Any) -> list[tuple[str, ...]]:
    """显存分解。**激活那一列必须单独可见。**

    ``activation_bytes`` 只依赖 ``(d_model, n_layers, seq_len, batch)``，
    与策略**无关**。于是「QLoRA 的显存只有 full 的 1/8」这句话在本项目的规模下
    是错的：基座那一项确实变成 1/8，但激活是四种策略共同的一块底，
    合起来的总显存差距远小于 8 倍。这一列藏进「其它」里，
    那个错误结论就会自动成立。
    """
    rows: list[tuple[str, ...]] = []
    for spec_id, _spec in _ordered_specs(result):
        cost = _cost(result, spec_id)
        cells = [fmt_bytes(_get(cost, key)) for key, _ in COST_COLUMNS]
        n_train = _get(cost, "n_trainable_params")
        n_base = _get(cost, "n_base_params")
        rows.append(
            (_label(result, spec_id), *cells, fmt_bytes(_get(cost, "vram_bytes")),
             f"{fmt_int(n_train)} / {fmt_int(n_base)}")
        )
    return rows


# ----------------------------------------------------------------------
# 3. 前沿
# ----------------------------------------------------------------------
def frontier_header() -> tuple[str, ...]:
    return ("spec",) + tuple(AXIS_LABEL.get(name, name) for name, _ in OBJECTIVES) + ("各轴冠军",)


def frontier_rows(result: Any) -> list[tuple[str, ...]]:
    """前沿上的点，**按所有可用点重新算一次目标值**，而不是读 ``frontier`` 列表。

    为什么重算：``frontier`` 是 ``run_pipeline`` 那一刻定下来的，而报告可能是在
    另一台机器、另一份 ``counters`` 上渲染的。重算的话，表里的数**必然**与
    「谁在前沿上」这句话一致；不重算的话，两者可以悄悄地分叉 ——
    而分叉的表现是「有一个明显更差的点也在前沿上」，读者会以为算法错了。
    """
    specs = getattr(result, "specs", None) or {}
    if not specs:
        return []
    points = {sid: compare.objective_values(result, sid) for sid in specs}
    front, _stats = compare.pareto_frontier(points)
    keys = compare.frontier_keys(points, front)
    champions: dict[str, list[str]] = {}
    for axis, sids in keys.items():
        for sid in sids:
            champions.setdefault(sid, []).append(axis)

    rows: list[tuple[str, ...]] = []
    for spec_id in front:
        p = points[spec_id]
        cells = [axis_fmt(name, p.get(name)) for name, _ in OBJECTIVES]
        champs = champions.get(spec_id, [])
        label = ", ".join(
            "前沿" if c == "pareto" else AXIS_LABEL.get(c, c) for c in sorted(champs)
        )
        rows.append((_label(result, spec_id), *cells, label))
    return rows


def constant_axes(result: Any) -> list[tuple[str, str]]:
    """在**整张网格**上取值恒定的目标轴，返回 ``(轴名, 那个值)``。

    一条恒定的轴对 Pareto 前沿**没有任何贡献**，但它看起来完全正常：
    它在表里是一列数，在其他轴比较时它不制造任何矛盾，前沿也照样算得出来。
    唯一的症状是「冠军表里这条轴的冠军是整个前沿」——读者会以为表坏了。

    本项目里这件事是**真的会发生的**：``train_flops`` 目前对每个 spec 都相同
    （所有格子跑同样的 steps × batch，而解析算子总数在四种策略下也是同一个数），
    于是四条目标轴实际上只有三条在起作用。自动把这条说出来，
    比让读者自己去比较 18 个八位数要可靠得多。

    判据用「不同的**取值个数**」而不是「标准差」：这里的数是计数，
    ``train_flops`` 一致时标准差是 0，但它和「差 1 个 FLOP」的区别在报告里
    是一样的（都是「没有区分度」），而标准差会把它俩分开。
    """
    specs = getattr(result, "specs", None) or {}
    if not specs:
        return []
    points = {sid: compare.objective_values(result, sid) for sid in specs}
    out: list[tuple[str, str]] = []
    for name, _direction in OBJECTIVES:
        vals = {
            float(p[name])
            for p in points.values()
            if name in p and not math.isnan(float(p.get(name, float("nan"))))
        }
        if len(vals) == 1:
            out.append((name, axis_fmt(name, next(iter(vals)))))
    return out


def champion_rows(result: Any) -> list[tuple[str, ...]]:
    """各轴冠军单独一张表。

    它和前沿是**两件事**，混在一张表里会让「前沿上有 5 个点」被读成
    「有 5 个并列最佳」。前沿回答「哪些方案没被完全压制」，
    冠军回答「只看这一条，谁最好」—— 冠军一定是前沿点，
    而前沿上大多数点不是任何一轴的冠军。
    """
    specs = getattr(result, "specs", None) or {}
    if not specs:
        return []
    points = {sid: compare.objective_values(result, sid) for sid in specs}
    front, _ = compare.pareto_frontier(points)
    keys = compare.frontier_keys(points, front)
    rows: list[tuple[str, ...]] = []
    for axis, _direction in OBJECTIVES:
        for sid in keys.get(axis, []):
            p = points[sid]
            rows.append((AXIS_LABEL.get(axis, axis), _label(result, sid), axis_fmt(axis, p.get(axis))))
    return rows


# ----------------------------------------------------------------------
# 4. 裁决与核对
# ----------------------------------------------------------------------
def verdict_rows(result: Any) -> list[tuple[str, ...]]:
    """四条设计期望的裁决。**``refuted`` 是结论，不是错误。**

    每一行都带 ``observed`` 的条数：读者不该只看到一个布尔，他应该能自己
    从证据行里算出结论。这里只给条数，证据本身在 ``result.json`` 里（``verdicts``）
    与 :func:`explain` 里。
    """
    mark = {"confirmed": "✅ 成立", "refuted": "❌ 被推翻", "unknown": "❔ 未判定"}
    rows: list[tuple[str, ...]] = []
    for v in getattr(result, "verdicts", None) or []:
        rows.append((
            str(v.get("name", "")),
            mark.get(str(v.get("verdict")), str(v.get("verdict"))),
            str(len(v.get("observed") or [])),
            str(v.get("reason", "")),
        ))
    return rows


def expectation_status_rows(result: Any) -> list[tuple[str, ...]]:
    """配置里**要求**的期望 vs 实际裁决出来的。

    差集非空就是一条真问题：配置里写了四条期望，而裁决只出来三条，
    说明有一条**没有被判**（裁决函数缺失，或者配对比不出来）。这种缺失
    在只看裁决表时是看不出来的 —— 少一行而已。
    """
    # 声明从 `result.expectations` 来（那是落盘的），**不是从 `cfg`** ——
    # 载入路径上 `cfg` 是空的（输出目录刻意不存整份配置），于是读 `cfg` 会让
    # 这张表在「报告从磁盘渲染」时整张消失。它曾经就正是这样：内存路径 6 行，
    # 载入路径一行「（空）」。而这张表要回答的恰恰是「有没有哪条期望被漏掉了」，
    # 它自己整张消失是最坏的一种表现。
    want = dict(getattr(result, "expectations", None) or {})
    if not want:
        want = expectations_of(getattr(result, "cfg", None) or {})
    got = {
        str(v.get("name")): str(v.get("verdict"))
        for v in (getattr(result, "verdicts", None) or [])
    }
    rows: list[tuple[str, ...]] = []
    for name, expected in sorted(want.items()):
        if name not in got:
            rows.append((name, "是" if expected else "否", "**缺失**", "配置里要求裁决，但结果里没有这一条"))
            continue
        v = got[name]
        if not expected:
            rows.append((name, "否", v, "配置里已关闭这条期望，裁决仅供旁观"))
        else:
            rows.append((name, "是", v, ""))
    return rows


def check_rows(result: Any) -> list[tuple[str, ...]]:
    """八项核对的结果。**「不适用」与「没通过」必须分开写。**

    ``merge`` / ``gradients`` 对 ``full`` 是**不适用**的（没有适配器），
    它们的 ``ok`` 是 ``True`` 而 ``applicable`` 是 ``False``。把两者都渲染成
    「通过」会让「8/8 通过」这句话在只有 5 项真的被验过的时候也成立。
    """
    checks = (getattr(result, "counters", None) or {}).get("checks") or {}
    labels = {
        "ir_roundtrip": "IR 序列化往返",
        "single_call_path": "单调用路径（无 if kind ==）",
        "merge": "适配器合并回基座",
        "macs": "解析算子表 vs 计数器（逐算子）",
        "three_way": "代价三路互校",
        "gradients": "梯度数值差分",
        "quant_error": "量化误差单调性",
        "artifact_bytes": "产物字节 = 声明值",
    }
    rows: list[tuple[str, ...]] = []
    for name in ("ir_roundtrip", "single_call_path", "merge", "macs", "three_way",
                 "gradients", "quant_error", "artifact_bytes"):
        c = checks.get(name)
        if not isinstance(c, Mapping):
            rows.append((labels[name], "**缺失**", "结果里没有这一项核对的记录"))
            continue
        ok = c.get("ok")
        mark = "✅" if ok else "❌"
        rows.append((labels[name], mark, _check_detail(name, c)))
    return rows


def _check_detail(name: str, c: Mapping[str, Any]) -> str:
    """一行的说明。**每个分支都从实际字段里取，没有一句是写死的话。**"""
    if name in ("merge", "gradients"):
        if not c.get("applicable", True):
            n_na = c.get("n_not_applicable", 0)
            return f"不适用（{n_na} 个规格没有可训练的适配器），已核对 {c.get('n_applicable', 0)} 个"
        if name == "merge":
            return (f"最大绝对差 {fmt(c.get('max_abs_diff'))}（容差 {fmt(c.get('tolerance'))}），"
                    f"核对 {c.get('n_applicable', 0)} 个")
        return (f"最差相对误差 {fmt(c.get('worst_rel_error'))}，"
                f"核对 {c.get('n_applicable', 0)} 个计划")
    if name == "macs":
        return f"逐算子比对 {c.get('n_checked', 0)} 个规格"
    if name == "three_way":
        return f"①解析 vs ②实测 单样本算子数，{c.get('n_checked', 0)} 个规格"
    if name == "ir_roundtrip":
        return f"往返 {c.get('n_checked', 0)} 份 Spec"
    if name == "single_call_path":
        return f"扫了 {len(c.get('scanned') or [])} 个模块，命中 {len(c.get('hits') or [])} 处"
    if name == "quant_error":
        comps = c.get("comparisons") or []
        if not comps:
            return "没有可比的位数对（网格里缺少同构的 4-bit / 8-bit 对照）"
        parts = [f"{x['low_bits']}-bit 误差是 {x['high_bits']}-bit 的 {fmt(x.get('ratio'))} 倍"
                 for x in comps]
        return "；".join(parts)
    if name == "artifact_bytes":
        return f"核对 {c.get('n_checked', 0)} 个产物"
    return ""


# ----------------------------------------------------------------------
# 5. 解释一个 spec
# ----------------------------------------------------------------------
def explain(result: Any, spec_id: str, *, max_capabilities: int = 12) -> str:
    """把一份 Spec 的「配了什么 / 跑成什么样 / 为什么在这个位置」写清楚。

    它和报告正文的分工：正文回答「64 个规格里哪个好」，``explain`` 回答
    「这一个是怎么来的」。所以它从 :class:`~finetunefactory.ir.TrainingSpec`
    的**每一个字段**开始 —— 一份报告里最容易被跳过、也最容易出错的就是
    「这一行到底用的是哪个配置」。
    """
    spec = (getattr(result, "specs", None) or {}).get(spec_id)
    if spec is None:
        known = sorted((getattr(result, "specs", None) or {}))
        raise KeyError(f"未知 spec_id：{spec_id}；已知 {len(known)} 个：{', '.join(known[:8])}")

    out: list[str] = []
    ft = spec.finetuning
    out.append(f"# {_label(result, spec_id)}")
    out.append("")
    out.append(f"`{spec_id}`")
    out.append("")

    # ---- 配了什么
    out.append("## 一、这份 Spec 配了什么")
    out.append("")
    rows: list[tuple[str, ...]] = [
        ("模型", spec.model.name, f"d_model={spec.model.d_model} n_layers={spec.model.n_layers} "
                                  f"n_heads={spec.model.n_heads} d_ff={spec.model.d_ff}"),
        ("数据集能力", spec.dataset.capability, ""),
        ("模板", spec.dataset.template, ""),
        ("序列长度", fmt_int(spec.dataset.seq_len), f"n_train={fmt_int(spec.dataset.n_train)} "
                                                     f"n_eval={fmt_int(spec.dataset.n_eval)}"),
        ("训练目标", spec.objective, ""),
        ("策略", ft.kind, _strategy_detail(ft)),
        ("优化器", spec.optimizer.name, f"lr={fmt(spec.optimizer.lr)} "
                                        f"weight_decay={fmt(spec.optimizer.weight_decay)}"),
        ("调度器", spec.scheduler.name, f"warmup={fmt_int(spec.scheduler.warmup_steps)} "
                                         f"total_steps={fmt_int(spec.scheduler.total_steps)}"),
        ("批", fmt_int(spec.batch.size), f"grad_accum={fmt_int(spec.batch.grad_accum)}"),
        ("精度 / 并行", f"{spec.precision} / {spec.distributed}", ""),
        ("种子", fmt_int(spec.seed), ""),
    ]
    out.append(markdown_table(("项", "值", "细节"), rows))
    out.append("")

    # ---- 跑成什么样
    run = _run(result, spec_id)
    ev = _eval(result, spec_id)
    art = (getattr(result, "artifacts", None) or {}).get(spec_id)
    out.append("## 二、跑成什么样")
    out.append("")
    if run is None and ev is None and art is None:
        out.append(f"{UNKNOWN}：这一份结果的运行记录 / 评测 / 产物都没有被载入。"
                   "只用 `result.json` 是渲染不出这一节的 —— "
                   "`runs.jsonl`、`evals.jsonl`、`artifacts/*.json` 缺一不可。")
        out.append("")
    else:
        losses = list(getattr(run, "losses", []) or []) if run is not None else []
        out.append(markdown_table(
            ("量", "值", "说明"),
            [
                ("步数", fmt_int(getattr(run, "steps", None)), ""),
                ("首损失", fmt(losses[0] if losses else None), "未训练模型在训练集上的起点"),
                ("末损失", fmt(losses[-1] if losses else None), "最后一步"),
                ("损失降幅", fmt_signed((losses[0] - losses[-1]) if len(losses) >= 2 else None),
                 "正 = 确实在学"),
                ("参与更新的参数", fmt_int(getattr(run, "n_updated_params", None)),
                 "从**真的拿到梯度的张量**数出来的，不是配置里的数"),
                ("训练耗时", f"{fmt(getattr(run, 'train_seconds', None), 3)} s" if run is not None else NA, ""),
                ("目标能力 acc", fmt(ev.target_acc if ev is not None else None), ""),
                ("gain", fmt_signed(result.gain_of(spec_id)) if ev is not None else NA,
                 "基准 = 同一个模型微调前的**实测**准确率"),
                ("干扰", fmt_signed(result.interference_of(spec_id)) if ev is not None else NA,
                 "基准 = chance；正 = 别的能力被踹到乱猜以下"),
                ("产物字节", fmt_bytes(art.payload_bytes) if art is not None else NA,
                 "裸 float32 小端，无头部；与算术值逐位相等"),
                ("产物摘要", (art.digest[:16] + "…") if art is not None and art.digest else NA, ""),
                ("合并最大绝对差",
                 fmt(art.merge_max_abs_diff) if art is not None else NA,
                 "不适用" if art is not None and art.merge_max_abs_diff is None else "合并回基座后与适配器直算的差"),
                ("量化相对误差", fmt(art.quant_error) if art is not None else NA,
                 "非量化策略为 0；QLoRA 的代价全在这一项上"),
            ],
        ))
        out.append("")

        per_cap = dict(getattr(ev, "per_capability", {}) or {}) if ev is not None else {}
        if per_cap:
            base = dict(getattr(result, "base_accs", None) or {})
            chance = dict(getattr(result, "chance_accs", None) or {})
            cap_rows = [
                (
                    cap,
                    fmt(v),
                    fmt(base.get(cap)),
                    fmt_signed(v - base[cap], 4) if cap in base else NA,
                    fmt(chance.get(cap)),
                    fmt_signed(chance[cap] - v, 4) if cap in chance else NA,
                    "← 目标" if cap == spec.dataset.capability else "",
                )
                for cap, v in sorted(per_cap.items())[:max_capabilities]
            ]
            out.append("### 逐能力")
            out.append("")
            out.append(markdown_table(
                ("能力", "微调后", "基座（实测）", "相对基座", "chance", "相对 chance", ""), cap_rows
            ))
            out.append("")
            out.append("`相对 chance` 那一列才是「有没有把这项能力踹到乱猜以下」的判据；"
                       "`相对基座` 那一列问的是「训练有没有帮上忙」。两个问题，两个基准。")
            out.append("")

    # ---- 为什么在这个位置
    out.append("## 三、它在整个网格里的位置")
    out.append("")
    specs = getattr(result, "specs", None) or {}
    if not specs:
        out.append(f"{UNKNOWN}：结果里没有任何 Spec。")
    else:
        points = {sid: compare.objective_values(result, sid) for sid in specs}
        p = points.get(spec_id, {})
        front, stats = compare.pareto_frontier(points)
        front_keys = compare.frontier_keys(points, front)
        on_front = spec_id in front
        out.append(markdown_table(
            ("量", "值"),
            [(AXIS_LABEL.get(name, name), axis_fmt(name, p.get(name))) for name, _ in OBJECTIVES]
            + [("在前沿上", "是" if on_front else "否"),
               ("前沿规模", f"{len(front)} / {stats.get('n_usable', 0)} 个可用点"
                            f"（{stats.get('n_dominated', 0)} 个被支配）")],
        ))
        out.append("")
        if not on_front:
            blockers = [
                other for other in front
                if compare.dominates(points[other], p, OBJECTIVES)
            ]
            if blockers:
                out.append("被以下前沿点支配（每一轴都不差、至少一轴严格更好）：")
                out.append("")
                out.append(markdown_table(
                    ("支配者",) + tuple(AXIS_LABEL.get(n, n) for n, _ in OBJECTIVES),
                    [
                        (_label(result, o), *[axis_fmt(n, points[o].get(n)) for n, _ in OBJECTIVES])
                        for o in sorted(blockers)[:6]
                    ],
                ))
                out.append("")
            else:
                out.append(f"**不在前沿上，但没有任何一个前沿点支配它。** "
                           f"这通常意味着它是被一组点的**组合**压住的，"
                           f"或者它的某个轴是 NaN（`skipped_nan` 里可能有它）。")
                out.append("")
        champs = sorted(
            axis for axis, sids in front_keys.items() if spec_id in sids
        )
        if champs:
            out.append("它是这些轴上的冠军：" + "、".join(
                "整体前沿" if a == "pareto" else AXIS_LABEL.get(a, a) for a in champs
            ) + "。")
            out.append("")
        skipped = (getattr(result, "counters", None) or {}).get("skipped") or {}
        if spec_id in (skipped.get("pareto") or []):
            out.append(f"⚠️ 它出现在 `skipped.pareto` 里 —— 有一个轴是 {UNKNOWN}，"
                       f"于是它**按设计**退出了前沿排序。这不等于它差。")
            out.append("")
    return "\n".join(out)


def _strategy_detail(ft: Any) -> str:
    """策略的细节。**四种 kind 都从这里展开，没有分支丢掉字段。**"""
    if ft.kind in ("lora", "qlora"):
        parts = [f"rank={ft.rank}", f"alpha={fmt(ft.alpha)}",
                 f"scale={fmt(ft.scale)}", f"挂载={','.join(ft.target_modules) or NA}"]
        if ft.kind == "qlora":
            parts.append(f"{ft.quant_bits}-bit")
            parts.append(f"group={ft.quant_group}")
        return " ".join(parts)
    if ft.kind == "freeze":
        return f"冻结 {ft.freeze_layers} 层"
    return "全部参数可训练"


# ----------------------------------------------------------------------
# 摘要与正文
# ----------------------------------------------------------------------
def summary_line(result: Any) -> str:
    """一行摘要。CLI 打完长表之后用它收尾。"""
    stats = (getattr(result, "counters", None) or {}).get("pareto_stats") or {}
    checks = (getattr(result, "counters", None) or {}).get("checks") or {}
    n_ok = sum(1 for c in checks.values() if isinstance(c, Mapping) and c.get("ok"))
    verdicts = getattr(result, "verdicts", None) or []
    n_conf = sum(1 for v in verdicts if v.get("verdict") == "confirmed")
    n_ref = sum(1 for v in verdicts if v.get("verdict") == "refuted")
    timings = (getattr(result, "counters", None) or {}).get("timings") or {}
    total = timings.get("total")
    return (
        f"{len(getattr(result, 'specs', None) or {})} 个规格 → "
        f"前沿 {stats.get('n_frontier', len(getattr(result, 'frontier', []) or []))} 个"
        f"（{stats.get('n_dominated', 0)} 个被支配）｜ "
        f"核对 {n_ok}/{len(checks)} 项通过｜ "
        f"期望 {n_conf} 条成立、{n_ref} 条被推翻"
        + (f"｜ 总耗时 {fmt(total, 3)} s" if isinstance(total, (int, float)) else "")
    )


def meta_payload(result: Any) -> dict[str, Any]:
    """给 ``result.json`` 附的元信息。**只放摘要，不放任何重算得出的量。**"""
    counters = getattr(result, "counters", None) or {}
    return {
        "protocol_digest": getattr(result, "protocol_digest", ""),
        "version_digest": getattr(result, "version_digest", ""),
        "n_specs": len(getattr(result, "specs", None) or {}),
        "pareto_stats": counters.get("pareto_stats", {}),
        "timings": counters.get("timings", {}),
    }


def render_markdown(result: Any) -> str:
    """报告正文。

    章节顺序**不是排版偏好，是阅读顺序**：基准 → 矩阵 → 代价 → 前沿 → 裁决 → 核对。
    把「核对」提到前面会让读者先看到一堆 ✅，然后才去看它们到底在核什么。
    """
    out: list[str] = []
    out.append("# FineTuneFactory 报告")
    out.append("")
    out.append(summary_line(result))
    out.append("")
    out.append(f"- 口径摘要 `protocol_digest`：`{getattr(result, 'protocol_digest', '') or UNKNOWN}`")
    out.append(f"- 版本摘要 `version_digest`：`{getattr(result, 'version_digest', '') or UNKNOWN}`")
    out.append("")
    out.append("口径摘要只跟评测口径有关；版本摘要跟模型注册表 + 数据集索引 + "
               "扇出网格有关。**后者变了必须整体重跑**，前者变了只需重出报告。")
    out.append("")

    # ---- 一、基准
    out.append("## 一、基准点：这份报告里的「涨了」和「掉了」各自对谁而言")
    out.append("")
    out.append(markdown_table(("能力", "chance（乱猜）", "基座（微调前实测）", "基座 − chance", "读作"),
                              baseline_rows(result)))
    out.append("")
    out.append("两个基准点对应两个问题，**不能合用一个**：")
    out.append("")
    out.append("- `gain = 微调后 − 基座实测` —— 训练有没有帮上忙；")
    out.append("- `interference = mean(chance − 微调后)` —— 训练有没有把**不相干**的能力踹到乱猜以下。")
    out.append("")
    out.append("为什么干扰不能用基座实测当基准：本项目从**随机初始化**的基座开始训，"
               "而随机基座在四项能力上全都低于 chance（见上表最后一列）。"
               "拿它当基准，「干扰」恒为负数，而且**参数越多的策略越像干扰最小** —— "
               "`full` 会算出「干扰最小」，把结论恰好反过来，而表里看不出任何异常。")
    out.append("")

    # ---- 二、矩阵
    out.append("## 二、对比矩阵")
    out.append("")
    out.append(markdown_table(matrix_header(), matrix_rows(result)))
    out.append("")
    out.append("`★` = 在 Pareto 前沿上。`前沿` 一列为空只表示「没进前沿」，不表示「差」——"
               "被支配的点里也可能有成本极低的那个。")
    out.append("")

    # ---- 三、显存
    out.append("## 三、显存分解")
    out.append("")
    out.append(markdown_table(cost_header(), cost_rows(result)))
    out.append("")
    out.append("**激活那一列与策略无关** —— 它只依赖 `d_model × n_layers × seq_len × batch`，"
               "是四种策略共同的一块底。所以「QLoRA 显存只有 full 的 1/8」这句话在本项目的"
               "规模下不成立：基座那一项确实变成 1/8，但合计的差距远小于 8 倍。"
               "激活这一列如果藏进「其它」里，那个错误结论就会自动成立。")
    out.append("")

    # ---- 四、前沿
    out.append("## 四、Pareto 前沿")
    out.append("")
    out.append("目标轴：" + "、".join(
        f"{AXIS_LABEL.get(n, n)}（{'越大越好' if d == 'max' else '越小越好'}）" for n, d in OBJECTIVES
    ))
    out.append("")
    out.append(markdown_table(frontier_header(), frontier_rows(result)))
    out.append("")
    stats = (getattr(result, "counters", None) or {}).get("pareto_stats") or {}
    if stats:
        out.append(f"可用点 {stats.get('n_usable', 0)} / {stats.get('n_points', 0)}，"
                   f"前沿 {stats.get('n_frontier', 0)}，被支配 {stats.get('n_dominated', 0)}。")
        if stats.get("skipped_nan"):
            out.append("")
            out.append(f"⚠️ 有 {len(stats['skipped_nan'])} 个点因为**某个轴取不到值**而退出了排序"
                       f"（`skipped_nan`）：它们不比任何点差，只是不可比。"
                       f"这通常意味着代价表没进 `counters` —— 那是载入路径的常见缺失，不是算法在筛人。")
        out.append("")
    out.append("### 各轴冠军")
    out.append("")
    out.append(markdown_table(("目标轴", "spec", "取值"), champion_rows(result)))
    out.append("")
    out.append("冠军一定是前沿点，但前沿上大多数点不是任何一轴的冠军 —— "
               "这正好说明前沿为什么是「一条线」而不是「一个点」。")
    out.append("")
    const = constant_axes(result)
    if const:
        out.append("⚠️ **本次网格里有轴是常数**，它对前沿没有任何贡献：")
        out.append("")
        out.append(markdown_table(("目标轴", "恒定取值"), [
            (AXIS_LABEL.get(n, n), v) for n, v in const
        ]))
        out.append("")
        out.append("这不是错误，也不是「这一列没算」——它是**真的**每个规格都取同一个值"
                   "（所有格子跑同样的步数与批，而结构上四种策略的单样本算子数也相同）。"
                   "后果是「谁进了前沿」这件事实际由剩下的几条轴决定。"
                   "要让它有区分度，得让网格里的 **steps / batch / 序列长度** 产生差异 —— "
                   "而在本项目的口径下它们刻意对全部规格统一，那正是「策略之间」可比的前提。")
        out.append("")

    # ---- 五、裁决
    out.append("## 五、四条设计期望的裁决")
    out.append("")
    out.append(markdown_table(("期望", "裁决", "证据条数", "说明"), verdict_rows(result)))
    out.append("")
    out.append("**被推翻不是错误。** 这四条是设计时的猜测，实测把它们推翻是这个项目"
               "最值钱的输出之一 —— 改掉期望去迁就结果，等于把实验做成了自我证实。"
               "证据行（`observed`）在 `result.json` 里，读者应当能自己算出结论。")
    out.append("")
    out.append("配置里要求 vs 实际裁决：")
    out.append("")
    out.append(markdown_table(("期望", "配置要求", "实际裁决", "备注"), expectation_status_rows(result)))
    out.append("")

    # ---- 六、核对
    out.append("## 六、验收核对")
    out.append("")
    out.append(markdown_table(("核对项", "结果", "说明"), check_rows(result)))
    out.append("")
    out.append("「不适用」与「没通过」是两件事：`full` / `freeze` 没有适配器，"
               "合并与梯度对它们是**不适用**（`ok=True, applicable=False`），"
               "把它们算作「通过」会让「全部通过」在只有一部分真的被验过时也成立。")
    out.append("")
    out.append("代价三路互校里，**② 是实测、③ 是估算**："
               "`③ = ① × steps × 有效批 × 3 × 2`，其中 3 = 前向 1 + 反向 2，"
               "2 = 每个 MAC 两个 FLOP。③ 的系数是经验值，核的是「它是不是真按这个式子算的」，"
               "不是「这个系数对不对」。")
    out.append("")

    # ---- 七、血缘
    dag = getattr(result, "dag", None)
    out.append("## 七、血缘")
    out.append("")
    if dag is None:
        out.append(f"{UNKNOWN}：`lineage.json` 不在或被显式关掉了。"
                   "血缘是**旁证**不是主产物，缺了它上面任何一节都照样成立。")
    else:
        kinds: dict[str, int] = {}
        for node in dag.nodes():
            kinds[node.kind] = kinds.get(node.kind, 0) + 1
        out.append(f"节点 {len(dag.nodes())} 个，边 {len(dag.edges())} 条，"
                   f"是否无环：{'是' if dag.is_acyclic() else '否'}。")
        out.append("")
        out.append(markdown_table(
            ("节点种类", "个数"),
            [(k, fmt_int(v)) for k, v in sorted(kinds.items())],
        ))
        out.append("")
        out.append("`pareto` 是整个图里**唯一**一个「吃很多、吐一个」的节点 —— "
                   "它收的是前沿候选的产物，出的是一条可以被引用、也可以被反驳的前沿。")
    return "\n".join(out) + "\n"
