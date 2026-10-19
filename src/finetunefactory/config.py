"""配置：默认值 + 深合并 + 校验 + 两类摘要 + 优化目标。

摘要分两种，**不能混**（`L2-moefactory` 的坑 70 就是混了）：

- :func:`protocol_digest` 只跟**口径**有关（评测批量、激活估算系数、报告阈值）。
  改这些会改变报告里的数字，但不会改变任何一个 Spec。
- :func:`version_digest` 跟**输入对象**有关（模型注册表 + 数据集索引 + 扇出网格）。
  它变了说明「这一轮实验的输入变了」，报告必须整体重跑。

与 L2 那几个项目的**唯一区别**：这个项目**有**优化目标（:func:`objectives_of`
返回四个），因为它的产物里有一条 Pareto 前沿。但目标只在 :mod:`compare` 里用，
**绝不允许渗进 :mod:`train`** —— 训练本身不关心哪个配置更好，它只负责跑出数字。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .rng import canonical_json, join_digest, sha256_text

__all__ = [
    "DEFAULTS",
    "OBJECTIVES",
    "cell_label",
    "deep_merge",
    "expectations_of",
    "load_config",
    "objectives_of",
    "protocol_digest",
    "resolve_path",
    "sweep_cells",
    "validate_config",
    "version_digest",
]

#: 四个目标。**方向是配置的一部分**，不是散落在比较代码里的 if。
OBJECTIVES: tuple[tuple[str, str], ...] = (
    ("gain", "max"),
    ("interference", "min"),
    ("vram_bytes", "min"),
    ("train_flops", "min"),
)

DEFAULTS: dict[str, Any] = {
    "model": {"registry": "data/models/models.json"},
    "dataset": {"index": "data/datasets/index.json"},
    "sweep": {
        # 扇出轴。**顺序即报告里的顺序**，也是「谁先被跑到」的顺序。
        "capabilities": ["upper", "shift", "reverse", "sort"],
        "kinds": ["full", "freeze", "lora", "qlora"],
        "ranks": [4, 8],
        "target_sets": {
            "qv": ["q_proj", "v_proj"],
            "qkvo": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "objectives": ["sft", "sft_response_only"],
        #: **显式网格**（可选）。给了它就**不再做叉积** ——
        #:
        #: 叉积的问题是它对每种策略一视同仁，于是「想看 lora 的 rank 效应」
        #: 这件事会同时把 qlora 的 rank 也乘上去，规格数按轴数**相乘**。
        #: 默认叉积是 4 能力 × 10 单元 × 2 objective = 80 个 spec，
        #: 而这个项目的落地约束是单机 4060 —— 80 个跑完要 8 分钟量级，
        #: 而报告真正需要的对照（四策略各一 + 两个 objective + rank 与
        #: target_set 各一对）只要 48 个。所以默认配置给的是**策展过的网格**：
        #: 每个 cell 显式写出它要哪一种策略、哪一个 rank、哪一组挂载点。
        "cells": [],
    },
    #: 训练与评测的口径。**它们不是扇出轴** —— 所有 spec 共用同一套，
    #: 于是「策略之间」的差异不会混进「跑了多少步」的差异。
    "train": {
        "steps": 48,
        "batch": 8,
        "grad_accum": 1,
        "lr": 3e-2,
        "warmup_steps": 8,
        "seq_len": 16,
        "template": "plain",
        "n_train": 160,
        "n_eval": 24,
        "eval_every": 24,
        "checkpoint_every": 24,
        "optimizer": "adamw",
        "weight_decay": 0.0,
        "scheduler": "cosine",
    },
    "cost": {
        # 显存估算的字节口径。**每一项都要能被单独指认**，否则显存表没人信。
        "base_bytes": 4,          # 基座权重（fp32）
        "grad_bytes": 4,          # 梯度
        "optimizer_bytes": 4,     # 优化器状态**每一项**（AdamW 有 2 项）
        "adapter_bytes": 4,       # LoRA 的 A / B（它们**永远**是 fp32）
        "activation_bytes": 4,    # 激活
        "activation_factor": 8.0,  # 每层每 token 的激活 ≈ factor × d_model × 字节
        # 固定开销（CUDA context / 碎片）。**默认给 0，不是漏了。**
        # 在本项目的规模下基座权重的全部状态是 ~21 KB，一个 1 MB 的常数
        # 是它的 30 倍 —— 它会把四种策略之间**全部**被测出来的差都埋掉，
        # 而显存那一列看起来仍然很专业。想让它有「真实感」就把它调大，
        # 但要知道自己是在把被测的差异埋掉。
        "overhead_bytes": 0,
    },
    "acceptance": {
        "min_specs": 40,
        "require_ir_roundtrip": True,
        "require_single_call_path": True,
        "require_merge_is_exact": True,
        "require_cost_three_way_agree": True,
        "merge_tolerance": 1e-5,
        # LoRA 的 gain 达到 full 的多少比例才算「够用」。**改它会改结论**，
        # 所以它是配置项而不是 compare.py 里的一个默认参数。
        "lora_gain_ratio": 0.9,
        "quant_error_decreasing": True,
    },
    "expectations": {
        # 设计期望。**允许被实测推翻** —— 被推翻的那些是附录 B 里最值得看的部分。
        "lora_interferes_less_than_full": True,
        "lora_reaches_full_gain_on_target": True,
        "qlora_trades_gain_for_memory": True,
        "response_only_beats_all_tokens": True,
    },
    "report": {"headline_specs": 6, "max_matrix_rows": 64, "pareto_max_points": 12},
}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def resolve_path(root: Path | str, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else Path(root) / p


def load_config(path: Path | str | None = None, *, root: Path | str) -> dict[str, Any]:
    """读配置。附加 `_config_path` / `_root`，并在校验失败时**立刻报错**。"""
    root = Path(root)
    cfg = dict(DEFAULTS)
    if path is not None:
        p = Path(path)
        raw = json.loads(p.read_text(encoding="utf-8"))
        cfg = deep_merge(cfg, raw)
    else:
        default = root / "configs" / "default.json"
        if default.exists():
            cfg = deep_merge(cfg, json.loads(default.read_text(encoding="utf-8")))
        p = default
    cfg["_config_path"] = str(p)
    cfg["_root"] = str(root)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Mapping[str, Any]) -> None:
    """配置校验。**宁可在这里炸，也不要跑到一半用一个荒唐的口径出报告。**"""
    errs: list[str] = []
    sweep = cfg.get("sweep") or {}
    caps = sweep.get("capabilities")
    if not isinstance(caps, list) or not caps:
        errs.append("sweep.capabilities 必须是非空列表")
    kinds = sweep.get("kinds")
    if not isinstance(kinds, list) or not kinds:
        errs.append("sweep.kinds 必须是非空列表")
    else:
        from .ir import STRATEGY_KINDS

        unknown = [k for k in kinds if k not in STRATEGY_KINDS]
        if unknown:
            errs.append(f"sweep.kinds 里有未知策略 {unknown}（可选：{list(STRATEGY_KINDS)}）")
    ranks = sweep.get("ranks")
    if not isinstance(ranks, list) or any(not isinstance(r, int) or r < 1 for r in ranks or []):
        errs.append("sweep.ranks 必须是正整数列表")
    targets = sweep.get("target_sets")
    if not isinstance(targets, Mapping) or not targets:
        errs.append("sweep.target_sets 必须是非空映射")
    else:
        from .ir import LORA_ABLE_MODULES

        for name, mods in targets.items():
            if not isinstance(mods, list) or not mods:
                errs.append(f"sweep.target_sets[{name!r}] 必须是非空列表")
                continue
            bad = [m for m in mods if m not in LORA_ABLE_MODULES]
            if bad:
                errs.append(f"sweep.target_sets[{name!r}] 里有不存在的模块 {bad}")
    cost = cfg.get("cost") or {}
    for key in (
        "base_bytes",
        "grad_bytes",
        "optimizer_bytes",
        "adapter_bytes",
        "activation_bytes",
    ):
        v = cost.get(key)
        if not isinstance(v, int) or v <= 0:
            errs.append(f"cost.{key} 必须是正整数，得到 {v!r}")
    # `overhead_bytes` 允许为 0，其余不允许。理由是这个数**与模型规模无关**：
    # 在一个参数量几万的小模型上，1MB 的固定开销是参数状态的十几倍，
    # 于是四种策略的显存差异会被一个常数整块盖住，而表里每一行看起来都很正常。
    # 想让它有「真实感」就把它调大，但要知道自己是在把被测的差异埋掉。
    oh = cost.get("overhead_bytes")
    if not isinstance(oh, int) or oh < 0:
        errs.append(f"cost.overhead_bytes 必须是非负整数（可以为 0），得到 {oh!r}")
    factor = cost.get("activation_factor")
    if not isinstance(factor, (int, float)) or factor <= 0:
        errs.append(f"cost.activation_factor 必须是正数，得到 {factor!r}")
    tol = (cfg.get("acceptance") or {}).get("merge_tolerance")
    if not isinstance(tol, (int, float)) or not (0 < tol <= 1e-2):
        errs.append(f"acceptance.merge_tolerance 应在 (0, 1e-2]，得到 {tol!r}")
    # 「LoRA 达到 full 多少比例的 gain 才算达标」是**口径**，不是常量。
    # 写死在 compare.py 里的话，读者看到 0.9 无法分辨它是标定的还是拍的，
    # 也没法在不改代码的前提下做敏感性分析。上界 1.0：超过 1 意味着
    # 「LoRA 必须比 full 更好」，那不是这条期望在说的意思。
    ratio = (cfg.get("acceptance") or {}).get("lora_gain_ratio")
    if not isinstance(ratio, (int, float)) or not (0 < ratio <= 1.0):
        errs.append(f"acceptance.lora_gain_ratio 应在 (0, 1]，得到 {ratio!r}")
    errs.extend(_validate_train(cfg.get("train") or {}))
    try:
        sweep_cells(cfg)
    except ValueError as exc:
        errs.append(str(exc))
    if errs:
        raise ValueError("配置校验失败：\n  - " + "\n  - ".join(errs))


def _validate_train(tr: Mapping[str, Any]) -> list[str]:
    """训练口径的校验。**这些数一旦荒唐，报告里的每个数字都跟着荒唐**，
    所以在配置阶段就挡住，而不是跑到一半才发现 `seq_len` 比序列还短。"""
    errs: list[str] = []
    # **键名校验。** `TrainSettings.from_config` 只挑它认识的字段，其余的**静默丢掉** ——
    # 于是把 `lr` 拼成 `learning_rate` 会让训练安静地跑在默认的 3e-2 上，
    # 而报告里只会显示「用了配置里的学习率」。这是本项目里最容易被漏掉的一类错：
    # 代码全对、测试全绿、数字是另一个配置的。
    from .train import TrainSettings as _TS

    known = set(_TS.__dataclass_fields__) | {"seed", "_comment"}
    unknown = sorted(set(tr) - known)
    if unknown:
        errs.append(f"train 里有不认识的键 {unknown}（认识：{sorted(known)}）")
    seed = tr.get("seed")
    if not isinstance(seed, int) or not (0 <= seed < (1 << 31)):
        errs.append(f"train.seed 应是 [0, 2^31) 里的整数，得到 {seed!r}")
    ints = {
        "steps": (1, 1 << 20),
        "batch": (1, 1 << 12),
        "grad_accum": (1, 1 << 12),
        "warmup_steps": (0, 1 << 20),
        "seq_len": (8, 1 << 14),
        "n_train": (1, 1 << 20),
        "n_eval": (1, 1 << 20),
        "eval_every": (1, 1 << 20),
        "checkpoint_every": (1, 1 << 20),
    }
    for key, (lo, hi) in ints.items():
        v = tr.get(key)
        if not isinstance(v, int) or not (lo <= v <= hi):
            errs.append(f"train.{key} 应是 [{lo}, {hi}] 里的整数，得到 {v!r}")
    lr = tr.get("lr")
    if not isinstance(lr, (int, float)) or not (1e-8 <= float(lr) <= 1.0):
        errs.append(f"train.lr 应在 [1e-8, 1.0]，得到 {lr!r}")
    wd = tr.get("weight_decay")
    if not isinstance(wd, (int, float)) or not (0.0 <= float(wd) <= 1.0):
        errs.append(f"train.weight_decay 应在 [0, 1]，得到 {wd!r}")
    if isinstance(tr.get("warmup_steps"), int) and isinstance(tr.get("steps"), int):
        if tr["warmup_steps"] > tr["steps"]:
            errs.append(f"train.warmup_steps({tr['warmup_steps']}) 不能超过 steps({tr['steps']})")
    from .ir import OBJECTIVES as IR_OBJECTIVES
    from .ir import TEMPLATES

    if tr.get("template") not in TEMPLATES:
        errs.append(f"train.template 应是 {list(TEMPLATES)} 之一，得到 {tr.get('template')!r}")
    return errs


def sweep_cells(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """把 `sweep` 展开成一张**显式单元表**：每个格子是一份待跑的配置。

    两种来源，二选一：

    - `sweep.cells` 非空 → **直接用它**，不做任何叉积。每个 cell 显式写出
      `kind` 以及（仅对 peft）`rank` / `targets`，可选 `objective`；
      qlora 还可以给 `bits` / `group`（默认 4 / 32）。
    - 否则 → `kinds` × `ranks` × `target_sets` × `objectives` 的完整叉积。
      这是「我想把整个设计空间铺开」的形态，规格数按轴数**相乘**。

    为什么 `bits` 只在显式网格里有：叉积那一支里，给 qlora 加上 bits 轴会让
    「4-bit 与 8-bit 各跑一遍全部格子」，规格数直接翻倍 —— 而
    `acceptance.quant_error_decreasing` 只要求**存在**一对同构的 4/8 位对照。
    在显式网格里手写那两格，是用两格换一倍运行时间。

    为什么要留着叉积那一支：它是**完备的**，而显式网格是**策展的**。
    完备网格告诉你「没漏」，策展网格告诉你「这几个对照是我要看的」。
    两者都是对的，但一个 80 规格的完备网格在单机上要跑 8 分钟量级，
    而它多出来的 32 个格子对报告里的四条期望**一条都不贡献**。

    返回的每一项都带上了 `index`（展开顺序）与 `label`（报告里的短名）。
    **顺序即「谁先被跑到」**，也是报告里矩阵的行序。
    """
    from .ir import OBJECTIVES as IR_OBJECTIVES
    from .ir import STRATEGY_KINDS

    sweep = cfg.get("sweep") or {}
    caps = list(sweep.get("capabilities") or [])
    if not caps:
        raise ValueError("sweep.capabilities 不能为空")
    dupes = sorted({c for c in caps if caps.count(c) > 1})
    if dupes:
        raise ValueError(f"sweep.capabilities 有重复 {dupes}（重复会让同一份 Spec 出现两次）")

    target_sets: Mapping[str, Any] = sweep.get("target_sets") or {}
    ranks = list(sweep.get("ranks") or [])
    objectives = list(sweep.get("objectives") or [])
    cells_in = list(sweep.get("cells") or [])

    units: list[dict[str, Any]] = []
    if cells_in:
        for i, cell in enumerate(cells_in):
            if not isinstance(cell, Mapping):
                raise ValueError(f"sweep.cells[{i}] 必须是对象，得到 {type(cell).__name__}")
            kind = cell.get("kind")
            if kind not in STRATEGY_KINDS:
                raise ValueError(f"sweep.cells[{i}].kind 应是 {list(STRATEGY_KINDS)} 之一，得到 {kind!r}")
            _allowed = {"kind", "rank", "targets", "objective", "bits", "group", "note"}
            _extra = sorted(set(cell) - _allowed)
            if _extra:
                raise ValueError(
                    f"sweep.cells[{i}] 有不认识的键 {_extra}（认识：{sorted(_allowed)}）。"
                    "一个被静默忽略的键会让这一格跑成**别的配置**，而标签上写的还是你以为的那个。"
                )
            rank = int(cell.get("rank") or 0)
            tname = cell.get("targets")
            bits = int(cell.get("bits") or 0)
            group = int(cell.get("group") or 0)
            objs = [cell["objective"]] if cell.get("objective") else objectives
            if kind in ("full", "freeze"):
                if rank or tname or bits or group:
                    raise ValueError(
                        f"sweep.cells[{i}]：kind={kind} 不接受 rank/targets/bits/group（它对这种策略没有意义）"
                    )
                mods: list[str] = []
                tname = ""
            else:
                if not ranks:
                    raise ValueError("sweep.ranks 不能为空：peft 策略需要一个 rank")
                rank = rank or int(ranks[0])
                if rank not in ranks:
                    raise ValueError(f"sweep.cells[{i}].rank={rank} 不在 sweep.ranks={ranks} 里")
                tname = tname or next(iter(target_sets), "")
                # 先查类型再查成员。`targets` 是一个**名字**（键），不是模块列表本身；
                # 写成 `["q_proj","v_proj"]` 是很自然的误解，而 `x in dict` 遇到 list
                # 会直接 `TypeError: unhashable type: 'list'` —— 报错在**成员测试**
                # 这一行炸掉，读者看不到下面那句「不在 sweep.target_sets 里」，
                # 于是拿到一条与真实原因无关的异常。先问「是不是个名字」。
                if not isinstance(tname, str):
                    raise ValueError(
                        f"sweep.cells[{i}].targets 应当是一个**名字**（{sorted(target_sets)} 之一），"
                        f"得到 {tname!r}。要按模块列表写，请先在 sweep.target_sets 里给它起个名字。"
                    )
                if tname not in target_sets:
                    raise ValueError(
                        f"sweep.cells[{i}].targets={tname!r} 不在 sweep.target_sets 里（可选：{sorted(target_sets)}）"
                    )
                mods = list(target_sets[tname])
                if kind == "lora" and (bits or group):
                    raise ValueError(
                        f"sweep.cells[{i}]：kind=lora 不接受 bits/group={bits}/{group}（那是 qlora 的字段）"
                    )
            if not objs:
                raise ValueError("sweep.objectives 不能为空")
            for o in objs:
                if o not in IR_OBJECTIVES:
                    raise ValueError(f"sweep.cells[{i}].objective={o!r} 不在 {list(IR_OBJECTIVES)} 里")
                units.append(
                    {
                        "kind": kind,
                        "rank": rank,
                        "targets": tname,
                        "target_modules": mods,
                        "objective": o,
                        "quant_bits": bits if kind == "qlora" else 0,
                        "quant_group": group if kind == "qlora" else 0,
                    }
                )
    else:
        for kind in sweep.get("kinds") or []:
            if kind not in STRATEGY_KINDS:
                raise ValueError(f"sweep.kinds 里有未知策略 {kind!r}")
            if kind in ("full", "freeze"):
                combos: list[tuple[int, str, list[str]]] = [(0, "", [])]
            else:
                combos = [
                    (int(r), tname, list(target_sets[tname]))
                    for r in ranks
                    for tname in target_sets
                ]
            for rank, tname, mods in combos:
                for o in objectives:
                    units.append(
                        {
                            "kind": kind,
                            "rank": rank,
                            "targets": tname,
                            "target_modules": mods,
                            "objective": o,
                            # 单元表**形状统一**：叉积那一支没有 bits 轴，但键要在。
                            # 少一个键的话，下游 `cell["quant_bits"]` 会在
                            # 「恰好用了叉积」的那次调用里 KeyError —— 而显式网格
                            # 那条路径一直是好的，于是问题只在另一种配置下出现。
                            "quant_bits": 4 if kind == "qlora" else 0,
                            "quant_group": 32 if kind == "qlora" else 0,
                        }
                    )
    if not units:
        raise ValueError("扇出网格展开后是空的：检查 sweep.cells / kinds / objectives")

    out: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for cap in caps:
        for unit in units:
            cell = dict(unit)
            cell["capability"] = cap
            cell["index"] = len(out)
            cell["label"] = cell_label(cell)
            if cell["label"] in seen:
                raise ValueError(
                    f"扇出网格里有重复的格子 {cell['label']!r}"
                    f"（第 {seen[cell['label']]} 与第 {cell['index']} 项）—— "
                    "两份相同的配置会落在同一个 spec_id 上，报告里只能看到一份，"
                    "而另一份会静默消失。"
                )
            seen[cell["label"]] = cell["index"]
            out.append(cell)
    return out


def cell_label(cell: Mapping[str, Any]) -> str:
    """一个格子的短名，例如 ``upper/full/sft`` 或 ``lora-r8-qv``。

    **它是报告里的行名，所以必须在所有格子里唯一。** 不唯一的话，
    同一张表里两行会看起来是同一份配置，而它们其实是两份 —— 报告会静默地骗人。
    """
    parts = [str(cell["capability"]), str(cell["kind"])]
    if cell.get("rank"):
        parts.append(f"r{cell['rank']}")
    if cell.get("targets"):
        parts.append(str(cell["targets"]))
    # 位数进标签：4-bit 与 8-bit 的两格除了 bits 之外**逐字相同**，
    # 不带位数的话 48 个标签会有一对完全相同 —— 而唯一性正是这张表的用处。
    if cell.get("quant_bits"):
        parts.append(f"q{cell['quant_bits']}")
    parts.append(str(cell["objective"]))
    return "/".join(parts)


def protocol_digest(cfg: Mapping[str, Any]) -> str:
    """口径摘要：只吃影响**数字口径**的配置（显存估算与容差），不吃网格。"""
    keep = {
        "cost": cfg.get("cost"),
        "acceptance": {
            "merge_tolerance": (cfg.get("acceptance") or {}).get("merge_tolerance"),
        },
        "report": cfg.get("report"),
    }
    return sha256_text(canonical_json(keep))


def version_digest(cfg: Mapping[str, Any], inputs: Iterable[tuple[str, Any]]) -> str:
    """版本摘要：吃**输入对象**的摘要（扇出网格 + 模型注册表 + 数据集索引）。

    网格进摘要的是 :func:`sweep_cells` **展开之后**的结果，不是 `sweep` 那个原始
    dict。这个区别是实质性的：cell 上可以带 `note`（写给读配置的人看的），
    它不影响任何一个 Spec，也就绝不能影响版本摘要 —— 否则改一句注释会让
    整份报告看起来「输入变了」，而它一个数字都没变。
    """
    parts = [sha256_text(canonical_json(sweep_cells(cfg)))]
    for name, payload in inputs:
        parts.append(sha256_text(f"{name}:{canonical_json(payload)}"))
    return join_digest(parts)


def objectives_of(cfg: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """四个优化目标（名字 + 方向）。

    **这个项目有目标**，与 L2 那几个测量型 / 比较型项目不同 —— 它的产物里有一条
    Pareto 前沿。但目标只活在 :mod:`compare` 里：:mod:`train` 不认识它，
    于是「训练」这件事永远是「跑出数字」，不是「跑出更好的数字」。
    """
    return OBJECTIVES


def expectations_of(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("expectations") or {})
