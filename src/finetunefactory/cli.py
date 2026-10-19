"""命令行入口。

设计约定（与同工作区其它项目共用一套，见 python-pipeline-scaffold skill）：

- **``--out`` 只属于 ``run``**。其余子命令是**只读观察者**，只读 ``out/`` 里的产物，
  不写任何东西 —— 观察者会改状态是最容易把人绕晕的一类设计。
- **只读子命令不做训练**，也**不重算任何东西**。它们从
  ``result.json`` + ``runs.jsonl`` + ``evals.jsonl`` + ``artifacts/*.json`` 拼出对象；
  两个例外都是**载入**而不是重算：``artifact`` 读 ``artifacts/<spec>.json``
  （那正是「导出的产物真的落盘了」这条承诺的可执行检查），``dag`` 从
  ``lineage.json`` 重建血缘图（否则 ``--dot`` 只能打印一句提示）。
- **未知 ``spec_id`` 返回 1 并给相近建议**，不抛栈。
- 刻意**没有** ``train`` 子命令：本项目没有「训练一个 checkpoint」这种单点操作，
  所有的训练都发生在一次扇出里，单独暴露它会让「训练」看起来是可以在网格之外
  发生的事情 —— 而报告里的每一个数字都绑着它所在的那一格配置。

### 「未知」与「0」在 CLI 里也必须分得开

``explain`` / ``compare`` 从磁盘载入时，如果 ``evals.jsonl`` 缺失，
``gain`` 一列会全是 ``未知``。这与「测出来是 0」是两件事，
而 CLI 是这两者最容易被压成同一个字符串的地方（两者都可能被渲染成 ``0``）。
所以打印一律过 :func:`report.fmt`，它把 ``None`` 和 ``NaN`` 分开写。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import report
from .config import load_config
from .pipeline import load_artifact, load_full, run_pipeline, write_outputs

__all__ = ["build_parser", "main"]


# ----------------------------------------------------------------------
# 读取产物
# ----------------------------------------------------------------------
def _load_result(out_dir: Path) -> Any:
    """拼一个**完整**的结果对象。

    用的是 :func:`pipeline.load_full` 而不是 ``load_result`` —— 两者的差别是
    **缺失的语义**：``result.json`` 不在就是「这不是一次运行的结果」（报错），
    而 ``runs.jsonl`` / ``evals.jsonl`` 不在只是「少了两列数字」（降级）。

    只读子命令**必须**看得见评测结果：``compare`` 里 gain 与 interference 两列
    的全部内容都来自 ``evals.jsonl``，而它缺失时的样子是一列 ``未知`` ——
    一列 ``未知`` 与一列 ``0`` 在终端里差别只有几个字，但它们的意思相反。
    """
    return load_full(out_dir)


def _print_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    print(report.markdown_table(header, rows))


def _suggest(result: Any, spec_id: str) -> str:
    known = sorted(getattr(result, "specs", None) or {})
    if not known:
        return "（结果里一个 spec 都没有）"
    # 先按「同一种策略 + 同一个能力」找相近的，那是读者最可能打错的两种字符。
    pool = [k for k in known if k.split("-")[:1] == spec_id.split("-")[:1]] or known
    return ", ".join(pool[:8]) + ("…" if len(pool) > 8 else "")


def _resolve_label(result: Any, text: str) -> str | None:
    """把用户输入解析成 spec_id。**先当 id，再当 label。**

    两种写法都接受是因为它们在报告里是**同一行**的两种叫法：报告用 label 当行名
    （``shift/lora/r8/qv/sft``），而 ``result.json`` 里只有 spec_id。
    只认一种的话，读者从报告里复制回来的那一行会「查无此 spec」——
    而这个报错会让人以为报告和结果对不上。
    """
    specs = getattr(result, "specs", None) or {}
    if text in specs:
        return text
    for sid in sorted(specs):
        if report._label(result, sid) == text:
            return sid
    return None


# ----------------------------------------------------------------------
# 子命令
# ----------------------------------------------------------------------
def _cmd_run(args: argparse.Namespace) -> int:
    # `root` 是**数据文件**的解析基准，必须与 `_anchor_args` 锚的 `--config` 同源。
    # 原来写死 `"."`（= cwd）⇒ 从项目目录外调用时配置锚到了项目根、数据却按 cwd 找，
    # 两半各指一处。**同一个 CLI 里的相对路径只能有一个基准。**
    cfg = load_config(args.config, root=str(_PKG_ROOT))
    if args.quiet:
        progress = None
    else:
        def progress(spec_id: str, label: str, index: int, total: int) -> None:
            print(f"  [{index + 1:>3}/{total}] {label}  ({spec_id})", file=sys.stderr)

    result = run_pipeline(cfg, root=str(_PKG_ROOT), progress=progress)
    out = Path(args.out)
    written = write_outputs(result, out)
    report_path = out / "report.md"
    report_path.write_text(report.render_markdown(result), encoding="utf-8")
    written["report"] = report_path

    print()
    print(report.summary_line(result))
    print()
    print(f"产物目录：{out.resolve()}")
    for name in sorted(written):
        print(f"  {name:<16} {written[name].stat().st_size:>10,} B")
    print()
    print(f"下一步：python main.py compare --out {out}")
    print(f"        python main.py frontier --out {out}")
    print(f"        python main.py list --out {out}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    result = _load_result(Path(args.out))
    specs = getattr(result, "specs", None) or {}
    if not specs:
        print("结果里一个 spec 都没有。", file=sys.stderr)
        return 1
    print("## 基准点")
    print()
    _print_table(("能力", "chance（乱猜）", "基座（微调前实测）", "基座 − chance", "读作"),
                 report.baseline_rows(result))
    print()
    print("## 对比矩阵")
    print()
    _print_table(report.matrix_header(), report.matrix_rows(result))
    print()
    print("## 显存分解")
    print()
    _print_table(report.cost_header(), report.cost_rows(result))
    print()
    if not getattr(result, "evals", None):
        print("⚠️ 结果里没有评测记录（`evals.jsonl` 不在）—— "
              "gain / 干扰 / acc 三列因此全是「未知」。这是**数据没载入**，"
              "不是「训练没有效果」。", file=sys.stderr)
    print(report.summary_line(result))
    return 0


def _cmd_frontier(args: argparse.Namespace) -> int:
    result = _load_result(Path(args.out))
    specs = getattr(result, "specs", None) or {}
    if not specs:
        print("结果里一个 spec 都没有。", file=sys.stderr)
        return 1
    print(report.markdown_table(report.frontier_header(), report.frontier_rows(result)))
    print()
    stats = (getattr(result, "counters", None) or {}).get("pareto_stats") or {}
    if stats:
        print(f"可用点 {stats.get('n_usable', 0)} / {stats.get('n_points', 0)}，"
              f"前沿 {stats.get('n_frontier', 0)}，被支配 {stats.get('n_dominated', 0)}")
        if stats.get("skipped_nan"):
            print()
            print(f"⚠️ {len(stats['skipped_nan'])} 个点因为某个轴取不到值而退出了排序"
                  f"（它们不比任何点差，只是不可比）：")
            for sid in stats["skipped_nan"][:10]:
                print(f"    {sid}")
        print()
    const = report.constant_axes(result)
    if const:
        print("⚠️ 这些轴在本次网格里是**常数**（每个规格都取同一个值），"
              "它们对前沿没有贡献：")
        for n, v in const:
            print(f"    {report.AXIS_LABEL.get(n, n)} = {v}")
        print()
    print("各轴冠军：")
    print()
    _print_table(("目标轴", "spec", "取值"), report.champion_rows(result))
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    result = _load_result(Path(args.out))
    spec_id = _resolve_label(result, args.spec_id)
    if spec_id is None:
        print(f"未知 spec：{args.spec_id}", file=sys.stderr)
        print(f"相近的可选项：{_suggest(result, args.spec_id)}", file=sys.stderr)
        return 1
    print(report.explain(result, spec_id))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    result = _load_result(Path(args.out))
    specs = getattr(result, "specs", None) or {}
    frontier = set(s.spec_id for s in (getattr(result, "frontier", None) or []))
    print(f"共 {len(specs)} 个 spec（★ 在前沿上）")
    print()
    for spec_id, spec in report._ordered_specs(result):
        ft = spec.finetuning
        detail = f"r{ft.rank}" if ft.rank else ""
        if ft.target_modules:
            detail += f"{'/' if detail else ''}{'+'.join(ft.target_modules)}"
        if ft.quant_bits:
            detail += f"{'/' if detail else ''}q{ft.quant_bits}"
        mark = "★" if spec_id in frontier else "·"
        print(f"  {mark} {report._label(result, spec_id):<44} "
              f"{ft.kind:<7} {detail:<22} "
              f"gain={report.fmt_signed(result.gain_of(spec_id)):<9} "
              f"干扰={report.fmt_signed(result.interference_of(spec_id)):<9} "
              f"{spec_id}")
    return 0


def _cmd_dag(args: argparse.Namespace) -> int:
    result = _load_result(Path(args.out))
    dag = getattr(result, "dag", None)
    if dag is None:
        print(f"{Path(args.out) / 'lineage.json'} 不在或里面没有节点。", file=sys.stderr)
        return 1
    if args.dot:
        if not args.frontier_only:
            print(dag.to_dot())
            return 0
        # 只画收拢那一半：从 `pareto` 往回走。**从 pareto 出发**，
        # 而不是从每个前沿 spec 出发 —— 前者天然包含那个唯一的收拢节点，
        # 后者会把图里最该被看见的那条边（artifact → pareto）漏掉。
        if not dag.has("pareto"):
            print("这张图里没有 pareto 节点；用不带 --frontier-only 的方式画全图。", file=sys.stderr)
            print(dag.to_dot())
            return 0
        keep = set(dag.ancestors("pareto")) | {"pareto"}
        print(dag.to_dot(nodes=sorted(keep)))
        return 0
    print(dag.to_ascii(limit=args.limit))
    kinds: dict[str, int] = {}
    for node in dag.nodes():
        kinds[node.kind] = kinds.get(node.kind, 0) + 1
    print()
    print(f"节点 {len(dag.nodes())} 个，边 {len(dag.edges())} 条，"
          f"无环：{'是' if dag.is_acyclic() else '否'}")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    return 0


def _cmd_artifact(args: argparse.Namespace) -> int:
    """把一份产物的元数据摊开 —— 「导出的东西到底是什么」的可执行检查。

    ⚠️ **两个 ``meta`` 不是一回事，这里是本项目最容易读错的一层嵌套。**

    ``artifacts/<spec>.json`` 是 ``Artifact.to_dict()`` 加一个展示用的 ``file``。
    ``Artifact`` 自己有一个字段就叫 ``meta``（一张任意旁证的字典），
    而它里面**又**装着两个键也叫 ``meta`` 的东西的近亲：``payload``（张量表）
    与 ``merge``（合并核验结果）。于是顶层 ``d["meta"]["meta"]`` 才是
    ``Artifact.meta``，而 ``d["meta"]["payload"]`` 是 ``None``。

    读错这一层**不会报错**：``.get("payload") or {}`` 给一个空字典，
    张量表整段消失、「布局 / dtype / 浮点数」三行全变成「未知」。
    之前这里就正是这么写的，而它在 ``full`` 的产物上看起来**完全正常** ——
    因为 ``full`` 本来就没有适配器可合并。只有在一份 LoRA 产物上它才会说出
    「合并：不适用 —— 没有适配器」这种话，而那句话是**编的**：
    真实原因是那个字段没被找到。

    所以下面一律走 ``Artifact.from_dict`` 反序列化之后按**字段**取，
    不再手写嵌套路径。
    """
    from .schema import Artifact

    out = Path(args.out)
    spec_id = _resolve_label(_load_result(out), args.spec_id)
    if spec_id is None:
        adir = out / "artifacts"
        known = sorted(p.stem for p in adir.glob("*.json")) if adir.is_dir() else []
        print(f"未知 spec：{args.spec_id}", file=sys.stderr)
        if known:
            print(f"可用的 spec：{', '.join(known[:10])}", file=sys.stderr)
        return 1

    raw = load_artifact(out, spec_id)
    fname = str(raw.get("file") or f"{spec_id}.bin")
    raw.pop("file", None)
    art = Artifact.from_dict(raw)
    payload = dict(art.meta.get("payload") or {})
    merge = dict(art.meta.get("merge") or {})

    print(f"spec      : {spec_id}")
    print(f"策略      : {art.kind}  可训练参数 {report.fmt_int(art.n_trainable_params)}")
    print(f"文件      : artifacts/{fname}")
    print()
    print(f"声明字节  : {report.fmt_bytes(art.n_bytes)}")
    print(f"实测字节  : {report.fmt_bytes(art.payload_bytes)}")
    verdict = "一致" if art.n_bytes == art.payload_bytes else "**不一致**"
    print(f"            {verdict}（这条等式是本项目唯一一处「声明 vs 实测」的强核对）")
    print()
    size = (out / "artifacts" / fname).stat().st_size if (out / "artifacts" / fname).exists() else None
    if size is not None:
        print(f"磁盘文件  : {report.fmt_bytes(size)}"
              f"　{'与声明一致' if size == art.payload_bytes else '**与声明不一致**'}")
        print()
    print(f"布局      : {payload.get('layout') or report.UNKNOWN}")
    print(f"dtype     : {payload.get('dtype') or report.UNKNOWN}"
          f" / {payload.get('endian') or report.UNKNOWN}")
    print(f"浮点数    : {report.fmt_int(payload.get('n_floats'))}")
    print()
    print(f"内容摘要  : {art.digest or report.UNKNOWN}")
    print()

    names = payload.get("names") or []
    shapes = payload.get("shapes") or []
    offsets = payload.get("offsets") or []
    if names:
        print(f"张量表（{len(names)} 个，按落盘顺序）：")
        print()
        rows = [
            (n,
             "×".join(str(s) for s in shapes[i]) if i < len(shapes) else report.UNKNOWN,
             report.fmt_int(offsets[i]) if i < len(offsets) else report.UNKNOWN)
            for i, n in enumerate(names)
        ]
        _print_table(("张量", "形状", "起始字节"), rows)
        print()
    else:
        print("张量表：**不在**这份 JSON 里（`meta.payload` 缺失）——"
              "这是元数据不完整，不是「产物没有张量」。")

    if merge:
        if merge.get("applicable"):
            print(f"合并最大绝对差：{report.fmt(merge.get('max_abs_diff'))}"
                  f"（容差 {report.fmt(merge.get('tolerance'))}），探测 {report.fmt_int(merge.get('n_probes'))} 次")
            if merge.get("identical"):
                print("            ⚠️ 差值**恰好为 0**：两份独立实现逐位相同，"
                      "通常意味着其中一份被换成了另一份（oracle 复用），而不是完美。")
        else:
            print(f"合并：不适用 —— {merge.get('reason') or '（没有给出理由）'}")
    else:
        print("合并核验：**不在**这份 JSON 里（`meta.merge` 缺失）。")
    print(f"量化相对误差  ：{report.fmt(art.quant_error)}")
    return 0


# ----------------------------------------------------------------------
#: 项目根：`src/{pkg}/cli.py` → 上三级。**所有相对路径默认值都锚到这里。**
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent


def _anchor_args(args: argparse.Namespace) -> None:
    """把 `--config` / `--out` / `--root` 的**相对路径**锚定到项目根。

    契约是 `cd <项目目录> && python main.py ...`，从项目目录里跑时锚不锚一个样
    —— 缺陷因此**长期隐形**。一旦从别处调用（例如跨项目的批量核对
    `python L5-xxx/main.py check`），`"configs/default.json"` 就指向了 cwd 下一个
    不存在的位置：严格 `load_config` 直接崩，宽松的会**静默回退 DEFAULTS**
    （更糟 —— 不报错，只是悄悄换了一套口径）。

    **在 `main()` 里紧跟 `parse_args` 锚一次**，比在每个调用点各锚一次可靠：
    漏掉一个调用点，就是一条只在某种调用方式下才现形的路径。

    绝对路径原样使用。语义与 `pytest` 的 rootdir / `cargo` / `npm --prefix` 一致。
    """
    for name in ("config", "out", "root"):
        value = getattr(args, name, None)
        if isinstance(value, str) and value:
            p = Path(value)
            if not p.is_absolute():
                setattr(args, name, str(_PKG_ROOT / p))

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finetunefactory",
        description="在统一的 Training IR 下扇出 SFT / freeze / LoRA / QLoRA 四条策略，"
                    "产出对比矩阵、Pareto 前沿与可复现的微调产物",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="跑完整流水线并落盘产物")
    p_run.add_argument("--config", default="configs/default.json", help="配置文件路径；相对路径按项目根解析")
    p_run.add_argument("--out", default="out", help="产物目录；相对路径按项目根解析")
    p_run.add_argument("--quiet", action="store_true", help="不打印逐个规格的进度")
    p_run.set_defaults(func=_cmd_run)

    p_cmp = sub.add_parser("compare", help="打印基准点、对比矩阵与显存分解（只读）")
    p_cmp.add_argument("--out", default="out", help="产物目录；相对路径按项目根解析")
    p_cmp.set_defaults(func=_cmd_compare)

    p_front = sub.add_parser("frontier", help="打印 Pareto 前沿与各轴冠军（只读）")
    p_front.add_argument("--out", default="out", help="产物目录；相对路径按项目根解析")
    p_front.set_defaults(func=_cmd_frontier)

    p_exp = sub.add_parser("explain", help="解释某一份 Spec 的配置、结果与位置（只读）")
    p_exp.add_argument("spec_id", help="spec_id 或报告里的行名（例如 shift/lora/r8/qv/sft）")
    p_exp.add_argument("--out", default="out", help="产物目录；相对路径按项目根解析")
    p_exp.set_defaults(func=_cmd_explain)

    p_list = sub.add_parser("list", help="列出全部 spec（只读）")
    p_list.add_argument("--out", default="out", help="产物目录；相对路径按项目根解析")
    p_list.set_defaults(func=_cmd_list)

    p_dag = sub.add_parser("dag", help="打印血缘图（只读）")
    p_dag.add_argument("--out", default="out", help="产物目录；相对路径按项目根解析")
    p_dag.add_argument("--dot", action="store_true", help="输出 Graphviz DOT")
    p_dag.add_argument("--frontier-only", action="store_true",
                       help="只画 pareto 节点的祖先链（配合 --dot）")
    p_dag.add_argument("--limit", type=int, default=40, help="ASCII 输出最多显示多少节点")
    p_dag.set_defaults(func=_cmd_dag)

    p_art = sub.add_parser("artifact", help="摊开一份产物的元数据（只读）")
    p_art.add_argument("spec_id")
    p_art.add_argument("--out", default="out", help="产物目录；相对路径按项目根解析")
    p_art.set_defaults(func=_cmd_artifact)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _anchor_args(args)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyError as exc:
        print(f"未知标识：{exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"配置或参数错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
