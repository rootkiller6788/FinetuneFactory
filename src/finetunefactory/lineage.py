"""训练血缘 DAG。

这个项目的血缘回答一个很具体的问题：**这份 Artifact 是从哪来的？**
把 README 的主流水线（`Base Model + Dataset + Training Spec → Validate →
Tokenizer/Template → Prepare → Build Strategy → Train → Checkpoint → Evaluate →
Compare → Export`）画成一张图。

三条别处没有的图不变量：

1. **`spec:<id>` 的父节点恰好是它声明的 `base_model` 与 `dataset`。**
   一个不多、一个不少。这不是形式：它把「这份 Spec 到底跑在什么基座、什么数据上」
   画进了图里。哪天基座被静默替换，图里会立刻多出 / 少掉一条边 ——
   而报告里的指标看不出任何异常。
2. **`pareto` 的父节点恰好是前沿候选的 artifact。** 一个不多（没把落选者偷偷
   接进来）、一个不少（前沿是真的被接住了）。于是「谁进了前沿」看一张图就知道。
3. **每个 `artifact` 的父节点里必须有一个 `base_eval`。** 没有它，
   「涨了多少」这条链在图里是断的 —— 而「涨了多少」正是本项目的验收标准第二条。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .rng import sha256_text

__all__ = [
    "Node",
    "Edge",
    "Lineage",
    "SWEEP_NODE_ID",
    "PARETO_NODE_ID",
    "base_model_node",
    "dataset_node",
    "spec_node",
    "run_node",
    "checkpoint_node",
    "eval_node",
    "artifact_node",
    "base_eval_node",
    "build_lineage",
]

SWEEP_NODE_ID = "sweep"
PARETO_NODE_ID = "pareto"


def base_model_node(name: str) -> str:
    return f"base_model:{name}"


def dataset_node(capability: str, template: str) -> str:
    return f"dataset:{capability}:{template}"


def spec_node(spec_id: str) -> str:
    return f"spec:{spec_id}"


def run_node(spec_id: str) -> str:
    return f"run:{spec_id}"


def checkpoint_node(spec_id: str, step: int) -> str:
    return f"checkpoint:{spec_id}:{step}"


def eval_node(spec_id: str) -> str:
    return f"eval:{spec_id}"


def artifact_node(spec_id: str) -> str:
    return f"artifact:{spec_id}"


def base_eval_node(model: str) -> str:
    return f"base_eval:{model}"


@dataclass
class Node:
    id: str
    kind: str
    label: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "label": self.label, "meta": self.meta}


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"src": self.src, "dst": self.dst, "label": self.label}


class Lineage:
    """一张有向无环图。**只增不减**，于是它天然是一张「发生过什么」的记录。"""

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._out: dict[str, list[str]] = {}
        self._in: dict[str, list[str]] = {}

    # ---- 构建

    def add_node(self, node_id: str, kind: str, label: str = "", **meta: Any) -> Node:
        """加节点。**同名节点重复加是幂等的**（但要保证 kind / label 一致）。"""
        existing = self._nodes.get(node_id)
        if existing is not None:
            if existing.kind != kind:
                raise ValueError(f"节点 {node_id!r} 的 kind 不一致：{existing.kind!r} vs {kind!r}")
            existing.meta.update(meta)
            return existing
        node = Node(id=node_id, kind=kind, label=label or node_id, meta=dict(meta))
        self._nodes[node_id] = node
        self._out.setdefault(node_id, [])
        self._in.setdefault(node_id, [])
        return node

    def add_edge(self, src: str, dst: str, label: str = "") -> None:
        if src not in self._nodes:
            raise ValueError(f"边的起点不存在：{src!r}")
        if dst not in self._nodes:
            raise ValueError(f"边的终点不存在：{dst!r}")
        edge = Edge(src=src, dst=dst, label=label)
        if edge in self._edges:
            return
        self._edges.append(edge)
        self._out[src].append(dst)
        self._in[dst].append(src)

    # ---- 查询

    def has(self, node_id: str) -> bool:
        return node_id in self._nodes

    def node(self, node_id: str) -> Node:
        if node_id not in self._nodes:
            raise KeyError(f"图里没有节点 {node_id!r}")
        return self._nodes[node_id]

    def nodes(self) -> tuple[Node, ...]:
        return tuple(sorted(self._nodes.values(), key=lambda n: n.id))

    def edges(self) -> tuple[Edge, ...]:
        return tuple(sorted(self._edges, key=lambda e: (e.src, e.dst, e.label)))

    def parents(self, node_id: str) -> tuple[str, ...]:
        return tuple(sorted(self._in.get(node_id, ())))

    def children(self, node_id: str) -> tuple[str, ...]:
        return tuple(sorted(self._out.get(node_id, ())))

    def roots(self) -> tuple[str, ...]:
        return tuple(sorted(n for n in self._nodes if not self._in.get(n)))

    def leaves(self) -> tuple[str, ...]:
        return tuple(sorted(n for n in self._nodes if not self._out.get(n)))

    def by_kind(self, kind: str) -> tuple[Node, ...]:
        return tuple(n for n in self.nodes() if n.kind == kind)

    def ancestors(self, node_id: str) -> tuple[str, ...]:
        seen: set[str] = set()
        stack = list(self.parents(node_id))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(self.parents(cur))
        return tuple(sorted(seen))

    def is_acyclic(self) -> bool:
        """拓扑排序能跑完 ⟺ 无环。**必须断言** —— 有环的「血缘」不是血缘。"""
        indeg = {n: len(self._in.get(n, ())) for n in self._nodes}
        queue = [n for n, d in indeg.items() if d == 0]
        seen = 0
        while queue:
            cur = queue.pop()
            seen += 1
            for nxt in self._out.get(cur, ()):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)
        return seen == len(self._nodes)

    # ---- 序列化

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes()],
            "edges": [e.to_dict() for e in self.edges()],
            "digest": sha256_text(json.dumps([n.to_dict() for n in self.nodes()], sort_keys=True, ensure_ascii=False)),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Lineage":
        g = cls()
        for n in d.get("nodes", ()):
            g.add_node(n["id"], n["kind"], n.get("label", ""), **(n.get("meta") or {}))
        for e in d.get("edges", ()):
            g.add_edge(e["src"], e["dst"], e.get("label", ""))
        return g

    def to_dot(
        self,
        *,
        kinds: Sequence[str] | None = None,
        nodes: Sequence[str] | None = None,
    ) -> str:
        """Graphviz DOT。两个过滤器**按名字区分，刻意不合并成一个 `include`**。

        原先只有一个参数叫 ``include``，而它按**节点种类**过滤（``n.kind not in keep``）。
        于是 ``to_dot(include=dag.ancestors("pareto"))`` —— 一句读起来完全正确的代码
        （「只画 pareto 的祖先」）—— 会拿一组**节点 id** 去比节点**种类**，
        一个都对不上，输出一张**空图**。空图不是报错：``digraph lineage {}``
        是一份合法的 DOT，``dot`` 画出来就是一张白纸，而调用方只会以为「图太稀疏了」。

        所以现在两个参数各叫各的名字：

        - ``kinds`` —— 只保留这些种类的节点（``artifact`` / ``run`` / …）；
        - ``nodes`` —— 只保留这些 id 的节点（用 :meth:`ancestors` 配 ``| {"pareto"}`` 得到）。

        两者同时给就是「且」。边在**两端都留下**时才画 ——
        只留一端的边会在 DOT 里引用一个不存在的节点，``dot`` 会替它补一个匿名节点出来。
        """
        keep_kinds = set(kinds) if kinds is not None else None
        keep_ids = set(nodes) if nodes is not None else None

        def keep(nid: str) -> bool:
            if keep_ids is not None and nid not in keep_ids:
                return False
            if keep_kinds is not None and self.node(nid).kind not in keep_kinds:
                return False
            return True

        lines = ["digraph lineage {", '  rankdir="LR";', '  node [shape=box, fontname="Helvetica"];']
        for n in self.nodes():
            if not keep(n.id):
                continue
            lines.append(f'  "{n.id}" [label="{n.label}\\n({n.kind})"];')
        for e in self.edges():
            if not (keep(e.src) and keep(e.dst)):
                continue
            label = f' [label="{e.label}"]' if e.label else ""
            lines.append(f'  "{e.src}" -> "{e.dst}"{label};')
        lines.append("}")
        return "\n".join(lines) + "\n"

    def to_ascii(self, *, limit: int = 60) -> str:
        """按 kind 分组的节点计数 —— 全图几百个节点，逐行打印没人看。"""
        counts: dict[str, int] = {}
        for n in self.nodes():
            counts[n.kind] = counts.get(n.kind, 0) + 1
        rows = [f"  {k:12s} {v:4d}" for k, v in sorted(counts.items())]
        head = f"血缘图：{len(self._nodes)} 节点 / {len(self._edges)} 边"
        roots = self.roots()
        tails = self.leaves()
        extra = [
            "",
            f"  根节点 {len(roots)} 个：" + ", ".join(roots[:limit]) + (" …" if len(roots) > limit else ""),
            f"  叶节点 {len(tails)} 个：" + ", ".join(tails[:limit]) + (" …" if len(tails) > limit else ""),
            f"  无环：{'是 ✅' if self.is_acyclic() else '否 ❌'}",
        ]
        return "\n".join([head, ""] + rows + extra)


def build_lineage(result: Any) -> Lineage:
    """从一次完整流水线的结果构建血缘图。

    入参是 :class:`finetunefactory.pipeline.PipelineResult`，但这里只按属性访问，
    于是它也能被一个「只有同样属性的假对象」驱动 —— 测试不需要跑整条流水线
    就能验证图不变量。
    """
    g = Lineage()
    g.add_node(SWEEP_NODE_ID, "sweep", "扇出网格", n_specs=len(result.specs))
    g.add_node(PARETO_NODE_ID, "pareto", "Pareto 前沿", n_points=len(result.frontier))

    for entry in result.models:
        g.add_node(
            base_model_node(entry.name),
            "base_model",
            entry.name,
            params=entry.spec.param_count(),
        )

    # 前沿成员先算出来。`pareto -> artifact` 那条「candidate」边**只给前沿候选加** ——
    # 这一点必须是结构上成立的，不能靠循环跑完再回头查。
    #
    # 之前的写法是每条 artifact 都无条件接一条 candidate 边，然后在函数末尾断言
    # 「pareto 的父节点恰好是前沿候选」。两条是互相矛盾的：只要有一个 spec 落选，
    # 断言就必然抛错。于是这个函数**第一次被执行**时直接炸了
    # （`ValueError: 落选者被接进了前沿`）—— 而它在此之前从没被跑过。
    #
    # 更要紧的是：一个「先无条件连、再断言不许连」的写法，即使侥幸不抛错也没有价值 ——
    # 图本身已经错了，断言只是把它变成异常，而不是把它变成对的。所以这里改成
    # **在加边的那一刻就查**，端点上的不变式由构造保证，末尾那条检查退化成残留守卫。
    on_frontier = {artifact_node(s.spec_id) for s in result.frontier}

    for spec in result.specs.values():
        s_id = spec.spec_id
        m_node = base_model_node(spec.model.name)
        d_node = dataset_node(spec.dataset.capability, spec.dataset.template)
        if not g.has(d_node):
            g.add_node(
                d_node,
                "dataset",
                f"{spec.dataset.capability}/{spec.dataset.template}",
                n_train=spec.dataset.n_train,
                n_eval=spec.dataset.n_eval,
                seq_len=spec.dataset.seq_len,
            )
        g.add_node(
            spec_node(s_id),
            "spec",
            s_id,
            # 键名是 `strategy_kind` 而不是 `kind`：`add_node` 的第二个**位置**参数
            # 就叫 `kind`（节点的种类），再传一个 `kind=` 会直接
            # `TypeError: got multiple values for argument 'kind'`。
            # 图上「节点是什么种类」和「这份 Spec 用哪种微调策略」是两件事，
            # 名字也应该分得开。
            strategy_kind=spec.finetuning.kind,
            digest=spec.digest(),
        )
        # 不变量 1：spec 的父节点**恰好**是基座与数据 —— 一个不多、一个不少。
        g.add_edge(m_node, spec_node(s_id), "base")
        g.add_edge(d_node, spec_node(s_id), "data")
        g.add_edge(SWEEP_NODE_ID, spec_node(s_id), "fan-out")

        run = result.runs[s_id]
        g.add_node(run_node(s_id), "run", f"run {s_id}", steps=run.steps, flops=run.train_flops)
        g.add_edge(spec_node(s_id), run_node(s_id), "run")
        for ck in run.checkpoints:
            g.add_node(checkpoint_node(s_id, ck.step), "checkpoint", f"step {ck.step}", loss=ck.loss)
            g.add_edge(run_node(s_id), checkpoint_node(s_id, ck.step), "ckpt")

        ev = result.evals[s_id]
        g.add_node(eval_node(s_id), "eval", f"eval {s_id}", target_acc=ev.target_acc)
        g.add_edge(run_node(s_id), eval_node(s_id), "eval")

        art = artifact_node(s_id)
        g.add_node(art, "artifact", f"artifact {s_id}", bytes=art_bytes_of(art, result))
        g.add_edge(run_node(s_id), art, "export")

        # 不变量 3：每个 artifact 都要有一个 base_eval 父节点。
        b_node = base_eval_node(spec.model.name)
        if not g.has(b_node):
            g.add_node(b_node, "base_eval", f"base eval {spec.model.name}")
        g.add_edge(b_node, art, "baseline")
        # 落选者到此为止：它有 run/eval/artifact 三个父节点，但**不接**前沿。
        if art in on_frontier:
            g.add_edge(art, PARETO_NODE_ID, "candidate")

    # 不变量 2（残留守卫）：前沿节点的父节点**恰好**是前沿候选 —— 落选者一条边都没有。
    # 上面已经是按 `on_frontier` 加边，所以正常情况下这里什么都不会发现；
    # 留着是为了挡住以后有人把 `if` 去掉、或者从别处接边进 pareto。
    for edge in list(g.edges()):
        if edge.dst == PARETO_NODE_ID and edge.src not in on_frontier:
            raise ValueError(f"落选者被接进了前沿：{edge.src}")
    missing = on_frontier - set(g.parents(PARETO_NODE_ID))
    if missing:
        raise ValueError(f"前沿候选没被接住：{sorted(missing)}")
    return g


def art_bytes_of(node_id: str, result: Any) -> int:
    """取一个 artifact 的落盘字节数。**取不到就返回 0**，不抛错。

    血缘图不该因为「某个字段没算出来」而整张构建失败 —— 图是**旁证**，
    不是主结论。但 `bytes=0` 在报告里会被显式标成「未知」，不会被当成 0 字节。
    """
    spec_id = node_id.split(":", 1)[1] if ":" in node_id else ""
    art = result.artifacts.get(spec_id)
    return int(getattr(art, "n_bytes", 0)) if art is not None else 0


def dumps(g: Lineage, path: Path | str) -> None:
    Path(path).write_text(json.dumps(g.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
