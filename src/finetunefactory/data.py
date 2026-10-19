"""数据：能力、模板、分词、掩码、划分、基线。

这个模块只回答一个问题：**给定 `DatasetSpec`，训练集和评测集长什么样。**
它不认识模型，也不认识策略 —— 于是同一份 `(capability, template, seed)`
在不同 `kind` 下拿到的训练数据**逐位相同**。这不是巧合，是可比性的前提：
如果 LoRA 和 full 用了不同的数据，两者的 gain 之差就同时包含了「策略差异」
和「数据差异」，报告里那两个数字就不再是同一个实验的两个格子。

### 序列格式（定长，于是只有一个 causal mask）

`:``plain``:` ``[BOS] task ':' p0 p1 p2 p3 '=' r0 r1 r2 r3 [EOS]``      长度 13
`:``chat`` :` ``[BOS] <u> task ':' p0 p1 p2 p3 <a> r0 r1 r2 r3 [EOS]``  长度 14

两者都补 `PAD` 到 `seq_len`。**补在后面**（而不是前面）是刻意的：causal mask 下
真实 token 只看得到自己左边，后面的 PAD 永远不会被真实 token 注意到，
所以不需要第二个「padding mask」算子。PAD 位置的输出是垃圾，但 `targets` 在那里
是 `-1`，掩码是 0，既不进损失也不进评测。

### 「准确率」指哪个准确率

`EvalResult.per_capability` 存的是**回答位置上的逐 token 准确率**，
不是整条回答全对的比率。两个原因：

1. 逐 token 是**平滑**的。4 个 token 全对是 all-or-nothing，一个刚学出点东西的
   小模型会长期停在 0，gain 曲线看起来像「什么都没发生」；而 token 准确率
   从 1/9 往 0.5 爬的过程是看得见的。
2. 它是训练**真正在优化**的那个量（掩码交叉熵的逐位置平均），
   于是「损失降了但准确率没动」这种诊断才有可能被发现。

整条全对的比率仍然会算，放在 ``meta["exact_match"]`` 里 —— 它更接近人的直觉，
报告里两个都给。**在同一个表格里混用这两个数是最容易犯的错**，
所以它们各有各的字段名，并且报告的表头会显式写出是哪一个。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from .config import resolve_path
from .rng import canonical_json, derive_seed, sha256_text

__all__ = [
    "DEFAULT_SPECIALS",
    "TASK_FNS",
    "Vocab",
    "Example",
    "DataIndex",
    "load_index",
    "build_vocab",
    "build_examples",
    "build_eval_sets",
    "build_train_set",
    "batch_of",
    "chance_level",
    "baselines",
    "content_positions",
    "fit_majority",
    "response_positions",
    "describe",
    "iter_capabilities",
]

DEFAULT_SPECIALS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", ":", "=", "<u>", "<a>")

PAD_TOK = "<pad>"
BOS_TOK = "<bos>"
EOS_TOK = "<eos>"


def _upper(p: str) -> str:
    return p.upper()


def _shift(p: str) -> str:
    return "".join(chr(ord(c) + 1) for c in p)


def _reverse(p: str) -> str:
    return p[::-1]


def _sort(p: str) -> str:
    return "".join(sorted(p))


#: 内置能力。**名字 → 一条纯函数**，没有例外、没有参数。
#:
#: 四个能力是刻意选的两组：`upper` / `shift` 是**逐位置**映射（不需要跨位置搬运
#: 信息），`reverse` / `sort` 是**置换**（必须靠 attention 把别的位置的信息搬过来）。
#: 这个分组让「LoRA 的干扰比 full 小」这条期望有一个可解释的机制：
#: 逐位置能力共享一套「读出输入字符」的底层特征，于是互相干扰；
#: 置换能力不共享，于是互不干扰。**如果实测出来的图景不是这样，那才是有意思的结果。**
TASK_FNS: dict[str, Callable[[str], str]] = {
    "upper": _upper,
    "shift": _shift,
    "reverse": _reverse,
    "sort": _sort,
}


# ----------------------------------------------------------------------
# 词表
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Vocab:
    """定长词表。**顺序即 id**，所以它必须来自文件而不是代码里的字面量。"""

    specials: tuple[str, ...]
    chars: str

    @property
    def tokens(self) -> tuple[str, ...]:
        return self.specials + tuple(self.chars)

    @property
    def size(self) -> int:
        return len(self.tokens)

    def id(self, tok: str) -> int:
        try:
            return self._stoi[tok]
        except KeyError as exc:  # pragma: no cover - 构造时就该挡住
            raise KeyError(f"词表里没有 token {tok!r}") from exc

    def has(self, tok: str) -> bool:
        return tok in self._stoi

    def __post_init__(self) -> None:
        toks = self.tokens
        dupes = sorted({t for t in toks if toks.count(t) > 1})
        if dupes:
            raise ValueError(f"词表里有重复 token（会静默改变 id 分配）：{dupes}")
        object.__setattr__(self, "_stoi", {t: i for i, t in enumerate(toks)})

    # dataclass(frozen=True) 上挂缓存字段：用 object.__setattr__ 在 __post_init__ 里放。
    _stoi: Mapping[str, int] = field(default_factory=dict, repr=False, compare=False)

    @property
    def pad(self) -> int:
        return self.id(PAD_TOK)

    @property
    def bos(self) -> int:
        return self.id(BOS_TOK)

    @property
    def eos(self) -> int:
        return self.id(EOS_TOK)


@dataclass(frozen=True)
class DataIndex:
    """`data/datasets/index.json` 的内容。

    它声明「有哪些能力、任务 token 是哪个、字符表多大、payload 多长」。
    **能力名必须在 :data:`TASK_FNS` 里有实现** —— 索引文件只能选，不能发明。
    这条限制让「加一个能力」永远是一次代码改动 + 一次数据改动，
    而不是一次「写进 JSON 然后运行时 KeyError」。
    """

    specials: tuple[str, ...]
    chars: str
    payload_chars: str
    payload_len: int
    tasks: Mapping[str, str]  # task_id token -> capability name
    source: str = ""

    @property
    def capabilities(self) -> tuple[str, ...]:
        return tuple(self.tasks.values())

    def task_id_of(self, capability: str) -> str:
        for tok, name in self.tasks.items():
            if name == capability:
                return tok
        raise KeyError(f"数据索引里没有能力 {capability!r}（可选：{sorted(self.tasks.values())}）")

    def to_dict(self) -> dict[str, Any]:
        """**不含 `source`** —— 路径是「从哪读的」，不是「读到了什么」。
        让它进指纹会让同一份索引在两个不同的挂载点下算出两个版本摘要。"""
        return {
            "specials": list(self.specials),
            "chars": self.chars,
            "payload_chars": self.payload_chars,
            "payload_len": self.payload_len,
            "tasks": dict(sorted(self.tasks.items())),
        }


def load_index(path: str | Mapping[str, Any], *, root: str = ".", source: str = "") -> DataIndex:
    """读数据索引。路径或已解析的 dict 都接受（后者供测试直接喂结构）。"""
    if isinstance(path, Mapping):
        raw = dict(path)
        src = source
    else:
        p = resolve_path(root, path)
        import json

        raw = json.loads(p.read_text(encoding="utf-8"))
        src = str(p)
    specials = tuple(raw.get("specials") or DEFAULT_SPECIALS)
    chars = str(raw.get("chars") or "")
    payload_chars = str(raw.get("payload_chars") or "")
    payload_len = int(raw.get("payload_len") or 4)
    tasks_in = raw.get("tasks") or {}
    if not chars:
        raise ValueError(f"数据索引 {src} 缺 chars")
    if not payload_chars:
        raise ValueError(f"数据索引 {src} 缺 payload_chars")
    if payload_len < 1:
        raise ValueError(f"数据索引 {src} 的 payload_len 必须 >= 1，得到 {payload_len}")
    if not tasks_in:
        raise ValueError(f"数据索引 {src} 缺 tasks")
    # 特殊 token 各自独立成词，**不必**出现在 chars 里 —— 它们是「结构」而不是「内容」。
    tasks: dict[str, str] = {}
    for tok, name in tasks_in.items():
        if tok not in chars:
            raise ValueError(f"任务 token {tok!r} 不在 chars 里，它需要一个 id")
        if name not in TASK_FNS:
            raise ValueError(f"能力 {name!r} 没有实现（可选：{sorted(TASK_FNS)}）")
        if name in tasks.values():
            raise ValueError(f"能力 {name!r} 被声明了两次")
        tasks[str(tok)] = str(name)
    v = Vocab(specials=specials, chars=chars)
    for c in payload_chars:
        if not v.has(c):
            raise ValueError(f"payload 字符 {c!r} 不在词表里")
    return DataIndex(
        specials=specials,
        chars=chars,
        payload_chars=payload_chars,
        payload_len=payload_len,
        tasks=tasks,
        source=src,
    )


def build_vocab(idx: DataIndex) -> Vocab:
    return Vocab(specials=idx.specials, chars=idx.chars)


# ----------------------------------------------------------------------
# 样本
# ----------------------------------------------------------------------
@dataclass
class Example:
    """一条定长样本。

    `ids` 与 `targets` 长度都是 `seq_len`；`targets[t]` 是「在位置 t+1 要预测的
    token」，PAD 处为 -1。`mask_all` / `mask_response` 是**同一段序列的两个掩码口径**，
    对应 Ir 里 `objective` 的两个取值 —— 它们在 :func:`batch_of` 里被选一个用。

    ### `eos_t`：为什么它要单独拿出来

    `targets` 里有一个位置的目标**永远是 EOS**（回答结束符），于是它 100% 可学，
    而且一条样本白送一次命中。把它算进「准确率」的后果是：所有能力的准确率被
    一起抬高 1/5，量程被压扁，而「模型学会了回答内容」与「模型学会了数到第 4 个
    就停」这两件难度天差地别的事在指标上无法区分。

    所以：**损失里保留 EOS**（不会停的模型没用），**准确率里剔除 EOS**
    （见 :func:`content_positions`），而 EOS 的命中率单独汇报。
    """

    ids: list[int]
    targets: list[int]
    mask_all: list[float]
    mask_response: list[float]
    capability: str
    response_start: int
    response_len: int
    #: 目标为 EOS 的那个位置下标；理论上有且仅有一个，找不到时为 -1。
    eos_t: int = -1

    def __len__(self) -> int:
        return len(self.ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ids": list(self.ids),
            "targets": list(self.targets),
            "capability": self.capability,
            "response_start": self.response_start,
            "response_len": self.response_len,
            "eos_t": self.eos_t,
            "text": " ".join(str(i) for i in self.ids),
        }


def _encode_sequence(
    vocab: Vocab, idx: DataIndex, task_id: str, payload: str, response: str, template: str
) -> tuple[list[int], int]:
    """把一条 (任务, 输入, 回答) 编成 token 串，返回（ids, 回答起点）。

    **模板的唯一作用是往 prompt 里加 / 换 token。** 回答段本身两种模板完全一样，
    于是「模板差异」在报告里是一个干净的变量：回答一样、prompt 框架不同。
    """
    s = [vocab.bos]
    if template == "chat":
        s.append(vocab.id("<u>"))
    s.append(vocab.id(task_id))
    s.append(vocab.id(":"))
    s.extend(vocab.id(c) for c in payload)
    s.append(vocab.id("<a>" if template == "chat" else "="))
    start = len(s)
    s.extend(vocab.id(c) for c in response)
    s.append(vocab.eos)
    return s, start


def _make_example(
    vocab: Vocab, idx: DataIndex, capability: str, payload: str, template: str, seq_len: int
) -> Example:
    task_id = idx.task_id_of(capability)
    response = TASK_FNS[capability](payload)
    ids, start = _encode_sequence(vocab, idx, task_id, payload, response, template)
    if len(ids) > seq_len:
        raise ValueError(
            f"序列长度 {len(ids)} 超过 dataset.seq_len={seq_len}（能力 {capability}、模板 {template}）"
        )
    pad = vocab.pad
    n_real = len(ids)
    ids = ids + [pad] * (seq_len - len(ids))
    # targets[t] = 位置 t+1 的 token。最后一个真实位置是 EOS，它后面是 PAD，
    # 于是「预测 EOS」这条监督信号被保留，而「在 PAD 后面预测 PAD」被丢弃。
    targets = [ids[t + 1] if t + 1 < n_real else -1 for t in range(seq_len)]
    mask_all = [1.0 if t >= 0 else 0.0 for t in targets]
    # 回答段 = 回答 token 本身 + 结尾的 EOS。EOS 必须算进去：
    # 不含 EOS 的训练会让模型永远学不会「该停了」，而推理时的贪心解码
    # 正是靠它停下来。这是 response_only 与 all_tokens 之外最容易漏的一格。
    mask_response = [1.0 if (t + 1 >= start and t + 1 < n_real) else 0.0 for t in range(seq_len)]
    eos_t = n_real - 2  # 目标为「最后一个真实 token（EOS）」的那个位置
    if eos_t < 0 or targets[eos_t] != vocab.eos:
        raise ValueError("样本里找不到 EOS 的监督位置：编码逻辑与 seq_len 对不上了")
    return Example(
        ids=ids,
        targets=targets,
        mask_all=mask_all,
        mask_response=mask_response,
        capability=capability,
        response_start=start,
        response_len=len(response) + 1,
        eos_t=eos_t,
    )


def _payloads(n: int, *, seed: int, alphabet: str, length: int, exclude: frozenset[str] = frozenset()) -> list[str]:
    """确定性抽取 n 个 payload。**用拒绝采样保证与 `exclude` 不相交。**

    `exclude` 是评测集与训练集不重叠的实现方式。没有它，一条训练样本会同时出现在
    评测集里，于是「涨了多少」这件事被记忆污染 —— 而这个错误不会报任何错，
    只会让所有 kind 的 gain 一起偏高、把策略之间的差异压扁。
    """
    rng = random.Random(seed)
    out: list[str] = []
    seen: set[str] = set()
    space = len(alphabet) ** length
    if n + len(exclude) > space:
        raise ValueError(f"要求 {n} 条且排除 {len(exclude)} 条，但样本空间只有 {space}")
    guard = 0
    while len(out) < n:
        guard += 1
        if guard > 100 * (n + 1):
            raise ValueError("拒绝采样没有收敛：样本空间几乎被 exclude 占满")
        p = "".join(rng.choice(alphabet) for _ in range(length))
        if p in seen or p in exclude:
            continue
        seen.add(p)
        out.append(p)
    return out


def build_examples(
    vocab: Vocab,
    idx: DataIndex,
    capability: str,
    *,
    template: str,
    seq_len: int,
    n: int,
    seed: int,
    exclude: frozenset[str] = frozenset(),
) -> list[Example]:
    """按 `seed` 抽 payload 并编码。**同一个 seed 必然给出同一批样本，顺序也一样。**"""
    pays = _payloads(
        n,
        seed=seed,
        alphabet=idx.payload_chars,
        length=idx.payload_len,
        exclude=exclude,
    )
    return [_make_example(vocab, idx, capability, p, template, seq_len) for p in pays]


# ----------------------------------------------------------------------
# 数据集构建（按 capability / template 缓存）
# ----------------------------------------------------------------------
_CACHE: dict[tuple[Any, ...], list[Example]] = {}


def _cache_key(idx: DataIndex, capability: str, template: str, seq_len: int, n: int, seed: int) -> tuple[Any, ...]:
    """缓存键。**用索引内容的摘要，不用 `id(idx)`。**

    `id()` 会在对象被回收后复用，于是「换了索引但缓存里还留着上一个索引的数据」
    这种错误会在某个完全无关的时刻冒出来 —— 而它表现成「数据和索引对不上」，
    离真正的原因隔着十万八千里。内容摘要没有这个危险，代价是一次 sha256。
    """
    return (sha256_text(canonical_json(idx.to_dict())), capability, template, seq_len, n, seed)


def build_eval_sets(
    idx: DataIndex,
    vocab: Vocab,
    *,
    template: str,
    seq_len: int,
    n_eval: int,
    seed: int,
    capabilities: Sequence[str] | None = None,
) -> dict[str, list[Example]]:
    """**每一个**能力都要有评测集，因为「干扰」就定义在别的能力上。

    评测 payload 与训练 payload 用**不同的种子**并互相排除，于是它们是真正的不相交集。
    """
    caps = list(capabilities) if capabilities is not None else list(idx.capabilities)
    out: dict[str, list[Example]] = {}
    for cap in caps:
        key = _cache_key(idx, cap, template, seq_len, n_eval, derive_seed("eval", cap, seed))
        if key not in _CACHE:
            _CACHE[key] = build_examples(
                vocab,
                idx,
                cap,
                template=template,
                seq_len=seq_len,
                n=n_eval,
                seed=derive_seed("eval", cap, seed),
            )
        out[cap] = _CACHE[key]
    return out


def build_train_set(
    idx: DataIndex,
    vocab: Vocab,
    *,
    capability: str,
    template: str,
    seq_len: int,
    n_train: int,
    seed: int,
    n_eval: int = 0,
) -> list[Example]:
    """训练集。**只依赖 (capability, template, seq_len, n_train, seed)** ——
    `kind` 不在其中，这正是「同一份数据、四种策略」的形式化版本。"""
    ev_seed = derive_seed("eval", capability, seed)
    excluded = frozenset(
        _payloads(
            n_eval,
            seed=ev_seed,
            alphabet=idx.payload_chars,
            length=idx.payload_len,
        )
        if n_eval
        else ()
    )
    key = _cache_key(idx, capability, template, seq_len, n_train, derive_seed("train", seed, n_eval))
    if key not in _CACHE:
        _CACHE[key] = build_examples(
            vocab,
            idx,
            capability,
            template=template,
            seq_len=seq_len,
            n=n_train,
            seed=derive_seed("train", seed),
            exclude=excluded,
        )
    return _CACHE[key]


# ----------------------------------------------------------------------
# 批
# ----------------------------------------------------------------------
def batch_of(
    examples: Sequence[Example], indices: Sequence[int], objective: str
) -> tuple[list[list[int]], list[list[int]], list[list[float]]]:
    """切一个 batch，返回 `(ids, targets, masks)`。

    `objective` 只在这里被翻译成掩码 —— 它是「策略是数据」在数据侧的唯一落点，
    而它下去之后再没有第二个分支。
    """
    if objective not in ("sft", "sft_response_only"):
        raise ValueError(f"未知 objective {objective!r}")
    src = [examples[i] for i in indices]
    if objective == "sft":
        masks = [list(e.mask_all) for e in src]
    else:
        masks = [list(e.mask_response) for e in src]
    ids = [list(e.ids) for e in src]
    targets = [list(e.targets) for e in src]
    for m in masks:
        if sum(m) <= 0:
            raise ValueError("batch 里有一条样本的掩码全 0：它没有任何监督信号")
    return ids, targets, masks


def response_positions(ex: Example) -> list[int]:
    """这条样本里**被回答掩码监督**的那些位置（回答内容 + EOS）。用于损失口径的核对。"""
    return [t for t in range(len(ex.targets)) if ex.mask_response[t] > 0 and ex.targets[t] >= 0]


def content_positions(ex: Example) -> list[int]:
    """这条样本里**算准确率**的那些位置 —— 只算回答内容，**剔除 EOS**。

    理由见 :class:`Example` 的类文档：EOS 的位置必然命中，白送一份命中，
    把量程压扁 1/5。剔除之后，准确率才真的在问「回答的内容对不对」。
    EOS 的命中率没有被丢掉，它进了 :class:`~finetunefactory.schema.EvalResult`
    的 `meta["eos_acc"]`。
    """
    return [t for t in response_positions(ex) if t != ex.eos_t]


# ----------------------------------------------------------------------
# 基线（见 skill 坑 47：机会 + 启发式，两个都要给）
# ----------------------------------------------------------------------
def chance_level(idx: DataIndex, n_positions: int = 0) -> float:
    """机会水平：逐位置均匀猜。

    **不是 1/词表大小。** 回答位置上可能的 token 只有「payload 字母 + 它的变换」
    那一小撮，用整个词表算会把机会水平压得毫无意义地低。这里按
    ``1 / |payload_chars|`` 计（`shift` 会用到 payload 字母之外的那一个字符，
    但那一个位置的分母仍然是同一量级），于是它是个**保守偏高的**机会下界 ——
    一个偏高（更难被超越）的基线是安全的，偏低则会让「已经学会」看起来像「还没学会」。
    """
    return 1.0 / max(1, len(idx.payload_chars))


def fit_majority(
    idx: DataIndex, train_sets: Mapping[str, Sequence[Example]]
) -> dict[str, list[int]]:
    """在**训练集**上拟合「每个回答位置出现最多的那一个 token」。

    **必须在训练集上拟合。** 在评测集上取每位置的众数是一个泄漏的基线：
    24 条样本 8 个字时，众数平均命中约 0.23，远高于真实的盲猜水平，
    于是「启发式基线」看起来比它实际能做到的高得多，而 gain 被系统性地低估。
    基线是「一个不看输入的策略能拿多少分」的度量，它的参数只能来自训练集。
    """
    out: dict[str, list[int]] = {}
    for cap, examples in train_sets.items():
        examples = list(examples)
        if not examples:
            continue
        counts: list[dict[int, int]] = [{} for _ in range(len(examples[0].targets))]
        for ex in examples:
            for t in content_positions(ex):
                d = counts[t]
                d[ex.targets[t]] = d.get(ex.targets[t], 0) + 1
        out[cap] = [(_argmax(c) if c else -1) for c in counts]
    return out


def baselines(
    idx: DataIndex,
    train_sets: Mapping[str, Sequence[Example]],
    eval_sets: Mapping[str, Sequence[Example]],
) -> dict[str, dict[str, float]]:
    """每个能力的两个基线（见 skill 坑 47：机会 + 启发式，两个都要给）。

    - `chance`：逐位置均匀猜（解析值，见 :func:`chance_level`）。
    - `majority`：**最强的不看输入的常数回答** —— 在训练集上拟合每个回答位置的众数，
      在评测集上打分。它把「这个任务是不是靠背一个常数就能拿分」变成可比较的数字。
    - `copy`：**照抄输入的启发式** —— 回答第 i 个 token 直接用输入第 i 个 payload
      字符。在 `sort` 这类「排序后常常跟原串有些位置一样」的任务上它不是 0，
      而它到底是几分正是这一项要量出来的东西。
    - `heuristic`：`max(majority, copy)`。报告里说的「启发式基线」就是它。

    **三个都按逐 token 准确率算**，且都**剔除 EOS**，与
    :class:`~finetunefactory.schema.EvalResult` 的主指标同口径。
    口径不同的基线与实测值不是同一个量，放在一起比较毫无意义。

    一个只看常数回答就能拿到高分的任务，其 gain 是没有信息量的。所以报告在给 gain
    的同时必须给这两个数，读者才知道那个 gain 该不该被当回事。
    """
    fitted = fit_majority(idx, train_sets)
    out: dict[str, dict[str, float]] = {}
    for cap, examples in eval_sets.items():
        examples = list(examples)
        if not examples:
            continue
        pos_runs: list[tuple[Example, list[int]]] = []
        n_pos = 0
        for ex in examples:
            pos = content_positions(ex)
            if not pos:
                continue
            pos_runs.append((ex, pos))
            n_pos += len(pos)
        mode = fitted.get(cap) or []
        n_major = 0
        n_copy = 0
        for ex, pos in pos_runs:
            # '=' 或 '<a>' 的位置，payload 就紧在它左边。
            dec = ex.response_start - 1
            payload_ids = ex.ids[dec - idx.payload_len : dec]
            for t in pos:
                if t < len(mode) and mode[t] >= 0 and ex.targets[t] == mode[t]:
                    n_major += 1
                i = t + 1 - ex.response_start  # 这是回答段的第几个 token
                if 0 <= i < len(payload_ids) and ex.targets[t] == payload_ids[i]:
                    n_copy += 1
        denom = n_pos or 1
        out[cap] = {
            "chance": chance_level(idx),
            "majority": n_major / denom,
            "copy": n_copy / denom,
            "heuristic": max(n_major, n_copy) / denom,
            "n_positions": float(n_pos),
        }
    return out


def _argmax(counts: Mapping[int, int]) -> int:
    """众数。**平局时取小的 token id** —— 平局判定必须是确定的，否则
    「同一份数据两次运行得到同一个基线」这句话就不成立（而基线会进报告）。"""
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def describe(idx: DataIndex) -> str:
    caps = ", ".join(f"{tok}={name}" for tok, name in sorted(idx.tasks.items()))
    return (
        f"数据索引 {idx.source or '<内存>'}：词表 {len(idx.specials)} 特殊 + {len(idx.chars)} 字符，"
        f"payload {idx.payload_len} 位 / 字母表 {len(idx.payload_chars)}，能力 {caps}"
    )


def iter_capabilities(idx: DataIndex, wanted: Iterable[str]) -> list[str]:
    """把配置里的能力名对齐到索引，**顺序照配置给**（顺序即报告里的顺序）。"""
    caps = list(wanted)
    known = set(idx.capabilities)
    unknown = [c for c in caps if c not in known]
    if unknown:
        raise ValueError(f"sweep.capabilities 里有数据索引不认识的能力 {unknown}（索引：{sorted(known)}）")
    seen: set[str] = set()
    out: list[str] = []
    for c in caps:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out
