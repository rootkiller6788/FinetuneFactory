"""确定性随机与稳定哈希。

**禁用内置 `hash()`**：`PYTHONHASHSEED` 会让它跨进程不稳定，直接破坏「同一份 Spec
跑两次得到逐位相同的 Artifact」这条验收标准。所有指纹一律走 `sha256`。

本模块是整个项目的**可复现性地基**：Spec 摘要、Artifact 摘要、数据划分种子、
初始化种子、batch 顺序全部从这里出去。四条纪律：

1. 任何要进指纹的数据，先过 :func:`canonical_json`（键排序 + 分隔符固定 + 非 ASCII
   原样），否则 dict 插入顺序会悄悄改变指纹。
2. :func:`derive_seed` 用 ``\\x1f`` 作分隔符（不可能出现在名字 / 路径里），避免
   ``("ab", "c")`` 与 ``("a", "bc")`` 撞到同一个种子。
3. :func:`short_digest` 只用于展示；**任何比较都必须用全长 digest**。
4. **浮点数不进指纹**。权重、损失、梯度都是浮点，而浮点在不同平台上不保证逐位一致。
   进指纹的一律是「结构 + 整数 + 字符串」（见 :func:`fingerprint_weights` 的反面用法：
   它把浮点先量化成整数再哈希，只在**同一台机器上**做等值比较时才用）。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

__all__ = [
    "canonical_json",
    "sha256_text",
    "sha256_bytes",
    "stable_hash64",
    "short_digest",
    "derive_seed",
    "digest_of",
    "join_digest",
    "quantize_for_hash",
]

_SEP = "\x1f"


def canonical_json(obj: Any) -> str:
    """把任意可序列化对象压成**唯一**字符串表示。

    排序键、固定分隔符、`ensure_ascii=False`、不允许 NaN。凡是进指纹的东西都必须
    先过这里 —— 这是「同一份 Spec 必然得到同一个摘要」的唯一保证。
    """
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        default=_fallback,
    )


def _fallback(obj: Any) -> Any:
    # tuple 会退化成 list，于是 (1,2) 与 [1,2] 指纹相同 —— 这正是我们想要的：
    # 指纹只关心「值」，不关心容器类型。
    if isinstance(obj, (tuple, set, frozenset)):
        return list(obj) if not isinstance(obj, set) else sorted(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"不可序列化，无法进指纹：{type(obj)!r}")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_of(obj: Any, *, length: int = 12) -> str:
    """对象的短指纹（展示用）。比较请用 `sha256_text(canonical_json(obj))` 全长。"""
    return sha256_text(canonical_json(obj))[:length]


def stable_hash64(*parts: Any) -> int:
    """把任意成分折成 64 位整数（供 `random.Random(seed)` 使用）。

    成分先过 `str()`：调用点很自然会传 `(spec_id, step)` 这种混合类型，
    而 `"\\x1f".join` 遇到 int 会抛 `TypeError` —— 报错的位置离真正的原因很远。
    """
    return int(sha256_text(_SEP.join(str(p) for p in parts))[:16], 16)


def short_digest(full: str, length: int = 12) -> str:
    return full[:length]


def derive_seed(*parts: Any) -> int:
    """从若干成分派生一个稳定种子。

    分隔符用 `\\x1f` 而不是 `":"` —— 名字里会有 `.` / `-` / `/`，一旦分隔符
    可能出现在成分里，`("a:b","c")` 与 `("a","b:c")` 就会撞种子。
    """
    return stable_hash64(*parts)


def join_digest(parts: Iterable[str]) -> str:
    """把若干 digest 合成一个（用于「版本摘要 = 全部输入摘要的函数」）。"""
    return sha256_text(_SEP.join(sorted(parts)))


def quantize_for_hash(value: float, *, digits: int = 6) -> int:
    """把浮点量化成整数，供**同机等值比较**使用。

    绝不把浮点直接塞进 :func:`canonical_json`：`0.1 + 0.2` 在不同平台上
    可能打印成 `0.30000000000000004` 或 `0.3`，而 `repr` 的稳定性是
    「实现细节」而不是语言保证。量化到固定小数位后，比较的是**同一个数字**
    在两次运行里是否一致 —— 这正是我们要问的问题。

    注意它**不是**「跨平台可复现」的许可证：量化之后仍然可能有 1 ulp 的差别
    落在边界上。所以本项目的产物里，浮点只出现在「实测值」字段，
    绝不出现在任何指纹的输入里。
    """
    scale = 10 ** int(digits)
    return int(round(float(value) * scale))
