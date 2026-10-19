"""CLI 契约：`--help`、每个子命令的 `--help`、以及 `check`（若有）都要 rc=0。

⚠️ 自包含：不依赖 conftest 夹具（有的项目自带别的 `conftest.py`）。
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "main.py"


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(MAIN), *args],
        cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )


def parse_subcommands(text: str) -> list[str]:
    """从 `--help` 输出里解析子命令名（两种写法都吃）。"""
    subs: set[str] = set()
    m = re.search(r"\{([a-z0-9,\-]+)\}", text)
    if m:
        subs |= set(m.group(1).split(","))
    for line in text.splitlines():
        m2 = re.match(r"^ {4}([a-z][a-z0-9\-]*)\s{2,}\S", line)
        if m2:
            subs.add(m2.group(1))
    return sorted(subs)


def test_top_level_help() -> None:
    r = run_cli("--help")
    assert r.returncode == 0, r.stderr


def test_every_subcommand_help() -> None:
    subs = parse_subcommands(run_cli("--help").stdout)
    assert subs, "没从 --help 里解析出任何子命令"
    bad = [f"{s}: rc={r.returncode}"
           for s in subs
           for r in [run_cli(s, "--help")]
           if r.returncode != 0]
    assert not bad, "有子命令的 --help 非零退出：\n" + "\n".join(bad)


def test_check_exit_zero() -> None:
    subs = parse_subcommands(run_cli("--help").stdout)
    if "check" not in subs:
        pytest.skip("本项目没有 check 子命令")
    r = run_cli("check")
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
