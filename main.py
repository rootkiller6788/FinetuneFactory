#!/usr/bin/env python
"""零安装入口：把 ``src/`` 塞进 ``sys.path`` 后转交 ``finetunefactory.cli:main``。

这样 ``python main.py run`` 不需要 ``pip install``，也不需要激活 venv ——
符合本工作区「核心链路零依赖」的约定。
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from finetunefactory.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
