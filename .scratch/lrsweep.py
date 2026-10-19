"""lr 敏感性：三种策略在多大步长下才学得动。**不是测试**。"""
import sys
sys.argv = ["x"]
exec(open(".scratch/calib.py", encoding="utf-8").read().split('if __name__ ==')[0])

for lr in (0.03, 0.1, 0.3, 1.0):
    for kind in ("full", "lora", "qlora", "freeze"):
        cell(kind, "upper", steps=48, lr=lr, rank=8)
