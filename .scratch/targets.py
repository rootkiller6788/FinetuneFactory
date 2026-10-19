"""LoRA 挂载点对结论的影响。**不是测试**。"""
import sys
sys.argv = ["x"]
exec(open(".scratch/calib.py", encoding="utf-8").read().split('if __name__ ==')[0])

ALL7 = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
TARGET_SETS.update({"all7": ALL7, "mlp": MLP})

for cap in ("upper", "reverse"):
    for rank in (8, 32):
        cell("lora", cap, steps=48, rank=rank, targets="all7")
    cell("lora", cap, steps=48, rank=8, targets="mlp")
    cell("lora", cap, steps=48, rank=32, targets="qv")
