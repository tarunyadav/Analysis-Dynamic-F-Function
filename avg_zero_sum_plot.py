#!/usr/bin/env python3
"""
avg_zero_sum_plot.py
--------------------
Parse SLIM-DDL 32-round integral distinguisher output (Result: {...} lines)
and plot average zero-sum counts per mask for masked vs random data.
"""

import re, ast
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ======== CONFIG ========
INPUT_LOG = "mask_executions_output.txt"     # <-- change to your .txt file name
OUT_PNG = "plots_mask_results/avg_zero_sum_barplot.png"
# =========================

def parse_result_line(line: str):
    """Extract dict from a line beginning with 'Result:'"""
    m = re.search(r"Result:\s*(\{.*\})", line)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(1))
    except Exception:
        return None

# ---- Parse all Result entries ----
rows = []
with open(INPUT_LOG, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        d = parse_result_line(line)
        if not d:
            continue
        if "mask" not in d or "zero_sum_count_mask" not in d:
            continue
        rows.append({
            "mask": hex(int(d["mask"]))[2:].zfill(4),
            "zero_sum_count_mask": d["zero_sum_count_mask"],
            "zero_sum_count_random": d["zero_sum_count_random"]
        })

if not rows:
    print("❌ No valid 'Result:' entries found in file.")
    exit(1)

df = pd.DataFrame(rows)

# ---- Compute average per mask ----
summary = (
    df.groupby("mask",sort=False)
    .agg(
        avg_zero_sum_mask=("zero_sum_count_mask", "mean"),
        avg_zero_sum_random=("zero_sum_count_random", "mean"),
        trials=("mask", "count")
    )
    .reset_index()
)

print("\n=== Average Zero-Sum Counts per Mask ===")

print(summary)

# ---- Plot grouped bar chart ----
plt.figure(figsize=(9, 5))
bar_width = 0.35
x = range(len(summary))

plt.bar(
    [i - bar_width / 2 for i in x],
    summary["avg_zero_sum_mask"],
    width=bar_width,
    label="Masked Data"
)
plt.bar(
    [i + bar_width / 2 for i in x],
    summary["avg_zero_sum_random"],
    width=bar_width,
    label="Random Data"
)

plt.xticks(x, summary["mask"], rotation=45)
plt.ylabel("Average Zero-Sum Count")
plt.xlabel("Mask (hex)")
plt.title("32-Round SLIM-DDL: Average Zero-Sum Count (Masked vs Random Data)")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig(OUT_PNG, dpi=150)
plt.show()

print(f"\n✅ Plot saved as {OUT_PNG}")