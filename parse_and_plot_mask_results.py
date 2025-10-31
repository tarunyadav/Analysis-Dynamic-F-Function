#!/usr/bin/env python3
"""
parse_and_plot_mask_logs.py

Parses terminal output containing lines like:
Result: {'mask': 4097, 'mask_size': 2, 'trials': 16384, 'total data': 65536,
         'zero_sum_count_mask': 8, 'zero_sum_count_random': 1, ...}

and builds:
 - mask_results.csv (mask, trial, zero_sum_count_mask, zero_sum_count_random)
 - plots in ./plots_mask_results/ for masks 0x1001,0x0003,0x0007
 - summary_stats.csv with per-mask mean/std for masked/random zero counts

Run:
    python3 parse_and_plot_mask_logs.py

If your file is named differently, edit INPUT_LOG below.
"""
import re
import csv
from pathlib import Path
import ast
import pandas as pd
import matplotlib.pyplot as plt

# -------- User params --------
INPUT_LOG = "mask_executions_output.txt"   # change to your uploaded .txt filename
OUT_DIR = Path("plots_mask_results")
OUT_DIR.mkdir(exist_ok=True)
CSV_OUT = OUT_DIR / "mask_results.csv"
SUMMARY_OUT = OUT_DIR / "summary_stats.csv"

# Masks to extract (hex strings used in your commands)
MASK_HEX = ["1001", "0003", "0007"]
# For matching numeric masks in logs (the logs show integers like 4097, 3, 7)
MASK_INTS = [int(m, 16) for m in MASK_HEX]

# -------- Helpers --------
def parse_result_line(line):
    """
    Given a line that contains "Result: { ... }", extract the dict and return a Python dict.
    Returns None if not a result line.
    """
    m = re.search(r"Result:\s*(\{.*\})", line)
    if not m:
        return None
    try:
        # Use ast.literal_eval to safely parse the Python-dict-like string
        d = ast.literal_eval(m.group(1))
        return d
    except Exception:
        # fallback: try to clean trailing commas or non-ASCII quotes
        s = m.group(1)
        try:
            d = eval(s, {"__builtins__": None}, {})
            return d
        except Exception:
            return None

# -------- Parse the log file --------
rows = []
if not Path(INPUT_LOG).exists():
    print(f"Input log {INPUT_LOG} not found. Please place your raw .txt in the same folder or edit INPUT_LOG.")
    raise SystemExit(1)

with open(INPUT_LOG, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

# We'll count trials per mask as they appear (sequence order) — there is no explicit trial number in the result lines.
counters = {m: 0 for m in MASK_INTS}

for ln in lines:
    ln = ln.strip()
    if not ln:
        continue
    d = parse_result_line(ln)
    if not d:
        continue
    # Ensure fields exist
    if 'mask' not in d:
        continue
    mask_val = int(d['mask'])
    if mask_val not in counters:
        # ignore other masks unless you want to capture them
        continue
    counters[mask_val] += 1
    trial_idx = counters[mask_val]
    row = {
        'mask': hex(mask_val)[2:].zfill(4),
        'mask_int': mask_val,
        'trial': trial_idx,
        'mask_size': d.get('mask_size'),
        'trials_param': d.get('trials'),
        'total_data': d.get('total data') or d.get('total_data') or d.get('total data'),
        'zero_sum_count_mask': d.get('zero_sum_count_mask'),
        'zero_sum_count_random': d.get('zero_sum_count_random'),
        'prob_zero_sum_mask': d.get('prob_zero_sum_mask'),
        'prob_zero_sum_random': d.get('prob_zero_sum_random'),
    }
    rows.append(row)

# Save parsed CSV
if not rows:
    print("No parsable Result: {...} lines found for the chosen masks.")
    raise SystemExit(1)

df = pd.DataFrame(rows)
df = df.sort_values(['mask','trial']).reset_index(drop=True)
df.to_csv(CSV_OUT, index=False)
print(f"Saved parsed results to: {CSV_OUT}")
print(df.groupby('mask').size())

# -------- Plotting --------
def plot_mask(df_mask, mask_hex):
    df_mask = df_mask.sort_values('trial')
    trials = df_mask['trial'].to_numpy()
    z_mask = df_mask['zero_sum_count_mask'].to_numpy()
    z_rand = df_mask['zero_sum_count_random'].to_numpy()
    plt.figure(figsize=(10,4))
    plt.plot(trials, z_mask, marker='o', linestyle='-', label=f"masked data ({mask_hex})")
    plt.plot(trials, z_rand, marker='x', linestyle='--', label="random data (baseline)")
    plt.xlabel("Trial")
    plt.ylabel("Zero-sum count")
    plt.title(f"Zero-sum counts across trials — mask {mask_hex}")
    plt.grid(alpha=0.3)
    plt.legend()
    outpng = OUT_DIR / f"zero_sum_per_trial_{mask_hex.replace('0x','')}.png"
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()
    print(f"Saved plot: {outpng}")

for m in MASK_HEX:
    df_m = df[df['mask'] == m]
    if df_m.empty:
        print(f"No data for mask {m}; skipping plot.")
        continue
    plot_mask(df_m, m)

# -------- Summary stats --------
summary = df.groupby('mask').agg(
    trials_count = ('trial','count'),
    mean_zero_mask = ('zero_sum_count_mask','mean'),
    std_zero_mask = ('zero_sum_count_mask','std'),
    mean_zero_random = ('zero_sum_count_random','mean'),
    std_zero_random = ('zero_sum_count_random','std'),
).reset_index()
summary.to_csv(SUMMARY_OUT, index=False)
print(f"Saved summary to: {SUMMARY_OUT}")
print(summary)