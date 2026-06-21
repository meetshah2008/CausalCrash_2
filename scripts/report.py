import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================
GEMINI_DIR = Path(os.getenv("GEMINI_DIR", "./outputs/judge_results/gemini"))
GPT_DIR    = Path(os.getenv("GPT_DIR", "./outputs/judge_results/gpt"))

LEVELS = [
    "level_1_perception",
    "level_2_temporal",
    # "level_3_causal_chain",  # optional
    "level_4_preventive",
    "level_5_counterfactual"
]

WEIGHTS = {
    "level_1_perception": 0.25,
    "level_2_temporal": 0.25,
    "level_4_preventive": 0.25,
    "level_5_counterfactual": 0.25,
}

OUT_DIR = Path(os.getenv("REPORT_OUT_DIR", "./outputs/reports"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Normalize weights (important)
total_w = sum(WEIGHTS.values())
WEIGHTS = {k: v / total_w for k, v in WEIGHTS.items()}

# =========================
# LOAD RESULTS
# =========================
def load_results(folder):
    data = {}
    for f in folder.glob("*_eval.json"):
        try:
            with open(f, "r") as fp:
                j = json.load(fp)
                vid = j["video_id"]
                scores = j["scores"]
                data[vid] = scores
        except Exception as e:
            print(f"Skipping {f}: {e}")
    return data

gemini = load_results(GEMINI_DIR)
gpt    = load_results(GPT_DIR)

common_videos = sorted(set(gemini.keys()) & set(gpt.keys()))
print(f"Total matched videos: {len(common_videos)}")

# =========================
# BUILD DATAFRAME
# =========================
rows = []

for vid in common_videos:
    row = {"video_id": vid}

    for lvl in LEVELS:
        if lvl not in gemini[vid] or lvl not in gpt[vid]:
            continue

        g1 = float(gemini[vid][lvl])
        g2 = float(gpt[vid][lvl])

        row[f"{lvl}_gemini"] = g1
        row[f"{lvl}_gpt"]    = g2
        row[f"{lvl}_avg"]    = (g1 + g2) / 2
        row[f"{lvl}_gap"]    = abs(g1 - g2)
        row[f"{lvl}_bias"]   = g2 - g1  # GPT - Gemini

    rows.append(row)

df = pd.DataFrame(rows)
df = df.apply(pd.to_numeric, errors="ignore")

# =========================
# COMPUTE CCS
# =========================
def compute_ccs(row, suffix):
    return sum(row[f"{lvl}_{suffix}"] * WEIGHTS[lvl] for lvl in LEVELS)

df["CCS_gemini"] = df.apply(lambda r: compute_ccs(r, "gemini"), axis=1)
df["CCS_gpt"]    = df.apply(lambda r: compute_ccs(r, "gpt"), axis=1)
df["CCS_avg"]    = df.apply(lambda r: compute_ccs(r, "avg"), axis=1)

# =========================
# SUMMARY TABLE
# =========================
summary = []

for lvl in LEVELS:
    summary.append({
        "level": lvl,
        "gemini_mean": df[f"{lvl}_gemini"].mean(),
        "gpt_mean": df[f"{lvl}_gpt"].mean(),
        "avg_mean": df[f"{lvl}_avg"].mean(),
        "gap_mean": df[f"{lvl}_gap"].mean(),
        "bias_mean": df[f"{lvl}_bias"].mean(),
        "corr": df[f"{lvl}_gemini"].corr(df[f"{lvl}_gpt"]),
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUT_DIR / "summary_table.csv", index=False)

print("\nSummary Table:\n", summary_df)

# =========================
# GLOBAL METRICS
# =========================
print("\n==== GLOBAL METRICS ====")
print("Mean CCS Gemini :", df["CCS_gemini"].mean())
print("Mean CCS GPT    :", df["CCS_gpt"].mean())
print("Mean CCS Avg    :", df["CCS_avg"].mean())

# identify unstable level
unstable_level = max(LEVELS, key=lambda lvl: df[f"{lvl}_gap"].mean())
print("Most unstable level:", unstable_level)

# =========================
# 📊 BAR CHART (LEVEL COMPARISON)
# =========================
x = np.arange(len(LEVELS))
width = 0.3

plt.figure()
plt.bar(x - width/2, summary_df["gemini_mean"], width, label="Gemini")
plt.bar(x + width/2, summary_df["gpt_mean"], width, label="GPT-4.1-mini")

plt.xticks(x, LEVELS, rotation=30)
plt.ylabel("Score")
plt.title("Per-Level Score Comparison")
plt.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "level_comparison.png")
plt.close()

# =========================
# 📉 GAP CHART
# =========================
plt.figure()

gap_vals = [df[f"{lvl}_gap"].mean() for lvl in LEVELS]
plt.bar(LEVELS, gap_vals)

plt.xticks(rotation=30)
plt.ylabel("Mean Absolute Gap")
plt.title("Judge Disagreement per Level")

plt.tight_layout()
plt.savefig(OUT_DIR / "judge_gap.png")
plt.close()

# =========================
# 📦 BOX PLOT
# =========================
plt.figure()

data = [df[f"{lvl}_avg"] for lvl in LEVELS]
plt.boxplot(data)

plt.xticks(range(1, len(LEVELS)+1), LEVELS, rotation=30)
plt.ylabel("Score")
plt.title("Score Distribution Across Levels")

plt.tight_layout()
plt.savefig(OUT_DIR / "boxplot_levels.png")
plt.close()

# =========================
# 📊 CCS DISTRIBUTION
# =========================
plt.figure()

plt.hist(df["CCS_avg"], bins=20)
plt.xlabel("Adjusted CCS")
plt.ylabel("Frequency")
plt.title("Final Score Distribution")

plt.tight_layout()
plt.savefig(OUT_DIR / "ccs_distribution.png")
plt.close()

# =========================
# 📉 BIAS PLOT (IMPORTANT)
# =========================
plt.figure()

bias_vals = [df[f"{lvl}_bias"].mean() for lvl in LEVELS]
plt.bar(LEVELS, bias_vals)

plt.xticks(rotation=30)
plt.ylabel("GPT - Gemini")
plt.title("Judge Bias per Level")

plt.tight_layout()
plt.savefig(OUT_DIR / "judge_bias.png")
plt.close()

print(f"\nSaved all outputs to: {OUT_DIR}")