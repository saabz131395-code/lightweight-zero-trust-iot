"""Generates Figures 6.5 and 6.6 (comparison with recent related work).
Published values: Zahoor et al. (2025, Sci. Rep.) Table 5; Ma et al. (2025, PLOS ONE).
Proposed values: results/baseline_runs.csv (alpha=0.7, seeds 41-80) and
results/sensitivity_runs.csv (alpha=0.8, seeds 1-40), produced by run_experiments.py."""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, LBLUE, GREY, LGREY = "#1f77b4", "#8fbbdf", "#6b6b6b", "#bdbdbd"
base = pd.read_csv("results/baseline_runs.csv")
sens = pd.read_csv("results/sensitivity_runs.csv")
df = pd.concat([base, sens[sens.alpha == 0.8]], ignore_index=True)
g = df.groupby("alpha")

# ---------------- Figure 6.5: detection comparison ----------------
f1 = lambda p, r: 2 * p * r / (p + r)
systems = [
    # published: F1 computed from the reported precision/recall
    ("Isolation Forest\n(Zahoor et al., 2025)", 0.37, 0.08, f1(0.37, 0.08), LGREY, "//", "{:.2f}"),
    ("CSAD\n(Zahoor et al., 2025)", 0.55, 0.12, f1(0.55, 0.12), GREY, "\\\\", "{:.2f}"),
    # proposed: seed-level means, identical to Tables 6.1 and 6.3
    ("Proposed\n(α = 0.7)", g.prec.mean()[0.7], g.rec.mean()[0.7], g.f1.mean()[0.7], LBLUE, "", "{:.3f}"),
    ("Proposed\n(α = 0.8)", g.prec.mean()[0.8], g.rec.mean()[0.8], g.f1.mean()[0.8], BLUE, "", "{:.3f}"),
]
metrics = ["Precision", "Recall", "F1-score"]
x = np.arange(len(metrics)); w = 0.2
fig, ax = plt.subplots(figsize=(7.2, 4.3))
for i, (name, p, r, f, c, h, fmt) in enumerate(systems):
    vals = [p, r, f]
    bars = ax.bar(x + (i - 1.5) * w, vals, w * 0.92, color=c, hatch=h,
                  edgecolor="white", linewidth=0.8, label=name)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, fmt.format(v),
                ha="center", va="bottom", fontsize=8.5, color="#222")
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylim(0, 1.0); ax.set_ylabel("Score")
ax.yaxis.grid(True, color="#e5e5e5"); ax.set_axisbelow(True)
ax.legend(ncol=2, fontsize=9, frameon=False, loc="upper center",
          bbox_to_anchor=(0.5, 1.22))
fig.tight_layout(); fig.savefig("figures/fig6_5_detection_comparison.png", dpi=300)

# ---------------- Figure 6.6: device-side energy ----------------
occ = df.groupby("alpha").reauth_cycles_normal.mean()
occ_sd = df.groupby("alpha").reauth_cycles_normal.std()
rows = [
    ("Per-cycle re-authentication\nRSA-2048 PKI", 180 * 11.7, 0, LGREY, "//"),
    ("Per-cycle re-authentication\nFAST-SM9 (Ma et al., 2025)", 180 * 4.2, 0, GREY, "\\\\"),
    ("Proposed (α = 0.7)\nFAST-SM9 + gateway monitoring", occ[0.7] * 4.2, occ_sd[0.7] * 4.2, LBLUE, ""),
    ("Proposed (α = 0.8)\nFAST-SM9 + gateway monitoring", occ[0.8] * 4.2, occ_sd[0.8] * 4.2, BLUE, ""),
]
fig, ax = plt.subplots(figsize=(7.4, 3.9))
y = np.arange(len(rows))[::-1]
for yi, (name, v, sd, c, h) in zip(y, rows):
    ax.barh(yi, v, 0.6, color=c, hatch=h, edgecolor="white", linewidth=0.8,
            xerr=sd if sd else None, error_kw=dict(ecolor="#333", capsize=4, lw=1))
    ax.text(v + (sd if sd else 0) + 25, yi, f"{v:.0f} mJ", va="center", fontsize=10)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=10)
ax.set_xlabel("Authentication energy per legitimate device, 180 cycles (mJ)")
ax.set_xlim(0, 2500)
ax.xaxis.grid(True, color="#e5e5e5"); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("figures/fig6_6_energy_comparison.png", dpi=300)
print("done", occ.round(2).to_dict())
