"""Regenerates Figures 6.3 and 6.4 from results/sensitivity_runs.csv
(seeds 1-40 for each alpha in 0.5..0.9, beta = 1 - alpha)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11})
BLUE, ORANGE = "#1f77b4", "#e8710a"
df = pd.read_csv("results/sensitivity_runs.csv")
g = df.groupby("alpha"); a = np.array(sorted(df.alpha.unique()))
m, s = g.mean(numeric_only=True), g.std(numeric_only=True)
ek = dict(capsize=4, elinewidth=1.2, capthick=1.2)

# Figure 6.3: detection and FPR vs alpha
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.errorbar(a, m.det*100, yerr=s.det*100, color=BLUE, marker="o", ms=7, lw=2.2, label="Detection rate", **ek)
ax.errorbar(a, m.fpr*100, yerr=s.fpr*100, color=ORANGE, marker="s", ms=7, lw=2.2, label="False-positive rate", **ek)
BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9)
for x, d, f in zip(a, m.det*100, m.fpr*100):
    # labels sit beside the marker, never on the vertical error bar
    ax.annotate(f"{d:.1f}%", (x, d), xytext=(8, 7), textcoords="offset points", ha="left", va="bottom",
                fontsize=10.5, color=BLUE, fontweight="bold", bbox=BOX)
    if round(x, 1) == 0.5:
        ax.annotate(f"{f:.1f}%", (x, f), xytext=(-9, -4), textcoords="offset points", ha="right", va="top",
                    fontsize=10.5, color=ORANGE, fontweight="bold", bbox=BOX)
    else:
        ax.annotate(f"{f:.1f}%", (x, f), xytext=(8, 6), textcoords="offset points", ha="left", va="bottom",
                    fontsize=10.5, color=ORANGE, fontweight="bold", bbox=BOX)
ax.set_xticks(a); ax.set_xlim(0.40, 0.97); ax.set_ylim(-5, 108)
ax.set_xlabel(r"Historical-trust weight $\alpha$   ($\beta = 1-\alpha$)"); ax.set_ylabel("Rate (%)")
ax.yaxis.grid(True, color="#e3e3e3"); ax.set_axisbelow(True)
ax.legend(loc="lower left", frameon=False, fontsize=11)
fig.tight_layout(); fig.savefig("figures/fig6_3_sensitivity.png", dpi=300)

# Figure 6.4: F1 and latency vs alpha (two panels, one axis each)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.9))
a1.errorbar(a, m.f1, yerr=s.f1, color=BLUE, marker="o", ms=7, lw=2.2, **ek)
BOX = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9)
for x, v in zip(a, m.f1): a1.annotate(f"{v:.3f}", (x, v), xytext=(8, -5), textcoords="offset points", ha="left", va="top", fontsize=10, fontweight="bold", color="#1b3d5c", bbox=BOX)
a1.set_ylim(0, 1.0); a1.set_ylabel("F1-score"); a1.set_title("(a) F1-score", fontsize=12)
a2.errorbar(a, m.lat, yerr=s.lat, color=ORANGE, marker="s", ms=7, lw=2.2, **ek)
for x, v in zip(a, m.lat): a2.annotate(f"{v:.2f}", (x, v), xytext=(8, -5), textcoords="offset points", ha="left", va="top", fontsize=10, fontweight="bold", color="#7a3a05", bbox=BOX)
a2.set_ylim(0, 18); a2.set_ylabel("Detection latency (cycles)"); a2.set_title("(b) Mean detection latency", fontsize=12)
for ax in (a1, a2):
    ax.set_xticks(a); ax.set_xlim(0.45, 0.99); ax.set_xlabel(r"Historical-trust weight $\alpha$")
    ax.yaxis.grid(True, color="#e3e3e3"); ax.set_axisbelow(True)
fig.tight_layout(w_pad=2.5); fig.savefig("figures/fig6_4_f1latency.png", dpi=300)
print("ok")
