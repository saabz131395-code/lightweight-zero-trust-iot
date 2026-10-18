"""Figure: testbed vs CICIoMT2024 real-device traffic (window statistics)."""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
store = pd.read_pickle("results/ciciomt_windows.pkl")
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.2))
for j, (tb, (c, t, real)) in enumerate(store.items()):
    for i, (col, lab, log) in enumerate((("rate", "Packets per second", True), ("size", "Mean frame size (bytes)", False))):
        ax = axs[i, j]
        bp = ax.boxplot([c[col].dropna(), t[col].dropna()], widths=0.55, showfliers=False, patch_artist=True)
        for patch, colr in zip(bp["boxes"], ["#B8C2D1", "#0E9F8E"]):
            patch.set_facecolor(colr); patch.set_alpha(0.8)
        for m in bp["medians"]: m.set_color("black")
        ax.set_xticks([1, 2]); ax.set_xticklabels(["CICIoMT2024\n(real)", "Testbed"])
        if log: ax.set_yscale("log")
        ax.set_ylabel(lab)
        if i == 0: ax.set_title(f"{tb}\nvs. {real}", fontsize=8.5)
fig.tight_layout(); fig.savefig("figures/fig_ciciomt.pdf"); fig.savefig("figures/fig_ciciomt.png", dpi=200)
print("ok")
