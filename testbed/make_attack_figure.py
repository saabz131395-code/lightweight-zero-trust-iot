"""One graph per attack type: traffic volume and trust score of a compromised device (one evaluation run)."""
import json, sqlite3, sys, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import analysis as A
import config as CFG

DB = sys.argv[1] if len(sys.argv) > 1 else "data/testbed.db"
RUN = int(sys.argv[2]) if len(sys.argv) > 2 else 11
best = json.load(open("results/stats.json"))["best_alpha"]
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8.5, "axes.spines.top": False})
A.BASE = CFG.BASELINE_CYCLES
con = sqlite3.connect(DB)
dev, runs, att = A.load_meta(con)
f = A.normalise(A.label(A.extract_features(con, RUN, dev, 240), att, RUN))
lg = A.run_engine(f, best, seed=RUN, calib="device")
COL = {"DDoS flooding": "#D64545", "Data exfiltration": "#E8952C", "Botnet C&C beaconing": "#4F6D99", "Port scanning": "#0E9F8E"}
fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True)
for ax, t in zip(axs.flat, COL):
    a = att[(att.run_id == RUN) & (att.attack_type == t)].sort_values("start_cycle").iloc[0]   # earliest window of this type
    g = lg[lg.device_id == a.device_id].sort_values("cycle")
    name = dev.set_index("device_id").name[a.device_id]
    ax.axvspan(a.start_cycle, a.end_cycle, color=COL[t], alpha=0.13, lw=0, label="Attack window")
    ax.axvspan(0, A.BASE, color="#000000", alpha=0.04, lw=0)
    ax.plot(g.cycle, (g.DV.rolling(5, min_periods=1).mean() / 1000).clip(lower=0.012), color="#8A94A6", lw=0.9, label="Data sent (kB per cycle, 5-cycle mean)")
    ax.set_yscale("log"); ax.set_ylim(1e-2, 2e3); ax.set_ylabel("Data sent (kB / cycle)")
    bx = ax.twinx()
    bx.plot(g.cycle, g.trust, color=COL[t], lw=1.6, label="Trust score")
    bx.axhline(70, color="#2E9E5B", lw=0.8, ls=":"); bx.axhline(40, color="#555555", lw=0.8, ls=":")
    bx.set_ylim(0, 100); bx.set_ylabel("Trust score")
    ax.set_title(f"{t} — {name}", fontsize=8.5)
    ax.text(2, 1.1e3, "baseline", fontsize=7, color="#555555")
for ax in axs[1]:
    ax.set_xlabel("Cycle (1 s)")
h1, l1 = axs[0, 0].get_legend_handles_labels(); h2, l2 = axs[0, 0].get_figure().axes[4].get_legend_handles_labels() if len(fig.axes) > 4 else ([], [])
fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig("figures/fig_attacks.pdf"); fig.savefig("figures/fig_attacks.png", dpi=200)
print("ok", best)
