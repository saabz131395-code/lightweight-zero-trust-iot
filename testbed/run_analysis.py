"""Run the full analysis on the captured testbed dataset and write results/ and figures/.

  python run_analysis.py --db data/testbed.db --train 1-10 --test 11-30
"""
import argparse, json, os, sqlite3, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analysis as A
import config as CFG

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9]
TYPES = ["DDoS flooding", "Data exfiltration", "Botnet C&C beaconing", "Port scanning"]
COL = {"DDoS flooding": "#D64545", "Data exfiltration": "#E8952C", "Botnet C&C beaconing": "#4F6D99", "Port scanning": "#0E9F8E"}


def rng_ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    b = np.random.default_rng(0).choice(x, (5000, len(x))).mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/testbed.db")
    ap.add_argument("--train", default="1-10")
    ap.add_argument("--test", default="11-30")
    ap.add_argument("--base", type=int, default=CFG.BASELINE_CYCLES)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    A.BASE = a.base
    os.makedirs(a.out, exist_ok=True); os.makedirs("figures", exist_ok=True)
    con = sqlite3.connect(a.db)
    dev, runs, att = A.load_meta(con)
    rr = lambda s: list(range(int(s.split("-")[0]), int(s.split("-")[1]) + 1))
    train, test = rr(a.train), rr(a.test)
    total = int(runs.total_cycles.iloc[0])

    # ---- features (pre-processing) --------------------------------------
    feats = {}
    for r in train + test:
        feats[r] = A.normalise(A.label(A.extract_features(con, r, dev, total), att, r))
    allf = pd.concat(feats.values())
    allf.to_csv(f"{a.out}/features.csv.gz", index=False)

    # ---- dataset statistics ------------------------------------------------
    st = pd.read_sql("SELECT run_id, proto, COUNT(*) n, SUM(ip_len) bytes FROM packets GROUP BY run_id, proto", con)
    st.to_csv(f"{a.out}/capture_stats.csv", index=False)
    by_dev = pd.read_sql("""SELECT d.device_id, d.name, d.protocol, d.port, COUNT(*) n_packets, SUM(p.ip_len) bytes
        FROM packets p JOIN devices d ON p.src_ip=d.ip GROUP BY d.device_id ORDER BY d.device_id""", con)
    by_dev.to_csv(f"{a.out}/packets_by_device.csv", index=False)
    norm_profile = allf[(allf.cycle < A.BASE)].merge(dev[["device_id", "dtype"]]).groupby("dtype")[A.FEATS].mean()
    norm_profile.to_csv(f"{a.out}/baseline_profile_by_type.csv")
    attack_profile = allf[allf.in_attack].groupby("attack_type")[A.FEATS].mean()
    attack_profile.to_csv(f"{a.out}/attack_profile.csv")

    # ---- stage 1: alpha sweep, IF fit variants, ablation --------------------
    # ---- design choices fixed on the TRAINING runs only -------------------
    # baseline length (20 vs 60 cycles), calibration of A(t) (pooled vs per device)
    # and the operating point alpha (maximum Youden's J)
    bl = []
    for base in (20, 60):
        A.BASE = base
        for r in train:
            fr = A.normalise(A.label(A.extract_features(con, r, dev, total), att, r))
            for cal in ("device", "global"):
                for al in ALPHAS:
                    m = A.device_metrics(A.run_engine(fr, al, seed=r, calib=cal), att, r, dev)
                    m.update(base=base, alpha=al, calib=cal); bl.append(m)
    A.BASE = a.base
    bl = pd.DataFrame(bl); bl.to_csv(f"{a.out}/design_choices_training_runs.csv", index=False)
    tr = bl[(bl.base == a.base) & (bl.calib == "device")].groupby("alpha")[["detection", "fpr"]].mean()
    Jt = (tr.detection - tr.fpr).round(6)
    best_pre = float(Jt[Jt == Jt.max()].index.max())   # ties -> the more persistent setting (fewer false alarms)
    print(bl.groupby(["base", "calib", "alpha"])[["detection", "fpr", "f1"]].mean().round(3).to_string())

    rows, logs = [], {}
    # main configuration: IF on all baseline cycles, per-device calibration of A(t)
    variants = [("full", "all", "device"), ("full", "all", "global"), ("full", "last", "global"),
                ("rule", "all", "device"), ("auth", "all", "device")]
    for r in test:
        for al in ALPHAS:
            lg = A.run_engine(feats[r], al, seed=r, if_fit="all", mode="full", calib="device")
            m = A.device_metrics(lg, att, r, dev); m.update(alpha=al, mode="full", if_fit="all", calib="device")
            rows.append(m); logs[(r, al)] = lg
    for r in test:
        for mode, fit, cal in variants[1:]:
            lg = A.run_engine(feats[r], best_pre, seed=r, if_fit=fit, mode=mode, calib=cal)
            m = A.device_metrics(lg, att, r, dev); m.update(alpha=best_pre, mode=mode, if_fit=fit, calib=cal)
            rows.append(m)
    res = pd.DataFrame(rows)
    res.to_csv(f"{a.out}/stage1_runs.csv", index=False)
    summ = res.groupby(["mode", "if_fit", "calib", "alpha"]).agg(["mean", "std"]).round(4)
    summ.to_csv(f"{a.out}/stage1_summary.csv")

    full = res[(res["mode"] == "full") & (res.if_fit == "all") & (res.calib == "device")]
    sweep = full.groupby("alpha")[["detection", "fpr", "precision", "f1", "latency"]].mean()
    sweep["J"] = sweep.detection - sweep.fpr
    for al in ALPHAS:
        sub = full[full.alpha == al]
        sweep.loc[al, "J_lo"], sweep.loc[al, "J_hi"] = rng_ci(sub.detection - sub.fpr)
        sweep.loc[al, "det_lo"], sweep.loc[al, "det_hi"] = rng_ci(sub.detection)
        sweep.loc[al, "fpr_lo"], sweep.loc[al, "fpr_hi"] = rng_ci(sub.fpr)
    best = best_pre   # chosen on the training runs
    sweep.to_csv(f"{a.out}/alpha_sweep.csv")
    per_attack = full[full.alpha == best][[f"det_{t}" for t in TYPES]].agg(["mean", "std"])
    per_attack.to_csv(f"{a.out}/per_attack_best_alpha.csv")

    stats = {}
    jb = full[full.alpha == best].set_index("run_id")
    for al in ALPHAS:
        if al == best:
            continue
        jo = full[full.alpha == al].set_index("run_id")
        d = (jb.detection - jb.fpr) - (jo.detection - jo.fpr)
        stats[f"J_best_vs_{al}"] = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
    fb = full[full.alpha == best].set_index("run_id")
    for (mode, fit, cal) in variants[1:]:
        o = res[(res["mode"] == mode) & (res.if_fit == fit) & (res.calib == cal) & (res.alpha == best)].set_index("run_id")
        for met in ("detection", "fpr"):
            d = fb[met] - o[met]
            stats[f"best_vs_{mode}_{fit}_{cal}_{met}_p"] = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
    stats["best_alpha"] = best
    stats["eval_J_argmax_alpha"] = float(sweep.J.idxmax())
    am = []
    for r in test:
        for cal in ("device", "global"):
            lg = A.run_engine(feats[r], best, seed=r, calib=cal)
            post = lg[(lg.cycle >= A.BASE) & (~lg.in_attack)]
            x = post.groupby("device_id").A.mean().rename("meanA").reset_index(); x["calib"] = cal; x["run_id"] = r; am.append(x)
    am = pd.concat(am).merge(dev[["device_id", "dtype"]]).groupby(["dtype", "calib"]).meanA.mean().unstack()
    am.to_csv(f"{a.out}/mean_anomaly_score_normal_cycles_by_type.csv")
    fpd = []
    for r in test:
        lg = logs[(r, best)]; post = lg[lg.cycle >= A.BASE]
        mal = set(att[att.run_id == r].device_id)
        mn = post[~post.device_id.isin(mal)].groupby("device_id").trust.min()
        for did, v in mn.items():
            fpd.append(dict(run_id=r, device_id=did, isolated=bool(v < A.MED)))
    fpd = pd.DataFrame(fpd).merge(dev[["device_id", "dtype"]])
    fpt = fpd.groupby("dtype").isolated.agg(["sum", "count"]); fpt["rate"] = fpt["sum"] / fpt["count"]
    fpt.sort_values("rate", ascending=False).to_csv(f"{a.out}/false_positives_by_type.csv")
    auc = []
    for r in test:
        lg = logs[(r, best)]; post = lg[lg.cycle >= A.BASE]
        auc.append(dict(run_id=r, auc_A=roc_auc_score(post.in_attack, post.A), auc_T=roc_auc_score(post.in_attack, -post.trust)))
    auc = pd.DataFrame(auc); auc.to_csv(f"{a.out}/device_cycle_auc.csv", index=False)
    stats["auc_A_mean"], stats["auc_T_mean"] = float(auc.auc_A.mean()), float(auc.auc_T.mean())
    stats["auc_T_gt_A_runs"] = int((auc.auc_T > auc.auc_A).sum())
    stats["auc_T_vs_A_p"] = float(wilcoxon(auc.auc_T - auc.auc_A).pvalue)

    # ---- stage 2: attack-type classification ------------------------------
    trn = allf[allf.run_id.isin(train) & (allf.cycle >= A.BASE)]
    clf = A.train_stage2(trn)
    tst_logs = pd.concat([logs[(r, best)] for r in test])
    susp = tst_logs[(tst_logs.cycle >= A.BASE) & (tst_logs.trust < A.HIGH)].copy()
    susp["pred"] = clf.predict(A.stage2_features(susp))
    labels = ["Normal"] + TYPES
    cm = confusion_matrix(susp.attack_type, susp.pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(f"{a.out}/stage2_confusion.csv")
    rep = classification_report(susp.attack_type, susp.pred, labels=labels, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(f"{a.out}/stage2_report.csv")
    # all attack cycles (classifier alone, independent of stage 1)
    allt = allf[allf.run_id.isin(test) & (allf.cycle >= A.BASE)]
    rep_all = classification_report(allt.attack_type, clf.predict(A.stage2_features(allt)), labels=labels, output_dict=True, zero_division=0)
    pd.DataFrame(rep_all).T.to_csv(f"{a.out}/stage2_report_all_cycles.csv")
    # device-level type assignment for detected attackers
    dl = []
    for r in test:
        lg = logs[(r, best)]
        for _, at in att[att.run_id == r].iterrows():
            g = lg[(lg.device_id == at.device_id) & (lg.cycle >= at.start_cycle) & (lg.cycle < at.end_cycle)]
            if (g.trust < A.MED).any():
                s = g[g.trust < A.HIGH]
                p = pd.Series(clf.predict(A.stage2_features(s)))
                p = p[p != "Normal"]
                dl.append(dict(run_id=r, true=at.attack_type, pred=p.mode().iloc[0] if len(p) else "Normal"))
    dl = pd.DataFrame(dl)
    dl.to_csv(f"{a.out}/stage2_device_level.csv", index=False)
    stats["stage2_device_accuracy"] = float((dl.true == dl.pred).mean()) if len(dl) else None
    stats["stage2_macro_f1_suspicious"] = rep["macro avg"]["f1-score"]
    stats["n_suspicious_cycles"] = int(len(susp))
    json.dump(stats, open(f"{a.out}/stats.json", "w"), indent=2)

    # ---- figures ----------------------------------------------------------
    # alpha sweep
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    x = sweep.index.values
    ax[0].errorbar(x, sweep.detection * 100, yerr=[(sweep.detection - sweep.det_lo) * 100, (sweep.det_hi - sweep.detection) * 100], marker="o", color="#0E9F8E", capsize=3, label="Detection rate")
    ax[0].errorbar(x, sweep.fpr * 100, yerr=[(sweep.fpr - sweep.fpr_lo) * 100, (sweep.fpr_hi - sweep.fpr) * 100], marker="s", color="#D64545", capsize=3, label="False-positive rate")
    ax[0].set_xlabel(r"Trust persistence $\alpha$"); ax[0].set_ylabel("%"); ax[0].set_ylim(0, 105); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("(a) Detection vs. false positives", fontsize=9)
    ax[1].plot(x, sweep.f1, marker="^", color="#4F6D99", label="F1-score")
    ax2 = ax[1].twinx(); ax2.plot(x, sweep.latency, marker="D", ls="--", color="#E8952C", label="Latency")
    ax[1].set_xlabel(r"Trust persistence $\alpha$"); ax[1].set_ylabel("F1-score"); ax2.set_ylabel("Detection latency (cycles)")
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="center right"); ax[1].set_title("(b) F1 vs. detection latency", fontsize=9)
    fig.tight_layout(); fig.savefig("figures/fig_alpha.pdf"); plt.close(fig)

    # one graph per attack type: see make_attack_figure.py

    # confusion matrix
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    short = ["Normal", "DDoS", "Exfiltration", "C&C", "Scanning"]
    ax.set_xticks(range(5)); ax.set_xticklabels(short, rotation=30, ha="right"); ax.set_yticks(range(5)); ax.set_yticklabels(short)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted type"); ax.set_ylabel("True type"); ax.spines[:].set_visible(True)
    fig.tight_layout(); fig.savefig("figures/fig_confusion.pdf"); plt.close(fig)

    print(sweep.round(3).to_string()); print(per_attack.round(3).to_string()); print(json.dumps(stats, indent=1))
    print(summ.loc[:, (["detection", "fpr", "f1", "latency"], "mean")].to_string())
    print(pd.DataFrame(rep).T.round(3).to_string())


if __name__ == "__main__":
    main()
