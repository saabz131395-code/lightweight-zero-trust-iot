"""Turn results/ into LaTeX macros and table rows, so no number is typed by hand."""
import json, os, sys
import numpy as np
import pandas as pd
import sqlite3, sklearn, platform

R = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "../article_v3"
DB = sys.argv[3] if len(sys.argv) > 3 else "data/testbed.db"
os.makedirs(f"{OUT}/tables", exist_ok=True)
from decimal import Decimal, ROUND_HALF_UP
def _r(x, nd):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))
pct = lambda x: _r(100 * x, 1) + "\\%"
f2 = lambda x: _r(x, 3)
M = {}

st = json.load(open(f"{R}/stats.json"))
sw = pd.read_csv(f"{R}/alpha_sweep.csv", index_col=0)
best = st["best_alpha"]
s1 = pd.read_csv(f"{R}/stage1_runs.csv")
main = s1[(s1["mode"] == "full") & (s1.if_fit == "all") & (s1.calib == "device")]

M["bestalpha"] = f"{best:g}"
b = sw.loc[best]
M["bestdet"], M["bestfpr"], M["bestf1"], M["bestlat"] = pct(b.detection), pct(b.fpr), f2(b.f1), _r(b.latency, 2)
M["bestprec"] = f2(b.precision)
M["bestJ"] = f2(b.J)
for al in sw.index:
    k = str(al).replace(".", "")
    M[f"det{k}"], M[f"fpr{k}"], M[f"fone{k}"], M[f"lat{k}"], M[f"J{k}"] = pct(sw.loc[al].detection), pct(sw.loc[al].fpr), f2(sw.loc[al].f1), _r(sw.loc[al].latency, 2), f2(sw.loc[al].J)

# alpha table rows
rows = []
for al, r in sw.iterrows():
    bold = (lambda t: f"\\textbf{{{t}}}") if al == best else (lambda t: t)
    rows.append(" & ".join(bold(x) for x in [f"{al:g}", f"{1 - al:.1f}", pct(r.detection), pct(r.fpr), f2(r.precision), f2(r.f1), _r(r.latency, 2), f"{_r(r.J, 3)} [{_r(r.J_lo, 3)}, {_r(r.J_hi, 3)}]"]) + " \\\\")
open(f"{OUT}/tables/alpha_rows.tex", "w").write("\n".join(rows) + "\n")

# per-attack at best alpha
pa = main[main.alpha == best]
types = ["DDoS flooding", "Data exfiltration", "Botnet C&C beaconing", "Port scanning"]
devs = {"DDoS flooding": "Room environment sensor, medicine fridge monitor", "Data exfiltration": "ECG monitor, infusion pump",
        "Botnet C&C beaconing": "IP security camera", "Port scanning": "Smart lighting controller"}
n_runs = main.run_id.nunique()
rows = []
for t in types:
    v = pa[f"det_{t}"]
    n = 2 if t in ("DDoS flooding", "Data exfiltration") else 1
    rows.append(f"{t.replace('&', chr(92) + '&')} & {devs[t]} & {n * n_runs} & {pct(v.mean())} \\\\")
    M["pa" + "".join(w[0] for w in t.split()).replace("&", "")] = pct(v.mean())
open(f"{OUT}/tables/perattack_rows.tex", "w").write("\n".join(rows) + "\n")

# variants at best alpha
lab = {("full", "all", "device"): "Proposed (IF on all baseline cycles, per-device calibration)",
       ("full", "all", "global"): "Pooled calibration of $A(t)$ (earlier design)",
       ("full", "last", "global"): "IF on last baseline cycle, pooled calibration (thesis design)",
       ("rule", "all", "device"): "Rule-based only (no Isolation Forest)",
       ("auth", "all", "device"): "Authentication only"}
rows = []
for key, name in lab.items():
    v = s1[(s1["mode"] == key[0]) & (s1.if_fit == key[1]) & (s1.calib == key[2]) & (s1.alpha == best)]
    pkey = f"best_vs_{key[0]}_{key[1]}_{key[2]}"
    pd_, pf_ = st.get(pkey + "_detection_p"), st.get(pkey + "_fpr_p")
    fmt = lambda p: "--" if p is None else ("n.s." if p >= 0.05 else ("$<$0.001" if p < 0.001 else f"{p:.3f}"))
    f1v = v.f1.mean()
    rows.append(f"{name} & {pct(v.detection.mean())} & {pct(v.fpr.mean())} & {f2(f1v)} & {('--' if np.isnan(v.latency.mean()) else _r(v.latency.mean(), 2))} & {fmt(pd_) if key != list(lab)[0] else '--'} & {fmt(pf_) if key != list(lab)[0] else '--'} \\\\")
    short = {"full_all_global": "glob", "full_last_global": "last", "rule_all_device": "rule", "auth_all_device": "auth", "full_all_device": "main"}["_".join(key)]
    M[f"v{short}det"], M[f"v{short}fpr"], M[f"v{short}fone"] = pct(v.detection.mean()), pct(v.fpr.mean()), f2(f1v)
open(f"{OUT}/tables/variants_rows.tex", "w").write("\n".join(rows) + "\n")

# stage 2
rep = pd.read_csv(f"{R}/stage2_report.csv", index_col=0)
cm = pd.read_csv(f"{R}/stage2_confusion.csv", index_col=0)
rows = []
for c in ["Normal"] + types:
    r = rep.loc[c]
    rows.append(f"{c.replace('&', chr(92) + '&')} & {f2(r.precision)} & {f2(r.recall)} & {f2(r['f1-score'])} & {int(r.support)} \\\\")
r = rep.loc["macro avg"]
rows.append(f"\\midrule Macro average & {f2(r.precision)} & {f2(r.recall)} & {f2(r['f1-score'])} & {int(r.support)} \\\\")
open(f"{OUT}/tables/stage2_rows.tex", "w").write("\n".join(rows) + "\n")
M["stmacrofone"] = f2(rep.loc["macro avg"]["f1-score"])
M["stacc"] = pct(rep.loc["accuracy"]["precision"])
M["stcncrec"], M["stcncprec"] = f2(rep.loc["Botnet C&C beaconing"].recall), f2(rep.loc["Botnet C&C beaconing"].precision)
dl = pd.read_csv(f"{R}/stage2_device_level.csv")
M["stdevacc"] = pct((dl.true == dl.pred).mean())
M["stdevn"] = str(len(dl)); M["stdevcorrect"] = str(int((dl.true == dl.pred).sum()))
M["nsusp"] = f"{st['n_suspicious_cycles']:,}".replace(",", "{,}")
ra = pd.read_csv(f"{R}/stage2_report_all_cycles.csv", index_col=0)
M["stallmacrofone"] = f2(ra.loc["macro avg"]["f1-score"])

# AUC
M["aucA"], M["aucT"] = f2(st["auc_A_mean"]), f2(st["auc_T_mean"])
M["aucgt"] = str(st["auc_T_gt_A_runs"]); M["aucp"] = (f"{st['auc_T_vs_A_p']:.1e}".replace("e-0", "\\times10^{-") + "}") if st["auc_T_vs_A_p"] < 0.001 else f"{st['auc_T_vs_A_p']:.3f}"
M["nruns"] = str(n_runs)
# J tests
for k, v in st.items():
    if k.startswith("J_best_vs_"):
        M["Jp" + k.split("_")[-1].replace(".", "")] = f"{v:.3f}" if v >= 0.001 else "<0.001"

# capture statistics
cs = pd.read_csv(f"{R}/capture_stats.csv")
tot = cs.groupby("run_id").n.sum()
M["pktsmean"] = f"{tot.mean():,.0f}".replace(",", "{,}")
M["pktstotal"] = f"{tot.sum():,.0f}".replace(",", "{,}")
M["pktsmin"], M["pktsmax"] = f"{tot.min():,}".replace(",", "{,}"), f"{tot.max():,}".replace(",", "{,}")
M["mbtotal"] = f"{cs.bytes.sum() / 1e6:,.0f}".replace(",", "{,}")
M["udpshare"] = pct(cs[cs.proto == "UDP"].n.sum() / cs.n.sum())
con = sqlite3.connect(DB)
M["dbmb"] = f"{os.path.getsize(DB) / 1e6:,.0f}".replace(",", "{,}")
M["ndevcycles"] = f"{30 * 240 * len(tot):,}".replace(",", "{,}")
M["pyver"] = platform.python_version(); M["sklver"] = sklearn.__version__; M["sqlitever"] = sqlite3.sqlite_version
M["dbrun"] = "11"

# false positives by device type
fpt = pd.read_csv(f"{R}/false_positives_by_type.csv", index_col=0)
tot_fp = int(fpt["sum"].sum())
top = fpt[fpt["sum"] > 0].sort_values("sum", ascending=False)
share = lambda names: int(top.loc[[n for n in names if n in top.index], "sum"].sum())
ev = share(["Nurse-call button", "Smart door sensor"])
def plural(n):
    n = n.replace("Pulse oximeter (SpO2)", "pulse oximeter")
    return (n if n[:2].isupper() else n[0].lower() + n[1:]) + "s"
_nl = [f"{plural(n)} ({int(v)})" for n, v in top["sum"].head(4).items()]
names = _nl[0] if len(_nl) == 1 else ", ".join(_nl[:-1]) + " and " + _nl[-1]
if ev == tot_fp:
    M["fptypes"] = (f"At the selected operating point, all {tot_fp} isolations of normal devices over the {n_runs} evaluation runs "
                    f"came from event-driven devices: {names}. No periodic or streaming device was ever isolated.")
else:
    M["fptypes"] = (f"Over the {n_runs} evaluation runs at $\\alpha={best:g}$, {tot_fp} normal-device isolations occurred; "
                    f"most came from {names}; event-driven devices account for {ev} of them.")
M["fptotal"] = str(tot_fp)
bl_all = pd.read_csv(f"{R}/design_choices_training_runs.csv")
bl = bl_all[bl_all.calib == "device"]
g = bl[bl.alpha == best].groupby("base")[["detection", "fpr"]].mean()
M["blfprtwenty"], M["blfprsixty"] = pct(g.loc[20].fpr), pct(g.loc[60].fpr)
M["bldettwenty"], M["bldetsixty"] = pct(g.loc[20].detection), pct(g.loc[60].detection)

gc = bl_all[(bl_all.base == 60) & (bl_all.alpha == best)].groupby("calib")[["detection", "fpr"]].mean()
M["trcaldevdet"], M["trcaldevfpr"] = pct(gc.loc["device"].detection), pct(gc.loc["device"].fpr)
M["trcalglobdet"], M["trcalglobfpr"] = pct(gc.loc["global"].detection), pct(gc.loc["global"].fpr)
am = pd.read_csv(f"{R}/mean_anomaly_score_normal_cycles_by_type.csv", index_col=0)
fast = ["Heart-rate monitor", "Pulse oximeter (SpO2)", "ECG monitor"]
slow = ["Fall-detection wearable", "Blood-pressure monitor", "Glucose monitor", "Patient temperature sensor", "Room environment sensor", "Medicine fridge monitor", "Smart bed sensor", "Smart infusion pump", "Smart lighting controller"]
per = [t for t in am.index if t not in ("Nurse-call button", "Smart door sensor")]
M["amfastlo"], M["amfasthi"] = f"{am.loc[fast, 'global'].min():.2f}", f"{am.loc[fast, 'global'].max():.2f}"
M["amslowlo"], M["amslowhi"] = f"{am.loc[slow, 'global'].min():.2f}", f"{am.loc[slow, 'global'].max():.2f}"
M["amdevlo"], M["amdevhi"] = f"{am.loc[per, 'device'].min():.2f}", f"{am.loc[per, 'device'].max():.2f}"
gt = bl[bl.base == 60].groupby("alpha")[["detection", "fpr"]].mean(); gt["J"] = gt.detection - gt.fpr
Jr = gt.J.round(6); M["trainbestalpha"] = f"{Jr[Jr == Jr.max()].index.max():g}"; M["trainJbest"] = f2(gt.J.max())
for al in gt.index:
    M["trainJ" + str(al).replace(".", "")] = f2(gt.loc[al].J)
cnc = main.groupby("alpha")["det_Botnet C&C beaconing"].mean()
for al in cnc.index:
    M["cnc" + str(al).replace(".", "")] = pct(cnc.loc[al])

cc = pd.read_csv(f"{R}/ciciomt_comparison.csv")
rows = []
for pair, g in cc.groupby("pair", sort=False):
    for k, (_, r) in enumerate(g.iterrows()):
        src = r.source.split(": ", 1)
        lab = ("CICIoMT2024: " + src[1]) if src[0] == "CICIoMT2024" else "Testbed"
        rows.append(f"{pair if k == 0 else ''} & {lab} & {int(r.windows)} & {r.rate_med:.2f} [{r.rate_q25:.2f}, {r.rate_q75:.2f}] & {r.size_med:.0f} [{r.size_q25:.0f}, {r.size_q75:.0f}] \\\\")
    rows.append("\\addlinespace")
open(f"{OUT}/tables/calib_rows.tex", "w").write("\n".join(rows[:-1]) + "\n")

with open(f"{OUT}/macros.tex", "w") as fh:
    fh.write("% generated by gen_results_tex.py -- do not edit\n")
    for k, v in M.items():
        fh.write(f"\\expandafter\\def\\csname val@{k}\\endcsname{{{v}}}\n")
    fh.write("\\newcommand{\\val}[1]{\\csname val@#1\\endcsname}\n")
print(json.dumps(M, indent=0))
