"""
revision_stats.py
-----------------
Statistical reporting requested by the internal examiner (Major comments 7 and 9).

1. Paired comparisons (full architecture vs each alternative), unit of analysis
   = one simulation seed; pairs = the same seed run under both configurations.
   For each metric: mean difference, median difference with a 95% bootstrap CI,
   Wilcoxon signed-rank test (zero differences dropped, 'wilcox' method; the
   number dropped is reported), Holm-corrected p-values across all tests, and
   the matched-pairs rank-biserial correlation as effect size.
2. Formal operating-point rule for alpha (Youden's J = recall - FPR, and the
   constrained rule "highest detection with FPR <= 5%").

Usage: python revision_stats.py
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

OUT = "results/revision"
rng = np.random.default_rng(0)


def rank_biserial(d):
    d = d[d != 0]
    if len(d) == 0:
        return np.nan
    r = stats.rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


def boot_ci_median(d, n=10000):
    idx = rng.integers(0, len(d), (n, len(d)))
    meds = np.median(d[idx], axis=1)
    return np.percentile(meds, [2.5, 97.5])


def paired_tests():
    abl = pd.read_csv("results/ablation_results.csv")
    frames = [abl]
    rf_path = f"{OUT}/rf_per_seed.csv"
    if os.path.exists(rf_path):
        frames.append(pd.read_csv(rf_path))
    df = pd.concat(frames, ignore_index=True)
    full = df[df["mode"] == "full"].set_index("seed").sort_index()
    rows = []
    for mode in [m for m in df["mode"].unique() if m != "full"]:
        other = df[df["mode"] == mode].set_index("seed").sort_index()
        for metric in ("detection_rate", "fpr", "f1"):
            a, b = full[metric], other.loc[full.index, metric]
            ok = a.notna() & b.notna()
            if ok.sum() == 0:
                rows.append(dict(comparison=f"full vs {mode}", metric=metric, n=0,
                                 note="not testable: metric undefined for every seed"))
                continue
            d = (a[ok] - b[ok]).values
            nz = int((d == 0).sum())
            if np.all(d == 0):
                rows.append(dict(comparison=f"full vs {mode}", metric=metric, n=int(ok.sum()),
                                 zero_diffs=nz, note="all differences zero"))
                continue
            w = stats.wilcoxon(d, zero_method="wilcox")
            lo, hi = boot_ci_median(d)
            rows.append(dict(comparison=f"full vs {mode}", metric=metric, n=int(ok.sum()),
                             zero_diffs=nz, mean_full=a[ok].mean(), mean_other=b[ok].mean(),
                             mean_diff=d.mean(), median_diff=np.median(d), ci_lo=lo, ci_hi=hi,
                             W=w.statistic, p=w.pvalue, effect_r=rank_biserial(d)))
    res = pd.DataFrame(rows)
    tested = res["p"].notna()
    p = res.loc[tested, "p"].values
    order = np.argsort(p)
    m = len(p)
    holm = np.empty(m)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, min(1, (m - rank) * p[i]))
        holm[i] = running
    res.loc[tested, "p_holm"] = holm
    res.to_csv(f"{OUT}/paired_tests.csv", index=False)
    return res


def alpha_rule():
    s = pd.read_csv("results/sensitivity_runs.csv")
    s["J"] = s["det"] - s["fpr"]
    g = s.groupby("alpha")
    out = pd.DataFrame({
        "det": g.det.mean(), "fpr": g.fpr.mean(), "f1": g.f1.mean(), "lat": g.lat.mean(),
        "J_mean": g.J.mean(),
        "J_ci_lo": g.J.apply(lambda x: x.mean() - 1.96 * x.std(ddof=1) / np.sqrt(len(x))),
        "J_ci_hi": g.J.apply(lambda x: x.mean() + 1.96 * x.std(ddof=1) / np.sqrt(len(x))),
    }).reset_index()
    out["fpr_le_5pct"] = out["fpr"] <= 0.05
    # paired test: J at 0.8 vs J at 0.9 and 0.7 (same seeds 1-40)
    piv = s.pivot(index="seed", columns="alpha", values="J")
    tests = {}
    for other in (0.7, 0.9):
        d = (piv[0.8] - piv[other]).values
        tests[other] = stats.wilcoxon(d).pvalue
    out.to_csv(f"{OUT}/alpha_selection.csv", index=False)
    return out, tests


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 200)
    print(paired_tests().round(4).to_string())
    a, t = alpha_rule()
    print(a.round(4).to_string())
    print("Wilcoxon J(0.8) vs J(0.7), J(0.9):", t)
