"""Device-cycle ROC-AUC analysis (article Section 6.4).

Every device at every post-baseline cycle is one sample; it is positive when
the device is inside its attack window. Compares the single-cycle anomaly
score A(t) with the trust score Ts(t) (lower trust = more suspicious) over the
40 baseline seeds (41-80, alpha = 0.7), plus cycle-level metrics at
alpha = 0.8 on the sensitivity-sweep seeds (1-40), and per-attack-type AUC of A(t) and
cycle-level FPR / precision / recall of the DENY decision.

Usage: python device_cycle_auc.py [--workers 4]
Writes results/device_cycle_auc.csv
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

import trust_engine_sim as sim

SEEDS = range(41, 81)


def one_seed(args):
    seed, alpha = args
    log, _, _ = sim.run_simulation(alpha=alpha, beta=round(1 - alpha, 10), verbose=False, seed=seed)
    d = log[log.t >= sim.BASELINE_STEPS]
    y = d.in_attack.astype(int).values
    row = dict(seed=seed, alpha=alpha, prevalence=y.mean(),
               auc_A=roc_auc_score(y, d.anomaly_A), auc_Ts=roc_auc_score(y, -d.trust_score))
    neg = d[~d.in_attack]
    for at in list(sim.ATTACK_TYPES):
        pos = d[d.in_attack & (d.attack_type == at)]
        if len(pos):
            yy = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            row[f"auc_A_{at}"] = roc_auc_score(yy, np.r_[pos.anomaly_A, neg.anomaly_A])
    flag = (d.decision == "DENY_ISOLATE").values
    tp = (flag & (y == 1)).sum(); fp = (flag & (y == 0)).sum()
    row.update(cyc_fpr=fp / (y == 0).sum(), cyc_prec=tp / max(tp + fp, 1), cyc_rec=tp / (y == 1).sum())
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    jobs = [(s, 0.7) for s in SEEDS] + [(s, 0.8) for s in range(1, 41)]  # 0.8 uses the sensitivity-sweep seeds
    with ProcessPoolExecutor(a.workers) as ex:
        df = pd.DataFrame(list(ex.map(one_seed, jobs)))
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/device_cycle_auc.csv", index=False)
    print(df.groupby("alpha").agg(["mean", "std"]).round(4).T.to_string())
    b = df[df.alpha == 0.7]
    print("Ts > A in", int((b.auc_Ts > b.auc_A).sum()), "of", len(b), "seeds; Wilcoxon p =",
          wilcoxon(b.auc_Ts, b.auc_A).pvalue)
