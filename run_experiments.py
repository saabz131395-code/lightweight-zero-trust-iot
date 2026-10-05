"""
run_experiments.py
------------------
Regenerates the per-seed result files used in Chapter 6 of the thesis and in
the article, using trust_engine_sim.py unchanged.

  results/baseline_runs.csv     alpha = 0.7, beta = 0.3, seeds 41-80   (40 runs)
  results/sensitivity_runs.csv  alpha in {0.5,...,0.9}, beta = 1-alpha,
                                seeds 1-40 for each alpha              (200 runs)

Together these are the 240 evaluation runs reported in the thesis.

Columns (one row per run):
  seed, alpha, det, fpr, prec, rec, f1, lat
      device-level detection rate, false-positive rate, precision, recall,
      F1-score and mean detection latency (cycles), from compute_metrics()
  reauth_entries_normal
      mean number of times a legitimate device ENTERED the RE-AUTHENTICATE
      band during the 180 monitoring cycles
  reauth_cycles_normal
      mean number of monitoring cycles a legitimate device SPENT in the
      RE-AUTHENTICATE band. Counting every such cycle as one FAST-SM9
      re-authentication gives the upper-bound device-side energy estimate
      used in the comparison with Ma et al. (2025).

Usage:
  python run_experiments.py              # all 240 runs (about 15-30 min)
  python run_experiments.py --quick      # 2 seeds per configuration, smoke test
  python run_experiments.py --workers 4  # run in parallel
"""

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

warnings.filterwarnings("ignore")

BASELINE = [(0.7, s) for s in range(41, 81)]
SENSITIVITY = [(a, s) for a in (0.5, 0.6, 0.7, 0.8, 0.9) for s in range(1, 41)]


def run_one(job):
    alpha, seed = job
    import trust_engine_sim as sim

    log, devices, m = sim.run_simulation(alpha=alpha, beta=round(1 - alpha, 2),
                                         seed=seed, verbose=False)
    mon = log[log.t >= sim.BASELINE_STEPS].sort_values(["device_id", "t"])
    normal = [d.id for d in devices if not d.is_malicious]

    entries = []
    for did in normal:
        prev, n = "ALLOW", 0
        for dec in mon.loc[mon.device_id == did, "decision"]:
            if dec == "RE_AUTHENTICATE" and prev != "RE_AUTHENTICATE":
                n += 1
            prev = dec
        entries.append(n)
    cycles = (mon[mon.device_id.isin(normal)].decision == "RE_AUTHENTICATE").sum()

    return dict(seed=seed, alpha=alpha,
                det=m["detection_rate"], fpr=m["false_positive_rate"],
                prec=m["precision"], rec=m["recall"], f1=m["f1_score"],
                lat=m["mean_detection_latency_steps"],
                reauth_entries_normal=sum(entries) / len(normal),
                reauth_cycles_normal=float(cycles) / len(normal))


def run(jobs, workers):
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(run_one, jobs))
    return [run_one(j) for j in jobs]


def summarize(df, name):
    print(f"\n{name}")
    print(df.groupby("alpha")[["det", "fpr", "f1", "lat", "reauth_cycles_normal"]]
            .mean().round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 seeds per configuration")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    base_jobs, sens_jobs = BASELINE, SENSITIVITY
    if args.quick:
        base_jobs = BASELINE[:2]
        sens_jobs = [j for j in SENSITIVITY if j[1] <= 2]

    base_df = pd.DataFrame(run(base_jobs, args.workers))
    sens_df = pd.DataFrame(run(sens_jobs, args.workers))

    suffix = "_quick" if args.quick else ""
    base_df.to_csv(f"results/baseline_runs{suffix}.csv", index=False)
    sens_df.to_csv(f"results/sensitivity_runs{suffix}.csv", index=False)
    summarize(base_df, "Baseline (seeds 41-80)")
    summarize(sens_df, "Sensitivity (seeds 1-40)")
