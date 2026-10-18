"""
revision_experiments.py
-----------------------
Additional analyses requested by the internal examiner (October 2026).
trust_engine_sim.py and ablation_study.py are NOT modified; every experiment
below reuses their device model, traffic and attack generator, and metrics.

  weights     Behavioural-risk weight sensitivity (Major comment 3)
  thresholds  Access-threshold sensitivity (Minor comment 8)
  iffit       Isolation Forest fitted on all 20 baseline cycles instead of the
              last baseline cycle only (Minor comment 7); also records the
              normalised feature vectors for the correlation analysis
              (Minor comment 9) and the attack prevalence (Minor comment 10)
  rf          Per-seed Random Forest results, needed for the full statistical
              reporting of Table 6.5 (Major comment 9)
  scale       Per-cycle cost of the gateway/cloud steps for 100 to 100,000
              devices (Major comment 14)

Usage:  python revision_experiments.py <experiment> [--workers N]
Outputs go to results/revision/.
"""

import argparse
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
OUT = "results/revision"

BASELINE_SEEDS = list(range(41, 81))      # alpha = 0.7 evaluation seeds
SENS_SEEDS = list(range(1, 41))           # alpha = 0.8 (sensitivity) seeds

WEIGHT_CONFIGS = {
    "default":        dict(RF=0.30, IAT=0.25, PS=0.20, PU=0.15, DV=0.10),
    "equal":          dict(RF=0.20, IAT=0.20, PS=0.20, PU=0.20, DV=0.20),
    "rate_heavy":     dict(RF=0.40, IAT=0.30, PS=0.10, PU=0.10, DV=0.10),
    "content_heavy":  dict(RF=0.15, IAT=0.15, PS=0.30, PU=0.20, DV=0.20),
}
THRESHOLD_CONFIGS = {          # (high, medium)
    "70/40 (default)": (70, 40),
    "70/30": (70, 30),
    "70/50": (70, 50),
    "65/35": (65, 35),
    "75/45": (75, 45),
}


def _metrics_row(m):
    return dict(det=m["detection_rate"], fpr=m["false_positive_rate"],
                prec=m["precision"], f1=m["f1_score"],
                lat=m["mean_detection_latency_steps"])


def _reauth_cycles(log, devices, baseline_steps):
    normal = [d.id for d in devices if not d.is_malicious]
    mon = log[(log.t >= baseline_steps) & (log.device_id.isin(normal))]
    return float((mon.decision == "RE_AUTHENTICATE").sum()) / len(normal)


# ---------------------------------------------------------------- weights / thresholds
def job_sim(args):
    kind, name, alpha, seed, extra = args
    import trust_engine_sim as sim
    kw = dict(alpha=alpha, beta=round(1 - alpha, 2), seed=seed, verbose=False)
    if kind == "weights":
        kw["weights"] = extra
    else:
        kw["high_thr"], kw["med_thr"] = extra
    log, devices, m = sim.run_simulation(**kw)
    row = dict(config=name, alpha=alpha, seed=seed, **_metrics_row(m))
    row["reauth_cycles_normal"] = _reauth_cycles(log, devices, sim.BASELINE_STEPS)
    return row


# ---------------------------------------------------------------- IF fitted on all baseline cycles
def job_iffit(args):
    alpha, seed, collect = args
    import trust_engine_sim as sim
    from sklearn.ensemble import IsolationForest
    from ablation_study import _compute_R

    beta = round(1 - alpha, 2)
    sim.RNG = np.random.default_rng(seed)
    devices = sim.build_population()
    hi, lo = sim.TRUST_THRESHOLDS["high"], sim.TRUST_THRESHOLDS["medium"]
    iso, R_hist, log_rows, vecs = None, [], [], []
    attack_cycles = total_cycles = 0

    for t in range(sim.TOTAL_STEPS):
        feats = []
        for d in devices:
            f = d.sample_features(t)
            f["device_id"] = d.id
            feats.append(f)
        batch = pd.DataFrame(feats).set_index("device_id")
        if t < sim.BASELINE_STEPS:
            for d in devices:
                d.record_baseline(batch.loc[d.id].to_dict())
        if t == sim.BASELINE_STEPS - 1:
            for d in devices:
                d.freeze_baseline()
        R, behav_trust, _ = _compute_R(devices, batch)

        if t < sim.BASELINE_STEPS - 1:
            A = {d.id: 0.0 for d in devices}
        else:
            if t == sim.BASELINE_STEPS - 1:
                # Re-normalise every recorded baseline cycle against the frozen
                # ranges, then fit on all 20 cycles (20 x 100 vectors).
                for tb in range(sim.BASELINE_STEPS):
                    rows = []
                    for d in devices:
                        h = {c: d.baseline_history[c][tb] for c in d.baseline_history}
                        br = d.baseline_range
                        rows.append([
                            np.clip((h["RF"] - br["RF"][0]) / (br["RF"][1] - br["RF"][0]), 0, 1),
                            np.clip(1 - (h["IAT"] - br["IAT"][0]) / (br["IAT"][1] - br["IAT"][0]), 0, 1),
                            np.clip((h["PS"] - br["PS"][0]) / (br["PS"][1] - br["PS"][0]), 0, 1),
                            np.clip((h["PU"] - br["PU"][0]) / (br["PU"][1] - br["PU"][0]), 0, 1),
                            np.clip((h["DV"] - br["DV"][0]) / (br["DV"][1] - br["DV"][0]), 0, 1)])
                    R_hist.append(rows)
                Xb = np.array(R_hist).reshape(-1, 5)
                iso = IsolationForest(**sim.ISOLATION_FOREST_PARAMS).fit(Xb)
                bs = -iso.score_samples(Xb)
                iso.mu_b, iso.sigma_b = float(bs.mean()), float(bs.std()) + 1e-9
            s = -iso.score_samples(R)
            A = dict(zip(batch.index, np.clip((s - iso.mu_b) / (3 * iso.sigma_b), 0, 1)))

        for i, d in enumerate(devices):
            s_auth = d.auth_score()
            t_total = (sim.TRUST_COMPONENT_WEIGHTS["auth"] * s_auth +
                       sim.TRUST_COMPONENT_WEIGHTS["behav"] * behav_trust[d.id])
            if t < sim.BASELINE_STEPS:
                ts = t_total * 100
            else:
                ts = float(np.clip(alpha * d.trust_score / 100 + beta * (1 - A[d.id]), 0, 1)) * 100
            d.trust_score = ts
            if ts >= hi:
                dec = "ALLOW"
            elif ts >= lo:
                dec = "RE_AUTHENTICATE"
            else:
                dec = "DENY_ISOLATE"
                if d.is_malicious and d.in_attack_window(t) and d.first_low_trust_step is None:
                    d.first_low_trust_step = t
            log_rows.append(dict(t=t, device_id=d.id, decision=dec))
            if t >= sim.BASELINE_STEPS:
                total_cycles += 1
                attack_cycles += int(d.in_attack_window(t))
            if collect:
                vecs.append(dict(seed=seed, t=t, device_type=d.device_type,
                                 attack=d.attack_type if d.in_attack_window(t) else "normal",
                                 RF=R[i][0], IAT=R[i][1], PS=R[i][2], PU=R[i][3], DV=R[i][4]))

    log = pd.DataFrame(log_rows)
    m = sim.compute_metrics(log, devices)
    row = dict(config="IF fitted on all 20 baseline cycles", alpha=alpha, seed=seed,
               **_metrics_row(m), attack_cycle_share=attack_cycles / total_cycles)
    return row, vecs


# ---------------------------------------------------------------- Random Forest per seed
def run_rf():
    from ablation_study import train_rf, run_variant
    t0 = time.time()
    clf = train_rf()
    print("RF trained in", round(time.time() - t0), "s")
    rows = []
    for seed in BASELINE_SEEDS:
        m = run_variant("rf_supervised", alpha=0.7, beta=0.3, seed=seed, rf_model=clf)
        rows.append(dict(seed=seed, mode="rf_supervised", detection_rate=m["detection_rate"] * 100,
                         fpr=m["false_positive_rate"] * 100, f1=m["f1_score"],
                         latency=m["mean_detection_latency_steps"]))
    pd.DataFrame(rows).to_csv(f"{OUT}/rf_per_seed.csv", index=False)


# ---------------------------------------------------------------- scalability timing
def run_scale():
    from sklearn.ensemble import IsolationForest
    import trust_engine_sim as sim
    rng = np.random.default_rng(0)
    w = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    iso = IsolationForest(**sim.ISOLATION_FOREST_PARAMS).fit(rng.random((2000, 5)) * 0.3)
    rows = []
    for n in (100, 1_000, 10_000, 100_000):
        x = rng.random((n, 5)) * 100
        lo_, hi_ = np.zeros((n, 5)), np.full((n, 5), 100.0)
        dt = rng.random(n)
        reps = 200 if n <= 10_000 else 20

        def timeit(fn):
            fn()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            return (time.perf_counter() - t0) / reps * 1000  # ms per cycle

        t_norm = timeit(lambda: np.clip((x - lo_) / (hi_ - lo_), 0, 1) @ w)
        R = np.clip((x - lo_) / (hi_ - lo_), 0, 1)
        t_if = timeit(lambda: iso.score_samples(R))
        a = rng.random(n)
        t_upd = timeit(lambda: np.clip(0.8 * dt + 0.2 * (1 - a), 0, 1))
        state_bytes = 8 * (10 + 1 + 1 + 1) + 8 * 5 * 2   # history, trust, flags, frozen min/max
        rows.append(dict(devices=n, normalise_risk_ms=t_norm, iforest_ms=t_if,
                         trust_update_ms=t_upd, total_ms=t_norm + t_if + t_upd,
                         per_device_us=(t_norm + t_if + t_upd) * 1000 / n,
                         gateway_state_kib=state_bytes * n / 1024,
                         feature_upload_kib=5 * 8 * n / 1024))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/scalability.csv", index=False)
    print(df.round(4).to_string())


def pool_map(fn, jobs, workers):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, jobs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["weights", "thresholds", "iffit", "rf", "scale"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if a.experiment in ("weights", "thresholds"):
        configs = WEIGHT_CONFIGS if a.experiment == "weights" else THRESHOLD_CONFIGS
        skip = "default" if a.experiment == "weights" else "70/40 (default)"
        jobs = [(a.experiment, name, alpha, s, cfg)
                for name, cfg in configs.items() if name != skip
                for alpha, seeds in ((0.7, BASELINE_SEEDS), (0.8, SENS_SEEDS)) for s in seeds]
        df = pd.DataFrame(pool_map(job_sim, jobs, a.workers))
        df.to_csv(f"{OUT}/{a.experiment}_sensitivity.csv", index=False)
        print(df.groupby(["config", "alpha"])[["det", "fpr", "f1", "lat"]].mean().round(3))
    elif a.experiment == "iffit":
        jobs = [(0.7, s, s <= 45) for s in BASELINE_SEEDS] + [(0.8, s, False) for s in SENS_SEEDS]
        out = pool_map(job_iffit, jobs, a.workers)
        pd.DataFrame([r for r, _ in out]).to_csv(f"{OUT}/iffit_all_baseline.csv", index=False)
        pd.DataFrame([v for _, vs in out for v in vs]).to_csv(f"{OUT}/feature_vectors_seeds41-45.csv", index=False)
        print(pd.DataFrame([r for r, _ in out]).groupby("alpha").mean(numeric_only=True).round(3))
    elif a.experiment == "rf":
        run_rf()
    else:
        run_scale()
