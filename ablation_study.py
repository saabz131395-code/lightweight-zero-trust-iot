"""
ablation_study.py
------------------
Additive companion to trust_engine_sim.py. Does NOT modify the original
engine or change any of its default behaviour -- trust_engine_sim.run_simulation()
is untouched and still reproduces every number already reported in the thesis
(Table 6.1: 81.0% detection, 31.9% FPR, F1=0.388, 3.97-cycle latency at
seeds 41-80, alpha=0.7/beta=0.3). This file only adds three comparison
configurations, run through the same population/traffic/attack generator,
so the results are directly comparable to the published baseline.

Three ablations:

  auth_only  - "authenticate once, trust forever". Trust is seeded from
               S_auth alone at the end of baseline and is NEVER updated
               again -- no behavioural monitoring, no anomaly detection,
               no dynamic trust engine at all. This is the conventional
               (non-Zero-Trust) alternative the thesis argues against.

  rule_only  - the dynamic trust update uses the already-computed weighted
               Behavioural Risk score (R_behav, Eq. 4.4) directly in place
               of the Isolation Forest anomaly evidence A(t). Isolation
               Forest is not fitted or used at all. This tests whether the
               ML-based anomaly detector actually adds value over the
               simpler rule-based behavioural-risk signal the architecture
               already computes.

  rf_supervised - Isolation Forest is replaced with a supervised Random
               Forest classifier that plays exactly the same role: it
               produces A(t) in [0,1], which feeds the *same* dynamic
               trust update (Eq. 4.7) and the *same* ALLOW / RE_AUTH /
               DENY thresholds. Unlike Isolation Forest, Random Forest
               needs labelled attack examples, so it is trained ONCE on a
               separate set of seeds (TRAIN_SEEDS, disjoint from the
               41-80 evaluation range used everywhere else in the thesis)
               before being evaluated on the same 40-seed baseline as
               every other configuration. This keeps the comparison fair:
               the supervised model is not trained and tested on the same
               attack instances.

Both ablations and the supervised comparison reuse Device,
build_population(), and compute_metrics() from trust_engine_sim.py
unchanged, so device population, traffic generation, attack injection,
and metric definitions are identical to the main results -- only the
trust-update / anomaly-scoring logic differs.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestClassifier

import trust_engine_sim as base
from trust_engine_sim import (
    build_population, compute_metrics, adaptive_minmax, BASELINE_STEPS, TOTAL_STEPS,
    TRUST_COMPONENT_WEIGHTS, BEHAVIOR_WEIGHTS, ISOLATION_FOREST_PARAMS,
    TRUST_THRESHOLDS,
)

TRAIN_SEEDS = range(1, 41)  # disjoint from the 41-80 evaluation range used everywhere else


def _compute_R(devices, batch_df):
    """Shared helper: per-device normalized risk vector [R_RF,R_IAT,R_PS,R_PU,R_DV]
    against each device's own frozen baseline (Eq. 4.2-4.3), plus behavioral
    trust (Eq. 4.4-4.5). Identical logic to trust_engine_sim.run_simulation."""
    w = BEHAVIOR_WEIGHTS
    behav_trust, r_behav_raw, R_rows = {}, {}, []
    for d in devices:
        row = batch_df.loc[d.id]
        br = d.baseline_range if d.baseline_range is not None else adaptive_minmax(batch_df)
        r_rf = np.clip((row["RF"] - br["RF"][0]) / (br["RF"][1] - br["RF"][0]), 0, 1)
        r_iat = np.clip(1 - (row["IAT"] - br["IAT"][0]) / (br["IAT"][1] - br["IAT"][0]), 0, 1)
        r_ps = np.clip((row["PS"] - br["PS"][0]) / (br["PS"][1] - br["PS"][0]), 0, 1)
        r_pu = np.clip((row["PU"] - br["PU"][0]) / (br["PU"][1] - br["PU"][0]), 0, 1)
        r_dv = np.clip((row["DV"] - br["DV"][0]) / (br["DV"][1] - br["DV"][0]), 0, 1)
        r_behav_local = (w["RF"] * r_rf + w["IAT"] * r_iat + w["PS"] * r_ps +
                          w["PU"] * r_pu + w["DV"] * r_dv)
        behav_trust[d.id] = 1 - r_behav_local
        r_behav_raw[d.id] = r_behav_local
        R_rows.append([r_rf, r_iat, r_ps, r_pu, r_dv])
    return np.array(R_rows), behav_trust, r_behav_raw


def collect_training_data(seeds=TRAIN_SEEDS):
    """Run the simulator on TRAIN_SEEDS purely to harvest labelled
    (feature_vector, is_attack_now) pairs for the Random Forest. Only the
    post-baseline cycles are used, matching when A(t) is actually consulted
    in the real trust update."""
    X_rows, y_rows = [], []
    for seed in seeds:
        base.RNG = np.random.default_rng(seed)
        devices = build_population()
        for t in range(TOTAL_STEPS):
            feats = []
            for d in devices:
                f = d.sample_features(t)
                f["device_id"] = d.id
                feats.append(f)
            batch_df = pd.DataFrame(feats).set_index("device_id")

            if t < BASELINE_STEPS:
                for d in devices:
                    d.record_baseline(batch_df.loc[d.id].to_dict())
            if t == BASELINE_STEPS - 1:
                for d in devices:
                    d.freeze_baseline()
            if t < BASELINE_STEPS:
                continue  # A(t) is only ever consulted after baseline

            R, _, _ = _compute_R(devices, batch_df)
            for i, d in enumerate(devices):
                X_rows.append(R[i])
                y_rows.append(1 if d.in_attack_window(t) else 0)

    return np.array(X_rows), np.array(y_rows)


def train_rf(seeds=TRAIN_SEEDS, random_state=0):
    X, y = collect_training_data(seeds)
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced",
        random_state=random_state, n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def run_variant(mode, alpha=0.7, beta=0.3, seed=42, rf_model=None, verbose=False):
    """mode in {'full', 'auth_only', 'rule_only', 'rf_supervised'}.

    'full' reproduces trust_engine_sim.run_simulation() exactly (kept here
    only as a same-file cross-check baseline for the paired comparison).
    rf_supervised requires a pre-trained rf_model (see train_rf()).
    """
    assert mode in ("full", "auth_only", "rule_only", "rf_supervised")
    if mode == "rf_supervised":
        assert rf_model is not None, "rf_supervised needs a pre-trained rf_model"

    base.RNG = np.random.default_rng(seed)
    high_thr = TRUST_THRESHOLDS["high"]
    med_thr = TRUST_THRESHOLDS["medium"]

    devices = build_population()
    iso_forest = None
    log_rows = []

    for t in range(TOTAL_STEPS):
        feats = []
        for d in devices:
            f = d.sample_features(t)
            f["device_id"] = d.id
            feats.append(f)
        batch_df = pd.DataFrame(feats).set_index("device_id")

        if t < BASELINE_STEPS:
            for d in devices:
                d.record_baseline(batch_df.loc[d.id].to_dict())
        if t == BASELINE_STEPS - 1:
            for d in devices:
                d.freeze_baseline()

        R, behav_trust, r_behav_raw = _compute_R(devices, batch_df)

        if mode == "full":
            if t == BASELINE_STEPS - 1:
                iso_forest = IsolationForest(**ISOLATION_FOREST_PARAMS).fit(R)
                base_s = -iso_forest.score_samples(R)
                iso_forest.mu_b = float(base_s.mean())
                iso_forest.sigma_b = float(base_s.std()) + 1e-9
            if iso_forest is not None:
                s_xn = -iso_forest.score_samples(R)
                a_scores = np.clip((s_xn - iso_forest.mu_b) / (3 * iso_forest.sigma_b), 0, 1)
                A = dict(zip(batch_df.index, a_scores))
            else:
                A = {d.id: 0.0 for d in devices}
        elif mode == "rf_supervised" and t >= BASELINE_STEPS:
            probs = rf_model.predict_proba(R)[:, 1]
            A = dict(zip(batch_df.index, probs))
        else:
            A = {d.id: 0.0 for d in devices}  # unused in auth_only / rule_only / RF baseline phase

        for d in devices:
            s_auth = d.auth_score()
            s_behav = behav_trust[d.id]
            t_total = (TRUST_COMPONENT_WEIGHTS["auth"] * s_auth +
                       TRUST_COMPONENT_WEIGHTS["behav"] * s_behav)

            if t < BASELINE_STEPS:
                ts_new = t_total * 100
            elif mode in ("full", "rf_supervised"):
                ts_prev = d.trust_score / 100.0
                ts_new_frac = alpha * ts_prev + beta * (1 - A[d.id])
                ts_new = float(np.clip(ts_new_frac, 0, 1)) * 100
            elif mode == "auth_only":
                # No re-evaluation after authentication: trust is frozen at
                # its baseline-derived value for the rest of the run.
                ts_new = d.trust_score
            elif mode == "rule_only":
                ts_prev = d.trust_score / 100.0
                ts_new_frac = alpha * ts_prev + beta * (1 - r_behav_raw[d.id])
                ts_new = float(np.clip(ts_new_frac, 0, 1)) * 100

            d.trust_score = ts_new

            if ts_new >= high_thr:
                decision = "ALLOW"
            elif ts_new >= med_thr:
                decision = "RE_AUTHENTICATE"
            else:
                decision = "DENY_ISOLATE"
                if d.is_malicious and d.in_attack_window(t) and d.first_low_trust_step is None:
                    d.first_low_trust_step = t

            log_rows.append(dict(t=t, device_id=d.id, decision=decision))

    log_df = pd.DataFrame(log_rows)
    metrics = compute_metrics(log_df, devices)
    if verbose:
        base.print_metrics(metrics)
    return metrics


def run_ablation_comparison(seeds=range(41, 81), alpha=0.7, beta=0.3, rf_model=None, modes=("full", "auth_only", "rule_only")):
    rows = []
    for seed in seeds:
        for mode in modes:
            m = run_variant(mode, alpha=alpha, beta=beta, seed=seed, rf_model=rf_model, verbose=False)
            rows.append(dict(
                seed=seed, mode=mode,
                detection_rate=m["detection_rate"] * 100,
                fpr=m["false_positive_rate"] * 100,
                f1=m["f1_score"],
                latency=m["mean_detection_latency_steps"],
            ))
    return pd.DataFrame(rows)


def summarize(df):
    summary = df.groupby("mode")[["detection_rate", "fpr", "f1"]].agg(["mean", "std"])
    print(summary)
    print()

    full = df[df["mode"] == "full"].sort_values("seed")
    other_modes = [m for m in df["mode"].unique() if m != "full"]
    for other_mode in other_modes:
        other = df[df["mode"] == other_mode].sort_values("seed")
        print(f"--- full vs {other_mode} (paired, n={len(full)} seeds) ---")
        for metric in ("detection_rate", "fpr", "f1"):
            a = full[metric].values
            b = other[metric].values
            stat, p = stats.wilcoxon(a, b)
            print(f"  {metric:16s} full_mean={a.mean():7.3f}  {other_mode}_mean={b.mean():7.3f}  "
                  f"Wilcoxon p={p:.2e}")
        print()


if __name__ == "__main__":
    df = run_ablation_comparison()
    df.to_csv("results/ablation_results.csv", index=False)
    summarize(df)


# ---------------------------------------------------------------------------
# Leave-one-attack-type-out generalization test for the supervised RF
# comparison. Trains RF with one attack type withheld entirely from the
# labelled training set, then evaluates specifically on that held-out
# attack type in the (disjoint) evaluation seeds. This tests whether RF's
# near-perfect performance reflects genuine anomaly generalization or
# memorization of the four specific attack signatures it was trained on --
# something Isolation Forest, being unsupervised, is not exposed to in
# either direction.
# ---------------------------------------------------------------------------

ATTACK_TYPE_LIST = ["ddos_flooding", "data_exfiltration", "botnet_cnc", "port_scanning"]


def collect_training_data_labeled(seeds=TRAIN_SEEDS, normal_subsample=100_000, random_state=0):
    """Like collect_training_data, but also records each row's attack_type
    (None for normal traffic) so a type can be excluded per held-out run.
    Subsamples normal (majority-class) rows for training speed; this is a
    standard imbalanced-learning practice and does not change what's being
    tested, only how fast four models can be trained."""
    rs = np.random.RandomState(random_state)
    X_rows, y_rows, atype_rows = [], [], []
    for seed in seeds:
        base.RNG = np.random.default_rng(seed)
        devices = build_population()
        for t in range(TOTAL_STEPS):
            feats = []
            for d in devices:
                f = d.sample_features(t)
                f["device_id"] = d.id
                feats.append(f)
            batch_df = pd.DataFrame(feats).set_index("device_id")
            if t < BASELINE_STEPS:
                for d in devices:
                    d.record_baseline(batch_df.loc[d.id].to_dict())
            if t == BASELINE_STEPS - 1:
                for d in devices:
                    d.freeze_baseline()
            if t < BASELINE_STEPS:
                continue
            R, _, _ = _compute_R(devices, batch_df)
            for i, d in enumerate(devices):
                attacking = d.in_attack_window(t)
                X_rows.append(R[i])
                y_rows.append(1 if attacking else 0)
                atype_rows.append(d.attack_type if attacking else None)

    X, y, atype = np.array(X_rows), np.array(y_rows), np.array(atype_rows, dtype=object)
    normal_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]
    if len(normal_idx) > normal_subsample:
        normal_idx = rs.choice(normal_idx, size=normal_subsample, replace=False)
    keep = np.concatenate([normal_idx, attack_idx])
    return X[keep], y[keep], atype[keep]


def train_rf_holdout(X, y, atype, held_out_type, random_state=0):
    """Train on everything except rows whose attack_type == held_out_type."""
    mask = atype != held_out_type
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced",
        random_state=random_state, n_jobs=-1,
    )
    clf.fit(X[mask], y[mask])
    return clf


def run_holdout_eval(rf_model, seeds, alpha=0.7, beta=0.3):
    """Evaluate a held-out-trained RF model across seeds, returning
    per-attack-type detection rates (via compute_metrics) for each seed."""
    rows = []
    for seed in seeds:
        m = run_variant("rf_supervised", alpha=alpha, beta=beta, seed=seed,
                         rf_model=rf_model, verbose=False)
        row = dict(seed=seed, detection_rate=m["detection_rate"] * 100,
                   fpr=m["false_positive_rate"] * 100, f1=m["f1_score"])
        for atype, stats_ in m["per_attack_type"].items():
            row[f"rate_{atype}"] = (stats_["rate"] * 100) if stats_["rate"] is not None else None
        rows.append(row)
    return pd.DataFrame(rows)
