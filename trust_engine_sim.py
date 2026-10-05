"""
trust_engine_sim.py
--------------------
Reference implementation of the Hybrid Zero Trust IoT simulation described
in the revised Chapter 3 (Methodology): FAST-SM9-authenticated devices,
gateway-side behavioral feature extraction, adaptive Behavior Risk
Assessment, cloud-offloaded Isolation Forest anomaly detection, and the
dynamic trust update Ts(t) = alpha*Ts(t-1) + beta*(1-A(t)).

Author: (thesis simulation module)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Configuration (mirrors Feature_Ranges_Reference.py / Chapter 3 Table 3.x)
# ---------------------------------------------------------------------------

TRUST_THRESHOLDS = {"high": 70, "medium": 40}         # Section 3.7.1
DYNAMIC_UPDATE = {"alpha": 0.7, "beta": 0.3}           # Eq. 3.7
BEHAVIOR_WEIGHTS = {                                    # Table 3.2 / Eq. 3.4
    "RF": 0.30, "IAT": 0.25, "PS": 0.20, "PU": 0.15, "DV": 0.10
}
TRUST_COMPONENT_WEIGHTS = {"auth": 0.4, "behav": 0.6}  # Eq. 3.6
AUTH_WEIGHTS = {"verify": 0.60, "history": 0.25, "freshness": 0.15}  # Eq. 3.1

ISOLATION_FOREST_PARAMS = dict(
    n_estimators=100, max_samples=256, contamination=0.05,
    bootstrap=False, random_state=42,
)

N_TOTAL_DEVICES = 100
N_MALICIOUS = 10
N_NORMAL = N_TOTAL_DEVICES - N_MALICIOUS

BASELINE_STEPS = 20      # Phase 1: normal-only baseline collection
ATTACK_STEPS = 180       # Phase 2: normal operation + attack injection
TOTAL_STEPS = BASELINE_STEPS + ATTACK_STEPS
T0 = 90.0                # initial trust score (0-100 scale)

# Normal device-type profiles (proportional down-scale of the 500-device
# design: 200/100/50/100/50 -> 36/18/9/18/9, summing to 90)
DEVICE_TYPES = {
    "temperature_sensor": dict(count=36, rf=(0.02, 0.04), ps=(60, 68),
                                dv=(400, 900), pu=1, iat=(28, 32)),
    "smart_bulb":          dict(count=18, rf=(0.0004, 0.0007), ps=(120, 136),
                                dv=(300, 800), pu=1, iat=(1700, 1900)),
    "security_camera":     dict(count=9,  rf=(28, 32), ps=(900, 1150),
                                dv=(30000, 45000), pu=1, iat=(0.03, 0.04)),
    "heart_rate_monitor":  dict(count=18, rf=(0.18, 0.22), ps=(60, 68),
                                dv=(600, 1400), pu=1, iat=(4.5, 5.5)),
    "door_sensor":         dict(count=9,  rf=(0.00002, 0.00004), ps=(28, 36),
                                dv=(50, 150), pu=1, iat=(3500, 4000)),
}
assert sum(v["count"] for v in DEVICE_TYPES.values()) == N_NORMAL

# Malicious device attack-type assignment (10 devices across 4 classes,
# per Threat Model Section 5 / Table 3.3)
ATTACK_TYPES = {
    "ddos_flooding":      dict(count=3, host_type="temperature_sensor"),
    "data_exfiltration":  dict(count=3, host_type="heart_rate_monitor"),
    "botnet_cnc":         dict(count=2, host_type="security_camera"),
    "port_scanning":      dict(count=2, host_type="smart_bulb"),
}
assert sum(v["count"] for v in ATTACK_TYPES.values()) == N_MALICIOUS


# ---------------------------------------------------------------------------
# 2. Device model
# ---------------------------------------------------------------------------

class Device:
    def __init__(self, device_id, is_malicious, device_type, attack_type=None):
        self.id = device_id
        self.is_malicious = is_malicious
        self.device_type = device_type
        self.attack_type = attack_type
        self.trust_score = T0
        self.auth_history = [1] * 10   # rolling success history buffer
        self.attack_start = None
        self.attack_end = None
        self.first_low_trust_step = None   # for detection latency

        if is_malicious:
            # attack window randomized within Phase 2
            dur = RNG.integers(20, 60)
            start = RNG.integers(BASELINE_STEPS + 5, TOTAL_STEPS - dur - 1)
            self.attack_start = int(start)
            self.attack_end = int(start + dur)

        # per-device adaptive baseline (populated during Phase 1, Section 3.8.7 Phase 1)
        self.baseline_history = {c: [] for c in ["RF", "PS", "DV", "PU", "IAT"]}
        self.baseline_range = None  # dict[col] = (min, max), frozen after Phase 1

    def in_attack_window(self, t):
        return self.is_malicious and self.attack_start <= t < self.attack_end

    # -- normal (benign) feature sampling --------------------------------
    def _sample_normal(self):
        p = DEVICE_TYPES[self.device_type]
        rf = RNG.uniform(*p["rf"])
        ps = RNG.uniform(*p["ps"])
        dv = RNG.uniform(*p["dv"])
        pu = p["pu"]
        iat = RNG.uniform(*p["iat"])
        return dict(RF=rf, PS=ps, DV=dv, PU=pu, IAT=iat)

    # -- attack feature sampling ------------------------------------------
    def _sample_attack(self):
        base = self._sample_normal()
        a = self.attack_type
        if a == "ddos_flooding":
            base.update(RF=RNG.uniform(500, 2000),
                        IAT=RNG.uniform(0.005, 0.05),
                        DV=RNG.uniform(1e7, 5e7))
        elif a == "data_exfiltration":
            base.update(PS=RNG.uniform(10240, 51200),
                        DV=RNG.uniform(1e5, 1e6),
                        PU=2)
        elif a == "botnet_cnc":
            # Threat model: device continues normal traffic but ALSO beacons
            # periodically to an additional, non-standard external port.
            base.update(PU=2, RF=base["RF"] + RNG.uniform(0.01, 0.02))
        elif a == "port_scanning":
            base.update(PU=RNG.integers(5, 20),
                        RF=RNG.uniform(50, 150),
                        IAT=RNG.uniform(0.3, 1.0))
        return base

    def sample_features(self, t):
        if self.in_attack_window(t):
            return self._sample_attack()
        return self._sample_normal()

    def record_baseline(self, feats):
        for c in self.baseline_history:
            self.baseline_history[c].append(feats[c])

    def freeze_baseline(self, margin=0.15):
        """Freeze the adaptive per-device normalization range at the end of
        Phase 1 (Section 3.8.7, Phase 1), with a small margin so that normal
        variance beyond the observed baseline window doesn't saturate R=1."""
        self.baseline_range = {}
        for c, vals in self.baseline_history.items():
            vmin, vmax = min(vals), max(vals)
            span = (vmax - vmin) if (vmax - vmin) > 1e-9 else max(vmax, 1e-9) * 0.1
            self.baseline_range[c] = (vmin - margin * span, vmax + margin * span)

    # -- authentication trust (Eq. 3.1) -----------------------------------
    def auth_score(self):
        # Devices are already authenticated at session start in this sim;
        # occasional freshness jitter models real re-auth variability.
        s_verify = 1
        s_history = float(np.mean(self.auth_history))
        s_freshness = float(np.clip(RNG.normal(0.97, 0.02), 0.8, 1.0))
        s_auth = (AUTH_WEIGHTS["verify"] * s_verify +
                  AUTH_WEIGHTS["history"] * s_history +
                  AUTH_WEIGHTS["freshness"] * s_freshness)
        return s_auth


def build_population():
    devices = []
    did = 0
    for dtype, spec in DEVICE_TYPES.items():
        for _ in range(spec["count"]):
            devices.append(Device(did, False, dtype))
            did += 1
    for atype, spec in ATTACK_TYPES.items():
        for _ in range(spec["count"]):
            devices.append(Device(did, True, spec["host_type"], attack_type=atype))
            did += 1
    RNG.shuffle(devices)
    return devices


# ---------------------------------------------------------------------------
# 3. Behavior Risk Assessment (Section 3.4.3 / 3.5.2, Eq. 3.2-3.5)
# ---------------------------------------------------------------------------

def adaptive_minmax(batch_df):
    """Compute per-feature (min,max) from the current batch (Eq. 3.2/3.3)."""
    ranges = {}
    for col in ["RF", "PS", "DV", "PU", "IAT"]:
        vmin, vmax = batch_df[col].min(), batch_df[col].max()
        if vmax - vmin < 1e-9:
            vmax = vmin + 1e-9
        ranges[col] = (vmin, vmax)
    return ranges


def behavior_risk(row, ranges):
    r = {}
    for col in ["RF", "PS", "DV", "PU"]:
        vmin, vmax = ranges[col]
        r[col] = np.clip((row[col] - vmin) / (vmax - vmin), 0, 1)
    vmin, vmax = ranges["IAT"]
    r["IAT"] = np.clip(1 - (row["IAT"] - vmin) / (vmax - vmin), 0, 1)

    r_behav = (BEHAVIOR_WEIGHTS["RF"] * r["RF"] +
               BEHAVIOR_WEIGHTS["IAT"] * r["IAT"] +
               BEHAVIOR_WEIGHTS["PS"] * r["PS"] +
               BEHAVIOR_WEIGHTS["PU"] * r["PU"] +
               BEHAVIOR_WEIGHTS["DV"] * r["DV"])
    return r_behav


# ---------------------------------------------------------------------------
# 4. Simulation loop
# ---------------------------------------------------------------------------

def run_simulation(alpha=None, beta=None, high_thr=None, med_thr=None, weights=None,
                    calibrated=True, verbose=True, seed=42):
    global RNG
    RNG = np.random.default_rng(seed)  # fixed population/traffic draw across sweep runs
    alpha = DYNAMIC_UPDATE["alpha"] if alpha is None else alpha
    beta = DYNAMIC_UPDATE["beta"] if beta is None else beta
    high_thr = TRUST_THRESHOLDS["high"] if high_thr is None else high_thr
    med_thr = TRUST_THRESHOLDS["medium"] if med_thr is None else med_thr
    w = BEHAVIOR_WEIGHTS if weights is None else weights

    devices = build_population()
    iso_forest = None
    log_rows = []

    for t in range(TOTAL_STEPS):
        # 1. sample features for every device this step
        feats = []
        for d in devices:
            f = d.sample_features(t)
            f["device_id"] = d.id
            feats.append(f)
        batch_df = pd.DataFrame(feats).set_index("device_id")

        # Phase 1: record each device's own behavior to build its adaptive baseline
        if t < BASELINE_STEPS:
            for d in devices:
                d.record_baseline(batch_df.loc[d.id].to_dict())
        if t == BASELINE_STEPS - 1:
            for d in devices:
                d.freeze_baseline()

        # 2. adaptive normalization (Eq 3.2-3.3) against EACH DEVICE'S OWN frozen
        # baseline range, + Behavior Risk / Behavioral Trust (Eq 3.4-3.5)
        behav_trust = {}
        R_rows = []
        for d in devices:
            row = batch_df.loc[d.id]
            rng = d.baseline_range if d.baseline_range is not None else adaptive_minmax(batch_df)
            if isinstance(rng, dict) and "RF" in rng and isinstance(rng["RF"], tuple):
                br = rng
            else:
                br = rng  # adaptive_minmax already returns the same shape
            r_rf = np.clip((row["RF"] - br["RF"][0]) / (br["RF"][1] - br["RF"][0]), 0, 1)
            r_iat = np.clip(1 - (row["IAT"] - br["IAT"][0]) / (br["IAT"][1] - br["IAT"][0]), 0, 1)
            r_ps = np.clip((row["PS"] - br["PS"][0]) / (br["PS"][1] - br["PS"][0]), 0, 1)
            r_pu = np.clip((row["PU"] - br["PU"][0]) / (br["PU"][1] - br["PU"][0]), 0, 1)
            r_dv = np.clip((row["DV"] - br["DV"][0]) / (br["DV"][1] - br["DV"][0]), 0, 1)

            r_behav_local = (w["RF"] * r_rf + w["IAT"] * r_iat + w["PS"] * r_ps +
                              w["PU"] * r_pu + w["DV"] * r_dv)
            behav_trust[d.id] = 1 - r_behav_local
            R_rows.append([r_rf, r_iat, r_ps, r_pu, r_dv])

        R = np.array(R_rows)

        # 3. Isolation Forest anomaly scoring (cloud, Section 3.6). Input is the
        # normalized per-device risk vector [R_RF,R_IAT,R_PS,R_PU,R_DV] so all
        # devices/features are on the same 0-1 scale regardless of device type.
        if t == BASELINE_STEPS - 1:
            iso_forest = IsolationForest(**ISOLATION_FOREST_PARAMS).fit(R)
            # calibrate A(t) against the trained baseline's OWN score distribution
            # (mu_b, sigma_b) rather than assuming the literature's fixed 0.5
            # midpoint, which compresses toward 0.5-0.65 for small forests (n~100)
            # even for clear outliers -- see Chapter 3.6 / sensitivity-analysis note.
            base_s = -iso_forest.score_samples(R)
            iso_forest.mu_b = float(base_s.mean())
            iso_forest.sigma_b = float(base_s.std()) + 1e-9
        if iso_forest is not None:
            s_xn = -iso_forest.score_samples(R)
            if calibrated:
                a_scores = np.clip((s_xn - iso_forest.mu_b) / (3 * iso_forest.sigma_b), 0, 1)
            else:
                a_scores = np.clip((s_xn - 0.5) * 2, 0, 1)      # Eq. 3.8, literal form
            A = dict(zip(batch_df.index, a_scores))
        else:
            A = {d.id: 0.0 for d in devices}  # no model yet during baseline warm-up

        # 4. trust score computation + dynamic update + access decision
        for d in devices:
            s_auth = d.auth_score()
            s_behav = behav_trust[d.id]
            t_total = (TRUST_COMPONENT_WEIGHTS["auth"] * s_auth +
                       TRUST_COMPONENT_WEIGHTS["behav"] * s_behav)  # Eq 3.6 (0-1)

            a_t = A[d.id]
            if t < BASELINE_STEPS:
                # Phase 1: seed trust directly from static score, no AI feedback yet
                ts_new = t_total * 100
            else:
                ts_prev = d.trust_score / 100.0
                ts_new_frac = alpha * ts_prev + beta * (1 - a_t)   # Eq 3.7
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

            log_rows.append(dict(
                t=t, device_id=d.id, is_malicious=d.is_malicious,
                device_type=d.device_type, attack_type=d.attack_type,
                in_attack=d.in_attack_window(t),
                s_auth=s_auth, s_behav=s_behav, t_total=t_total,
                anomaly_A=a_t, trust_score=ts_new, decision=decision,
            ))

    log_df = pd.DataFrame(log_rows)
    metrics = compute_metrics(log_df, devices)
    if verbose:
        print_metrics(metrics)
    return log_df, devices, metrics


# ---------------------------------------------------------------------------
# 5. Metrics (Section 3.8.5)
# ---------------------------------------------------------------------------

def compute_metrics(log_df, devices):
    malicious = [d for d in devices if d.is_malicious]
    normal = [d for d in devices if not d.is_malicious]

    detected = [d for d in malicious if d.first_low_trust_step is not None]
    detection_rate = len(detected) / len(malicious)

    latencies = [d.first_low_trust_step - d.attack_start for d in detected]
    mean_latency = float(np.mean(latencies)) if latencies else None

    # False positive: a NORMAL device ever denied/isolated
    fp_devices = []
    for d in normal:
        sub = log_df[(log_df.device_id == d.id) & (log_df.t >= BASELINE_STEPS)]
        if (sub.decision == "DENY_ISOLATE").any():
            fp_devices.append(d)
    fpr = len(fp_devices) / len(normal)

    fn = len(malicious) - len(detected)
    fnr = fn / len(malicious)

    precision = len(detected) / (len(detected) + len(fp_devices)) if (len(detected) + len(fp_devices)) > 0 else float("nan")
    recall = detection_rate
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")

    per_attack = {}
    for atype in ATTACK_TYPES:
        subset = [d for d in malicious if d.attack_type == atype]
        det = [d for d in subset if d.first_low_trust_step is not None]
        per_attack[atype] = dict(n=len(subset), detected=len(det),
                                  rate=len(det) / len(subset) if subset else None)

    return dict(
        detection_rate=detection_rate, mean_detection_latency_steps=mean_latency,
        false_positive_rate=fpr, false_negative_rate=fnr,
        precision=precision, recall=recall, f1_score=f1,
        n_malicious=len(malicious), n_normal=len(normal),
        n_detected=len(detected), n_false_positive=len(fp_devices),
        per_attack_type=per_attack,
    )


def print_metrics(m):
    print("=" * 60)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Detection Rate:            {m['detection_rate']*100:.1f}%  "
          f"({m['n_detected']}/{m['n_malicious']} malicious devices)")
    if m["mean_detection_latency_steps"] is not None:
        print(f"Mean Detection Latency:    {m['mean_detection_latency_steps']:.2f} monitoring cycles")
    print(f"False Positive Rate:       {m['false_positive_rate']*100:.1f}%  "
          f"({m['n_false_positive']}/{m['n_normal']} normal devices)")
    print(f"False Negative Rate:       {m['false_negative_rate']*100:.1f}%")
    print(f"Precision:                 {m['precision']:.3f}")
    print(f"Recall:                    {m['recall']:.3f}")
    print(f"F1-score:                  {m['f1_score']:.3f}")
    print("-" * 60)
    print("Per-attack-type detection rate:")
    for atype, r in m["per_attack_type"].items():
        rate = f"{r['rate']*100:.0f}%" if r["rate"] is not None else "n/a"
        print(f"  {atype:20s} {r['detected']}/{r['n']}  ({rate})")
    print("=" * 60)


if __name__ == "__main__":
    run_simulation()
