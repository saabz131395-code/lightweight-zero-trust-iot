"""Pre-processing, trust engine and two-stage detection on the captured testbed data.

Pipeline (per run):
  packets (SQLite)  ->  per-device, per-cycle features RF, IAT, PS, PU, DV
  -> per-device baseline normalisation (cycles 0-19, 15 % margin)
  -> behavioural risk / trust (Eqs. 1-3)  -> Isolation Forest anomaly score A(t)
  -> dynamic trust DT(t) = a*DT(t-1) + b*(1-A(t))  -> ALLOW / RE-AUTH / DENY
Stage 2: a Random Forest assigns an attack type to suspicious device-cycles.
"""
import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier

FEATS = ["RF", "IAT", "PS", "PU", "DV"]
W = {"RF": 0.30, "IAT": 0.25, "PS": 0.20, "PU": 0.15, "DV": 0.10}
AUTHW = {"verify": 0.60, "history": 0.25, "fresh": 0.15}
IF_PARAMS = dict(n_estimators=100, max_samples=256, contamination=0.05, random_state=42)
HIGH, MED = 70, 40
BASE = 20
IAT_CAP = 30.0


# ------------------------------------------------------------------ features
def load_meta(con):
    dev = pd.read_sql("SELECT * FROM devices", con)
    runs = pd.read_sql("SELECT * FROM runs", con)
    att = pd.read_sql("SELECT * FROM attacks", con)
    return dev, runs, att


def extract_features(con, run_id, dev, total_cycles, cycle_s=1.0):
    """Turn captured packets into one feature vector per device per cycle.

    Only packets a device originates count: packets whose source is the device,
    excluding TCP resets (replies to someone else's scan) and pure ACKs.
      RF  requests per cycle (packets carrying payload, or connection attempts = SYN)
      IAT mean gap (s) between consecutive requests; if no request in the cycle,
          the time since the last request (capped at 30 s)
      PS  mean payload size (bytes) of payload-carrying packets (0 if none)
      PU  number of distinct destination ports contacted
      DV  total bytes (IP length) sent in the cycle
    """
    q = """SELECT src_ip, t_rel, dst_port, ip_len, payload_len, tcp_flags, proto
           FROM packets WHERE run_id=? AND src_ip LIKE '127.0.1.%' ORDER BY t_rel"""
    p = pd.read_sql(q, con, params=(run_id,))
    p = p[(p.t_rel >= 0) & (p.t_rel < total_cycles * cycle_s)]
    p = p[~p.tcp_flags.str.contains("R")]
    p = p[~((p.proto == "TCP") & (p.payload_len == 0) & (p.tcp_flags == "A"))]
    p["cycle"] = (p.t_rel // cycle_s).astype(int)
    p["req"] = (p.payload_len > 0) | (p.tcp_flags == "S")
    ip2id = dict(zip(dev.ip, dev.device_id))
    p["device_id"] = p.src_ip.map(ip2id)
    rows = []
    for did, g in p.groupby("device_id"):
        r = g[g.req]
        gaps = r.t_rel.diff()
        r = r.assign(gap=gaps)
        last = None
        by_c = {c: x for c, x in r.groupby("cycle")}
        dv = g.groupby("cycle").ip_len.sum()
        for c in range(total_cycles):
            x = by_c.get(c)
            if x is not None and len(x):
                gp = x.gap.dropna()
                if last is not None and np.isnan(x.gap.iloc[0]):
                    gp = pd.concat([pd.Series([x.t_rel.iloc[0] - last]), gp])
                iat = float(gp.mean()) if len(gp) else min(IAT_CAP, (c + 1) * cycle_s)
                pl = x[x.payload_len > 0].payload_len
                rows.append((did, c, len(x), min(iat, IAT_CAP), float(pl.mean()) if len(pl) else 0.0,
                             x.dst_port.nunique(), float(dv.get(c, 0))))
                last = x.t_rel.iloc[-1]
            else:
                since = ((c + 1) * cycle_s - last) if last is not None else (c + 1) * cycle_s
                rows.append((did, c, 0, min(since, IAT_CAP), 0.0, 0, float(dv.get(c, 0))))
    f = pd.DataFrame(rows, columns=["device_id", "cycle"] + FEATS)
    # devices that sent nothing at all still get rows
    full = pd.MultiIndex.from_product([dev.device_id, range(total_cycles)], names=["device_id", "cycle"])
    f = f.set_index(["device_id", "cycle"]).reindex(full).reset_index()
    f[["RF", "PS", "PU", "DV"]] = f[["RF", "PS", "PU", "DV"]].fillna(0)
    f["IAT"] = f["IAT"].fillna(IAT_CAP)
    f["run_id"] = run_id
    return f


def label(f, att, run_id):
    f = f.copy()
    f["in_attack"] = False
    f["attack_type"] = "Normal"
    for _, a in att[att.run_id == run_id].iterrows():
        m = (f.device_id == a.device_id) & (f.cycle >= a.start_cycle) & (f.cycle < a.end_cycle)
        f.loc[m, "in_attack"] = True
        f.loc[m, "attack_type"] = a.attack_type
    return f


# ------------------------------------------------------------------ engine
def normalise(f, margin=0.15):
    """Per-device min-max against the frozen baseline (cycles 0..BASE-1)."""
    out = f.copy()
    b = f[f.cycle < BASE].groupby("device_id")[FEATS]
    lo, hi = b.min(), b.max()
    span = (hi - lo)
    small = span < 1e-9
    span = span.where(~small, np.maximum(hi.abs(), 1e-9) * 0.1)
    lo2, hi2 = lo - margin * span, hi + margin * span
    for c in FEATS:
        L = out.device_id.map(lo2[c]); H = out.device_id.map(hi2[c])
        v = ((out[c] - L) / (H - L)).clip(0, 1)
        out["r_" + c] = (1 - v) if c == "IAT" else v
    return out


def run_engine(fn, alpha, seed=0, if_fit="all", mode="full", if_input="minmax", calib="global"):
    """fn: normalised features of one run. Returns per device-cycle trust log.

    mode: full (Isolation Forest evidence), rule (R_behav used as the anomaly
    evidence, no Isolation Forest), auth (authenticate once, trust never updated).
    """
    rng = np.random.default_rng(seed)
    R = if_matrix(fn, if_input)
    rb = sum(W[c] * fn["r_" + c].to_numpy() for c in FEATS)
    fn = fn.assign(r_behav=rb)
    base_mask = (fn.cycle < BASE) if if_fit == "all" else (fn.cycle == BASE - 1)
    iso = IsolationForest(**IF_PARAMS).fit(R[base_mask.to_numpy()])
    s_all = -iso.score_samples(R)
    bm = base_mask.to_numpy()
    if calib == "device":
        sb = pd.Series(s_all[bm]).groupby(fn.device_id.to_numpy()[bm])
        mu = fn.device_id.map(sb.mean()).to_numpy(); sd = fn.device_id.map(sb.std()).to_numpy() + 1e-9
    else:
        mu, sd = s_all[bm].mean(), s_all[bm].std() + 1e-9
    A = np.clip((s_all - mu) / (3 * sd), 0, 1)
    fn = fn.assign(A=A)
    fn = fn.sort_values(["device_id", "cycle"])
    out = []
    for did, g in fn.groupby("device_id"):
        ts = None
        for row in g.itertuples():
            s_auth = AUTHW["verify"] + AUTHW["history"] + AUTHW["fresh"] * float(np.clip(rng.normal(0.97, 0.02), 0.8, 1.0))
            t_total = 0.4 * s_auth + 0.6 * (1 - row.r_behav)
            if row.cycle < BASE:
                ts = t_total * 100
            elif mode == "auth":
                pass
            elif mode == "rule":
                ts = float(np.clip(alpha * ts / 100 + (1 - alpha) * (1 - row.r_behav), 0, 1)) * 100
            else:
                ts = float(np.clip(alpha * ts / 100 + (1 - alpha) * (1 - row.A), 0, 1)) * 100
            out.append(ts)
    fn["trust"] = out
    fn["decision"] = np.where(fn.trust >= HIGH, "ALLOW", np.where(fn.trust >= MED, "RE-AUTH", "DENY"))
    return fn


def if_matrix(fn, if_input):
    """Input representation for Isolation Forest.
    minmax : per-device min-max risk vector (thesis design)
    logdev : log-scaled raw features plus each feature's deviation from the
             device's own baseline mean (log scale), unclipped
    """
    if if_input == "minmax":
        return fn[["r_" + c for c in FEATS]].to_numpy()
    L = np.log1p(fn[FEATS].astype(float))
    bm = fn.cycle < BASE
    mb = L[bm].groupby(fn.device_id[bm]).mean()
    D = L.to_numpy() - mb.reindex(fn.device_id).to_numpy()
    return np.hstack([L.to_numpy(), D])


def device_metrics(log, att, run_id, dev):
    mal = att[att.run_id == run_id].set_index("device_id")
    post = log[log.cycle >= BASE]
    det, lat, per = 0, [], {}
    for did, a in mal.iterrows():
        g = post[(post.device_id == did) & (post.cycle >= a.start_cycle) & (post.cycle < a.end_cycle)]
        hit = g[g.trust < MED]
        ok = len(hit) > 0
        det += ok
        per.setdefault(a.attack_type, []).append(ok)
        if ok:
            lat.append(int(hit.cycle.iloc[0]) - int(a.start_cycle))
    normal = set(dev.device_id) - set(mal.index)
    fp = post[post.device_id.isin(normal)].groupby("device_id").trust.min().lt(MED).sum()
    n_mal, n_norm = len(mal), len(normal)
    tp = det
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / n_mal
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return dict(run_id=run_id, detection=rec, fpr=fp / n_norm, precision=prec, f1=f1,
                latency=np.mean(lat) if lat else np.nan, n_fp=int(fp),
                **{f"det_{k}": np.mean(v) for k, v in per.items()})


# ------------------------------------------------------------------ stage 2
def stage2_features(fn):
    X = fn[["r_" + c for c in FEATS]].copy()
    for c in FEATS:
        X["log_" + c] = np.log1p(fn[c].astype(float))
    return X


def train_stage2(train_df):
    X, y = stage2_features(train_df), train_df.attack_type
    clf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=0)
    clf.fit(X, y)
    return clf
