"""Compare testbed device traffic with real IoMT device traffic from CICIoMT2024 profiling captures.

CICIoMT2024 profiling CSVs summarise each window of 10 consecutive packets of a capture
(both directions): Rate (packets/s), Tot size (mean frame size, bytes), IAT (mean gap, s).
The same window statistics are computed here from the testbed database for the matching
testbed devices, using normal (non-attack) traffic of the evaluation runs.
"""
import glob, sqlite3, sys
import numpy as np, pandas as pd

DB = sys.argv[2] if len(sys.argv) > 2 else "data/testbed.db"; CIC = sys.argv[1]
RUNS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else range(11, 31)

def windows(p, n=10):
    p = p.sort_values("t_rel")
    t = p.t_rel.to_numpy(); fs = np.maximum(p.ip_len.to_numpy() + 14, 60)   # Ethernet frame, 60 B minimum
    out = []
    for i in range(0, len(p) - n + 1, n):
        tt = t[i:i + n]; dur = tt[-1] - tt[0]
        if dur <= 0: continue
        out.append(dict(rate=(n - 1) / dur, size=fs[i:i + n].mean(), iat=np.diff(tt).mean()))
    return pd.DataFrame(out)

con = sqlite3.connect(DB)
SCANNER = "127.0.1.40"   # exclude traffic of the port scan, which also reaches benign devices
dev = pd.read_sql("SELECT device_id, name, dtype, ip FROM devices", con)
att = pd.read_sql("SELECT * FROM attacks", con)
def testbed(dtype):
    rows = []
    for _, d in dev[dev.dtype == dtype].iterrows():
        for r in RUNS:
            p = pd.read_sql("SELECT t_rel, ip_len FROM packets WHERE run_id=? AND (src_ip=? OR dst_ip=?) AND t_rel>=0 AND src_ip<>? AND dst_ip<>?", con, params=(r, d.ip, d.ip, SCANNER, SCANNER))
            a = att[(att.run_id == r) & (att.device_id == d.device_id)]
            if len(a):
                p = p[(p.t_rel < a.start_cycle.iloc[0]) | (p.t_rel >= a.end_cycle.iloc[0])]
            rows.append(windows(p))
    return pd.concat(rows)

pairs = [("IP security camera", ["Blink_Camera_LAN_WATCH.pcap.csv"], "Blink camera (live view)"),
         ("Nurse-call button", ["Singcall_LAN_PHYSICAL.pcap.csv", "Multifunctional_Pager_LAN_PHYSICAL.pcap.csv"], "Singcall + pager buttons"),
         ("Fall-detection wearable", ["SenseU_LAN_EMERGENCY.pcap.csv"], "SenseU sensor")]
res = []
store = {}
for tb, files, real in pairs:
    c = pd.concat([pd.read_csv(glob.glob(f"{CIC}/*{f}")[0]) for f in files]).rename(columns={"Rate": "rate", "Tot size": "size", "IAT": "iat"})
    t = testbed(tb)
    store[tb] = (c, t, real)
    for src, x in (("CICIoMT2024: " + real, c), ("Testbed: " + tb, t)):
        res.append(dict(pair=tb, source=src, windows=len(x),
                        rate_med=x.rate.median(), rate_q25=x.rate.quantile(.25), rate_q75=x.rate.quantile(.75),
                        size_med=x["size"].median(), size_q25=x["size"].quantile(.25), size_q75=x["size"].quantile(.75)))
res = pd.DataFrame(res)
res.to_csv(sys.argv[4] if len(sys.argv) > 4 else "results/ciciomt_comparison.csv", index=False)
pd.to_pickle(store, "results/ciciomt_windows.pkl")
print(res.round(2).to_string())
