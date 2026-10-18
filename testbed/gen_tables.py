"""Write LaTeX table rows (device and attack tables) from config.py."""
import os, sys
import config as C

out = sys.argv[1] if len(sys.argv) > 1 else "../article_v3/tables"
os.makedirs(out, exist_ok=True)
devs = C.build_devices()
rows = []
for dtype, (proto, count, mode, interval, payload) in C.DEVICE_TYPES.items():
    ds = [d for d in devs if d["dtype"] == dtype]
    sr = f"{ds[0]['device_id']}" if len(ds) == 1 else f"{ds[0]['device_id']}--{ds[-1]['device_id']}"
    ips = ds[0]["ip"] if len(ds) == 1 else f"{ds[0]['ip']}--.{ds[-1]['ip'].split('.')[-1]}"
    mark = "$^\\dagger$" if any(d["attack"] for d in ds) else ""
    tr = C.SERVICES[proto][0].upper()
    iv = {"periodic": f"{interval:g} s", "poisson": f"mean {interval:g} s", "burst": f"bursts, mean {interval:g} s", "stream": f"{interval*1000:g} ms (stream)"}[mode]
    pl = payload if mode == "stream" else (payload[0] + C.SECURE_OVERHEAD[0], payload[1] + C.SECURE_OVERHEAD[1])
    name = dtype.replace("(SpO2)", "(SpO$_2$)")
    rows.append(f"{sr} & {name} ($\\times${count}){mark} & {ips} & {proto} & {C.SERVICES[proto][1]}/{tr.lower()} & {iv} & {pl[0]}--{pl[1]} \\\\")
open(f"{out}/devices_rows.tex", "w").write("\n".join(rows) + "\n")

desc = {
    "DDoS flooding": "UDP flood, 300--500 pkt/s to gateway 5683",
    "Data exfiltration": "8--16 kB every 0.2 s to 127.0.2.10:8443",
    "Botnet C\\&C beaconing": "60--120 B beacon every 5 s to 127.0.2.20:6667",
    "Port scanning": "50 SYN/s to ports 1--1024 of gateway and devices",
}
arows = []
for d in devs:
    if d["attack"]:
        a = d["attack"].replace("&", "\\&")
        nm = d["name"].replace("#", "\\#")
        arows.append(f"{nm} & {d['ip']} & {a} & {desc[a]} \\\\")
open(f"{out}/attacks_rows.tex", "w").write("\n".join(arows) + "\n")
print("\n".join(rows + arows))
