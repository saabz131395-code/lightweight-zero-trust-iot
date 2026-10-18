"""Screenshots of the real testbed database (SQLite), rendered as a database-browser view.

Every value shown comes from a live query on data/testbed.db; the query text is
printed in the window so the screenshot can be reproduced.
"""
import sqlite3, html, sys
from playwright.sync_api import sync_playwright

DB = sys.argv[1] if len(sys.argv) > 1 else "data/testbed.db"
RUN = int(sys.argv[2]) if len(sys.argv) > 2 else 11

CSS = """
body{margin:0;background:#e9edf2;font-family:Segoe UI,Arial,sans-serif}
.win{margin:14px;background:#fff;border:1px solid #9aa5b4;border-radius:6px;box-shadow:0 2px 6px #0002;overflow:hidden;display:inline-block}
.bar{background:#2f3e52;color:#fff;font-size:13px;padding:7px 12px}
.q{font-family:Consolas,monospace;font-size:12px;background:#f6f8fa;border-bottom:1px solid #d0d7de;padding:8px 12px;color:#0b2545;white-space:pre}
table{border-collapse:collapse;font-size:12px}
th{background:#dde3ea;border:1px solid #b8c2d1;padding:4px 8px;text-align:left;font-weight:600}
td{border:1px solid #d6dce4;padding:3px 8px;font-family:Consolas,monospace;white-space:nowrap}
tr.atk td{background:#fdecec}
.foot{font-size:11px;color:#556;padding:6px 12px}
"""


def render(page, title, query, rows, cols, out, highlight=None, foot=""):
    trs = []
    for r in rows:
        cls = ' class="atk"' if highlight and highlight(r) else ""
        trs.append(f"<tr{cls}>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>")
    doc = f"""<html><head><style>{CSS}</style></head><body><div class="win">
      <div class="bar">DB Browser &mdash; testbed.db &mdash; {html.escape(title)}</div>
      <div class="q">{html.escape(query)}</div>
      <table><tr>{''.join(f'<th>{html.escape(c)}</th>' for c in cols)}</tr>{''.join(trs)}</table>
      <div class="foot">{html.escape(foot)}</div></div></body></html>"""
    page.set_content(doc)
    page.locator(".win").screenshot(path=out)


con = sqlite3.connect(DB)
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(device_scale_factor=2, viewport={"width": 1400, "height": 900})

    q1 = "SELECT device_id, name, ip, protocol, transport, port, attack_type FROM devices;"
    rows = con.execute(q1).fetchall()
    rows = [tuple("-" if v is None else v for v in r) for r in rows]
    render(pg, "devices", q1, rows, ["device_id", "name", "ip", "protocol", "transport", "port", "attack_type"],
           "figures/db_devices.png", highlight=lambda r: r[-1] != "-", foot=f"{len(rows)} rows")

    att = con.execute("SELECT device_id, attack_type, start_cycle FROM attacks WHERE run_id=?", (RUN,)).fetchall()
    parts, q2 = [], ""
    # 3 normal packets then 2 packets of each attack, taken from the attack windows
    norm = con.execute("""SELECT run_id, ROUND(t_rel,3), src_ip, src_port, dst_ip, dst_port, proto, ip_len, payload_len, tcp_flags
        FROM packets WHERE run_id=? AND src_ip IN ('127.0.1.11','127.0.1.24','127.0.1.25') AND payload_len>0 AND t_rel>100 LIMIT 3""", (RUN,)).fetchall()
    parts += norm
    filt = {"DDoS flooding": "proto='UDP'", "Data exfiltration": "dst_port=8443",
            "Botnet C&C beaconing": "dst_port=6667 AND payload_len>0", "Port scanning": "tcp_flags='S'"}
    for did, at, sc in att:
        ip = con.execute("SELECT ip FROM devices WHERE device_id=?", (did,)).fetchone()[0]
        parts += con.execute(f"""SELECT run_id, ROUND(t_rel,3), src_ip, src_port, dst_ip, dst_port, proto, ip_len, payload_len, tcp_flags
            FROM packets WHERE run_id=? AND src_ip=? AND t_rel>=? AND {filt[at]} LIMIT 2""", (RUN, ip, sc + 2)).fetchall()
    mal_ips = {con.execute("SELECT ip FROM devices WHERE device_id=?", (d,)).fetchone()[0] for d, _, _ in att}
    q2 = ("SELECT run_id, t_rel, src_ip, src_port, dst_ip, dst_port, proto, ip_len, payload_len, tcp_flags\n"
          f"FROM packets WHERE run_id={RUN} AND ...   -- sample: 3 normal packets, then 2 from each attack window")
    n = con.execute("SELECT COUNT(*) FROM packets WHERE run_id=?", (RUN,)).fetchone()[0]
    render(pg, "packets", q2, parts, ["run_id", "t_rel (s)", "src_ip", "src_port", "dst_ip", "dst_port", "proto", "ip_len", "payload_len", "tcp_flags"],
           "figures/db_packets.png", highlight=lambda r: r[2] in mal_ips and r[1] > 0,
           foot=f"run {RUN}: {n:,} captured packets; red rows are from compromised devices during their attack window")
    b.close()
print("ok")
