"""Virtual hospital IoT testbed.

Three cooperating processes per run:
  1. gateway  - TCP/UDP services the devices talk to (MQTT 1883, HL7 2575,
                CoAP 5683, RTSP 554, HTTP 80) plus the attacker's external hosts
                (exfiltration sink 127.0.2.10:8443, C&C server 127.0.2.20:6667);
  2. capture  - a raw-socket packet sniffer on the loopback interface that writes
                every packet to or from a device into the SQLite table `packets`;
  3. devices  - one thread per IoT device, each bound to its own IP address,
                sending its normal traffic and, if compromised, its attack traffic
                during a random attack window.

Usage:  python testbed.py --runs 1-30 --db data/testbed.db
"""
import argparse, os, random, socket, sqlite3, struct, threading, time
import multiprocessing as mp
import socketserver

import config as C

SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (device_id INTEGER PRIMARY KEY, name TEXT, dtype TEXT, ip TEXT,
  protocol TEXT, transport TEXT, port INTEGER, mode TEXT, interval_s REAL, payload_min INTEGER,
  payload_max INTEGER, compromised INTEGER, attack_type TEXT);
CREATE TABLE IF NOT EXISTS runs (run_id INTEGER PRIMARY KEY, seed INTEGER, t0 REAL, cycle_s REAL,
  total_cycles INTEGER, baseline_cycles INTEGER, n_packets INTEGER);
CREATE TABLE IF NOT EXISTS attacks (run_id INTEGER, device_id INTEGER, attack_type TEXT,
  start_cycle INTEGER, end_cycle INTEGER);
CREATE TABLE IF NOT EXISTS packets (run_id INTEGER, ts REAL, t_rel REAL, src_ip TEXT, src_port INTEGER,
  dst_ip TEXT, dst_port INTEGER, proto TEXT, ip_len INTEGER, payload_len INTEGER, tcp_flags TEXT);
"""


# ============================== gateway ======================================
class _Sink(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(30.0)
        try:
            while True:
                data = self.request.recv(65536)
                if not data:
                    break
                if self.server.reply:
                    self.request.sendall(b"CMD:sleep;" + os.urandom(16).hex().encode())
        except (socket.timeout, OSError):
            pass


class _TCP(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 256

    def __init__(self, addr, reply=False):
        self.reply = reply
        super().__init__(addr, _Sink)


def gateway_proc(ready):
    servers = []
    for name, (tr, port) in C.SERVICES.items():
        if tr == "tcp":
            servers.append(_TCP((C.GATEWAY_IP, port)))
    servers.append(_TCP(C.EXFIL_SERVER))
    servers.append(_TCP(C.CNC_SERVER, reply=True))
    for s in servers:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
    udp.bind((C.GATEWAY_IP, C.SERVICES["CoAP"][1]))
    ready.set()
    while True:
        udp.recvfrom(65536)


# ============================== capture ======================================
FLAGS = "FSRPAU"


def capture_proc(db, run_id, t0_value, stop, ready):
    s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 << 20)
    s.bind(("lo", 0))
    s.settimeout(0.2)
    con = sqlite3.connect(db, timeout=60)
    buf, n = [], 0
    ready.set()
    t0 = t0_value.value
    while not stop.is_set():
        try:
            frame, addr = s.recvfrom(65535)
        except socket.timeout:
            frame = None
        if frame is not None and addr[2] == 0 and frame[12:14] == b"\x08\x00":  # PACKET_HOST, IPv4
            ts = time.time()
            ip = frame[14:]
            ihl = (ip[0] & 0x0F) * 4
            ip_len = struct.unpack("!H", ip[2:4])[0]
            proto = ip[9]
            src, dst = socket.inet_ntoa(ip[12:16]), socket.inet_ntoa(ip[16:20])
            if (src.startswith("127.0.1.") or dst.startswith("127.0.1.")) and proto in (6, 17):
                sport, dport = struct.unpack("!HH", ip[ihl:ihl + 4])
                if proto == 6:
                    off = (ip[ihl + 12] >> 4) * 4
                    fl = ip[ihl + 13]
                    flags = "".join(c for i, c in enumerate(FLAGS) if fl & (1 << i))
                    pl = ip_len - ihl - off
                    pname = "TCP"
                else:
                    flags, pl, pname = "", ip_len - ihl - 8, "UDP"
                buf.append((run_id, ts, ts - t0, src, sport, dst, dport, pname, ip_len, pl, flags))
        if len(buf) >= 2000 or (frame is None and buf):
            con.executemany("INSERT INTO packets VALUES (?,?,?,?,?,?,?,?,?,?,?)", buf)
            con.commit(); n += len(buf); buf = []
    if buf:
        con.executemany("INSERT INTO packets VALUES (?,?,?,?,?,?,?,?,?,?,?)", buf)
        con.commit(); n += len(buf)
    con.execute("UPDATE runs SET n_packets=? WHERE run_id=?", (n, run_id)); con.commit()


# ============================== devices ======================================
def _sleep_until(t):
    d = t - time.time()
    if d > 0:
        time.sleep(d)


def _tcp(src_ip, dst):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((src_ip, 0))
    s.connect(dst)
    return s


def normal_traffic(dev, t0, t_end, rng):
    lo, hi = dev["payload"]
    dst = (C.GATEWAY_IP, dev["port"])
    if dev["transport"] == "udp":
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind((dev["ip"], 0))
        send = lambda b: s.sendto(b, dst)
    else:
        s = _tcp(dev["ip"], dst)
        send = s.sendall
    t = t0 + rng.uniform(0, dev["interval"])
    while t < t_end:
        _sleep_until(t)
        try:
            send(os.urandom(rng.randint(lo, hi)))
        except OSError:
            pass
        if dev["mode"] == "poisson":
            t += rng.expovariate(1.0 / dev["interval"])
        elif dev["mode"] == "stream":
            t += dev["interval"]
        else:
            t += dev["interval"] * rng.uniform(0.9, 1.1)
    s.close()


def attack_traffic(dev, start, end, rng):
    a = dev["attack"]
    _sleep_until(start)
    if a == "DDoS flooding":
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind((dev["ip"], 0))
        rate = rng.randint(*C.DDOS_RATE)
        tick = 0.01
        t = time.time()
        while t < end:
            for _ in range(max(1, int(rate * tick))):
                s.sendto(os.urandom(rng.randint(*C.DDOS_PAYLOAD)), (C.GATEWAY_IP, C.SERVICES["CoAP"][1]))
            t += tick
            _sleep_until(t)
    elif a == "Data exfiltration":
        s = _tcp(dev["ip"], C.EXFIL_SERVER)
        t = time.time()
        while t < end:
            s.sendall(os.urandom(rng.randint(*C.EXFIL_CHUNK)))
            t += C.EXFIL_INTERVAL
            _sleep_until(t)
        s.close()
    elif a == "Botnet C&C beaconing":
        t = time.time() + rng.uniform(0, C.CNC_INTERVAL)
        while t < end:
            _sleep_until(t)
            try:
                s = _tcp(dev["ip"], C.CNC_SERVER)
                s.sendall(os.urandom(rng.randint(*C.CNC_PAYLOAD)))
                s.settimeout(1.0); s.recv(256); s.close()
            except OSError:
                pass
            t += C.CNC_INTERVAL * rng.uniform(0.9, 1.1)
    elif a == "Port scanning":
        targets = [C.GATEWAY_IP] + [f"127.0.1.{i}" for i in range(11, 41)]
        rng.shuffle(targets)
        ports = list(C.SCAN_PORTS)
        t = time.time()
        i = 0
        while t < end:
            host, port = targets[(i // len(ports)) % len(targets)], ports[i % len(ports)]
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind((dev["ip"], 0)); s.setblocking(False)
            try:
                s.connect_ex((host, port))
            finally:
                s.close()
            i += 1
            t += 1.0 / C.SCAN_RATE
            _sleep_until(t)


def devices_proc(t0, windows, seed):
    rng0 = random.Random(seed)
    t_end = t0 + C.TOTAL_CYCLES * C.CYCLE_S
    threads = []
    for dev in C.build_devices():
        r = random.Random(rng0.random())
        threads.append(threading.Thread(target=normal_traffic, args=(dev, t0, t_end, r), daemon=True))
        if dev["attack"]:
            sc, ec = windows[dev["device_id"]]
            threads.append(threading.Thread(target=attack_traffic,
                args=(dev, t0 + sc * C.CYCLE_S, t0 + ec * C.CYCLE_S, random.Random(rng0.random())), daemon=True))
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=max(0.1, t_end - time.time() + 5))


# ============================== orchestration ================================
def init_db(db):
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    if con.execute("SELECT COUNT(*) FROM devices").fetchone()[0] == 0:
        for d in C.build_devices():
            con.execute("INSERT INTO devices VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (d["device_id"], d["name"], d["dtype"], d["ip"], d["protocol"], d["transport"], d["port"],
                 d["mode"], d["interval"], d["payload"][0], d["payload"][1], int(bool(d["attack"])), d["attack"]))
    con.commit(); con.close()


def attack_windows(seed):
    rng = random.Random(10_000 + seed)
    w = {}
    for d in C.build_devices():
        if d["attack"]:
            dur = min(rng.randint(*C.ATTACK_DURATION), C.TOTAL_CYCLES - C.ATTACK_EARLIEST - 2)
            start = rng.randint(C.ATTACK_EARLIEST, C.TOTAL_CYCLES - dur - 1)
            w[d["device_id"]] = (start, start + dur)
    return w


def run_once(db, run_id, total_cycles=None):
    if total_cycles:
        C.TOTAL_CYCLES = total_cycles
    seed = run_id
    windows = attack_windows(seed)
    con = sqlite3.connect(db, timeout=60)
    con.execute("DELETE FROM packets WHERE run_id=?", (run_id,))
    con.execute("DELETE FROM attacks WHERE run_id=?", (run_id,))
    con.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
    t0 = time.time() + 1.5
    con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,NULL)", (run_id, seed, t0, C.CYCLE_S, C.TOTAL_CYCLES, C.BASELINE_CYCLES))
    for did, (sc, ec) in windows.items():
        atype = next(d["attack"] for d in C.build_devices() if d["device_id"] == did)
        con.execute("INSERT INTO attacks VALUES (?,?,?,?,?)", (run_id, did, atype, sc, ec))
    con.commit(); con.close()

    stop, ready = mp.Event(), mp.Event()
    t0v = mp.Value("d", t0)
    cap = mp.Process(target=capture_proc, args=(db, run_id, t0v, stop, ready)); cap.start(); ready.wait()
    dev = mp.Process(target=devices_proc, args=(t0, windows, seed)); dev.start()
    dev.join(timeout=C.TOTAL_CYCLES * C.CYCLE_S + 30)
    if dev.is_alive():
        dev.terminate()
    time.sleep(1.0)
    stop.set(); cap.join(timeout=60)
    con = sqlite3.connect(db)
    n = con.execute("SELECT n_packets FROM runs WHERE run_id=?", (run_id,)).fetchone()[0]
    con.close()
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="1-1")
    ap.add_argument("--db", default="data/testbed.db")
    ap.add_argument("--cycles", type=int, default=None, help="override total cycles (quick test)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.db) or ".", exist_ok=True)
    init_db(a.db)
    g_ready = mp.Event()
    gw = mp.Process(target=gateway_proc, args=(g_ready,), daemon=True); gw.start(); g_ready.wait()
    lo, hi = map(int, a.runs.split("-"))
    for r in range(lo, hi + 1):
        t = time.time()
        n = run_once(a.db, r, a.cycles)
        print(f"run {r}: {n} packets captured in {time.time() - t:.0f} s", flush=True)
    gw.terminate()
