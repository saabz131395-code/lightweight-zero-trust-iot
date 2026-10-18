"""Virtual hospital IoT testbed: device catalogue, network addressing and attack plan.

All hosts live on the Linux loopback network (127.0.0.0/8), so every device has
its own IP address and the traffic is real TCP/UDP traffic between sockets that
can be captured at packet level. Nothing here is a real hospital capture: the
reporting intervals and payload sizes are configured to reflect typical device
behaviour (see the article's device table).
"""

# ---- network -------------------------------------------------------------
GATEWAY_IP = "127.0.0.1"
EXFIL_SERVER = ("127.0.2.10", 8443)   # attacker-controlled external host
CNC_SERVER = ("127.0.2.20", 6667)     # botnet command-and-control host

# gateway services (protocol, port)
SERVICES = {
    "MQTT": ("tcp", 1883),
    "HL7": ("tcp", 2575),
    "CoAP": ("udp", 5683),
    "RTSP": ("tcp", 554),
    "HTTP": ("tcp", 80),
}

# ---- device types ----------------------------------------------------------
# name: (protocol, count, mode, interval_s, payload_bytes(min,max))
#   mode "periodic": one message every interval (+-10 % jitter)
#   mode "poisson":  event-driven, exponential gaps with this mean
#   mode "stream":   continuous media stream, one packet every interval
#   mode "burst":    event-driven bursts (button press): exponential gaps with this
#                    mean between events, each event = BURST_MSGS messages BURST_GAP apart
#
# Calibration against CICIoMT2024 profiling captures (Dadkhah et al., 2024):
#   * camera stream rate and payload range follow the Blink camera live-view capture;
#   * nurse-call bursts follow the Singcall / multifunctional pager button captures;
#   * the fall-detection wearable's reporting interval follows the SenseU sensor capture;
#   * every non-video message carries SECURE_OVERHEAD extra bytes for the TLS/DTLS record
#     layer and application framing, so that small-device frame sizes match the real
#     devices (median 126-146 B), which all communicate over encrypted channels.
DEVICE_TYPES = {
    "Heart-rate monitor":         ("MQTT", 3, "periodic", 1.0, (40, 80)),
    "Pulse oximeter (SpO2)":      ("MQTT", 3, "periodic", 1.0, (30, 60)),
    "Blood-pressure monitor":     ("MQTT", 2, "periodic", 5.0, (50, 100)),
    "ECG monitor":                ("HL7", 2, "periodic", 0.25, (300, 600)),
    "Smart infusion pump":        ("HL7", 2, "periodic", 2.0, (150, 300)),
    "Glucose monitor":            ("MQTT", 2, "periodic", 5.0, (40, 80)),
    "Patient temperature sensor": ("CoAP", 2, "periodic", 2.0, (20, 40)),
    "Room environment sensor":    ("CoAP", 2, "periodic", 3.0, (30, 50)),
    "Medicine fridge monitor":    ("CoAP", 2, "periodic", 3.0, (20, 40)),
    "Smart door sensor":          ("MQTT", 2, "poisson", 4.0, (20, 30)),
    "IP security camera":         ("RTSP", 2, "stream", 0.025, (1000, 1400)),
    "Nurse-call button":          ("MQTT", 2, "burst", 8.0, (20, 40)),
    "Smart bed sensor":           ("MQTT", 2, "periodic", 2.0, (60, 120)),
    "Fall-detection wearable":    ("MQTT", 1, "periodic", 2.0, (30, 60)),
    "Smart lighting controller":  ("HTTP", 1, "periodic", 10.0, (100, 200)),
}

SECURE_OVERHEAD = (60, 100)     # bytes added to every non-video message
BURST_MSGS = (4, 7)             # messages per button-press event
BURST_GAP = (0.1, 0.4)          # seconds between messages within a burst

# ---- compromised devices -------------------------------------------------
# (device type, index within type) -> attack type
ATTACK_PLAN = {
    ("Room environment sensor", 1): "DDoS flooding",
    ("Medicine fridge monitor", 1): "DDoS flooding",
    ("ECG monitor", 1): "Data exfiltration",
    ("Smart infusion pump", 1): "Data exfiltration",
    ("IP security camera", 1): "Botnet C&C beaconing",
    ("Smart lighting controller", 1): "Port scanning",
}

# attack parameters
DDOS_RATE = (300, 500)          # packets per second (UDP flood to the gateway)
DDOS_PAYLOAD = (20, 60)
EXFIL_CHUNK = (8000, 16000)     # bytes per chunk
EXFIL_INTERVAL = 0.2            # seconds between chunks
CNC_INTERVAL = 5.0              # beacon every 5 s
CNC_PAYLOAD = (60, 120)
SCAN_RATE = 50                  # connection attempts per second
SCAN_PORTS = range(1, 1025)

# ---- run schedule -------------------------------------------------------
CYCLE_S = 1.0          # one monitoring cycle = one second of traffic
BASELINE_CYCLES = 60
TOTAL_CYCLES = 240
ATTACK_DURATION = (20, 59)      # cycles
ATTACK_EARLIEST = 65             # first possible attack cycle (after the 60-cycle baseline)


def build_devices():
    """Return the list of devices with id, name, type, IP, protocol, port, attack."""
    devices, n = [], 0
    for dtype, (proto, count, mode, interval, payload) in DEVICE_TYPES.items():
        for k in range(1, count + 1):
            n += 1
            devices.append(dict(
                device_id=n,
                name=f"{dtype} #{k}",
                dtype=dtype,
                ip=f"127.0.1.{10 + n}",
                protocol=proto,
                transport=SERVICES[proto][0],
                port=SERVICES[proto][1],
                mode=mode, interval=interval, payload=payload,
                attack=ATTACK_PLAN.get((dtype, k)),
            ))
    return devices


if __name__ == "__main__":
    ds = build_devices()
    print(len(ds), "devices,", len(DEVICE_TYPES), "types,", sum(d["attack"] is not None for d in ds), "compromised")
    for d in ds:
        print(d["device_id"], d["name"], d["ip"], d["protocol"], d["port"], d["attack"] or "")
