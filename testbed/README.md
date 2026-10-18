# Virtual hospital IoT testbed

Code, captured dataset and results for the article *A Lightweight Zero Trust
Architecture for Resource-Constrained Medical IoT: Behaviour-Based Trust Scoring
and Isolation Forest Anomaly Detection Evaluated on a Virtual Hospital Testbed*.

The testbed emulates a hospital IoT network on one Linux machine. Every device,
the gateway and the attacker's hosts have their own IPv4 address on the loopback
network (127.0.0.0/8) and exchange **real TCP/UDP packets**. A raw-socket packet
sniffer captures every packet to or from a device into an SQLite database.
Device behaviour (reporting interval, payload size) is configured per device class;
the camera, nurse-call button and fall-detection wearable profiles and the encryption
overhead on every non-video message are calibrated against real medical IoT devices
in the public CICIoMT2024 profiling captures (`compare_ciciomt.py`). It is not a
capture from a real hospital.

## Network

| Host | Address | Services / ports |
|---|---|---|
| Edge gateway | 127.0.0.1 | MQTT 1883/tcp, HL7 2575/tcp, CoAP 5683/udp, RTSP 554/tcp, HTTP 80/tcp |
| 30 IoT devices (15 types) | 127.0.1.11 – 127.0.1.40 | see `config.py` and Table 1 of the article |
| Exfiltration server (attacker) | 127.0.2.10 | 8443/tcp |
| C&C server (attacker) | 127.0.2.20 | 6667/tcp |

Compromised devices (device #1 of each type): ECG monitor and infusion pump
(data exfiltration), room environment sensor and medicine fridge monitor (DDoS
flooding), IP security camera (botnet C&C beaconing), smart lighting controller
(port scanning). Each attacks during one random window of 20–59 s per run.

## Files

| File | Purpose |
|---|---|
| `config.py` | Devices, addresses, protocols, ports, traffic profiles, attack plan, run schedule |
| `testbed.py` | Gateway services, packet capture (AF_PACKET raw socket) and device threads; writes `testbed.db` |
| `analysis.py` | Pre-processing (packets -> RF, IAT, PS, PU, DV per device per second), trust engine, Isolation Forest, metrics, stage-2 classifier |
| `run_analysis.py` | Full evaluation: design choices on training runs 1–10, alpha sweep, ablations, stage 2, AUC, figures |
| `make_attack_figure.py` | One graph per attack type (Figure 4) |
| `db_screenshots.py` | Screenshots of the real database tables (Figure 3) |
| `compare_ciciomt.py`, `make_ciciomt_figure.py` | Compare testbed traffic with CICIoMT2024 real-device captures (calibration / validation) |
| `gen_tables.py`, `gen_results_tex.py` | Generate the article's tables and numbers from `config.py` and `results/` |
| `data/testbed.db.gz` | Captured dataset: 30 runs x 240 s, 2,431,902 packets (161 MB uncompressed) |
| `results/` | Per-run results and summaries behind every number in the article |

Database tables: `devices`, `runs`, `attacks` (ground truth: device, attack type,
start and end cycle) and `packets` (run_id, ts, t_rel, src_ip, src_port, dst_ip,
dst_port, proto, ip_len, payload_len, tcp_flags).

## Reproduce

Linux, Python 3.10+, root privileges (raw sockets and ports below 1024).

```bash
pip install numpy pandas scikit-learn scipy matplotlib
sudo python testbed.py --runs 1-30 --db data/testbed.db     # about 2 hours (30 x 4 min)
python run_analysis.py --db data/testbed.db --train 1-10 --test 11-30
python make_attack_figure.py data/testbed.db 11
```

To analyse the released data instead of capturing new traffic:

```bash
gunzip -k data/testbed.db.gz
python run_analysis.py --db data/testbed.db --train 1-10 --test 11-30
```

## Main results (evaluation runs 11–30, alpha = 0.9 chosen on training runs)

| Metric | Value |
|---|---|
| Detection of compromised devices | 79.2% |
| Device-level false-positive rate | 4.6% (all from event-driven devices) |
| F1-score | 0.803 |
| Mean detection latency | 7.66 s |
| Detection by attack | DDoS 100%, exfiltration 87.5%, port scan 100%, C&C 0% |
| Device-cycle ROC-AUC, trust vs anomaly score | 0.961 vs 0.920 |
| Stage-2 attack type, isolated attackers | 95 / 95 correct |

To compare with CICIoMT2024, download the profiling CSVs (Blink_Camera_LAN_WATCH,
Singcall_LAN_PHYSICAL, Multifunctional_Pager_LAN_PHYSICAL, SenseU_LAN_EMERGENCY) from
http://cicresearch.ca/IOTDataset/CICIoMT2024/ and run
`python compare_ciciomt.py <folder with the CSVs>` (file names may need adjusting).
