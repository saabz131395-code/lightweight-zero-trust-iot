# Lightweight Zero Trust Architecture for Resource-Constrained IoT Devices

Simulation code for the MS thesis and research article by **Saba Nawaz**
(Department of Computer Science, University of Management and Technology, Lahore):

> *A Lightweight Zero Trust Architecture for Resource-Constrained IoT Devices:
> Integrating FAST-SM9 Authentication, Behaviour-Based Trust Scoring, and
> Isolation Forest Anomaly Detection*

The architecture treats authentication as identity assurance rather than
permanent authorisation. Devices authenticate with FAST-SM9; an edge gateway
monitors five network-visible behavioural features and keeps a per-device
trust score; a cloud-hosted, unsupervised Isolation Forest supplies anomaly
evidence that updates the trust score every cycle and drives a three-level
access decision (ALLOW / RE-AUTHENTICATE / DENY-ISOLATE). No anomaly-detection
work runs on the constrained device.

## Virtual hospital testbed (real traffic)

The folder [`testbed/`](testbed/) contains a second evaluation on **real network traffic**:
a virtual hospital network of 30 IoT devices (15 types) with their own IP addresses
and ports, packet-level capture into an SQLite database, real attack traffic, and a
second stage that classifies the attack type. The captured dataset (30 runs,
1.85 million packets) is included. See [`testbed/README.md`](testbed/README.md).

## What the simulation models

- 100 devices: 90 normal (temperature sensors, smart bulbs, security cameras,
  heart-rate monitors, door sensors) and 10 malicious.
- 200 cycles per run: a 20-cycle baseline followed by 180 monitoring cycles.
- Four post-authentication attacks: DDoS flooding, data exfiltration, botnet
  C&C beaconing (low-signature stress test), and port scanning / lateral movement.
- Features: request frequency (RF), inter-arrival time (IAT), payload size (PS),
  port usage (PU), data volume (DV), normalised against a frozen per-device baseline.
- Dynamic trust update: `DT(t) = α·DT(t−1) + β·(1 − A(t))`, with `α + β = 1`.
- Access thresholds: `Ts ≥ 70` ALLOW, `40 ≤ Ts < 70` RE-AUTHENTICATE, `Ts < 40` DENY/ISOLATE.

FAST-SM9 is represented at the trust-model level (its authentication outcome),
not executed cryptographically. All traffic and attacks are synthetic.

## Files

| File | Purpose |
|---|---|
| `trust_engine_sim.py` | The simulation engine: device model, traffic and attack generation, normalisation, Behavioural Risk, Isolation Forest, trust update, access decision, metrics. |
| `run_experiments.py` | Runs the 240 evaluation runs (baseline + α sensitivity) and saves per-seed results. |
| `ablation_study.py` | Ablation (authentication-only, rule-based-only), supervised Random Forest comparison, and leave-one-attack-type-out test. |
| `make_sensitivity_figures.py` | Draws the α-sensitivity figures (detection/FPR, F1/latency). |
| `make_comparison_figures.py` | Draws the comparison figures with Zahoor et al. (2025) and Ma et al. (2025). |
| `revision_experiments.py` | Additional analyses from the internal review: feature-weight and threshold sensitivity, Isolation Forest fitted on all baseline cycles, feature correlation, Random Forest per-seed results, scalability timing. |
| `device_cycle_auc.py` | Device-cycle ROC-AUC of the anomaly score vs. the trust score, per-attack AUC and cycle-level FPR/precision (article Section 6.4). |
| `revision_stats.py` | Paired Wilcoxon tests with Holm correction, rank-biserial effect sizes and bootstrap CIs; formal selection of α (Youden's J). |
| `results/` | Per-seed result files (CSV) behind every reported number. |
| `figures/` | Figures as they appear in the thesis (Chapter 6). |

## Installation

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

## Reproducing the results

Run all commands from the repository folder.

```bash
# 1. Baseline (α = 0.7, seeds 41-80) and α sensitivity (α = 0.5-0.9, seeds 1-40): 240 runs
python run_experiments.py --workers 4

# 2. Ablation study (writes results/ablation_results.csv)
python ablation_study.py

# 3. Additional analyses (results/revision/)
python revision_experiments.py weights
python revision_experiments.py thresholds
python revision_experiments.py iffit
python revision_experiments.py rf
python revision_experiments.py scale
python revision_stats.py
python device_cycle_auc.py

# 4. Figures
python make_sensitivity_figures.py
python make_comparison_figures.py
```

`python run_experiments.py --quick` runs two seeds per configuration as a fast check.
A single run can also be started directly with `python trust_engine_sim.py`.

## Where each reported result comes from

| Result (thesis / article) | Source |
|---|---|
| Baseline performance, Table 6.1 / Table 5 | `results/baseline_runs.csv` |
| Per-attack detection, Table 6.2 / Table 6 | `trust_engine_sim.py` per-attack metrics, seeds 41-80 |
| α sensitivity, Table 6.3 / Table 7, Figures 6.3-6.4 / 3-4 | `results/sensitivity_runs.csv` |
| Ablation, Table 6.4 / Table 8 | `results/ablation_results.csv` |
| Random Forest and leave-one-attack-type-out, Tables 6.5-6.6 / 9-10 | `ablation_study.py` (`train_rf`, `train_rf_holdout`, `run_holdout_eval`) |
| Device-cycle ROC-AUC (article Section 6.4, Figure 3) | `results/device_cycle_auc.csv` |
| Comparison with recent work, Figures 6.5-6.6 / 5-6 | `results/baseline_runs.csv`, `results/sensitivity_runs.csv`, plus published values from the cited papers |

Headline results: at α = 0.7, 81.0% detection, 31.9% device-level false-positive
rate, F1 0.388, 3.97-cycle mean latency; at the recommended α = 0.8, 74.5%
detection, 4.4% false-positive rate, F1 0.711.

The published values used for comparison (Zahoor et al., 2025: Isolation Forest
precision 0.37, recall 0.08; Ma et al., 2025: 4.2 mJ per FAST-SM9 authentication,
11.7 mJ for RSA-2048) are taken as reported in those papers. The device-side
energy figures are estimates from these per-operation costs, not hardware measurements.

## Citation

If you use this code, please cite the article (details will be added once
published) or this repository; see `CITATION.cff`.

## License

MIT License. See `LICENSE`.
