"""Track S, main study: threshold tuning on validation seeds, main comparison on 200 test seeds, scenarios,
weight ablations and the anytime-valid e-detector.
Outputs results/S/*.json.  Usage: python exps/s1_main.py
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ns import sim, runner

VAL = range(1000, 1100)        # validation seeds (tuning / calibration only)
TEST = range(0, 200)           # test seeds
GRID = tuple(np.round(np.arange(0.25, 0.651, 0.025), 3))


def tune(summary, prefix, min_det=0.9):
    """theta* = smallest false-episode rate subject to attributable detection >= min_det (else max detection)."""
    cands = []
    for th in GRID:
        m = summary["methods"][f"{prefix}@{th:.2f}"]
        cands.append((th, m["false_episodes_per_1000"]["mean"], m["attributable_detection"]["rate"]))
    ok = [c for c in cands if c[2] >= min_det]
    best = min(ok, key=lambda c: (c[1], -c[2])) if ok else max(cands, key=lambda c: c[2])
    return float(best[0]), cands


def main():
    # 1. tuning on validation seeds
    val = runner.run("val_grid", seeds=VAL, kw=dict(thetas=GRID))
    th_fdc, c_fdc = tune(val["summary"], "FDC")
    th_fil, c_fil = tune(val["summary"], "FDC+filter")
    # calibration of p for the e-detector (validation seeds, outages, no drift)
    tuning = dict(theta_fdc=th_fdc, theta_filter=th_fil, candidates_fdc=c_fdc, candidates_filter=c_fil,
                  rule="min false episodes s.t. attributable detection >= 0.9 on validation seeds 1000-1099")
    json.dump(tuning, open(os.path.join(runner.RESULTS, "S", "tuning.json"), "w"), indent=1)
    print("tuned theta: FDC", th_fdc, " FDC+filter", th_fil)
    thetas = tuple(sorted(set([0.35, 0.45, 0.55, th_fdc, th_fil])))
    # e-detector p: weighted epoch-alarm rate on calibration runs + margin
    edp = calibrate_edet(VAL[:30])
    json.dump(edp, open(os.path.join(runner.RESULTS, "S", "edet_calibration.json"), "w"), indent=1)
    ed = dict(p=edp["p"], lambdas=[5, 10, 20, 40, 80], c=[100, 1000])
    # 2. main comparison + scenarios
    runner.run("main_outage_drift", seeds=TEST, kw=dict(thetas=thetas, edet=ed))
    runner.run("faults_drift", seeds=TEST, kw=dict(thetas=thetas, scenario="faults_drift", edet=ed))
    runner.run("outage_nodrift", seeds=TEST, kw=dict(thetas=thetas, scenario="outage_nodrift", edet=ed))
    # 3. weight ablations
    for w in ["equal", "random_rho", "volume_x_calibrated"]:
        runner.run(f"weights_{w}", overrides=dict(weights=w), seeds=TEST, kw=dict(thetas=thetas))
    print("done")


def calibrate_edet(seeds, margin=0.02):
    cfg = dict(sim.DEFAULT)
    rates = {"raw": [], "filtered": []}
    for s in seeds:
        g = sim.generate(cfg, s, drift=False, outage=True)
        sg = sim.decision_signals(cfg, g, (0.35,))
        for kind, al in [("raw", sg["raw_alarm"]), ("filtered", sg["raw_alarm"] & sg["adj_alarm"])]:
            rates[kind].append(float(sim.epoch_votes(al, sg["alpha"], cfg["W"], 2 * cfg["MB"]).mean()))
    return dict(p={k: float(np.mean(v) + margin) for k, v in rates.items()},
                mean_weighted_epoch_alarm_rate={k: float(np.mean(v)) for k, v in rates.items()},
                margin=margin, seeds=[int(s) for s in seeds],
                note="p for the e-detector = weighted mean epoch alarm rate on calibration seeds (outages, no drift) + margin")


if __name__ == "__main__":
    main()
