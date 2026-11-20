"""Track S sweeps: threshold curve, federation size, affected fraction, drift magnitude, covariate failure modes,
Byzantine voters, private filter and Sparse-Vector gate.  Seeds 0-99 (200 for the threshold curve).
Usage: python exps/s2_sweeps.py
"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ns import sim, runner

SEEDS = range(0, 100)
GRID = tuple(np.round(np.arange(0.25, 0.651, 0.025), 3))
TUNED = json.load(open(os.path.join(runner.RESULTS, "S", "tuning.json")))
TH = tuple(sorted(set([0.35, 0.45, TUNED["theta_fdc"], TUNED["theta_filter"]])))


def main(only=None):
    if only == "privacy":
        return privacy()
    runner.run("theta_curve", seeds=range(0, 200), kw=dict(thetas=GRID), subdir="S_sweeps")
    for K in [10, 25, 50, 100, 200, 400]:
        runner.run(f"K_{K}", overrides=dict(K=K), seeds=SEEDS, kw=dict(thetas=TH), subdir="S_sweeps")
    for fd in [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        runner.run(f"frac_{fd}", overrides=dict(frac_drift=fd), seeds=SEEDS, kw=dict(thetas=TH), subdir="S_sweeps")
    for dm in [0.33, 0.5, 0.67, 1.0]:
        runner.run(f"mag_{dm}", overrides=dict(drift_mag=dm), seeds=SEEDS, kw=dict(thetas=TH), subdir="S_sweeps")
    # covariate failure modes
    for zs in [0.03, 0.1, 0.2, 0.3, 0.5]:
        runner.run(f"zsd_{zs}", overrides=dict(z_sd=zs), seeds=SEEDS, kw=dict(thetas=TH), subdir="S_fail")
    for zo in [1.0, 0.75, 0.5, 0.25, 0.0]:
        runner.run(f"zobs_{zo}", overrides=dict(z_obs=zo), seeds=SEEDS, kw=dict(thetas=TH), subdir="S_fail")
    runner.run("nonlinear", overrides=dict(nonlinear=True), seeds=SEEDS, kw=dict(thetas=TH), subdir="S_fail")
    runner.run("spoofed_feed", overrides=dict(spoof_feed=True), seeds=SEEDS, kw=dict(thetas=TH), subdir="S_fail")
    # Byzantine voters
    for mode in ["flood", "suppress"]:
        for fr in [0.05, 0.1, 0.2, 0.3]:
            runner.run(f"byz_{mode}_{fr}", seeds=SEEDS, kw=dict(thetas=TH, byz=(mode, fr)), subdir="S_attack")
    privacy()
    print("done")


def privacy():
    # private filter: Gaussian noise on the private window sums
    c = sim.DEFAULT
    delta2 = math.sqrt(c["C"] * (1 + c["zmax"] ** 2)) / c["NB"]
    for sf in [1, 3, 10, 30, 100, 300]:
        runner.run(f"privfilter_{sf}", seeds=SEEDS, kw=dict(thetas=TH, private_sd=sf * delta2), subdir="S_privacy")
    json.dump(dict(l2_sensitivity=delta2, note="private_sd = noise multiplier x l2_sensitivity of (sum_new Y, sum_old Y, "
                   "sum Z*Y) per record in [0,1]^C with Z clipped to [0, zmax]"),
              open(os.path.join(runner.RESULTS, "S_privacy", "privfilter_sensitivity.json"), "w"), indent=1)
    # Sparse-Vector gate (fixed implementation)
    for eps, cc in [(0.2, 1), (0.5, 1), (1.0, 1), (0.5, 5), (0.2, 5), (0.2, 10)]:
        runner.run(f"svt_{eps}_{cc}", seeds=SEEDS, kw=dict(thetas=TH, svt=(eps, cc)), subdir="S_privacy")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
