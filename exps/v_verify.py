"""Independent check: recompute key paper numbers from per-seed / per-run records (not from stored summaries) and
compare with paper/numbers.tex. Exit code 1 on any mismatch.  Usage: python exps/v_verify.py"""
import json, os, re, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
N = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", open(os.path.join(ROOT, "paper", "numbers.tex")).read()))
J = lambda *p: json.load(open(os.path.join(R, *p)))
bad, ok = [], 0


def chk(name, value, fmt):
    global ok
    s = fmt.format(value)
    if N.get(name) != s:
        bad.append((name, N.get(name), s))
    else:
        ok += 1


m = J("S", "main_outage_drift.json")
ps = m["per_seed"]
rate = lambda k: np.mean([r["methods"][k]["false_episodes"] / r["npre"] * 1000 for r in ps])
det = lambda k: np.mean([r["methods"][k]["detected"] for r in ps])
chk("SfdcFE", rate("FDC@0.35"), "{:.2f}"); chk("SfdcDet", det("FDC@0.35"), "{:.3f}")
chk("SfdcRawDet", np.mean([r["methods"]["FDC@0.35"]["detected_raw"] for r in ps]), "{:.3f}")
chk("SfilDet", det("FDC+filter@0.35"), "{:.3f}")
assert sum(r["methods"]["FDC+filter@0.35"]["false_episodes"] for r in ps) == 0
th = J("S", "tuning.json")["theta_fdc"]
chk("SfdctFE", rate(f"FDC@{th:.2f}"), "{:.2f}")
chk("SlocalW", np.mean([r["methods"]["local"]["wasted_node_retrains"] / r["npre"] * 1000 for r in ps]), "{:.1f}")
# exact Poisson UCB recomputed
from scipy.stats import chi2
exp_k = sum(r["npre"] for r in ps) / 1000
chk("SfilUCB", chi2.ppf(0.95, 2) / 2 / exp_k, "{:.3f}")
# theory
t = J("T", "theory.json")
chk("TindViol", sum(1 for r in t["edetector_arl0"] if r["scenario"] == "independent" and r["alarms"] and r["epochs"] / r["alarms"] < r["c"]), "{}")
# accounting re-derived with dp_accounting directly
import dp_accounting as dpa
from dp_accounting import rdp as rdpm
def rounds(sig):
    lo, hi = 0, 3000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        a = rdpm.RdpAccountant(); a.compose(dpa.GaussianDpEvent(sig), mid)
        lo, hi = (mid, hi) if a.get_epsilon(1e-5) <= 4 else (lo, mid - 1)
    return lo
chk("AccRdp", rounds(25.0), "{}")
# open loop: recompute tuned-theta false rates from per-seed records
cnt_lower = 0
for d, K in [("covtype", 10), ("covtype", 50), ("covtype", 100), ("airlines", 10), ("airlines", 50), ("airlines", 100), ("elec", 5), ("elec", 10)]:
    o = J("A", f"open_{d}_K{K}.json")
    f = lambda key: np.mean([r["methods"][key]["false_episodes"] / r["npre"] * 1000 for r in o["per_seed"]])
    tf, tc = o["tuned"]["theta_fdc"], o["tuned"]["theta_filter"]
    cnt_lower += f(f"FDC+filter@{tc:.2f}") < f(f"FDC@{tf:.2f}")
chk("AnLower", cnt_lower, "{}")
# closed loop from per-run
for d, pre in [("covtype", "Cov"), ("airlines", "Air")]:
    o = J("A", f"closed_{d}_K50.json")
    g = lambda pol, k: np.mean([r[k] for r in o["per_run"] if r["policy"] == pol])
    chk(pre + "ClFdcOut", g("fdc", "retrains_during_outage"), "{:.1f}"); chk(pre + "ClFilOut", g("fdc_filter", "retrains_during_outage"), "{:.1f}")
    chk(pre + "ClFdAcc", g("feddrift", "acc_all"), "{:.3f}"); chk(pre + "ClFilAcc", g("fdc_filter", "acc_all"), "{:.3f}")
# C-MAPSS from per-engine
c = J("B_real", "cmapss.json")
chk("CmAdjDetTwo", np.mean([e["detected_degraded"] for e in c["FD002"]["per_engine"]["adjusted"]]), "{:.3f}")
chk("CmRawDetTwo", np.mean([e["detected_degraded"] for e in c["FD002"]["per_engine"]["raw"]]), "{:.3f}")
# DP utility and Byzantine from per-run
dpu = J("P", "dp_utility.json")
cov_rows = dpu["per_run"][:50]          # jobs were issued covtype-first: 10 seeds x 5 arms
assert all(r["lr"] in (1.0, dpu["lr_clipped"]["covtype"]) for r in cov_rows)
chk("DpCovTen", np.mean([r["acc"] for r in cov_rows if r["mode"] == "dp" and r["sigma"] == 10.0]), "{:.3f}")
chk("DpCovClip", np.mean([r["acc"] for r in cov_rows if r["mode"] == "clip"]), "{:.3f}")
bz = J("B", "byzantine.json")
chk("ByzMeanNone", np.mean([r["acc"] for r in bz["per_run"] if r["agg"] == "mean" and r["attack"] == "none"]), "{:.3f}")
print(f"{ok} checks passed; {len(bad)} mismatches")
for b in bad:
    print("MISMATCH", b)
sys.exit(1 if bad else 0)
