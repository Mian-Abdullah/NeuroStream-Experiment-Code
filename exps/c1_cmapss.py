"""Track B: a real observed common cause. NASA C-MAPSS FD002/FD004 (six operating conditions, three operational
settings recorded with every cycle). Node-level test only: engines are not synchronised, so operating-condition
changes are not shared across nodes at the same time (stated in the paper).

Raw detector: two-window test (MB cycles per half) on per-condition-agnostic sensor values, standardised with the
fleet's first 30 cycles. Adjusted detector: identical test after subtracting the fleet's per-condition means and
dividing by per-condition standard deviations, both estimated on the first 30 cycles of every training engine
(conditioning on the observed operating condition). Cut: Gaussian approximation |diff| > z_{1-d'/2}*sqrt(2/MB)
(per-cycle values are not [0,1]-bounded, so the Hoeffding cut of the stream experiments is not applicable).
Proxy ground truth: healthy = RUL > 125 cycles, degraded = RUL <= 30 cycles (run-to-failure training trajectories).
Usage: python exps/c1_cmapss.py -> results/B_real/cmapss.json
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ns import stats, runner

RAW = os.environ.get("NS_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"))
SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]


def load(fd):
    d = pd.read_csv(os.path.join(RAW, "cmapss", f"train_{fd}.txt"), sep=r"\s+", header=None)
    d.columns = ["unit", "cycle", "set1", "set2", "set3"] + [f"s{i}" for i in range(1, 22)]
    d["rul"] = d.groupby("unit")["cycle"].transform("max") - d["cycle"]
    d["cond"] = d.groupby([d.set1.round(0), d.set2.round(2), d.set3.round(0)]).ngroup()
    return d


def run(fd, MB=15, dprime=0.002):
    d = load(fd)
    cols = [f"s{i}" for i in SENSORS]
    early = d[d.cycle <= 30]
    gm, gs = early[cols].mean(), early[cols].std().replace(0, 1)
    cm = early.groupby("cond")[cols].mean()
    cs = early.groupby("cond")[cols].std().fillna(1.0).replace(0, 1)
    zc = norm.ppf(1 - dprime / 2)
    res = {"raw": [], "adjusted": []}
    for u, g in d.groupby("unit"):
        Xr = ((g[cols] - gm) / gs).to_numpy()
        Xa = ((g[cols].to_numpy() - cm.loc[g.cond].to_numpy()) / cs.loc[g.cond].to_numpy())
        rul = g.rul.to_numpy()
        T = len(g)
        for kind, X in [("raw", Xr), ("adjusted", Xa)]:
            cs_ = np.concatenate([np.zeros((1, X.shape[1])), np.cumsum(X, 0)])
            b = np.arange(2 * MB - 1, T)
            diff = np.abs((cs_[b + 1] - cs_[b + 1 - MB]) - (cs_[b + 1 - MB] - cs_[b + 1 - 2 * MB])) / MB
            # Bonferroni across the 14 sensors for a per-window false-alarm rate of about dprime
            al = (diff > norm.ppf(1 - dprime / (2 * len(cols))) * np.sqrt(2.0 / MB)).any(1)
            healthy = rul[b] > 125
            degraded = rul[b] <= 30
            res[kind].append(dict(unit=int(u), healthy_windows=int(healthy.sum()),
                                  healthy_alarm_rate=float(al[healthy].mean()) if healthy.any() else float("nan"),
                                  detected_degraded=bool(al[degraded].any()) if degraded.any() else False))
    summ = {}
    for kind, rows in res.items():
        summ[kind] = dict(healthy_alarm_rate=stats.summarize_rate([r["healthy_alarm_rate"] for r in rows]),
                          degraded_detection=stats.summarize_prop([r["detected_degraded"] for r in rows]),
                          engines=len(rows))
    return dict(fd=fd, rows=len(d), engines=int(d.unit.nunique()), conditions=int(d.cond.nunique()), MB=MB,
                dprime=dprime, sensors=SENSORS, summary=summ, per_engine=res)


def main():
    out = {fd: run(fd) for fd in ["FD002", "FD004"]}
    out["note"] = __doc__
    os.makedirs(os.path.join(runner.RESULTS, "B_real"), exist_ok=True)
    json.dump(out, open(os.path.join(runner.RESULTS, "B_real", "cmapss.json"), "w"), indent=1)
    for fd in ["FD002", "FD004"]:
        o = out[fd]
        print(fd, "rows", o["rows"], "engines", o["engines"], "conditions", o["conditions"])
        for k, v in o["summary"].items():
            print("  ", k, "healthy alarm rate %.4f %s" % (v["healthy_alarm_rate"]["mean"], [round(x, 4) for x in v["healthy_alarm_rate"]["ci95"]]),
                  "degraded detection %.3f %s" % (v["degraded_detection"]["rate"], [round(x, 3) for x in v["degraded_detection"]["ci95"]]))


if __name__ == "__main__":
    main()
