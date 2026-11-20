"""Track A, closed loop: accuracy and cost of retraining policies on real streams with labelled injected events.
A federation-wide drift (gamma = 1) is injected so that one global model is the right response; regional drift is
studied in the open loop. Thresholds come from the open-loop validation seeds (results/A/open_<d>_K<K>.json).
Usage: python exps/a2_closed_loop.py -> results/A/closed_<dataset>_K<K>.json
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ns import stream, stats, runner

POLICIES = ["static", "periodic", "oracle", "local_adwin", "fdc", "fdc_filter", "feddrift"]
CONFIGS = [("covtype", 50, 600), ("airlines", 50, 600)]
SEEDS = range(0, 10)


def job(a):
    name, K, B, seed, pol, th = a
    t = time.time()
    r = stream.closed_loop(name, K, B, seed, pol, theta=th, gamma=1.0, L=10, R_tr=10)
    r["runtime_s"] = round(time.time() - t, 1)
    return r


def main():
    for name, K, B in CONFIGS:
        tuned = json.load(open(os.path.join(runner.RESULTS, "A", f"open_{name}_K{K}.json")))["tuned"]
        jobs = []
        for s in SEEDS:
            for pol in POLICIES:
                th = tuned["theta_filter"] if pol == "fdc_filter" else tuned["theta_fdc"]
                jobs.append((name, K, B, s, pol, th))
        t = time.time()
        with Pool(2) as p:
            rows = p.map(job, jobs)
        summ = {}
        for pol in POLICIES:
            rs = [r for r in rows if r["policy"] == pol]
            summ[pol] = {k: stats.summarize_rate([r[k] for r in rs]) for k in
                         ["acc_all", "acc_pre", "acc_post", "acc_adapt", "retrains", "retrains_pre_t0",
                          "retrains_during_outage", "node_retrains", "releases_per_node"]}
            if pol == "feddrift":
                for k in ["fd_isolations", "fd_isolations_pre_t0", "fd_models_final"]:
                    summ[pol][k] = stats.summarize_rate([r[k] for r in rs])
        out = dict(dataset=name, K=K, B=B, seeds=list(SEEDS), thresholds=tuned, gamma=1.0, L=10, R_tr=10, P=50,
                   versions=runner.versions(), runtime_s=round(time.time() - t, 1), summary=summ, per_run=rows)
        json.dump(out, open(os.path.join(runner.RESULTS, "A", f"closed_{name}_K{K}.json"), "w"), indent=1, default=str)
        for pol in POLICIES:
            s = summ[pol]
            print(name, K, pol, "acc %.4f pre %.4f post %.4f adapt %.4f | retrains %.1f (pre %.1f, outage %.1f) node %.1f rel %.1f" % (
                s["acc_all"]["mean"], s["acc_pre"]["mean"], s["acc_post"]["mean"], s["acc_adapt"]["mean"], s["retrains"]["mean"],
                s["retrains_pre_t0"]["mean"], s["retrains_during_outage"]["mean"], s["node_retrains"]["mean"], s["releases_per_node"]["mean"]),
                flush=True)


if __name__ == "__main__":
    main()
