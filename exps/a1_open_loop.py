"""Track A, open loop: decision quality on real streams with labelled injected outages and regional drift.
Validation seeds 100-109 tune theta (same rule as Track S); test seeds 0-19 are reported.
Usage: python exps/a1_open_loop.py  -> results/A/open_<dataset>_K<K>.json
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ns import stream, stats, runner

GRID = tuple(np.round(np.arange(0.15, 0.551, 0.05), 3))
CONFIGS = [("covtype", 10, 600), ("covtype", 50, 600), ("covtype", 100, 600),
           ("airlines", 10, 600), ("airlines", 50, 600), ("airlines", 100, 600),
           ("elec", 5, 150), ("elec", 10, 150)]
VAL, TEST = range(100, 110), range(0, 20)


def job(a):
    name, K, B, seed, rk = a
    return stream.open_loop(name, K, B, seed, thetas=GRID, river_kinds=rk)


def summarize(rows):
    npre = [r["npre"] for r in rows]
    s = {"k_eff": stats.summarize_rate([r["k_eff"] for r in rows]),
         "mean_error_R0": stats.summarize_rate([r["mean_error_R0"] for r in rows]),
         "outage_block_frac": stats.summarize_rate([r["outage_block_frac"] for r in rows]), "methods": {}}
    for m in rows[0]["methods"]:
        ms = [r["methods"][m] for r in rows]
        if "false_episodes" in ms[0]:
            dl = [x["delay"] for x in ms if x["delay"] is not None]
            s["methods"][m] = dict(
                false_episodes_per_1000=stats.summarize_counts_per_1000([x["false_episodes"] for x in ms], npre),
                natural_episodes_per_1000=stats.summarize_counts_per_1000([x["natural_episodes"] for x in ms], npre),
                attributable_detection=stats.summarize_prop([x["detected"] for x in ms]),
                delay_median=float(np.median(dl)) if dl else None)
        else:
            s["methods"][m] = dict(
                wasted_node_retrains_per_1000=stats.summarize_rate([x["wasted_node_retrains"] / r["npre"] * 1000 for x, r in zip(ms, rows)]),
                natural_node_episodes_per_1000=stats.summarize_rate([x["natural_node_episodes"] / r["npre"] * 1000 for x, r in zip(ms, rows)]),
                node_detect_frac=stats.summarize_rate([x["node_detect_frac"] for x in ms]))
    return s


def tune(s, prefix):
    c = [(th, s["methods"][f"{prefix}@{th:.2f}"]["false_episodes_per_1000"]["mean"],
          s["methods"][f"{prefix}@{th:.2f}"]["attributable_detection"]["rate"]) for th in GRID]
    ok = [x for x in c if x[2] >= 0.9]
    return float((min(ok, key=lambda x: (x[1], -x[2])) if ok else max(c, key=lambda x: x[2]))[0]), c


def main():
    os.makedirs(os.path.join(runner.RESULTS, "A"), exist_ok=True)
    for name, K, B in CONFIGS:
        t = time.time()
        rk = ("adwin", "ddm", "ph") if K in (10, 50) and name != "elec" or (name == "elec" and K == 10) else ()
        with Pool(2) as p:
            val = p.map(job, [(name, K, B, s, ()) for s in VAL])
            test = p.map(job, [(name, K, B, s, rk) for s in TEST])
        sv = summarize(val)
        th_f, cf = tune(sv, "FDC")
        th_c, cc = tune(sv, "FDC+filter")
        out = dict(dataset=name, K=K, B=B, val_seeds=list(VAL), test_seeds=list(TEST), grid=GRID,
                   tuned=dict(theta_fdc=th_f, theta_filter=th_c, candidates_fdc=cf, candidates_filter=cc),
                   versions=runner.versions(), runtime_s=round(time.time() - t, 1),
                   summary=summarize(test), per_seed=test)
        json.dump(out, open(os.path.join(runner.RESULTS, "A", f"open_{name}_K{K}.json"), "w"), indent=1, default=str)
        sm = out["summary"]["methods"]
        print(name, K, "tuned", th_f, th_c, "| FDC* fe %.2f det %.2f | filter* fe %.2f det %.2f | FDC.35 fe %.2f det %.2f | filt.35 fe %.2f det %.2f | %.0fs" % (
            sm[f"FDC@{th_f:.2f}"]["false_episodes_per_1000"]["mean"], sm[f"FDC@{th_f:.2f}"]["attributable_detection"]["rate"],
            sm[f"FDC+filter@{th_c:.2f}"]["false_episodes_per_1000"]["mean"], sm[f"FDC+filter@{th_c:.2f}"]["attributable_detection"]["rate"],
            sm["FDC@0.35"]["false_episodes_per_1000"]["mean"], sm["FDC@0.35"]["attributable_detection"]["rate"],
            sm["FDC+filter@0.35"]["false_episodes_per_1000"]["mean"], sm["FDC+filter@0.35"]["attributable_detection"]["rate"],
            time.time() - t), flush=True)


if __name__ == "__main__":
    main()
