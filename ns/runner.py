"""Seeded experiment runner: executes run_seed over seeds in parallel, writes per-seed logs and a summary with CIs."""
import hashlib, json, os, platform, time
from multiprocessing import Pool
import numpy as np
from . import sim, stats

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def versions():
    import importlib.metadata as m
    out = {"python": platform.python_version()}
    for p in ["numpy", "scipy", "dp-accounting", "river", "pandas", "matplotlib"]:
        try:
            out[p] = m.version(p)
        except Exception:
            pass
    return out


def cfg_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _job(args):
    cfg, seed, kw = args
    return sim.run_seed(cfg, seed, **kw)


def run(name, overrides=None, seeds=range(100), kw=None, procs=2, subdir="S"):
    cfg = dict(sim.DEFAULT)
    cfg.update(overrides or {})
    kw = kw or {}
    t = time.time()
    with Pool(procs) as p:
        rows = p.map(_job, [(cfg, s, kw) for s in seeds], chunksize=4)
    summ = summarize(rows, cfg)
    out = dict(name=name, config=cfg, kwargs={k: v for k, v in kw.items()}, seeds=[int(s) for s in seeds],
               config_hash=cfg_hash([cfg, kw]), versions=versions(), runtime_s=round(time.time() - t, 1),
               summary=summ, per_seed=rows)
    d = os.path.join(RESULTS, subdir)
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, f"{name}.json"), "w"), indent=1, default=str)
    return out


def summarize(rows, cfg):
    names = list(rows[0]["methods"].keys())
    npre = np.array([r["npre"] for r in rows])
    s = {"k_eff": stats.summarize_rate([r["k_eff"] for r in rows]),
         "p_hat_raw": stats.summarize_rate([r["p_hat_raw"] for r in rows]),
         "p_hat_filtered": stats.summarize_rate([r["p_hat_filtered"] for r in rows]),
         "local_scope_per_1000": stats.summarize_rate([r["local_scope_episodes"] / r["npre"] * 1000 for r in rows]),
         "local_scope_filtered_per_1000": stats.summarize_rate([r["local_scope_episodes_filtered"] / r["npre"] * 1000 for r in rows]),
         "filter_releases_per_node": stats.summarize_rate([r["filter_releases_mean"] for r in rows]),
         "methods": {}}
    for key in ["svt_active_frac", "svt_positives_per_node_per_1000", "svt_instances_mean", "svt_active_frac_filtered",
                "svt_positives_filtered_per_node_per_1000", "svt_instances_filtered_mean"]:
        if key in rows[0]:
            s[key] = stats.summarize_rate([r[key] for r in rows])
    for n in names:
        ms = [r["methods"][n] for r in rows]
        a = {}
        if ms[0].get("false_episodes") is not None:
            a["false_episodes_per_1000"] = stats.summarize_counts_per_1000([m["false_episodes"] for m in ms], npre)
        if "wasted_node_retrains" in ms[0]:
            a["wasted_node_retrains_per_1000"] = stats.summarize_rate([m["wasted_node_retrains"] / r["npre"] * 1000
                                                                     for m, r in zip(ms, rows)])
        if "false_alarms" in ms[0]:
            fa = [m["false_alarms"] for m in ms]
            ep = [m["epochs_pre"] for m in ms]
            a["false_alarms_per_1000_epochs"] = stats.summarize_counts_per_1000(fa, ep)
        if "detected" in ms[0]:
            a["attributable_detection"] = stats.summarize_prop([m["detected"] for m in ms])
            dl = [m.get("delay", m.get("delay_epochs")) for m in ms]
            dl = [x for x in dl if x is not None]
            a["delay_median"] = float(np.median(dl)) if dl else None
            a["delay_iqr"] = [float(np.percentile(dl, 25)), float(np.percentile(dl, 75))] if dl else None
        if "detected_raw" in ms[0]:
            a["raw_detection"] = stats.summarize_prop([m["detected_raw"] for m in ms])
        if "node_detect_frac" in ms[0]:
            a["node_detect_frac"] = stats.summarize_rate([m["node_detect_frac"] for m in ms])
        s["methods"][n] = a
    return s
