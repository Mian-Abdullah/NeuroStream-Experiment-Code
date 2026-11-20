"""Dataset loading with provenance, preprocessing without look-ahead, and non-IID federated stream partitioning.

Files are the GitHub mirrors listed in data/PROVENANCE.md (SHA-256 checked at load time). No dataset is resampled,
extended or synthesised; injected events (outages, drift) are applied in ns/stream.py and are always labelled as such.
"""
import hashlib, os
import numpy as np
import pandas as pd

RAW = os.environ.get("NS_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"))
SHA = {
    "elec.csv": "cdf901433885f29eca6911f70c0eeafb50d90596c879c30c5b99f5a2e8e734ff",
    "covtype.csv": "ea35ca6820fcbe0bdc6707e9ed31e11a9de84f07d88da73b511c737383c271a6",
    "airlines.csv": "0bbf1719c7a5c82a1144723c0bdb6d7f563118057e5f979a3c34f824fb40b980",
}


def _check(fname):
    p = os.path.join(RAW, fname)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if fname in SHA and h != SHA[fname]:
        raise RuntimeError(f"checksum mismatch for {fname}")
    return p


def load(name):
    """Returns X (float, original order), y (int, 0..k-1), group (int per row, used for partitioning), meta."""
    if name == "elec":
        d = pd.read_csv(_check("elec.csv"))
        y = d["class"].astype(int).to_numpy()
        X = d.drop(columns=["class"]).to_numpy(float)
        group = None
        meta = dict(features=list(d.columns[:-1]), classes=2)
    elif name == "covtype":
        d = pd.read_csv(_check("covtype.csv"))
        y = d.iloc[:, -1].astype(int).to_numpy() - 1
        X = d.iloc[:, :-1].to_numpy(float)
        group = np.argmax(d[[f"Wilderness_Area{i}" for i in range(1, 5)]].to_numpy(), 1)
        meta = dict(features=list(d.columns[:-1]), classes=7)
    elif name == "airlines":
        d = pd.read_csv(_check("airlines.csv"))
        y = d["Delay"].astype(int).to_numpy()
        al = pd.get_dummies(d["Airline"], prefix="al").to_numpy(float)
        dw = pd.get_dummies(d["DayOfWeek"], prefix="dw").to_numpy(float)
        X = np.concatenate([al, dw, d[["Time", "Length"]].to_numpy(float)], 1)
        group = d["AirportFrom"].to_numpy()
        meta = dict(features="Airline one-hot, DayOfWeek one-hot, Time, Length", classes=2)
    else:
        raise ValueError(name)
    return X, y, group, meta


def standardize_from_prefix(X, n_prefix):
    mu = X[:n_prefix].mean(0)
    sd = X[:n_prefix].std(0)
    sd[sd < 1e-8] = 1.0
    return (X - mu) / sd


def assign_nodes(name, y, group, K, rng, alpha=0.5):
    """Node index per row (or -1 = unused).
    elec: Dirichlet(alpha) label skew (each class's rows spread over nodes with Dirichlet proportions).
    covtype: node i belongs to wilderness area i mod 4; rows of an area go uniformly to that area's nodes.
    airlines: nodes are the K busiest departure airports; other rows unused."""
    n = len(y)
    node = np.full(n, -1)
    if name == "elec":
        k = y.max() + 1
        P = rng.dirichlet(alpha * np.ones(K), size=k)          # per class: distribution over nodes
        u = rng.random(n)
        cum = np.cumsum(P, 1)
        node = np.minimum((u[:, None] > cum[y]).sum(1), K - 1)
    elif name == "covtype":
        area_nodes = {a: [i for i in range(K) if i % 4 == a] for a in range(4)}
        for a, nodes in area_nodes.items():
            idx = np.nonzero(group == a)[0]
            node[idx] = rng.choice(nodes, size=len(idx))
    elif name == "airlines":
        vals, cnt = np.unique(group, return_counts=True)
        top = vals[np.argsort(-cnt)[:K]]
        m = {a: i for i, a in enumerate(top)}
        node = np.array([m.get(a, -1) for a in group])
    return node


def blocks_of(n_rows, B):
    edges = np.linspace(0, n_rows, B + 1).astype(int)
    return np.searchsorted(edges, np.arange(n_rows), side="right") - 1
