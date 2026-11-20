"""Federated softmax regression in NumPy: local training, DP updates, robust aggregators and Byzantine attacks."""
import numpy as np
from scipy.stats import norm


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Softmax:
    """Multinomial logistic regression; parameters flattened as a vector of size (d+1)*k."""

    def __init__(self, d, k):
        self.d, self.k = d, k

    def size(self):
        return (self.d + 1) * self.k

    def unpack(self, w):
        W = w.reshape(self.d + 1, self.k)
        return W[:-1], W[-1]

    def predict(self, w, X):
        Wm, b = self.unpack(w)
        return np.argmax(X @ Wm + b, 1)

    def per_example_grads(self, w, X, y):
        Wm, b = self.unpack(w)
        P = softmax(X @ Wm + b)
        P[np.arange(len(y)), y] -= 1.0
        # gradient wrt W for each example: x_i outer P_i  -> flatten with bias
        Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
        return (Xb[:, :, None] * P[:, None, :]).reshape(len(X), -1)

    def grad(self, w, X, y, l2=1e-4):
        Wm, b = self.unpack(w)
        P = softmax(X @ Wm + b)
        P[np.arange(len(y)), y] -= 1.0
        gW = X.T @ P / len(y)
        gb = P.mean(0)
        return np.concatenate([gW, gb[None, :]], 0).ravel() + l2 * w


def local_update(model, w, X, y, steps=5, lr=0.5, dp=None, rng=None):
    """Full-batch local gradient descent. dp = (clip C, noise multiplier sigma): per-example clipping, noise
    N(0, (sigma C)^2) added to the summed gradient at every step (record-level Gaussian mechanism per step).
    Returns the model delta."""
    w0 = w.copy()
    for _ in range(steps):
        if len(y) == 0:
            break
        if dp is None:
            g = model.grad(w, X, y)
        else:
            C, sigma = dp
            G = model.per_example_grads(w, X, y)
            nrm = np.linalg.norm(G, axis=1, keepdims=True)
            G = G * np.minimum(1.0, C / (nrm + 1e-12))
            g = (G.sum(0) + rng.normal(0, sigma * C, G.shape[1])) / max(len(y), 1)
        w = w - lr * g
    return w - w0


# ------------------------------------------------------------------------------------------------- aggregators
def agg_mean(U, **k):
    return U.mean(0)


def agg_median(U, **k):
    return np.median(U, 0)


def agg_trimmed(U, beta=0.1, **k):
    s = np.sort(U, 0)
    t = int(np.floor(beta * len(U)))
    return s[t:len(U) - t].mean(0)


def _krum_scores(U, f):
    D = ((U[:, None, :] - U[None, :, :]) ** 2).sum(2)
    m = max(len(U) - f - 2, 1)
    return np.sort(D, 1)[:, 1:m + 1].sum(1)


def agg_krum(U, f=0, **k):
    return U[np.argmin(_krum_scores(U, f))]


def agg_multikrum(U, f=0, **k):
    sc = _krum_scores(U, f)
    m = max(len(U) - f, 1)
    return U[np.argsort(sc)[:m]].mean(0)


def agg_cclip(U, prev=None, tau=1.0, iters=3, **k):
    """Centered clipping (Karimireddy et al. 2021) around the previous aggregate."""
    v = np.zeros(U.shape[1]) if prev is None else prev.copy()
    for _ in range(iters):
        dlt = U - v
        nr = np.linalg.norm(dlt, axis=1, keepdims=True)
        v = v + (dlt * np.minimum(1.0, tau / (nr + 1e-12))).mean(0)
    return v


def agg_fltrust(U, root_update=None, **k):
    """FLTrust (Cao et al. 2021): ReLU cosine trust scores w.r.t. a server update on a small root set,
    client updates rescaled to the server update's norm."""
    g0 = root_update
    n0 = np.linalg.norm(g0) + 1e-12
    nr = np.linalg.norm(U, axis=1) + 1e-12
    ts = np.maximum(0.0, U @ g0 / (nr * n0))
    if ts.sum() == 0:
        return np.zeros_like(g0)
    return (ts[:, None] * U * (n0 / nr)[:, None]).sum(0) / ts.sum()


AGG = dict(mean=agg_mean, median=agg_median, trimmed=agg_trimmed, krum=agg_krum, multikrum=agg_multikrum,
           cclip=agg_cclip, fltrust=agg_fltrust)


# ------------------------------------------------------------------------------------------------- attacks
def attack(U_benign, n_bad, kind):
    """Return the n_bad malicious updates given the benign updates (omniscient attacker)."""
    mu, sd = U_benign.mean(0), U_benign.std(0) + 1e-12
    K = len(U_benign) + n_bad
    if kind == "signflip":
        return -3.0 * U_benign[:n_bad]
    if kind == "alie":       # Baruch et al. 2019, "A little is enough"
        s = int(np.floor(K / 2 + 1)) - n_bad
        z = norm.ppf((K - s) / K) if K > s else 1.0
        return np.repeat((mu - z * sd)[None, :], n_bad, 0)
    if kind == "ipm":        # inner-product manipulation (Xie et al. 2020), epsilon = 0.5
        return np.repeat((-0.5 * mu)[None, :], n_bad, 0)
    if kind == "minmax":     # Shejwalkar & Houmansadr 2021, unit-vector deviation
        dev = -mu / (np.linalg.norm(mu) + 1e-12)
        D = np.sqrt(((U_benign[:, None] - U_benign[None]) ** 2).sum(2)).max()
        lo, hi = 0.0, 50.0
        for _ in range(30):
            g = (lo + hi) / 2
            m = mu + g * dev
            if np.sqrt(((U_benign - m) ** 2).sum(1)).max() <= D:
                lo = g
            else:
                hi = g
        return np.repeat((mu + lo * dev)[None, :], n_bad, 0)
    raise ValueError(kind)
