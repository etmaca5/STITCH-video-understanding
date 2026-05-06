"""Single-vector chunk aggregation over window embeddings.

Given L2-normalized window embeddings {e_1, ..., e_n} in a chunk, produce
one unit vector c summarizing the chunk, plus the resultant length R
(vMF concentration / coherence in [0, 1]).

MaxSim s(q, C) = max_i cos(q, e_i) has no exact single-vector surrogate
since q -> max_i q.e_i is piecewise linear in q. The methods here are
principled single-vector approximations:

    mean             spherical / vMF mean direction
    gem              signed generalized (power) mean
    coherence_mean   medoid-flavored weighted spherical mean
    lse              coordinate-wise log-sum-exp (smooth max)
"""

import numpy as np


_METHODS = ("mean", "gem", "coherence_mean", "lse")


def aggregate(
    window_embs: np.ndarray,
    method: str = "mean",
    gem_p: float = 3.0,
    coherence_tau: float = 10.0,
    lse_tau: float = 10.0,
) -> tuple[np.ndarray, float]:
    """Aggregate (n, d) window embeddings into a single unit vector.

    Args:
        window_embs: (n, d) array of L2-normalized window embeddings.
        method: one of {"mean", "gem", "coherence_mean", "lse"}.
        gem_p: exponent for GeM (p=1 -> mean, p->inf -> coord-wise max).
        coherence_tau: inverse temperature for coherence_mean weights.
        lse_tau: inverse temperature for log-sum-exp.

    Returns:
        (c, R) where c is a (d,) unit vector and R is the resultant length
        ||(1/n) sum_i e_i|| in [0, 1] (chunk coherence, independent of
        the chosen method).
    """
    E = np.asarray(window_embs, dtype=np.float64)
    if E.ndim != 2:
        raise ValueError(f"window_embs must be 2-D, got shape {E.shape}")
    n = E.shape[0]
    if n == 0:
        raise ValueError("window_embs must contain at least one window")

    mean_vec = E.mean(axis=0)
    R = float(np.linalg.norm(mean_vec))

    if method == "mean":
        c = mean_vec
    elif method == "gem":
        c = _gem(E, gem_p)
    elif method == "coherence_mean":
        c = _coherence_mean(E, coherence_tau)
    elif method == "lse":
        c = _lse(E, lse_tau)
    else:
        raise ValueError(
            f"Unknown aggregation method: {method!r}. "
            f"Must be one of {_METHODS}."
        )

    c = _l2_normalize(c)
    return c.astype(np.float32), R


def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < eps:
        return v
    return v / norm


def _gem(E: np.ndarray, p: float) -> np.ndarray:
    """Signed generalized mean: c_j = sign(m_j) |m_j|^(1/p),
    m_j = mean_i sign(e_ij) |e_ij|^p."""
    if p <= 0:
        raise ValueError(f"gem_p must be positive, got {p}")
    signs = np.sign(E)
    powered = signs * np.power(np.abs(E), p)
    m = powered.mean(axis=0)
    return np.sign(m) * np.power(np.abs(m), 1.0 / p)


def _coherence_mean(E: np.ndarray, tau: float) -> np.ndarray:
    """Weighted spherical mean with w_i = sum_j exp(tau * e_i . e_j).

    tau = 0 reduces to the plain spherical mean (equal weights).
    Numerically stable: shift the exponent by its row max.
    """
    logits = tau * (E @ E.T)
    logits = logits - logits.max(axis=1, keepdims=True)
    w = np.exp(logits).sum(axis=1)
    w = w / w.sum()
    return w @ E


def _lse(E: np.ndarray, tau: float) -> np.ndarray:
    """Coordinate-wise log-sum-exp: c_j = (1/tau) log mean_i exp(tau e_ij).

    tau -> 0: recovers the mean (by L'Hopital).
    tau -> inf: recovers the coordinate-wise max.
    Stable: shift by per-coordinate max before exp.
    """
    if tau <= 0:
        raise ValueError(f"lse_tau must be positive, got {tau}")
    n = E.shape[0]
    M = E.max(axis=0)
    shifted = np.exp(tau * (E - M))
    return M + np.log(shifted.mean(axis=0)) / tau
