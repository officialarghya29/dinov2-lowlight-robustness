"""Statistical machinery: everything the paper's significance claims rest on.

- Unbiased linear CKA (sample-size bias corrected; matters at n=300).
- Participation ratio (effective dimensionality of the embedding cloud).
- Two null models for the accuracy-vs-severity curve (linearity and
  midpoint-symmetry) with exact permutation tests.
- Paired permutation tests between conditions.
- Multiset-distance permutation test for AOPC curves.
- Bootstrap and Wilson intervals.

Design note: permutation tests are exact under exchangeability and make no
distributional assumptions — appropriate when accuracy is a bounded proportion
computed on a small test fold.
"""

import numpy as np

from src.common import wilson_ci


# ---------------------------------------------------------------------------
# Representation similarity
# ---------------------------------------------------------------------------

def linear_cka_unbiased(X: np.ndarray, Y: np.ndarray) -> float:
    """Unbiased estimator of linear CKA (Kornblith et al. 2019, Appendix A).

    Uses only the OFF-DIAGONAL entries of the feature-centered Gram matrices:

        CKA_u = sum_{i≠j} K_ij L_ij / sqrt( sum_{i≠j} K_ij^2 * sum_{i≠j} L_ij^2 )

    The standard (biased) estimator includes the diagonal (self-similarity)
    terms, which inflates similarity toward 1 when n is small relative to d —
    at n=300, d=384 this bias is material (independent representations score
    ~0.88 under the biased estimator vs ~0 under this one).
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    n = X.shape[0]
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    K = Xc @ Xc.T  # Gram matrix, samples x samples
    L = Yc @ Yc.T
    off = ~np.eye(n, dtype=bool)
    hsic = float((K[off] * L[off]).sum())
    var_x = float((K[off] ** 2).sum())
    var_y = float((L[off] ** 2).sum())
    return float(hsic / np.sqrt(var_x * var_y + 1e-12))


def participation_ratio(X: np.ndarray) -> float:
    """PR = (sum lambda)^2 / sum lambda^2 of covariance eigenvalues.

    The effective number of dimensions carrying variance. A drop means the
    embedding cloud collapses toward a low-dim manifold (e.g. a 'dark blob');
    stability with low cosine-to-clean means rotation instead.
    """
    S = np.cov(X.astype(np.float64).T)
    eig = np.linalg.eigvalsh(S)[::-1]
    eig = np.clip(eig, 0, None)
    return float((eig.sum() ** 2) / ((eig ** 2).sum() + 1e-12))


# ---------------------------------------------------------------------------
# Null models + permutation tests for the severity curve
# ---------------------------------------------------------------------------

def _curve_ssr(acc: np.ndarray) -> float:
    """Sum of squared residuals of the least-squares line fit. Large SSR =
    structured non-linearity (cliff + floor)."""
    s = np.arange(len(acc), dtype=np.float64)
    A = np.stack([np.ones_like(s), s], axis=1)
    coef, *_ = np.linalg.lstsq(A, acc, rcond=None)
    resid = acc - A @ coef
    return float(np.sum(resid ** 2))


def _curve_midpoint_gap(acc: np.ndarray) -> float:
    """|acc(2) - acc(4)| — the symmetry contrast."""
    return float(abs(acc[2] - acc[4]))


def _permute_curve(acc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permutation scheme: resample each severity's correctness vector.

    Each severity's test correctness vector is shuffled in place — preserves
    the per-severity Bernoulli structure while breaking any relationship
    between severities. Exact under the null that severities are exchangeable.
    """
    out = np.empty_like(acc)
    for j in range(len(acc)):
        out[j] = rng.permutation(acc[j]).mean() if isinstance(acc[j], np.ndarray) else acc[j]
    return out


def curve_permutation_test(correctness_by_severity, n_perm: int = 5000, seed: int = 42) -> dict:
    """Parametric-bootstrap null tests for the severity curve.

    Test 1 (non-linearity): fit the least-squares line to the observed
    accuracies; under H0 ('the curve IS linear'), simulate correctness vectors
    as Bernoulli(p_lin(s)) at each severity and compute the null distribution
    of SSR. p = P(SSR_null >= SSR_obs). Shuffling rows within a severity would
    be a no-op (row means are permutation-invariant), so resampling under the
    fitted null is the correct construction.

    Test 2 (midpoint asymmetry): H0 = acc(2) == acc(4). Pool severities 2 and 4
    into one Bernoulli rate and bootstrap the |difference| distribution.

    correctness_by_severity: list of 6 boolean arrays (test-fold correctness).
    """
    rng = np.random.default_rng(seed)
    corr = np.stack([np.asarray(c, dtype=float) for c in correctness_by_severity])
    n = corr.shape[1]
    accs = corr.mean(axis=1)

    obs_ssr = _curve_ssr(accs)
    s = np.arange(len(accs), dtype=np.float64)
    A = np.stack([np.ones_like(s), s], axis=1)
    coef, *_ = np.linalg.lstsq(A, accs, rcond=None)
    p_lin = np.clip(A @ coef, 0.0, 1.0)

    null_ssr = np.empty(n_perm)
    for i in range(n_perm):
        sim = np.stack([(rng.random(n) < p_lens_s) for p_lens_s in p_lin]).astype(float)
        null_ssr[i] = _curve_ssr(sim.mean(axis=1))

    p_pool = (corr[2].sum() + corr[4].sum()) / (2 * n)
    obs_gap = _curve_midpoint_gap(accs)
    null_gap = np.empty(n_perm)
    for i in range(n_perm):
        a = (rng.random(n) < p_pool).mean()
        b = (rng.random(n) < p_pool).mean()
        null_gap[i] = abs(a - b)

    return {
        "observed_ssr": obs_ssr,
        "p_not_linear": float((np.sum(null_ssr >= obs_ssr) + 1) / (n_perm + 1)),
        "observed_midpoint_gap": obs_gap,
        "p_midpoint_asymmetric": float((np.sum(null_gap >= obs_gap) + 1) / (n_perm + 1)),
        "n_perm": n_perm,
    }


def paired_permutation_test(scores_a: np.ndarray, scores_b: np.ndarray,
                            n_perm: int = 5000, seed: int = 42) -> dict:
    """Exact paired permutation test on per-example scores (correctness vectors,
    or per-example margins). H0: A and B are exchangeable per example.
    """
    rng = np.random.default_rng(seed)
    diff = np.mean(scores_a) - np.mean(scores_b)
    n = len(scores_a)
    count = 0
    for _ in range(n_perm):
        mask = rng.random(n) < 0.5
        d = (np.where(mask, scores_a, scores_b) - np.where(mask, scores_b, scores_a)).mean()
        if abs(d) >= abs(diff):
            count += 1
    return {
        "observed_diff": float(diff),
        "p_value": float((count + 1) / (n_perm + 1)),
        "n_perm": n_perm,
        "n": n,
    }


def multiset_distance_test(curve_a: np.ndarray, curve_b: np.ndarray,
                           n_perm: int = 5000, seed: int = 42) -> dict:
    """Permutation test for AOPC-style curves: H0 = the two curves are draws
    from the same multiset (permutation-invariant distance). Addresses the
    failure mode of endpoint-only comparisons.
    """
    rng = np.random.default_rng(seed)
    obs = float(np.abs(np.sort(curve_a) - np.sort(curve_b)).mean())
    pooled = np.concatenate([curve_a, curve_b])
    na = len(curve_a)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        d = np.abs(np.sort(perm[:na]) - np.sort(perm[na:])).mean()
        if d >= obs:
            count += 1
    return {
        "observed_distance": obs,
        "p_value": float((count + 1) / (n_perm + 1)),
        "n_perm": n_perm,
    }


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------

def bootstrap_accuracy_ci(correct: np.ndarray, n_boot: int = 2000, ci: int = 95,
                          seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(correct)
    boots = correct[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return {"mean": float(correct.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "wilson_low": float(wilson_ci(int(correct.sum()), n)[0]),
            "wilson_high": float(wilson_ci(int(correct.sum()), n)[1])}


def benjamini_hochberg(pvals: list) -> list:
    """BH FDR correction. Returns adjusted p-values (same order)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out.tolist()
