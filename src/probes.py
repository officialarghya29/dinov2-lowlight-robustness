"""Probes: fixed, severity-adapted, and n-trainable linear readouts (for AOPC).

A note on the severity-adapted probe: it is trained on the SAME images as the
fixed probe's training fold, only degraded. This is a stricter control than
reusing the test fold — it isolates "the probe's decision boundary is stale"
from "the model simply saw different images".
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_probe(X: np.ndarray, y: np.ndarray, C: float = 1.0, max_iter: int = 2000,
              class_weight=None) -> LogisticRegression:
    """Linear readout. class_weight='balanced' reweights classes inversely to
    their training frequency, removing prior dominance (used by the
    class-balanced floor test, corollary 2 of the paper)."""
    probe = LogisticRegression(max_iter=max_iter, C=C, class_weight=class_weight)
    probe.fit(X, y)
    return probe


def fit_probe_on_severity(extractor, images, labels, idx_train, severity, low_light_fn,
                          C: float = 1.0):
    """Train a probe on degraded versions of the fixed probe's own train fold."""
    degraded = [low_light_fn(images[i], severity) for i in idx_train]
    E = extractor(degraded)
    return fit_probe(E, labels[idx_train], C=C)


class NTrainableProbe:
    """Logistic head trained on n per-class examples — measures sample efficiency.

    Used for the 'information is present but needs more supervision to read out'
    analysis. k=1 per class is the extreme few-shot readout.
    """

    def __init__(self, n_per_class: int, seed: int = 42):
        self.n_per_class = n_per_class
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.linear_model import SGDClassifier

        rng = np.random.default_rng(self.seed)
        classes = np.unique(y)
        idx = []
        for c in classes:
            c_idx = np.where(y == c)[0]
            take = min(self.n_per_class, len(c_idx))
            idx.extend(rng.choice(c_idx, size=take, replace=False))
        idx = np.array(idx)
        self._clf = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2000,
                                  tol=1e-4, random_state=self.seed)
        self._clf.fit(X[idx], y[idx])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict(X)


def stratified_split(labels: np.ndarray, test_fraction: float, seed: int):
    from sklearn.model_selection import train_test_split

    n = len(labels)
    idx = np.arange(n)
    _, _, y_tr, y_te, idx_tr, idx_te = train_test_split(
        np.zeros(n), labels, idx, test_size=test_fraction,
        random_state=seed, stratify=labels)
    return idx_tr, idx_te
