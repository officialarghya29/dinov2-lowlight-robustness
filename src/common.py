"""Shared utilities: seeding, device selection, config loading, output paths.

This module must stay importable without torch: the analysis stack
(src.analysis, tests, CI) uses it while CI deliberately runs torch-free.
Torch-specific behavior (RNG seeding, CUDA device selection) activates
only when torch is installed.
"""

import hashlib
import json
import os
import random

import numpy as np


def _torch():
    """Return the torch module, or None when torch is not installed."""
    try:
        import torch
        return torch
    except ImportError:
        return None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch = _torch()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml_config(path: str) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def save_csv(rows, path: str, fieldnames=None) -> None:
    import csv

    ensure_dir(os.path.dirname(path) or ".")
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv_rows(path: str):
    import csv

    with open(path) as f:
        return list(csv.DictReader(f))


def matplotlib_agg():
    """Headless matplotlib for scripts/notebook runners."""
    import matplotlib

    matplotlib.use("Agg")
    return matplotlib


def format_perm_pvalue(p: float) -> str:
    return "p < 0.0001" if p < 1e-4 else f"p = {p:.4f}"


def wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple:
    """Wilson score interval for a binomial proportion (better than naive Wald)."""
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    # fp guard: for k=0 / k=n the exact limits are 0 and 1, but center±half can
    # land at 0.9999999999999999-style values that never clamp to the boundary.
    if hi > 1.0 - 1e-9:
        hi = 1.0
    if lo < 1e-9:
        lo = 0.0
    return (float(lo), float(hi))
