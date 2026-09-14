"""Reproducibility manifest: config hash, package versions, dataset indices.

Every experiment call appends a line to {results_dir}/manifest.jsonl so any
number in the paper can be traced to the exact config, code state, and
environment that produced it.
"""

import json
import platform
import sys

import numpy as np
import torch

from src.common import ensure_dir, sha256_of_file


def write_manifest(results_dir: str, config_path: str, experiment: str, extra: dict = None):
    manifest = {
        "experiment": experiment,
        "config_sha256": sha256_of_file(config_path),
        "config_path": config_path,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if extra:
        manifest.update(extra)
    ensure_dir(results_dir)
    with open(f"{results_dir}/manifest.jsonl", "a") as f:
        f.write(json.dumps(manifest, sort_keys=True) + "\n")
    return manifest
