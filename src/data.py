"""Dataset loading with deterministic, documented subset selection.

Subset identity is part of the experimental record: the harness writes the
selected indices to results/manifests/ so any table can be traced back to
exact images.
"""

import numpy as np
import torchvision


def load_cifar10_subsets(n_test: int = 1000, n_train: int = 5000, seed: int = 42):
    """Returns (test_images, test_labels, test_indices, train_images, train_labels, train_indices).

    - test pool: CIFAR-10 test split (10k), seeded choice of n_test.
    - train pool: CIFAR-10 train split (50k), seeded choice of n_train.
    Disjointness is structural (different splits), stated here so reviewers
    don't have to dig for it.
    """
    raw_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    raw_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)

    rng = np.random.default_rng(seed)
    test_idx = rng.choice(len(raw_test), size=n_test, replace=False)
    train_idx = rng.choice(len(raw_train), size=n_train, replace=False)

    test_images, test_labels = [], []
    for i in test_idx:
        img, label = raw_test[i]
        test_images.append(np.array(img))
        test_labels.append(label)

    train_images, train_labels = [], []
    for i in train_idx:
        img, label = raw_train[i]
        train_images.append(np.array(img))
        train_labels.append(label)

    return (test_images, np.array(test_labels), test_idx,
            train_images, np.array(train_labels), train_idx)
