"""Expanded candidate-feature pool for the scaling study (Option A).

The 13 raw Cleveland features carry little redundancy, so naive top-k mutual
information is already near-optimal (test AUC 0.885 vs 0.896 for all 13 features).
To study what happens when candidates genuinely overlap, this module builds a
LARGER pool:

    columns 0..12   : the 13 raw features, exactly as in data.py
    columns 13..    : one-hot indicator columns for each level of the multi-level
                      categorical variables (cp, restecg, slope, ca, thal), added in
                      a fixed, seeded-shuffled order so any prefix is a balanced mix

Taking the first n columns therefore gives NESTED problem instances
(n = 13, 16, 20, 25, ... up to 13 + number of dummy columns = 30 on Cleveland).

OPTION A CAVEAT (important when interpreting results): a one-hot column is not a
separate clinical measurement. Picking `cp=3` and `cp=4` uses ONE measurement
(chest-pain type) but counts as two features. We treat the expanded pool as a
generic feature-selection benchmark and report, for every selected subset, how many
DISTINCT underlying measurements it covers (see ``distinct_measurements``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from data import load_heart_disease

# Same discrete/continuous split used in the data.py fix.
DISCRETE_RAW = {"sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}
# Multi-level categoricals that get one-hot expanded.
ONE_HOT = ("cp", "restecg", "slope", "ca", "thal")


@dataclass
class PoolData:
    X: np.ndarray               # (m, n) feature matrix
    y: np.ndarray               # (m,) binary labels
    names: list[str]            # column names, e.g. "cp" or "cp=3"
    groups: list[str]           # underlying measurement of each column ("cp" for "cp=3")
    discrete: np.ndarray        # (n,) bool mask, for mutual_info_classif
    relevance: np.ndarray       # (n,) MI with label, normalized to max 1 (full data)
    redundancy: np.ndarray      # (n, n) |Pearson corr|, zero diagonal (full data)

    @property
    def n(self) -> int:
        return self.X.shape[1]

    def distinct_measurements(self, subset) -> int:
        """How many different underlying measurements a column subset covers."""
        return len({self.groups[i] for i in subset})


def relevance_and_redundancy(X, y, discrete, seed: int = 0):
    """Normalized mutual information and |correlation| for a feature matrix.

    Used on the full data here, and inside each training split by the AUC
    evaluation (so selection never sees test labels).
    """
    r = mutual_info_classif(X, y, discrete_features=list(discrete), random_state=seed)
    r = r / r.max() if r.max() > 0 else r
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.abs(np.corrcoef(X, rowvar=False))
    c = np.nan_to_num(c, nan=0.0)               # constant column in a split -> 0
    np.fill_diagonal(c, 0.0)
    return r, c


def max_pool_size() -> int:
    base = load_heart_disease()
    extra = 0
    for var in ONE_HOT:
        extra += len(np.unique(base.X[:, base.feature_names.index(var)]))
    return base.n_features + extra


def load_expanded(n_features: int = 25, order_seed: int = 0, mi_seed: int = 0) -> PoolData:
    """Build the first ``n_features`` columns of the expanded pool (n >= 13)."""
    base = load_heart_disease()
    names, X, y = list(base.feature_names), base.X, base.y

    derived_cols, derived_names, derived_groups = [], [], []
    for var in ONE_HOT:
        j = names.index(var)
        for lvl in np.unique(X[:, j]):
            derived_cols.append((X[:, j] == lvl).astype(float))
            derived_names.append(f"{var}={lvl:g}")
            derived_groups.append(var)

    total = len(names) + len(derived_cols)
    if not (len(names) <= n_features <= total):
        raise ValueError(f"n_features must be between {len(names)} and {total}")

    order = np.random.default_rng(order_seed).permutation(len(derived_cols))
    keep = order[: n_features - len(names)]

    pool_X = np.column_stack([X] + [derived_cols[i] for i in keep])
    pool_names = names + [derived_names[i] for i in keep]
    pool_groups = names + [derived_groups[i] for i in keep]     # raw columns are their own group
    discrete = np.array([nm.split("=")[0] in DISCRETE_RAW for nm in pool_names])

    r, c = relevance_and_redundancy(pool_X, y, discrete, seed=mi_seed)
    return PoolData(pool_X, y, pool_names, pool_groups, discrete, r, c)


if __name__ == "__main__":
    for n in (13, 16, 20, 25):
        pool = load_expanded(n)
        print(f"n={n}: {pool.n} columns; added: {pool.names[13:]}")
    print(f"max pool size: {max_pool_size()}")