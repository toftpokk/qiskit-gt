"""Load and featurize the UCI Heart Disease dataset for the feature-selection QUBO.

This turns the raw dataset into the two quantities the QUBO needs (see
``README.md``):

* ``relevance``  — ``r_i``, the mutual information between feature ``i`` and the
  (binarized) label.
* ``redundancy`` — ``c_ij``, the absolute Pearson correlation between features
  ``i`` and ``j``.

Everything else in the pipeline (the QUBO, the QAOA ansatz, the optimizer) is
dataset-agnostic and lives in ``qaoa.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from ucimlrepo import fetch_ucirepo


@dataclass
class HeartData:
    """Cleaned Heart Disease dataset plus its QUBO features."""

    relevance: np.ndarray      # r_i (n,)      mutual information with label
    redundancy: np.ndarray     # c_ij (n, n)   |Pearson correlation|
    X: np.ndarray              # feature matrix (m, n)
    y: np.ndarray              # binarized labels (m,) 0 = no disease, 1 = disease
    feature_names: list[str]
    n_features: int
    n_samples: int

    def relevance_ranking(self) -> list[tuple[str, float]]:
        """Features ordered by decreasing relevance."""
        order = np.argsort(self.relevance)[::-1]
        return [(self.feature_names[i], float(self.relevance[i])) for i in order]


def load_heart_disease(seed: int = 42) -> HeartData:
    """Fetch, clean, and featurize the UCI Heart Disease dataset (id=45).

    * Drops rows with any missing entry (Cleveland subset -> 297 patients).
    * Binarizes the 0-4 target: ``0`` = no disease, ``1-4`` = disease.
    * ``relevance``  = ``mutual_info_classif`` (KNN estimator, ``random_state``
      set for reproducibility).
    * ``redundancy`` = ``|corr|`` from ``np.corrcoef``; diagonal zeroed.
    """
    heart = fetch_ucirepo(id=45)
    X = heart.data.features.copy()
    y = heart.data.targets["num"].copy()

    mask = X.notna().all(axis=1)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    # ``ca`` / ``thal`` are integer-coded but become float once NaN is present.
    for col in ("ca", "thal"):
        X[col] = X[col].astype(int)

    feature_names = list(X.columns)
    Xm = X.to_numpy(dtype=float)
    y_bin = (y > 0).astype(int).to_numpy()

    relevance = mutual_info_classif(
        Xm, y_bin, discrete_features="auto", random_state=seed
    )

    redundancy = np.abs(np.corrcoef(Xm, rowvar=False))
    np.fill_diagonal(redundancy, 0.0)

    return HeartData(
        relevance=relevance,
        redundancy=redundancy,
        X=Xm,
        y=y_bin,
        feature_names=feature_names,
        n_features=len(feature_names),
        n_samples=len(Xm),
    )
