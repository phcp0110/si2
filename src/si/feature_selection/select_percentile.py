"""
SelectPercentile 
-------------------------------
Feature selector that keeps the top percentile of features according to a
univariate scoring function (e.g., ANOVA F for classification).

API contract (per slides):
- class SelectPercentile(Transformer)
- parameters: score_func (defaults to f_classification), percentile 
- estimated parameters: F (scores), p (p-values)
- methods: _fit(dataset) -> self, _transform(dataset) -> Dataset
- ties: if the threshold falls on a value shared by multiple features, include
        as many tied features (in stable order) as needed to reach the exact k.
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple, List
import numpy as np

from src.si.base.transformer import Transformer
from src.si.data.dataset import Dataset
from src.si.statistics.f_classification import f_classification


class SelectPercentile(Transformer):
    """
    Selects the top `percentile` percent features according to `score_func`.

    Parameters
    ----------
    score_func : Callable[[Dataset], Tuple[np.ndarray, np.ndarray]]
        Scoring function returning (F, p), each shaped (n_features,).
        Defaults to f_classification.
    percentile : int
        Percent of features to keep in [0, 100].

    Estimated Parameters
    --------------------
    F : np.ndarray | None
        Scores (F-values) per feature, set in _fit().
    p : np.ndarray | None
        P-values per feature, set in _fit().
    """

    def __init__(
        self,
        score_func: Callable[[Dataset], Tuple[np.ndarray, np.ndarray]] = f_classification,
        percentile: int = 10,
    ) -> None:
        super().__init__()
        if not (0 <= int(percentile) <= 100):
            raise ValueError("percentile must be in [0, 100]")
        self.score_func = score_func
        self.percentile = int(percentile)
        self.F: Optional[np.ndarray] = None
        self.p: Optional[np.ndarray] = None
        self._support_mask: Optional[np.ndarray] = None

    # -------------------------- core API per slides --------------------------

    def _fit(self, dataset: Dataset) -> "SelectPercentile":
        """
        Estimates F and p for each feature using `score_func`; returns self.
        """
        if dataset.X.ndim != 2:
            raise ValueError("Dataset.X must be 2D (n_samples, n_features).")

        F, p = self.score_func(dataset)
        F = np.asarray(F)
        p = np.asarray(p)

        if F.ndim != 1:
            raise ValueError("score_func must return a 1D array of scores.")
        if F.shape[0] != dataset.X.shape[1]:
            raise ValueError("scores length must equal n_features.")
        if p.shape != F.shape:
            raise ValueError("p-values must have the same shape as scores.")

        self.F, self.p = F, p
        self._support_mask = None
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects a given percentage of features based on their F-values,
        handling ties at the threshold to keep exactly k features.
        """
        if self.F is None:
            raise RuntimeError("Call fit() before transform().")

        n_features = dataset.X.shape[1]

        # Special case: 0% keeps 0 features
        if self.percentile == 0:
            mask = np.zeros(n_features, dtype=bool)
            self._support_mask = mask
            X_new = dataset.X[:, mask]
            feats = [] if dataset.features is not None else None
            return Dataset(X_new, dataset.y, feats, dataset.label)

        # k = ceil(percentile% of n_features) and at least 1
        k = max(1, int(np.ceil(n_features * (self.percentile / 100.0))))

        # Threshold is the (100 - percentile)th percentile of F
        # NumPy <1.22 compatibility: use interpolation="linear"
        threshold = np.percentile(self.F, 100 - self.percentile, interpolation="linear")

        # First take strictly greater than threshold
        greater_idx = np.where(self.F > threshold)[0].tolist()
        selected: List[int] = list(greater_idx)

        # Then, if needed to reach k, include ties (== threshold) in stable order
        if len(selected) < k:
            ties_idx = np.where(self.F == threshold)[0].tolist()
            needed = k - len(selected)
            selected.extend(ties_idx[:needed])

        # Keep exactly k, in ascending (stable) order
        selected = sorted(selected)[:k]

        mask = np.zeros(n_features, dtype=bool)
        mask[selected] = True
        self._support_mask = mask

        X_new = dataset.X[:, mask]
        new_features = [dataset.features[i] for i in selected] if dataset.features is not None else None
        return Dataset(X_new, dataset.y, new_features, dataset.label)

   