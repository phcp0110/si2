"""
StackingClassifier
------------------
Ensemble classifier that trains a final model using the predictions
of multiple base classifiers.

Workflow
--------
1. Train base models on the original dataset.
2. Use base model predictions as meta-features.
3. Train a final model on these meta-features.
4. Predict by repeating steps 2 and 3 on new data.
"""

from __future__ import annotations
from typing import List
import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class StackingClassifier(Model):
    """
    Stacking ensemble classifier.

    Parameters
    ----------
    models : list of Model
        Base classifiers (level-0 models).
    final_model : Model
        Meta-classifier (level-1 model).
    """

    def __init__(self, models: List[Model], final_model: Model):
        super().__init__()

        if not models:
            raise ValueError("models must be a non-empty list")
        if final_model is None:
            raise ValueError("final_model must not be None")

        self.models = models
        self.final_model = final_model

    def _meta_features(self, dataset: Dataset) -> np.ndarray:
        """
        Generate meta-features from base model predictions.

        Each column corresponds to predictions from one base model.
        """
        predictions = []

        for model in self.models:
            y_pred = model._predict(dataset)
            predictions.append(np.asarray(y_pred).reshape(-1))

        return np.column_stack(predictions)

    def _fit(self, dataset: Dataset) -> "StackingClassifier":
        """
        Fit base models and then the final model.
        """
        if dataset.y is None:
            raise ValueError("Dataset must have labels")

        # Train base models
        for model in self.models:
            model._fit(dataset)

        # Create meta-dataset
        Z = self._meta_features(dataset)
        meta_dataset = Dataset(Z, dataset.y)

        # Train final model
        self.final_model._fit(meta_dataset)

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict using the stacking ensemble.
        """
        Z = self._meta_features(dataset)
        meta_dataset = Dataset(Z, None)

        return self.final_model._predict(meta_dataset)

    def _score(self, dataset: Dataset) -> float:
        """
        Compute accuracy on a labeled dataset.
        """
        y_pred = self._predict(dataset)
        return accuracy(dataset.y, y_pred)


