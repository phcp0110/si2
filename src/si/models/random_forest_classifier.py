import numpy as np
from typing import List, Tuple, Optional

from si.data.dataset import Dataset
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.metrics.accuracy import accuracy


class RandomForestClassifier:
    """
    Random forest classifier using an ensemble of decision trees.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of decision trees in the forest.
    max_features : Optional[int], default=None
        Maximum number of features to use per tree. If None, uses sqrt(n_features).
    min_sample_split : int, default=2
        Minimum number of samples required to split an internal node.
    max_depth : Optional[int], default=None
        Maximum depth of each decision tree. If None, trees are expanded until pure.
    mode : str, default="gini"
        Impurity criterion used in the decision trees ("gini" or "entropy").
    seed : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_features: Optional[int] = None,
        min_sample_split: int = 2,
        max_depth: Optional[int] = None,
        mode: str = "gini",
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed

        # estimated parameters
        # list of tuples: (feature_indices, trained_tree)
        self.trees: List[Tuple[np.ndarray, DecisionTreeClassifier]] = []

    # ------------------------------------------------------------------ #
    #                            CORE METHODS                            #
    # ------------------------------------------------------------------ #

    def _fit(self, dataset: Dataset) -> "RandomForestClassifier":
        """
        Train the random forest on the given dataset using bootstrap samples
        and random feature subsets for each tree.

        Parameters
        ----------
        dataset : Dataset
            Training dataset.

        Returns
        -------
        self : RandomForestClassifier
        """
        np.random.seed(self.seed)

        X, y = dataset.X, dataset.y
        n_samples, n_features = X.shape

        # if max_features is not set, use sqrt(n_features) as in the slides
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        self.trees = []

        for _ in range(self.n_estimators):
            # 1) bootstrap samples (with replacement)
            sample_idx = np.random.randint(0, n_samples, size=n_samples)

            # 2) random subset of features (without replacement)
            feature_idx = np.random.choice(
                n_features, size=self.max_features, replace=False
            )

            # build bootstrap dataset with selected samples and features
            X_bootstrap = X[sample_idx][:, feature_idx]
            y_bootstrap = y[sample_idx]

            if dataset.features is not None:
                feat_names = [dataset.features[j] for j in feature_idx]
            else:
                feat_names = None

            bootstrap_ds = Dataset(
                X_bootstrap, y_bootstrap, features=feat_names, label=dataset.label
            )

            # 3) train a decision tree on the bootstrap dataset
            tree = DecisionTreeClassifier(
                min_sample_split=self.min_sample_split,
                max_depth=self.max_depth,
                mode=self.mode,
            )
            tree._fit(bootstrap_ds)

            # 4) store (features used, trained tree)
            self.trees.append((feature_idx, tree))

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class labels for the given dataset using majority voting
        over all trees in the forest.

        Parameters
        ----------
        dataset : Dataset
            Dataset to predict.

        Returns
        -------
        y_pred : np.ndarray
            Predicted class labels.
        """
        if not self.trees:
            raise RuntimeError("RandomForestClassifier must be fitted before calling _predict().")

        X = dataset.X
        n_samples = X.shape[0]

        # collect predictions from all trees
        all_preds = []

        for feature_idx, tree in self.trees:
            X_sub = X[:, feature_idx]
            ds_sub = Dataset(X_sub, None, features=None, label=None)
            y_tree = tree._predict(ds_sub)
            all_preds.append(y_tree)

        all_preds = np.array(all_preds)  # shape: (n_trees, n_samples)

        # majority vote for each sample
        y_final = np.empty(n_samples, dtype=all_preds.dtype)
        for i in range(n_samples):
            votes = all_preds[:, i]
            values, counts = np.unique(votes, return_counts=True)
            y_final[i] = values[np.argmax(counts)]

        return y_final

    def _score(self, dataset: Dataset) -> float:
        """
        Compute the accuracy of the random forest on the given dataset.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        score : float
            Accuracy between predicted and true labels.
        """
        y_pred = self._predict(dataset)
        return accuracy(dataset.y, y_pred)
