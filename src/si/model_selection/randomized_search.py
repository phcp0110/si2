"""
randomized_search_cv
--------------------
Randomized search with cross validation for hyperparameter optimization.

This function evaluates a fixed number of randomly selected hyperparameter
combinations using cross validation and returns the best configuration.
"""

from typing import Dict, Callable, Any
import numpy as np

from si.model_selection.cross_validate import k_fold_cross_validation
from si.data.dataset import Dataset
from si.base.model import Model


def randomized_search_cv(
    model: Model,
    dataset: Dataset,
    hyperparameter_grid: Dict[str, np.ndarray],
    scoring: Callable,
    cv: int = 3,
    n_iter: int = 10,
    random_state: int | None = None
) -> Dict[str, Any]:
    """
    Perform randomized search with cross validation.

    Parameters
    ----------
    model : Model
        Model to optimize.
    dataset : Dataset
        Dataset used for cross validation.
    hyperparameter_grid : dict
        Dictionary mapping hyperparameter names to arrays of possible values.
    scoring : callable
        Scoring function.
    cv : int
        Number of folds.
    n_iter : int
        Number of random hyperparameter combinations to evaluate.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - 'hyperparameters'
        - 'scores'
        - 'best_hyperparameters'
        - 'best_score'
    """

    if random_state is not None:
        np.random.seed(random_state)

    # 1. Validate hyperparameters
    for param in hyperparameter_grid:
        if not hasattr(model, param):
            raise ValueError(f"Model does not have hyperparameter '{param}'")

    param_names = list(hyperparameter_grid.keys())
    param_values = list(hyperparameter_grid.values())

    # 2. Sample random combinations
    sampled_params = []
    scores = []

    for _ in range(n_iter):
        params = {
            name: np.random.choice(values)
            for name, values in zip(param_names, param_values)
        }

        # 3. Set model hyperparameters
        for name, value in params.items():
            setattr(model, name, value)

        # 4. Cross validation
        cv_scores = k_fold_cross_validation(
            model=model,
            dataset=dataset,
            scoring=scoring,
            k=cv
        )

        mean_score = float(np.mean(cv_scores))

        # 5. Store results
        sampled_params.append(params.copy())
        scores.append(mean_score)

    # 6. Identify best result
    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]
    best_params = sampled_params[best_idx]

    return {
        "hyperparameters": sampled_params,
        "scores": scores,
        "best_hyperparameters": best_params,
        "best_score": best_score,
    }
