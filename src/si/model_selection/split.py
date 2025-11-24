from typing import Tuple

import numpy as np

from si.data.dataset import Dataset

from si.statistics.euclidean_distance import euclidean_distance



def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test

#Exercicio 6

def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into stratified training and testing sets.

    Parameters
    ----------
    dataset: Dataset
        The dataset to split.
    test_size: float
        The proportion of the dataset to include in the test split.
    random_state: int
        The seed of the random number generator.

    Returns
    -------
    train: Dataset
        The training dataset.
    test: Dataset
        The testing dataset.
    """
    # Set random state
    np.random.seed(random_state)

    # Get unique class labels and their counts
    y = dataset.y
    unique_labels, label_counts = np.unique(y, return_counts=True)

    train_indices = []
    test_indices = []

    for label, count in zip(unique_labels, label_counts):
        # Get indices for the current label
        label_indices = np.where(y == label)[0]

        # Shuffle the indices
        np.random.shuffle(label_indices)

        # Calculate the number of test samples for the current class
        n_test = int(test_size * count)

        # Split into train and test
        test_indices.extend(label_indices[:n_test])
        train_indices.extend(label_indices[n_test:])

    # Create train and test datasets
    train = Dataset(dataset.X[train_indices], dataset.y[train_indices],
                    features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_indices], dataset.y[test_indices],
                   features=dataset.features, label=dataset.label)

    return train, test