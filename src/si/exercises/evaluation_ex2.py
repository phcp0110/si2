"""
Exercise 2 - Dataset methods 
--------------------------------------------
Demonstrates how to use the new Dataset methods:
- dropna()
- fillna()
- remove_by_index()
"""

import numpy as np
from src.si.data.dataset import Dataset

def main():
    # Example dataset with NaN values
    X = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, np.nan], [5.0, 6.0]])
    y = np.array([0.0, 1.0, np.nan, 0.0])

    print("=== Original Dataset ===")
    ds = Dataset(X, y)
    print(ds.to_dataframe(), "\n")

    print("=== dropna() ===")
    ds_clean = Dataset(X.copy(), y.copy()).dropna()
    print(ds_clean.to_dataframe(), "\n")

    print("=== fillna('mean') ===")
    ds_filled = Dataset(X.copy()).fillna("mean")
    print(ds_filled.to_dataframe(), "\n")

    print("=== remove_by_index(1) ===")
    ds_removed = Dataset(X.copy(), y.copy()).remove_by_index(1)
    print(ds_removed.to_dataframe(), "\n")

if __name__ == "__main__":
    main()
