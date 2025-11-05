"""
Exercise 1 - Dataset exploration and indexing 
----------------------------------------------------------
Performs simple dataset operations using the Iris dataset:
- Accessing columns and slices
- Filtering samples
- Computing simple statistics
"""

import numpy as np
from src.si.io.csv_file import read_csv  



def main():
    # Load the dataset
    ds = read_csv("datasets/iris.csv", sep=",", features=True, label=True)

    # 1.2) Penultimate feature
    penultimate_feature = ds.X[:, -2]
    print("Penultimate feature shape:", penultimate_feature.shape)

    # 1.3) Last 10 samples and mean per feature
    last_10 = ds.X[-10:, :]
    print("Mean of last 10 samples per feature:", last_10.mean(axis=0))

    # 1.4) Samples where ALL feature values ≤ 6
    mask_all_le6 = np.all(ds.X <= 6, axis=1)
    print("Samples with all features ≤ 6:", int(mask_all_le6.sum()))

    # 1.5) Samples whose label is not 'Iris-setosa'
    if ds.has_label():
        mask_not_setosa = ds.y != "Iris-setosa"
        print("Samples not Iris-setosa:", int(mask_not_setosa.sum()))

if __name__ == "__main__":
    main()
