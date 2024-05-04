import time
from typing import List
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
import logging


def load_features(srcs: List[Path]) -> np.ndarray:
    res = []
    for path in srcs:
        with open(path, "rt") as f:
            for line in f.readlines():
                res.append(list(map(float, line.split(","))))
    return np.array(res)


def convert_to_2d(vectors: np.ndarray) -> np.ndarray:
    pca_2d = PCA(n_components=2)
    vectors_2d = pca_2d.fit_transform(vectors)
    return vectors_2d

