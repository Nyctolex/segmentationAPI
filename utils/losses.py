import numpy as np

def MSE(vector1: np.ndarray, vector2: np.ndarray) -> float:
    assert len(vector1.shape) == len(vector2.shape)
    assert all([s[0] == s[1] for s in zip(vector1.shape, vector2.shape) ])
    return ((vector1 - vector2)**2).mean()

    