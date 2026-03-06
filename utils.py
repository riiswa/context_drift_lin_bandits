import numpy as np


def randargmax(arr, rng):
    return np.argmax(arr)
    return rng.choice(np.where(arr == arr.max())[0])

def randargmin(arr, rng):
    return rng.choice(np.where(arr == arr.min())[0])