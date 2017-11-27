import numpy as np

def calculate_stats(dataset):
    return (
        np.mean(dataset),
        np.std(dataset),
        np.min(dataset),
        np.max(dataset)
    )
