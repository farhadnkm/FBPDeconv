import numpy as np

def normalize_perc(x, percentile=(0.01, 99.99)):
    p_small, p_large = np.percentile(x, percentile)
    a =  1 / (p_large - p_small)
    b = p_small / (p_small - p_large)
    return a *x + b

def normalize_minmax(x):
    return (x-x.min()) / (x.max() - x.min())

