from typing import Union, Optional

import numpy as np
from scipy.stats import skew as scipy_skew

def compute_gradient(arr: np.ndarray) -> np.ndarray:
    """
    Calculate the derivative of a given ndarray along the second axis.
    
    Parameters:
        arr: Input array.

    Output:
        An array holding the derivatives along the second axis.
    """
    return np.gradient(arr, axis=1)

def apply_threshold(arr: np.ndarray, th_value: float = 10, reset_value: float = 0) -> np.ndarray:
    """
    Implement thresholding on an input array.

    Parameters:
        arr: Input array.
        th_value: Percentile under which values will be set to zero.
        reset_value: The value to assign to elements under the threshold.

    Output:
        An array where elements under the threshold are replaced.
    """
    arr[arr < np.abs(np.percentile(arr.flatten(), th_value))] = reset_value
    return arr

def apply_clipping(arr: np.ndarray, std_devs: float = 3, method: str = 'std_devs') -> np.ndarray:
    """
    Cplis an array symmetrically based on the selected method.

    Parameters:
        arr: Input array.
        std_devs: Count of standard deviations for clipping.
        method: The strategy to use for clipping ('std_devs' or 'mean').

    Output:
        An array where values have been symmetrically clipped.
    """
    arr_copy = np.copy(arr)
    avg = np.mean(arr)
    stddev = np.std(arr)
    normalized_arr = (arr - avg) / stddev

    if method.lower() == 'std_devs':
        arr_copy[normalized_arr < -std_devs] = -std_devs * stddev + avg
        arr_copy[normalized_arr > std_devs] = std_devs * stddev + avg
    elif method.lower() == 'mean':
        arr_copy[normalized_arr < -std_devs] = avg
        arr_copy[normalized_arr > std_devs] = avg
    else:
        raise ValueError(f'Invalid "method": {method}. Allowed methods: std_devs, mean')

    return arr_copy

def correct_skewness(arr: np.ndarray) -> np.ndarray:
    """
    Inverts an array based on its skewness metric.(effective for gradient data)

    Parameters:
        arr: Input array.

    Output:
        An array where the skewness sign has been applied.
    """
    skew_direction = np.sign(scipy_skew(np.ravel(arr)))
    return arr * skew_direction

def normalize_zscore(arr: np.ndarray) -> np.ndarray:
    """
    Normalizes an array using Z-score normalization.

    Parameters:
        arr: Input array.

    Output:
        A Z-score normalized array.
    """
    return (arr - arr.mean()) / arr.std()
