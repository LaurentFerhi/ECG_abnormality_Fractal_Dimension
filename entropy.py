"""Entropy functions"""
import numpy as np
from math import factorial, log
from sklearn.neighbors import KDTree
from scipy.signal import periodogram, welch

from utils import _embed, _xlogx

all = [
    "perm_entropy",
    "spectral_entropy",
    "svd_entropy",
    "app_entropy",
    "sample_entropy",
    "lziv_complexity",
    "num_zerocross",
    "hjorth_params",
]


def perm_entropy(x, order=3, delay=1, normalize=False):
    """Permutation Entropy.

    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times)
    order : int
        Order of permutation entropy. Default is 3.
    delay : int, list, np.ndarray or range
        Time delay (lag). Default is 1. If multiple values are passed
        (e.g. [1, 2, 3]), AntroPy will calculate the average permutation
        entropy across all these delays.
    normalize : bool
        If True, divide by log2(order!) to normalize the entropy between 0
        and 1. Otherwise, return the permutation entropy in bit.

    Returns
    -------
    pe : float
        Permutation Entropy.
    """
    # If multiple delay are passed, return the average across all d
    if isinstance(delay, (list, np.ndarray, range)):
        return np.mean([perm_entropy(x, order=order, delay=d, normalize=normalize) for d in delay])
    x = np.array(x)
    ran_order = range(order)
    hashmult = np.power(order, ran_order)
    assert delay > 0, "delay must be greater than zero."
    # Embed x and sort the order of permutations
    sorted_idx = _embed(x, order=order, delay=delay).argsort(kind="quicksort")
    # Associate unique integer to each permutations
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    # Return the counts
    _, c = np.unique(hashval, return_counts=True)
    # Use np.true_divide for Python 2 compatibility
    p = np.true_divide(c, c.sum())
    pe = -_xlogx(p).sum()
    if normalize:
        pe /= np.log2(factorial(order))
    return pe


def spectral_entropy(x, sf, method="fft", nperseg=None, normalize=False, axis=-1):
    """Spectral Entropy.

    Parameters
    ----------
    x : list or np.array
        1D or N-D data.
    sf : float
        Sampling frequency, in Hz.
    method : str
        Spectral estimation method:

        * ``'fft'`` : Fourier Transform (:py:func:`scipy.signal.periodogram`)
        * ``'welch'`` : Welch periodogram (:py:func:`scipy.signal.welch`)
    nperseg : int or None
        Length of each FFT segment for Welch method.
        If None (default), uses scipy default of 256 samples.
    normalize : bool
        If True, divide by log2(psd.size) to normalize the spectral entropy
        between 0 and 1. Otherwise, return the spectral entropy in bit.
    axis : int
        The axis along which the entropy is calculated. Default is -1 (last).

    Returns
    -------
    se : float
        Spectral Entropy
    """
    x = np.asarray(x)
    # Compute and normalize power spectrum
    if method == "fft":
        _, psd = periodogram(x, sf, axis=axis)
    elif method == "welch":
        _, psd = welch(x, sf, nperseg=nperseg, axis=axis)
    psd_norm = psd / psd.sum(axis=axis, keepdims=True)
    se = -_xlogx(psd_norm).sum(axis=axis)
    if normalize:
        se /= np.log2(psd_norm.shape[axis])
    return se


def svd_entropy(x, order=3, delay=1, normalize=False):
    """Singular Value Decomposition entropy.

    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times)
    order : int
        Order of SVD entropy (= length of the embedding dimension).
        Default is 3.
    delay : int
        Time delay (lag). Default is 1.
    normalize : bool
        If True, divide by log2(order!) to normalize the entropy between 0
        and 1. Otherwise, return the permutation entropy in bit.

    Returns
    -------
    svd_e : float
        SVD Entropy
    """
    x = np.array(x)
    mat = _embed(x, order=order, delay=delay)
    W = np.linalg.svd(mat, compute_uv=False)
    # Normalize the singular values
    W /= sum(W)
    svd_e = -_xlogx(W).sum()
    if normalize:
        svd_e /= np.log2(order)
    return svd_e


def _app_samp_entropy(x, order, metric="chebyshev", approximate=True):
    """Utility function for `app_entropy`` and `sample_entropy`."""
    _all_metrics = KDTree.valid_metrics
    if metric not in _all_metrics:
        raise ValueError(
            "The given metric (%s) is not valid. The valid "
            "metric names are: %s" % (metric, _all_metrics)
        )
    phi = np.zeros(2)
    r = 0.2 * np.std(x, ddof=0)

    # compute phi(order, r)
    _emb_data1 = _embed(x, order, 1)
    if approximate:
        emb_data1 = _emb_data1
    else:
        emb_data1 = _emb_data1[:-1]
    count1 = (
        KDTree(emb_data1, metric=metric)
        .query_radius(emb_data1, r, count_only=True)
        .astype(np.float64)
    )
    # compute phi(order + 1, r)
    emb_data2 = _embed(x, order + 1, 1)
    count2 = (
        KDTree(emb_data2, metric=metric)
        .query_radius(emb_data2, r, count_only=True)
        .astype(np.float64)
    )
    if approximate:
        phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
        phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
    return phi

def _numba_sampen(sequence, order, r):
    """
    Fast evaluation of the sample entropy using Numba.
    """

    size = sequence.size
    # sequence = sequence.tolist()

    numerator = 0
    denominator = 0

    for offset in range(1, size - order):
        n_numerator = int(abs(sequence[order] - sequence[order + offset]) >= r)
        n_denominator = 0

        for idx in range(order):
            n_numerator += abs(sequence[idx] - sequence[idx + offset]) >= r
            n_denominator += abs(sequence[idx] - sequence[idx + offset]) >= r

        if n_numerator == 0:
            numerator += 1
        if n_denominator == 0:
            denominator += 1

        prev_in_diff = int(abs(sequence[order] - sequence[offset + order]) >= r)
        for idx in range(1, size - offset - order):
            out_diff = int(abs(sequence[idx - 1] - sequence[idx + offset - 1]) >= r)
            in_diff = int(abs(sequence[idx + order] - sequence[idx + offset + order]) >= r)
            n_numerator += in_diff - out_diff
            n_denominator += prev_in_diff - out_diff
            prev_in_diff = in_diff

            if n_numerator == 0:
                numerator += 1
            if n_denominator == 0:
                denominator += 1

    if denominator == 0:
        return 0  # use 0/0 == 0
    elif numerator == 0:
        return np.inf
    else:
        return -log(numerator / denominator)


def app_entropy(x, order=2, metric="chebyshev"):
    """Approximate Entropy.

    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times).
    order : int
        Embedding dimension. Default is 2.
    metric : str
        Name of the distance metric function used with
        :py:class:`sklearn.neighbors.KDTree`. Default is to use the
        `Chebyshev <https://en.wikipedia.org/wiki/Chebyshev_distance>`_
        distance.

    Returns
    -------
    ae : float
        Approximate Entropy.
    """
    phi = _app_samp_entropy(x, order=order, metric=metric, approximate=True)
    return np.subtract(phi[0], phi[1])


def sample_entropy(x, order=2, metric="chebyshev"):
    """Sample Entropy.

    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times).
    order : int
        Embedding dimension. Default is 2.
    metric : str
        Name of the distance metric function used with
        :py:class:`sklearn.neighbors.KDTree`. Default is to use the
        `Chebyshev <https://en.wikipedia.org/wiki/Chebyshev_distance>`_
        distance.

    Returns
    -------
    se : float
        Sample Entropy.
    """
    x = np.asarray(x, dtype=np.float64)
    if metric == "chebyshev" and x.size < 5000:
        return _numba_sampen(x, order=order, r=(0.2 * x.std(ddof=0)))
    else:
        phi = _app_samp_entropy(x, order=order, metric=metric, approximate=False)
        return -np.log(np.divide(phi[1], phi[0]))

def _lz_complexity(binary_string):
    """Internal Numba implementation of the Lempel-Ziv (LZ) complexity.
    https://github.com/Naereen/Lempel-Ziv_Complexity/blob/master/src/lziv_complexity.py
    - Updated with strict integer typing instead of strings
    - Slight restructuring based on Yacine Mahdid's notebook:
    https://github.com/BIAPT/Notebooks/blob/master/features/Lempel-Ziv%20Complexity.ipynb
    """
    # Initialize variables
    complexity = 1
    prefix_len = 1
    len_substring = 1
    max_len_substring = 1
    pointer = 0

    # Iterate until the entire string has not been parsed
    while prefix_len + len_substring <= len(binary_string):
        # Given a prefix length, find the largest substring
        if (
            binary_string[pointer + len_substring - 1]
            == binary_string[prefix_len + len_substring - 1]  # noqa: W503
        ):
            len_substring += 1  # increase the length of the substring
        else:
            max_len_substring = max(len_substring, max_len_substring)
            pointer += 1
            # Since all pointers have been scanned, pick largest as the jump
            # size
            if pointer == prefix_len:
                # Increment complexity
                complexity += 1
                # Set prefix length to the max substring size found so far
                # (jump size)
                prefix_len += max_len_substring
                # Reset pointer and max substring size
                pointer = 0
                max_len_substring = 1
            # Reset length of current substring
            len_substring = 1

    # Check if final iteration occurred in the middle of a substring
    if len_substring != 1:
        complexity += 1

    return complexity


def lziv_complexity(sequence, normalize=False):
    """
    Lempel-Ziv (LZ) complexity of (binary) sequence.

    .. versionadded:: 0.1.1

    Parameters
    ----------
    sequence : str or array
        A sequence of character, e.g. ``'1001111011000010'``,
        ``[0, 1, 0, 1, 1]``, or ``'Hello World!'``.
    normalize : bool
        If ``True``, returns the normalized LZ (see Notes).

    Returns
    -------
    lz : int or float
        LZ complexity, which corresponds to the number of different
        substrings encountered as the stream is viewed from the
        beginning to the end. If ``normalize=False``, the output is an
        integer (counts), otherwise the output is a float.
    """
    assert isinstance(sequence, (str, list, np.ndarray))
    assert isinstance(normalize, bool)
    if isinstance(sequence, (list, np.ndarray)):
        sequence = np.asarray(sequence)
        if sequence.dtype.kind in "bfi":
            # Convert [True, False] or [1., 0.] to [1, 0]
            s = sequence.astype("uint32")
        else:
            # Treat as numpy array of strings
            # Map string characters to utf-8 integer representation
            s = np.fromiter(map(ord, "".join(sequence.astype(str))), dtype="uint32")
            # Can't preallocate length (by specifying count) due to string
            # concatenation
    else:
        s = np.fromiter(map(ord, sequence), dtype="uint32")

    if normalize:
        # 1) Timmermann et al. 2019
        # The sequence is randomly shuffled, and the normalized LZ
        # is calculated as the ratio of the LZ of the original sequence
        # divided by the LZ of the randomly shuffled LZ. However, the final
        # output is dependent on the random seed.
        # sl_shuffled = list(s)
        # rng = np.random.RandomState(None)
        # rng.shuffle(sl_shuffled)
        # s_shuffled = ''.join(sl_shuffled)
        # return _lz_complexity(s) / _lz_complexity(s_shuffled)
        # 2) Zhang et al. 2009
        n = len(s)
        base = sum(np.bincount(s) > 0)  # Number of unique characters
        base = 2 if base < 2 else base
        return _lz_complexity(s) / (n / log(n, base))
    else:
        return _lz_complexity(s)

def num_zerocross(x, normalize=False, axis=-1):
    """Number of zero-crossings.

    .. versionadded: 0.1.3

    Parameters
    ----------
    x : list or np.array
        1D or N-D data.
    normalize : bool
        If True, divide by the number of samples to normalize the output
        between 0 and 1. Otherwise, return the absolute number of zero
        crossings.
    axis : int
        The axis along which to perform the computation. Default is -1 (last).

    Returns
    -------
    nzc : int or float
        Number of zero-crossings.
    """
    x = np.asarray(x)
    # https://stackoverflow.com/a/29674950/10581531
    nzc = np.diff(np.signbit(x), axis=axis).sum(axis=axis)
    if normalize:
        nzc = nzc / x.shape[axis]
    return nzc


def hjorth_params(x, axis=-1):
    """Calculate Hjorth mobility and complexity on given axis.

    .. versionadded: 0.1.3

    Parameters
    ----------
    x : list or np.array
        1D or N-D data.
    axis : int
        The axis along which to perform the computation. Default is -1 (last).

    Returns
    -------
    mobility, complexity : float
        Mobility and complexity parameters.
    """
    x = np.asarray(x)
    # Calculate derivatives
    dx = np.diff(x, axis=axis)
    ddx = np.diff(dx, axis=axis)
    # Calculate variance
    x_var = np.var(x, axis=axis)  # = activity
    dx_var = np.var(dx, axis=axis)
    ddx_var = np.var(ddx, axis=axis)
    # Mobility and complexity
    mob = np.sqrt(dx_var / x_var)
    com = np.sqrt(ddx_var / dx_var) / mob
    return mob, com
