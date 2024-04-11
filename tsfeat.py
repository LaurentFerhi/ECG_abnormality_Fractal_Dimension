from functools import partial
from itertools import groupby
from math import log, e
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
from statsmodels.api import add_constant, OLS
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf, kpss
from typing import List, Dict, Optional, Callable

def embed(x: np.array, p: int) -> np.array:
    """Embeds the time series x into a low-dimensional Euclidean space.

    Parameters
    ----------
    x: numpy array
        Time series.
    p: int
        Embedding dimension.

    References
    ----------
    https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/embed
    """
    x = np.transpose(np.vstack(list((np.roll(x, k) for k in range(p)))))
    x = x[p - 1:]

    return x

def arch_stat(x: np.array, freq: int = 1,
              lags: int = 12, demean: bool = True) -> Dict[str, float]:
    """Arch model features.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'arch_lm': R^2 value of an autoregressive model of order lags applied to x**2.
    """
    if len(x) <= lags + 1:
        return {'arch_lm': np.nan}
    if demean:
        x = x - np.mean(x)

    size_x = len(x)
    mat = embed(x ** 2, lags + 1)
    X = mat[:, 1:]
    y = np.vstack(mat[:, 0])

    try:
        r_squared = LinearRegression().fit(X, y).score(X, y)
    except:
        r_squared = np.nan

    return {'arch_lm': r_squared}

def sparsity(x: np.array, freq: int = 1) -> Dict[str, float]:
    """Sparsity.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'sparsity': Average obs with zero values.
    """

    return {'sparsity': np.mean(x == 0)}

def stability(x: np.array, freq: int = 1) -> Dict[str, float]:
    """Stability.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'stability': Variance of the means of tiled windows.
    """
    if freq == 1:
        width = 10
    else:
        width = freq

    nr = len(x)
    lo = np.arange(0, nr, width)
    up = lo + width
    nsegs = nr / width
    meanx = [np.nanmean(x[lo[idx]:up[idx]]) for idx in np.arange(int(nsegs))]

    if len(x) < 2 * width:
        stability = 0
    else:
        stability = np.nanvar(meanx, ddof=1)

    return {'stability': stability}

def heterogeneity(x: np.array, freq: int = 1) -> Dict[str, float]:
    """Heterogeneity.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'arch_acf': Sum of squares of the first 12 autocorrelations of the
                    residuals of the AR model applied to x
        'garch_acf': Sum of squares of the first 12 autocorrelations of the
                    residuals of the GARCH model applied to x
        'arch_r2': Function arch_stat applied to the residuals of the
                   AR model applied to x.
        'garch_r2': Function arch_stat applied to the residuals of the GARCH
                    model applied to x.
    """
    m = freq

    size_x = len(x)
    order_ar = min(size_x - 1, np.floor(10 * np.log10(size_x)))
    order_ar = int(order_ar)

    try:
        x_whitened = AutoReg(x,lags=order_ar).fit().resid
    except:
        output = {
             'arch_acf': np.nan,
             'garch_acf': np.nan,
            'arch_r2': np.nan,
            'garch_r2': np.nan
        }

        return output
    # arch and box test
    x_archtest = arch_stat(x_whitened, m)['arch_lm']
    LBstat = (acf(x_whitened ** 2, nlags=12, fft=False)[1:] ** 2).sum()

    output = {'arch_acf': LBstat}
    
    return output

def holt_parameters(x: np.array, freq: int = 1) -> Dict[str, float]:
    """Fitted parameters of a Holt model.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'alpha': Level paramater of the Holt model.
        'beta': Trend parameter of the Hold model.
    """
    try :
        fit = ExponentialSmoothing(x, trend='add', seasonal=None).fit()
        params = {
            'alpha': fit.params['smoothing_level'],
            'beta': fit.params['smoothing_trend']
        }
    except:
        params = {
            'alpha': np.nan,
            'beta': np.nan
        }

    return params