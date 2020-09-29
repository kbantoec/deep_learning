import os
import numpy as np
from pandas.core.frame import Series, DataFrame
from numpy.core import ndarray, float64
from typing import Union, List
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from time import time
from datetime import datetime


def normabspath(basedir: str, filename: str):
    return os.path.normpath(os.path.join(basedir, filename))


def show_ellapsed_time(start: float, end: float) -> None:
    ellapsed_secs: int = round(end - start)

    if ellapsed_secs > 59:
        minutes: int = int(ellapsed_secs / 60)
        seconds_msg: str = ' and ' + str(ellapsed_secs % 60) + ' seconds' if ellapsed_secs % 60 != 0 else ''
        print(f"Ellapsed time: {minutes} {'minutes' if minutes > 1 else 'minute'}{seconds_msg}")
    else:
        print(f"Ellapsed time: {ellapsed_secs} seconds")


def lagged_features(features_tensor: ndarray, num_lags: int, fill_value: Union[int, float] = np.nan) -> ndarray:
    """Creates a tensor of lagged features and appends them to it.

    Parameters
    ----------
    features_tensor : numpy.ndarray
        NumPy array of feature data with shape (N, T, D=1), where
        N is the number of samples, T is the period-window size, and D
        is the number of features.

    num_lags : int
        The number of lags to be generated.

    fill_value : Union[int, float]
        The values the lag vector should be filled of when being shifted.

    Returns
    -------
    numpy.ndarray
        Extended features' tensor with its lags.
    """
    N, T, D = features_tensor.shape  # D = 1
    filled = np.full((N, num_lags, D), fill_value)
    extended_feature = np.concatenate((filled, features_tensor), axis=1)
    return extended_feature[:, :T]


def single_autocorr(series: ndarray, num_lags: int) -> float64:
    """Computes the autocorrelation between the series and its t-lagged version.

    Parameters
    ----------
    series : numpy.ndarray
        Time series.

    num_lags : int
        Number of lags.

    Returns
    -------
    numpy.float64
        The serial correlation of the series.
    """

    x0 = series[num_lags:]
    x1 = series[:-num_lags]
    mu0 = np.nanmean(x0)
    mu1 = np.nanmean(x1)
    dx0 = x0 - mu0
    dx1 = x1 - mu1
    sigma0 = np.sqrt(np.nansum(dx0 * dx0))
    sigma1 = np.sqrt(np.nansum(dx1 * dx1))
    return np.nansum(dx0 * dx1) / (sigma0 * sigma1) if (sigma0 * sigma1) != 0 else 0


def batch_autocorr(time_series: ndarray, num_lags: int, lookback_periods: int, verbose: bool = True) -> ndarray:
    """Create a batch of autocorrelation features.

    Parameters
    ----------
    time_series : numpy.ndarray
        Multiple time series, where rows are samples, and columns are time periods.
        Shape is (N, T).

    num_lags : int
        Number of lags.

    lookback_periods : int
        The lookback rolling window's length.

    verbose : bool, optional
        Print the progression of autocorrelation computations.

    Returns
    -------
    autocorrs : numpy.ndarray
        A vector containing autocorrelations of each series of the time series data for
        a given number of lags and a given lookback rolling window.
        Output shape is (N*lookback_periods,).
    """
    N, D = time_series.shape

    collect_autocorrs: List[float64] = []
    for i in range(N):
        if verbose:
            if i % 10 == 0:
                print(f"Series {i}/{N}")
        r = single_autocorr(time_series[i, :], num_lags)
        collect_autocorrs.append(r)

    autocorrs: ndarray = np.array(collect_autocorrs).reshape(-1, 1)  # Shape: (N, 1)
    autocorrs = np.expand_dims(autocorrs, -1)  # Shape: (N, 1, 1)

    # Propagate the autocorrelation
    autocorrs = np.repeat(autocorrs, lookback_periods, axis=1)  # Shape: (N, lookback_periods, 1)
    return autocorrs


def one_hot_encode_series(series: Union[Series, List[str]]) -> ndarray:
    """One-hot encode series."""
    encoded_series: ndarray = LabelEncoder().fit_transform(series).reshape(-1, 1)
    one_hot_series: ndarray = OneHotEncoder(sparse=False).fit_transform(encoded_series)
    return one_hot_series


def one_hot_and_dims(series: Series, lookback_periods: int) -> ndarray:
    """One-hot encode features and expand dimensions."""
    encoded_series: ndarray = one_hot_encode_series(series)
    encoded_series: ndarray = np.expand_dims(encoded_series, 1)
    encoded_series: ndarray = np.repeat(encoded_series, lookback_periods, axis=1)
    return encoded_series


def create_medians_tensor(time_series_matrix: ndarray, lookback_periods: int):
    medians: ndarray = np.median(time_series_matrix, axis=1)  # WHY SERIES, NOT LOG_SERIES?? # Shape: (N,)
    medians = np.expand_dims(medians, -1)  # Shape: (N, 1)
    medians = np.expand_dims(medians, -1)  # Shape: (N, 1, 1)
    medians = np.repeat(medians, lookback_periods, axis=1)
    return medians


def get_weekdays(series: Series, start: int, end: int) -> List[str]:
    """Get weekday names from a series with dates in its columns."""
    return [datetime.strptime(date, '%Y-%m-%d').strftime('%a') for date in series.iloc[:, start:end].columns.to_numpy()]


def get_batch(time_series: DataFrame,
              global_features: DataFrame = None,
              start: int = 0,
              lookback_periods: int = 100,
              lags: tuple = None,
              show_runtime: bool = True) -> tuple:
    """Generate one batch of data. A batch is a cube (3D tensor).

    Parameters
    ----------
    time_series : pandas.DataFrame
        Time series DataFrame with samples as rows and dates as columns.

    global_features : pandas.DataFrame or None
        Include other features.

    start : int, optional
        Index that specifies at which column of the time series DataFrame we should start.

    lookback_periods : int, optional
        Window size. Specifies how much periods the window has.

    lags: tuple of int or None
        Include period lags as features.

    show_runtime: bool, optional
        Print message of ellapsed time.

    Notes
    -----
    log_series : numpy.ndarray
        The natural logarithm of the time series passed as argument to the `time_series` parameter.
        Since the logarithm of zero is undefined, we use `numpy.log1p` which is the natural logarithm function.
        Expressed as a tensor with shape (N, lookback_periods, 1).

    days : numpy.ndarray
        One-hot encoded weekdays and expressed as a tensor with shape (N, lookback_periods, 7).

    medians : numpy.ndarray
        The median of the time series over the lookback period.

    Returns
    -------
    batch : numpy.ndarray
        Batch

    target : numpy.ndarray
        The natural log of the time series coming right after the training time series data.

    """

    if show_runtime:
        start_time: float = time()

        # N is the number of samples
    # T is the number of periods
    N, T = time_series.shape

    end: int = start + lookback_periods
    assert end <= T, f"End of lookback out of bounds. End of lookback: {end}, but end of your time series: {T}."

    time_series_matrix: ndarray = time_series.iloc[:, start:end].to_numpy()
    target: ndarray = np.log1p(time_series.iloc[:, end].to_numpy())

    log_series: ndarray = np.log1p(time_series_matrix)  # Shape: (N, lookback_periods)
    log_series = np.expand_dims(log_series, axis=-1)  # Shape: (N, lookback_periods, 1)

    weekdays: List[str] = get_weekdays(time_series, start, end)
    days_one_hot: ndarray = one_hot_encode_series(weekdays)  # Shape: (lookback_periods, 7)
    days_one_hot = np.expand_dims(days_one_hot, 0)  # Shape: (1, lookback_periods, 7)
    days: ndarray = np.repeat(days_one_hot, repeats=N, axis=0)  # Shape: (N, lookback_periods, 7)

    batch: ndarray = np.concatenate((log_series, days), axis=2)  # Shape: (N, lookback_periods, 8)

    if lags is not None:
        for lag in lags:
            batch = np.concatenate((batch, lagged_features(log_series, lag, np.nan)), axis=2)
            batch = np.concatenate((batch, batch_autocorr(time_series_matrix, lag, lookback_periods, verbose=False)),
                                   axis=2)  # WHY SERIES, NOT LOG_SERIES??

    if global_features is not None:
        assert not isinstance(global_features,
                              Series), "You passed a 'pandas.Series' object instead of a 'pandas.DataFrame'"

        N_, D = global_features.shape
        assert N == N_, ("'time_series' and 'global_features' must have same number of samples."
                         + f" You gave {N} samples for 'time_series', but {N_} for 'global_features'.")

        for feature in global_features.columns:
            batch = np.concatenate((batch, one_hot_and_dims(global_features[feature], lookback_periods)), axis=2)

    medians: ndarray = create_medians_tensor(time_series_matrix,
                                             lookback_periods)  # WHY SERIES, NOT LOG_SERIES?? # Shape (N, lookback_periods, 1)
    batch = np.concatenate((batch, medians), axis=2)

    if show_runtime:
        end_time: float = time()
        show_ellapsed_time(start_time, end_time)

    return batch, target


def generate_batches(time_series: DataFrame,
                     global_features: DataFrame = None,
                     batch_size: int = 32,
                     lookback_periods: int = 100,
                     lags=(365, 182, 91),
                     show_runtime: bool = True):
    """Generate batches.

    batch_size : int, optional
        The height of the rolling cube.
    """

    # N is the number of samples
    # T is the number of periods (or the number of steps)
    N, T = time_series.shape

    num_batches: int = N // batch_size

    while True:
        for i in range(num_batches):
            batch_start: int = i * batch_size
            batch_end: int = batch_start + batch_size

            seq_start: int = np.random.randint(T - lookback_periods)

            if show_runtime:
                print("-" * 30)
                print(f"Batch {i + 1}/{num_batches}")

            X, y = get_batch(time_series.iloc[batch_start:batch_end],
                             global_features=global_features.iloc[batch_start:batch_end],
                             start=seq_start,
                             lookback_periods=lookback_periods,
                             lags=lags,
                             show_runtime=show_runtime)

            yield X, y