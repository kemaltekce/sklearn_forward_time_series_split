import numpy as np

import pandas as pd
from sklearn.model_selection._split import _BaseKFold


class ForwardTimeSeriesSplit(_BaseKFold):
    """Forwardtesting Time Series cross-validator

    Provides train/test indices to split time series data samples. This allows
    you to split your data in a forward moving fashion, which increases the
    train set from one split to the next. Additionally, you can select the
    size of your test set and specify an offset between train set and test set
    if needed.

    Parameters
    ----------

    X : DataFrame
        Data set with time series index.

    start : str
        Start date of forward testing.

    end : str, default=None
        End date of forward testing. If not specified, the latest date in X
        will be set to end.

    pred_window : int, default=7
        Size of prediction window (size of test set).

    pred_offset : int, default=0
        Amount of units to skip before prediction window (test set) starts.

    unit : str, default='D'
        This defines the unit of pred_window and pred_offset.

    Example
    -------
    >>> import pandas as pd
    >>> from forwardtester import ForwardTimeSeriesSplit
    >>> X = pd.DataFrame(
    ...    index=pd.date_range('01-01-2020', '14-01-2020', freq='D'),
    ...    data=range(0, 14),
    ...    columns=['value'])
    >>> ftscv = ForwardTimeSeriesSplit(
    ...    X, start='2020-01-08', pred_window=2)
    >>> for train_index, test_index in ftscv.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    print(
    ...        "TRAIN:", X.iloc[
    ...            train_index].index.strftime('%Y-%m-%d').tolist(),
    ...        "TEST:", X.iloc[test_index].index.strftime('%Y-%m-%d').tolist())
    TRAIN: [0 1 2 3 4 5 6] TEST: [7 8]
    TRAIN: ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
            '2020-01-05', '2020-01-06', '2020-01-07']
    TEST: ['2020-01-08', '2020-01-09']
    TRAIN: [0 1 2 3 4 5 6 7 8] TEST: [ 9 10]
    TRAIN: ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
            '2020-01-05', '2020-01-06', '2020-01-07', '2020-01-08',
            '2020-01-09']
    TEST: ['2020-01-10', '2020-01-11']
    TRAIN: [ 0  1  2  3  4  5  6  7  8  9 10] TEST: [11 12]
    TRAIN: ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
            '2020-01-05', '2020-01-06', '2020-01-07', '2020-01-08',
            '2020-01-09', '2020-01-10', '2020-01-11']
    TEST: ['2020-01-12', '2020-01-13']
    """

    def __init__(self, X, start, end=None, pred_window=7,
                 pred_offset=0, unit='D'):
        self.start = pd.Timestamp(start)
        self.pred_window = pd.Timedelta(pred_window, unit=unit)
        self.pred_offset = pd.Timedelta(pred_offset, unit=unit)
        self.unit = unit
        self.end = pd.Timestamp(end) if end else X.index.max()

        n_splits_total = self._calculate_n_splits_total()
        splits = self._calculate_splits(X)
        n_splits = len(splits[0])

        if n_splits != n_splits_total:
            print(
                "Only %d of %d splits are possible because of "
                "empty test sets." % (n_splits, n_splits_total))
        super().__init__(n_splits, shuffle=False, random_state=None)

    def _calculate_n_splits_total(self):
        delta = (self.end - self.start)
        n_splits_total = (
            delta - self.pred_offset
        ) // self.pred_window
        return n_splits_total

    def _calculate_splits(self, X_):
        n_splits_total = self._calculate_n_splits_total()

        X = X_.copy()
        X['ids'] = np.arange(len(X))

        start_ids = []
        test_start_ids = []
        test_end_ids = []

        for window in range(n_splits_total):
            start = self.start + self.pred_window * window

            start_id = X[X.index <= start].ids.max()
            test_start_id = X[
                X.index <= (start + self.pred_offset)].ids.max()
            test_end_id = X[
                X.index <= (start + self.pred_window + self.pred_offset)
            ].ids.max()

            if test_start_id == test_end_id:
                continue
            else:
                start_ids.append(start_id)
                test_start_ids.append(test_start_id)
                test_end_ids.append(test_end_id)
        return start_ids, test_start_ids, test_end_ids

    def split(self, X_, y=None, groups=None):
        X = X_.copy()
        splits = self._calculate_splits(X)
        for start_id, test_start_id, test_end_id in zip(*splits):
            yield (np.arange(0, start_id),
                   np.arange(test_start_id, test_end_id))
