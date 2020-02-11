# Forwardtesting Time Series cross-validator

I built a train test splitter for time series data, to appropriately forwardtest forecasting models. Sklearn only provides the TimeSeriesSplit which isn't enough in a lot of cases.

This forwardtesting time series cross-validator allows you to start at a given date, set the size of the test set and specify an offset between train and test set if needed.

## Try out the forwardtesting validator
To try out the validator yourself, run the following:
```
python exmaple.py
```

## Example
Code:
```
import pandas as pd

from forwardtester import ForwardTimeSeriesSplit


X = pd.DataFrame(
    index=pd.date_range('01-01-2020', '14-01-2020', freq='D'),
    data=range(0, 14),
    columns=['value'])

ftscv = ForwardTimeSeriesSplit(
    X, start='2020-01-08', pred_window=2, unit='D')
for train_index, test_index in ftscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    print(
        "TRAIN:", X.iloc[
            train_index].index.strftime('%Y-%m-%d').tolist(),
        "TEST:", X.iloc[
            test_index].index.strftime('%Y-%m-%d').tolist())

```
Output:
```
TRAIN: [0 1 2 3 4 5 6]
TEST: [7 8]
TRAIN: ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06', '2020-01-07']
TEST: ['2020-01-08', '2020-01-09']

TRAIN: [0 1 2 3 4 5 6 7 8]
TEST: [ 9 10]
TRAIN: ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06', '2020-01-07', '2020-01-08', '2020-01-09']
TEST: ['2020-01-10', '2020-01-11']

TRAIN: [ 0  1  2  3  4  5  6  7  8  9 10]
TEST: [11 12]
TRAIN: ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06', '2020-01-07', '2020-01-08', '2020-01-09', '2020-01-10', '2020-01-11']
TEST: ['2020-01-12', '2020-01-13']
```

