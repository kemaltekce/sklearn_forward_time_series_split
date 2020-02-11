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
        "TRAIN:", X.iloc[train_index].index.strftime('%Y-%m-%d').tolist(),
        "TEST:", X.iloc[test_index].index.strftime('%Y-%m-%d').tolist())
