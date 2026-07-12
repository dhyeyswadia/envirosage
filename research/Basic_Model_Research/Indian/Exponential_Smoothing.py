import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Nadam

# Load and preprocess data
df = pd.read_csv("cleaned_bin_data_synthetic.csv")
df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
df.sort_values(by="datetime", inplace=True)

train_end = pd.Timestamp("2025-03-05 23:59:59")
test_start = pd.Timestamp("2025-03-06 00:00:00")
test_end = pd.Timestamp("2025-03-07 00:00:00")

bins = df["Bin_ID"].unique()
time_steps = df.groupby("date")["time"].nunique().mode()[0]  # dynamic

# Helper functions


def prepare_data(series, time_steps):
    X, y = [], []
    for i in range(len(series) - time_steps):
        X.append(series[i : i + time_steps])
        y.append(series[i + time_steps])
    return np.array(X), np.array(y)


def create_model(time_steps):
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(time_steps, 1)))
    model.add(Dense(1))
    return model


results_exp = {}

for bin_id in bins:
    bin_data = df[df["Bin_ID"] == bin_id].set_index("datetime")
    train = bin_data[bin_data.index <= train_end]["Fullness"]
    test = bin_data[(bin_data.index >= test_start) & (bin_data.index < test_end)][
        "Fullness"
    ]

    model = ExponentialSmoothing(train, seasonal="add", seasonal_periods=30)
    fit = model.fit()
    forecast = fit.forecast(len(test))

    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    results_exp[bin_id] = {"RMSE": rmse, "MAE": mae}

    plt.figure(figsize=(10, 5))
    plt.plot(test.index, test, label="Actual")
    plt.plot(test.index, forecast, label="Exponential Smoothing")
    plt.title(f"Exponential Smoothing Forecast - Bin {bin_id}")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"[Exp Smoothing] Bin {bin_id} → RMSE: {rmse:.2f}, MAE: {mae:.2f}")
