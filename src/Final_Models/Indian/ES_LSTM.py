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
df = pd.read_csv("cleaned_bin_data.csv")
df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
df.sort_values(by="datetime", inplace=True)

train_end = pd.Timestamp("2025-03-05 23:59:59")
test_start = pd.Timestamp("2025-03-06 00:00:00")
test_end = pd.Timestamp("2025-03-07 00:00:00")

# bins = df['Bin_ID'].unique()


bins = [1001, 1002, 1003, 1004, 1005, 1016, 1017, 1020, 1021, 1022, 1023]


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


results_exp_lstm = {}


def prepare_lstm_data(series, time_steps=10):
    X, y = [], []
    for i in range(len(series) - time_steps):
        X.append(series[i : i + time_steps])
        y.append(series[i + time_steps])
    return np.array(X), np.array(y)


for bin_id in bins:
    bin_data = df[df["Bin_ID"] == bin_id].set_index("datetime")
    train = bin_data[bin_data.index <= train_end]["Fullness"]
    test = bin_data[(bin_data.index >= test_start) & (bin_data.index < test_end)][
        "Fullness"
    ]
    forecast_steps = len(test)

    exp_model = ExponentialSmoothing(train, seasonal="add", seasonal_periods=30)
    exp_fit = exp_model.fit()
    exp_forecast = exp_fit.forecast(forecast_steps)

    residuals = train - exp_fit.fittedvalues
    scaler = MinMaxScaler()
    scaled_resid = scaler.fit_transform(residuals.values.reshape(-1, 1))

    X, y = prepare_lstm_data(scaled_resid.flatten(), time_steps=10)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    lstm_model = Sequential(
        [LSTM(50, activation="relu", input_shape=(X.shape[1], 1)), Dense(1)]
    )
    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(X, y, epochs=20, verbose=0)

    last_resid = scaled_resid[-10:].reshape(1, 10, 1)
    lstm_forecast_scaled = []

    for _ in range(forecast_steps):
        pred = lstm_model.predict(last_resid, verbose=0)
        lstm_forecast_scaled.append(pred[0, 0])
        last_resid = np.roll(last_resid, -1)
        last_resid[0, -1, 0] = pred[0, 0]

    lstm_forecast = scaler.inverse_transform(
        np.array(lstm_forecast_scaled).reshape(-1, 1)
    ).flatten()

    hybrid_forecast = exp_forecast.values + lstm_forecast
    hybrid_forecast = np.maximum(hybrid_forecast, 0)
    hybrid_forecast = np.minimum(hybrid_forecast, 5)

    rmse = np.sqrt(mean_squared_error(test, hybrid_forecast))
    mae = mean_absolute_error(test, hybrid_forecast)
    mape = np.mean(np.abs((test.values - hybrid_forecast) / test.values)) * 100

    results_exp_lstm[bin_id] = {"RMSE": rmse, "MAE": mae, "MAPE": mape}

    plt.figure(figsize=(8, 4))
    plt.plot(
        test.index,
        test,
        label="Actual Fullness",
        color="blue",
        linewidth=2.5,
        marker="o",
        markersize=6,
    )
    plt.plot(
        test.index,
        hybrid_forecast,
        label="Exp + LSTM Predicted Fullness",
        color="red",
        linewidth=2.5,
        marker="o",
        markersize=6,
        linestyle="--",
    )
    plt.title(f"Hybrid Exp Smoothing + LSTM - Bin {bin_id}", fontsize=12)
    plt.xlabel("Date", fontsize=10)
    plt.ylabel("Fullness", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=9)
    plt.ylim(0, 5)
    plt.tight_layout()
    plt.show()

    print(
        f"[Exp+LSTM Residual] Bin {bin_id} → RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%"
    )
