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


results_exp_sarima = {}

for bin_id in bins:
    bin_data = df[df["Bin_ID"] == bin_id].set_index("datetime")
    train = bin_data[bin_data.index <= train_end]["Fullness"]
    test = bin_data[(bin_data.index >= test_start) & (bin_data.index < test_end)][
        "Fullness"
    ]
    forecast_steps = len(test)

    # --- Step 1: Exponential Smoothing ---
    exp_model = ExponentialSmoothing(train, seasonal="add", seasonal_periods=30)
    exp_fit = exp_model.fit()
    exp_forecast = exp_fit.forecast(forecast_steps)

    # --- Step 2: Get Residuals from Training Data ---
    exp_train_fitted = exp_fit.fittedvalues
    residuals = train - exp_train_fitted

    # --- Step 3: SARIMA on Residuals ---
    sarima_resid_model = SARIMAX(
        residuals, order=(1, 1, 1), seasonal_order=(1, 1, 1, 30)
    )
    sarima_resid_fit = sarima_resid_model.fit(disp=False)
    resid_forecast = sarima_resid_fit.forecast(forecast_steps)

    # --- Step 4: Final Forecast = Exp + Residual Model Forecast ---
    hybrid_forecast = exp_forecast.values + resid_forecast.values
    hybrid_forecast = np.maximum(hybrid_forecast, 0)
    hybrid_forecast = np.minimum(hybrid_forecast, 5)

    # --- Evaluation ---
    rmse = np.sqrt(mean_squared_error(test, hybrid_forecast))
    mae = mean_absolute_error(test, hybrid_forecast)
    mape = np.mean(np.abs((test.values - hybrid_forecast) / test.values)) * 100

    results_exp_sarima[bin_id] = {"RMSE": rmse, "MAE": mae, "MAPE": mape}

    # --- Plot ---
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
        label="Exp + SARIMA Predicted Fullness",
        color="red",
        linewidth=2.5,
        marker="o",
        markersize=6,
        linestyle="--",
    )
    plt.title(f"Hybrid Exp Smoothing + SARIMA - Bin {bin_id}", fontsize=12)
    plt.xlabel("Date", fontsize=10)
    plt.ylabel("Fullness", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=9)
    plt.ylim(0, 5)
    plt.tight_layout()
    plt.show()

    print(
        f"[Exp+SARIMA Residual] Bin {bin_id} → RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%"
    )
