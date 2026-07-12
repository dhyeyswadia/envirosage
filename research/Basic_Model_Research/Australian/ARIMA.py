import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("cleaned_bin_data.csv", parse_dates=["timestamp"])

# Select bins for forecasting
bins_to_forecast = [1511215, 1511194, 1511191]
df = df[df["Bin ID"].isin(bins_to_forecast)]

# Filter data up to May 3, 2021
train_data = df[df["timestamp"] < "2021-04-26"]
print(train_data)


results_arima = {}
results_arima_mae = {}
for bin_id in bins_to_forecast:
    # Prepare training data for this bin
    bin_train = train_data[train_data["Bin ID"] == bin_id].sort_values(by="timestamp")
    bin_train["timestamp"] = pd.to_datetime(bin_train["timestamp"])
    bin_train = bin_train.set_index("timestamp")

    ts = bin_train["Fullness"]

    # Fit an ARIMA model (order chosen here is (1,1,1); adjust if needed)
    model_arima = ARIMA(ts, order=(1, 1, 1))
    model_arima_fit = model_arima.fit()

    # Forecast next 7 days
    forecast_arima = model_arima_fit.forecast(steps=7)
    predictions_arima = forecast_arima.values.reshape(-1, 1)

    # Define prediction period: 27th April to 3rd May
    prediction_start = pd.to_datetime("2021-04-27")
    prediction_end = pd.to_datetime("2021-05-03")
    prediction_dates = pd.date_range(
        start=prediction_start, end=prediction_end, freq="D"
    )

    # Use the full dataset to get actual values for this bin and prediction period
    bin_full = df[df["Bin ID"] == bin_id].sort_values(by="timestamp")
    bin_full["timestamp"] = pd.to_datetime(bin_full["timestamp"])
    bin_full = bin_full.set_index("timestamp")
    daily_actual = bin_full["Fullness"].resample("D").last()
    daily_actual = (
        daily_actual.reindex(prediction_dates)
        .fillna(method="ffill")
        .fillna(method="bfill")
    )
    prediction_actual = daily_actual.values.reshape(-1, 1)

    # Calculate RMSE for ARIMA predictions
    rmse_arima = np.sqrt(mean_squared_error(prediction_actual, predictions_arima))
    results_arima[bin_id] = rmse_arima
    mae_arima = mean_squared_error(prediction_actual, predictions_arima)
    results_arima_mae[bin_id] = mae_arima
    # Plot Actual vs ARIMA Predicted Fullness
    plt.figure(figsize=(10, 5))
    plt.plot(
        prediction_dates, prediction_actual, "bo-", label="Actual Fullness", linewidth=2
    )
    plt.plot(
        prediction_dates,
        predictions_arima,
        "ro--",
        label="ARIMA Predicted Fullness",
        linewidth=2,
    )
    plt.ylim(0, 5)
    plt.title(f"Actual vs ARIMA Predicted Fullness - Bin {bin_id}")
    plt.xlabel("Date")
    plt.ylabel("Fullness Level")
    plt.legend()
    plt.show()

# Display ARIMA RMSE values
rmse_arima_df = pd.DataFrame.from_dict(results_arima, orient="index", columns=["RMSE"])
mae_arima_df = pd.DataFrame.from_dict(
    results_arima_mae, orient="index", columns=["MAE"]
)
rmse_arima_df.index.name = "Bin ID"
mae_arima_df.index.name = "Bin ID"
print("ARIMA RMSE:")
print(rmse_arima_df)
print(mae_arima_df)
