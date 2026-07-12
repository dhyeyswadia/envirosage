import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv("cleaned_bin_data.csv", parse_dates=["timestamp"])
bins_to_forecast = [1511208, 1511199, 1510830]
df = df[df["Bin ID"].isin(bins_to_forecast)]
train_data = df[df["timestamp"] <= "2021-04-26"]

results_es_sarima = {}
results_es_sarima_mae = {}

for bin_id in bins_to_forecast:
    bin_train = train_data[train_data["Bin ID"] == bin_id].sort_values("timestamp")
    bin_train.set_index("timestamp", inplace=True)

    # Exponential Smoothing
    es_model = ExponentialSmoothing(
        bin_train["Fullness"], trend="add", seasonal="add", seasonal_periods=7
    )
    es_fit = es_model.fit()
    es_forecast = es_fit.forecast(7)

    # SARIMA on residuals
    residuals = bin_train["Fullness"] - es_fit.fittedvalues
    sarima_model = SARIMAX(residuals, order=(1, 0, 1), seasonal_order=(1, 1, 1, 35))
    sarima_fit = sarima_model.fit()
    residual_forecast = sarima_fit.forecast(7)

    # Hybrid forecast
    hybrid_forecast = es_forecast.values + residual_forecast.values

    # Date alignment
    last_actual_date = bin_train.index[-1]
    prediction_dates = pd.date_range(
        start=last_actual_date + pd.Timedelta(days=1), periods=7, freq="D"
    )

    # Actual data
    bin_full = (
        df[df["Bin ID"] == bin_id].sort_values("timestamp").set_index("timestamp")
    )
    daily_actual = (
        bin_full["Fullness"]
        .resample("D")
        .last()
        .reindex(prediction_dates)
        .fillna(method="ffill")
        .fillna(method="bfill")
    )

    # Plot results
    plt.figure(figsize=(8, 4))
    plt.plot(
        prediction_dates,
        daily_actual.values,
        "bo-",
        label="Actual Fullness",
        linewidth=2,
    )
    plt.plot(
        prediction_dates,
        hybrid_forecast,
        "ro--",
        label="Hybrid ES + SARIMA Predicted Fullness",
        linewidth=2,
    )
    plt.ylim(0, 5)
    plt.title(f"Actual vs Hybrid ES + SARIMA Predicted Fullness - Bin {bin_id}")
    plt.xlabel("Date")
    plt.ylabel("Fullness Level")
    plt.legend()
    plt.grid()
    plt.show()

    # RMSE
    rmse = np.sqrt(mean_squared_error(daily_actual.values, hybrid_forecast))
    results_es_sarima[bin_id] = rmse
    # MAE calculation
    mae = mean_absolute_error(daily_actual.values, hybrid_forecast)
    results_es_sarima_mae[bin_id] = mae

# Display RMSE and MAE
results_df = pd.DataFrame(
    {
        "ES + Sarima (Residual) RMSE": pd.Series(results_es_sarima),
        "ES + Sarima (Residual) MAE": pd.Series(results_es_sarima_mae),
    }
)
results_df.index.name = "Bin ID"
print(results_df)
