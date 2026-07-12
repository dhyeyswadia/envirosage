import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load data
df = pd.read_csv("cleaned_bin_data.csv", parse_dates=["timestamp"])
bins_to_forecast = [1511208, 1511194, 1511191]
df = df[df["Bin ID"].isin(bins_to_forecast)]
train_data = df[df["timestamp"] <= "2021-04-26"]

results_df = pd.DataFrame(columns=["Bin ID", "RMSE", "MAE"])

# LSTM helper function


def lstm_forecast(residual_series, forecast_days=7):
    scaler = MinMaxScaler()
    residual_scaled = scaler.fit_transform(residual_series.values.reshape(-1, 1))

    # Prepare data for LSTM
    X, y = [], []
    seq_len = 7
    for i in range(len(residual_scaled) - seq_len):
        X.append(residual_scaled[i : i + seq_len])
        y.append(residual_scaled[i + seq_len])
    X, y = np.array(X), np.array(y)

    # LSTM model
    model = Sequential()
    model.add(LSTM(32, activation="relu", input_shape=(seq_len, 1)))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")

    model.fit(X, y, epochs=50, batch_size=8, verbose=0)

    # Forecast next 7 days
    pred_input = residual_scaled[-seq_len:].reshape(1, seq_len, 1)
    forecasts = []
    for _ in range(forecast_days):
        pred = model.predict(pred_input, verbose=0)[0][0]
        forecasts.append(pred)
        pred_input = np.append(pred_input[:, 1:, :], [[[pred]]], axis=1)

    forecasts = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
    return forecasts


# Main Forecast Loop
for bin_id in bins_to_forecast:
    bin_train = train_data[train_data["Bin ID"] == bin_id].sort_values("timestamp")
    bin_train.set_index("timestamp", inplace=True)

    # Exponential Smoothing (ES)
    es_model = ExponentialSmoothing(
        bin_train["Fullness"], trend="add", seasonal="add", seasonal_periods=7
    )
    es_fit = es_model.fit()
    es_forecast = es_fit.forecast(7)

    # SARIMA on residuals of ES
    residuals_es = bin_train["Fullness"] - es_fit.fittedvalues
    sarima_model = SARIMAX(residuals_es, order=(1, 0, 1), seasonal_order=(1, 1, 1, 35))
    sarima_fit = sarima_model.fit()
    sarima_forecast = sarima_fit.forecast(7)

    # Hybrid forecast: ES + SARIMA
    hybrid_forecast_es_sarima = es_forecast.values + sarima_forecast.values

    # Residuals after SARIMA
    residuals_sarima = residuals_es - sarima_fit.fittedvalues

    # LSTM forecast on SARIMA residuals
    lstm_residual_forecast = lstm_forecast(residuals_sarima, 7)

    # Final Hybrid Forecast (ES + SARIMA + LSTM residual correction)
    final_hybrid_forecast = hybrid_forecast_es_sarima + lstm_residual_forecast

    # Dates for predictions
    last_date = bin_train.index[-1]
    pred_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=7, freq="D"
    )

    # Actual values for evaluation
    bin_full = (
        df[df["Bin ID"] == bin_id].sort_values("timestamp").set_index("timestamp")
    )
    daily_actual = (
        bin_full["Fullness"]
        .resample("D")
        .last()
        .reindex(pred_dates)
        .fillna(method="ffill")
        .fillna(method="bfill")
    )

    # Plot actual vs predicted
    plt.figure(figsize=(8, 4))
    plt.plot(pred_dates, daily_actual, "bo-", label="Actual", linewidth=2)
    plt.plot(
        pred_dates,
        final_hybrid_forecast,
        "go--",
        label="ES+SARIMA+LSTM Prediction",
        linewidth=2,
    )
    plt.title(f"Bin {bin_id}: Actual vs ES+SARIMA+LSTM")
    plt.xlabel("Date")
    plt.ylabel("Fullness")
    plt.legend()
    plt.grid()
    plt.ylim(0, 5)
    plt.show()

    # Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(daily_actual, final_hybrid_forecast))
    mae = mean_absolute_error(daily_actual, final_hybrid_forecast)

    # Store results
    results_df = pd.concat(
        [results_df, pd.DataFrame([{"Bin ID": bin_id, "RMSE": rmse, "MAE": mae}])],
        ignore_index=True,
    )


# Final RMSE and MAE results
results_df.set_index("Bin ID", inplace=True)
print("\nFinal Forecasting Performance (ES + SARIMA + LSTM):")
print(results_df)
