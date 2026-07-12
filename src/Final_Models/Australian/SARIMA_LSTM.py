import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

# Load resampled dataset
df = pd.read_csv(
    "/content/sample_data/cleaned_bin_data_new.csv", parse_dates=["timestamp"]
)

# Select bins for forecasting
bins_to_forecast = [1511208, 1511199, 1510830]
df = df[df["Bin ID"].isin(bins_to_forecast)]

# Filter data up to a specific date
train_data = df[df["timestamp"] <= "2021-04-26"]


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    # avoid NaN where both actual and forecast are zero
    diff[denominator == 0] = 0
    return np.mean(diff) * 100


def prepare_data(series, time_steps=10):
    X, y = [], []
    for i in range(len(series) - time_steps):
        X.append(series[i : i + time_steps])
        y.append(series[i + time_steps])
    return np.array(X), np.array(y)


results_sarima_lstm = {}
results_sarima_lstm_mae = {}
results_es_lstm_mape = {}
results_es_lstm_smape = {}
for bin_id in bins_to_forecast:
    bin_train = train_data[train_data["Bin ID"] == bin_id].sort_values(by="timestamp")
    bin_train.set_index("timestamp", inplace=True)

    # SARIMA model
    sarima_model = SARIMAX(
        bin_train["Fullness"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)
    )
    sarima_fit = sarima_model.fit(disp=False)
    sarima_fitted_values = sarima_fit.fittedvalues
    sarima_forecast = sarima_fit.forecast(steps=7)

    # Calculate residuals from SARIMA
    residuals = bin_train["Fullness"] - sarima_fitted_values

    # Scale residuals
    scaler = MinMaxScaler()
    residuals_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))

    # Prepare data for LSTM
    time_steps = 10
    X_train, y_train = prepare_data(residuals_scaled, time_steps)
    X_train = X_train.reshape((X_train.shape[0], time_steps, 1))

    # LSTM model
    model = Sequential(
        [
            LSTM(
                100,
                activation="relu",
                return_sequences=True,
                input_shape=(time_steps, 1),
            ),
            Dropout(0.2),
            LSTM(100, activation="relu"),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

    # Predict residuals with LSTM
    predictions = []
    last_sequence = residuals_scaled[-time_steps:].reshape(1, time_steps, 1)
    for _ in range(7):
        pred = model.predict(last_sequence, verbose=0)
        predictions.append(pred[0, 0])
        last_sequence = np.append(last_sequence[:, 1:, :], [[[pred[0, 0]]]], axis=1)

    residual_predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()

    # Combine SARIMA forecast and LSTM residual predictions
    hybrid_forecast = sarima_forecast.values + residual_predictions

    # Date alignment
    last_actual_date = bin_train.index[-1]
    prediction_dates = pd.date_range(
        start=last_actual_date + pd.Timedelta(days=1), periods=7, freq="D"
    )

    # Actual data alignment
    bin_full = (
        df[df["Bin ID"] == bin_id].sort_values(by="timestamp").set_index("timestamp")
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
        label="SARIMA + LSTM (Residual) Predicted Fullness",
        linewidth=2,
    )
    plt.ylim(0, 5)
    plt.title(f"Actual vs Predicted (SARIMA + LSTM) - Bin {bin_id}")
    plt.xlabel("Date")
    plt.ylabel("Fullness Level")
    plt.legend()
    plt.grid()
    plt.show()

    # RMSE calculation
    rmse = np.sqrt(mean_squared_error(daily_actual.values, hybrid_forecast))
    results_sarima_lstm[bin_id] = rmse
    # MAE calculation
    mae = mean_absolute_error(daily_actual.values, hybrid_forecast)
    results_sarima_lstm_mae[bin_id] = mae
    mape = mean_absolute_percentage_error(daily_actual.values, hybrid_forecast) * 100
    results_es_lstm_mape[bin_id] = mape
    smape_value = smape(daily_actual.values, hybrid_forecast)
    results_es_lstm_smape[bin_id] = smape_value

# Display RMSE and MAE
results_df = pd.DataFrame(
    {
        "SARIMA + LSTM (Residual) RMSE": pd.Series(results_es_lstm),
        "SARIMA + LSTM (Residual) MAE": pd.Series(results_es_lstm_mae),
        # 'ES + LSTM (Residual) MAPE (%)': pd.Series(results_es_lstm_mape)
        "SARIMA + LSTM (Residual) SMAPE (%)": pd.Series(results_es_lstm_smape),
    }
)
results_df.index.name = "Bin ID"

print(results_df)
