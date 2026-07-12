import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

df = pd.read_csv("cleaned_bin_data.csv", parse_dates=["timestamp"])

# Select bins for forecasting
bins_to_forecast = [1511215, 1511194, 1511191]
df = df[df["Bin ID"].isin(bins_to_forecast)]

# Filter data up to May 3, 2021
train_data = df[df["timestamp"] < "2021-04-26"]
print(train_data)


def prepare_data(series, time_steps=7):
    X, y = [], []
    for i in range(len(series) - time_steps):
        X.append(series[i : i + time_steps])
        y.append(series[i + time_steps])
    return np.array(X), np.array(y)


results = {}
results_mae = {}

for bin_id in bins_to_forecast:
    bin_train = train_data[train_data["Bin ID"] == bin_id].sort_values(by="timestamp")

    # Normalize fullness values (scaling)
    scaler = MinMaxScaler()
    fullness_train = scaler.fit_transform(bin_train[["Fullness"]])

    # Prepare training data
    time_steps = 7
    X_train, y_train = prepare_data(fullness_train, time_steps)

    # Reshape input for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    # Build improved LSTM model
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
    model.compile(optimizer="nadam", loss="mse")

    # Train the model with more epochs
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    # Predict the next 7 days
    predictions = []
    last_sequence = fullness_train[-time_steps:].reshape(1, time_steps, 1)

    for _ in range(7):
        pred = model.predict(last_sequence)
        predictions.append(pred[0, 0])
        last_sequence = np.append(last_sequence[:, 1:, :], [[[pred[0, 0]]]], axis=1)

    # Inverse transform predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Define prediction period: 27th April to 3rd May
    prediction_start = pd.to_datetime("2021-04-27")
    prediction_end = pd.to_datetime("2021-05-03")
    prediction_dates = pd.date_range(
        start=prediction_start, end=prediction_end, freq="D"
    )

    # Use the full dataset for this bin and ensure 'timestamp' is datetime
    bin_full = df[df["Bin ID"] == bin_id].sort_values(by="timestamp")
    bin_full["timestamp"] = pd.to_datetime(bin_full["timestamp"])
    bin_full = bin_full.set_index("timestamp")

    # Resample to daily frequency, taking the last available 'Fullness' value of each day
    daily_actual = bin_full["Fullness"].resample("D").last()

    # Reindex to the complete prediction period (fills missing dates with NaN, then forward-fill if needed)
    daily_actual = daily_actual.reindex(prediction_dates).fillna(method="ffill")
    prediction_actual = daily_actual.values.reshape(-1, 1)

    # Calculate RMSE using the actual values for the prediction period
    rmse = np.sqrt(mean_squared_error(prediction_actual, predictions))
    results[bin_id] = rmse
    mae = mean_absolute_error(prediction_actual, predictions)
    results_mae[bin_id] = mae

    # Generate timestamps for the prediction period
    prediction_dates = pd.date_range(start="2021-04-27", end="2021-05-03", freq="D")
    print(predictions)

    # Plot actual vs predicted values for the prediction period
    plt.figure(figsize=(10, 5))
    plt.plot(
        prediction_dates, prediction_actual, "bo-", label="Actual Fullness", linewidth=2
    )
    plt.plot(
        prediction_dates, predictions, "ro--", label="Predicted Fullness", linewidth=2
    )
    plt.ylim(0, 5)
    plt.title(f"Actual vs Predicted Fullness - Bin {bin_id}")
    plt.xlabel("Date")
    plt.ylabel("Fullness Level")
    plt.legend()
    plt.show()


rmse_df = pd.DataFrame.from_dict(results, orient="index", columns=["RMSE"])
rmse_df.index.name = "Bin ID"
mae_df = pd.DataFrame.from_dict(results_mae, orient="index", columns=["MAE"])
mae_df.index.name = "Bin ID"
# Print RMSE values
print(rmse_df)
print(mae_df)
