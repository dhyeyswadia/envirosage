import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("cleaned_bin_data.csv", parse_dates=["timestamp"])

# Bins to forecast
bins_to_forecast = [
    1511195,
    1511196,
    1511199,
    1511204,
    1511209,
    1511210,
    1511218,
    1511221,
]
df = df[df["Bin ID"].isin(bins_to_forecast)]
train_data = df[df["timestamp"] <= "2021-04-26"]


def prepare_data(series, time_steps=10):
    X, y = [], []
    for i in range(len(series) - time_steps):
        X.append(series[i : i + time_steps])
        y.append(series[i + time_steps])
    return np.array(X), np.array(y)


results = []

for bin_id in bins_to_forecast:
    bin_train = train_data[train_data["Bin ID"] == bin_id].sort_values(by="timestamp")
    bin_train.set_index("timestamp", inplace=True)

    sarima_model = SARIMAX(
        bin_train["Fullness"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)
    )
    sarima_fit = sarima_model.fit(disp=False)
    sarima_fitted = sarima_fit.fittedvalues
    sarima_forecast = sarima_fit.forecast(steps=1)

    residuals = bin_train["Fullness"] - sarima_fitted
    scaler = MinMaxScaler()
    residuals_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))

    X_train, y_train = prepare_data(residuals_scaled)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    model = Sequential(
        [
            LSTM(
                100,
                activation="relu",
                return_sequences=True,
                input_shape=(X_train.shape[1], 1),
            ),
            Dropout(0.2),
            LSTM(100, activation="relu"),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

    last_seq = residuals_scaled[-10:].reshape(1, 10, 1)
    pred_residual = model.predict(last_seq, verbose=0)[0, 0]
    pred_residual = scaler.inverse_transform([[pred_residual]])[0, 0]

    hybrid_pred = sarima_forecast.values[0] + pred_residual
    last_date = bin_train.index[-1]
    prediction_date = last_date + pd.Timedelta(days=1)

    bin_full = (
        df[df["Bin ID"] == bin_id].sort_values(by="timestamp").set_index("timestamp")
    )
    actual_value = (
        bin_full["Fullness"]
        .resample("D")
        .last()
        .reindex([prediction_date])
        .fillna(method="ffill")
        .fillna(method="bfill")
        .values[0]
    )

    results.append(
        {
            "Bin ID": bin_id,
            "Date": prediction_date,
            "Actual Fullness": actual_value,
            "Predicted Fullness": hybrid_pred,
        }
    )

# Convert to DataFrame
result_df = pd.DataFrame(results)

# Priority calculation
min_val = result_df["Predicted Fullness"].min()
max_val = result_df["Predicted Fullness"].max()
interval = (max_val - min_val) / 5
result_df["Priority"] = (
    ((result_df["Predicted Fullness"] - min_val) // interval + 1)
    .clip(upper=5)
    .astype(int)
)

print(
    result_df[["Bin ID", "Date", "Actual Fullness", "Predicted Fullness", "Priority"]]
)
