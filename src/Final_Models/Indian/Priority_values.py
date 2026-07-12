import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Nadam
import math

# Load bin data and cluster information
df = pd.read_csv("cleaned_bin_data.csv")
df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
df.sort_values(by="datetime", inplace=True)

# Load cluster information
clusters_df = pd.read_csv("clusters_of_mumbai_dataset.csv")

# Group bins by cluster
cluster_bins = {}
for cluster in sorted(clusters_df["knn_cluster"].unique()):
    cluster_bins[cluster] = clusters_df[clusters_df["knn_cluster"] == cluster][
        "Bin Id's"
    ].tolist()

# Define time periods for training and testing
train_end = pd.Timestamp("2025-03-05 23:59:59")
test_start = pd.Timestamp("2025-03-06 00:00:00")
test_end = pd.Timestamp("2025-03-07 00:00:00")

# Get all bin IDs from the cluster data
bins = clusters_df["Bin Id's"].tolist()
time_steps = df.groupby("date")["time"].nunique().mode()[0]  # dynamic based on data

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


# Main processing for all bins
results_hybrid = {}
slopes = {}

print("Processing forecasts for all bins...")
for bin_id in bins:
    try:
        bin_data = df[df["Bin_ID"] == bin_id].set_index("datetime")

        # Skip bins with insufficient data
        if len(bin_data) < time_steps + 1:
            print(f"Skipping Bin {bin_id} due to insufficient data")
            continue

        train = bin_data[bin_data.index <= train_end]["Fullness"]
        test = bin_data[(bin_data.index >= test_start) & (bin_data.index < test_end)][
            "Fullness"
        ]

        # Skip if test data is empty
        if len(test) == 0:
            print(f"Skipping Bin {bin_id} due to no test data available")
            continue

        forecast_steps = len(test)

        # --- SARIMA ---
        sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 30))
        sarima_fit = sarima_model.fit(disp=False)

        # Get fitted values for the training period
        sarima_fitted = sarima_fit.fittedvalues

        # Calculate residuals for the training period
        residuals = train.values - sarima_fitted.values

        # Prepare residuals for LSTM training
        scaler = MinMaxScaler()
        scaled_residuals = scaler.fit_transform(residuals.reshape(-1, 1))

        # --- LSTM on Residuals ---
        X_train, y_train = prepare_data(scaled_residuals, time_steps)
        X_train = X_train.reshape((X_train.shape[0], time_steps, 1))

        lstm_model = create_model(time_steps)
        lstm_model.compile(optimizer=Nadam(), loss="mse")
        lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

        # Generate SARIMA forecasts for the test period
        sarima_forecast = sarima_fit.forecast(steps=forecast_steps).values

        # Generate LSTM residual forecasts
        last_seq = scaled_residuals[-time_steps:].reshape(1, time_steps, 1)
        lstm_preds = []
        for _ in range(forecast_steps):
            pred = lstm_model.predict(last_seq, verbose=0)
            lstm_preds.append(pred[0, 0])
            # Update the sequence for the next prediction
            last_seq = np.append(last_seq[:, 1:, :], [[[pred[0, 0]]]], axis=1)

        # Inverse transform LSTM predictions
        lstm_residual_forecast = scaler.inverse_transform(
            np.array(lstm_preds).reshape(-1, 1)
        ).flatten()

        # --- Combined Forecast (SARIMA + Residual LSTM) ---
        hybrid_forecast = sarima_forecast + lstm_residual_forecast

        # --- Calculate slope of the forecast ---
        X = np.arange(len(hybrid_forecast)).reshape(-1, 1)
        y = hybrid_forecast.reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        slope = reg.coef_[0][0]
        slopes[bin_id] = slope

        print(f"Bin {bin_id}: Forecast slope = {slope:.4f}")

    except Exception as e:
        print(f"Error processing Bin {bin_id}: {str(e)}")
        continue

# Calculate priorities by cluster
print("\nCalculating priorities by cluster...")
priorities_data = []

for cluster, cluster_bin_ids in cluster_bins.items():
    # Extract slopes for this cluster's bins
    cluster_slopes = {
        bin_id: slopes.get(bin_id) for bin_id in cluster_bin_ids if bin_id in slopes
    }

    if not cluster_slopes:
        print(f"No valid forecast data for any bin in Cluster {cluster}")
        continue

    # Calculate min and max slopes within this cluster
    min_slope = min(cluster_slopes.values())
    max_slope = max(cluster_slopes.values())
    slope_range = max_slope - min_slope

    # Assign priorities within this cluster
    for bin_id, slope in cluster_slopes.items():
        location = clusters_df[clusters_df["Bin Id's"] == bin_id]["Location"].values[0]

        # Avoid division by zero if all slopes in cluster are the same
        if slope_range == 0:
            priority = 3  # Middle priority if all same
        else:
            # Normalize the slope to 0-1 range within this cluster
            normalized_slope = (slope - min_slope) / slope_range
            # Convert to priority 1-5
            priority = math.ceil(normalized_slope * 4 + 1)

        priorities_data.append(
            {
                "Bin_ID": bin_id,
                "Location": location,
                "Cluster": cluster,
                "Slope": slope,
                "Priority": priority,
            }
        )

# Create and export the final DataFrame
priorities_df = pd.DataFrame(priorities_data)
priorities_df = priorities_df.sort_values(
    by=["Cluster", "Priority"], ascending=[True, False]
)

# Export to CSV
csv_filename = "bin_priorities_by_cluster.csv"
priorities_df.to_csv(csv_filename, index=False)

print(f"\nPriorities calculated and exported to {csv_filename}")
print(f"Total bins processed: {len(priorities_data)} out of {len(bins)}")

# Display a summary of the results
print("\nSummary of bin priorities by cluster:")
for cluster in sorted(cluster_bins.keys()):
    cluster_data = priorities_df[priorities_df["Cluster"] == cluster]
    if len(cluster_data) > 0:
        print(f"\nCluster {cluster} ({len(cluster_data)} bins):")
        print(
            cluster_data[["Bin_ID", "Location", "Priority", "Slope"]].to_string(
                index=False
            )
        )
