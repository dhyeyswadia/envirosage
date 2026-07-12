import pandas as pd

# Load dataset
file_path = "wyndham_smartbin_filllevel.csv"  # Replace with the actual file path
df = pd.read_csv(file_path)

# Convert timestamp column to datetime format
df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")

# Drop rows with missing timestamps
df = df.dropna(subset=["timestamp"])

# Sort the dataframe by Bin ID and timestamp
df = df.sort_values(by=["Bin ID", "timestamp"])
df = df.replace(
    {"Fullness": {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5}}
)

# Extract useful time-based features
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
df["day_of_week"] = df["timestamp"].dt.dayofweek

# Compute the change in fullness level (trend indicator) but **DO NOT normalize fullness values**
df["fullness_change"] = df.groupby("Bin ID")["Fullness"].diff().fillna(0)

# Save the cleaned dataset
cleaned_file_path = "cleaned_bin_data.csv"
df.to_csv(cleaned_file_path, index=False)

# Display success message
print(f"Cleaned dataset saved as: {cleaned_file_path}")

# Display the first few rows
print(df.head())
