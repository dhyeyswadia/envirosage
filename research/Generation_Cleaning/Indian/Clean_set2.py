import pandas as pd

# Load dataset
file_path = "cleaned_waste_data.csv"  # Replace with the actual file path
df = pd.read_csv(file_path)

# Convert timestamp column to datetime format

# Drop rows with missing timestamps

# Sort the dataframe by Bin ID and timestamp
df = df.replace(
    {"Fullness": {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4, 10: 5}}
)

# Compute the change in fullness level (trend indicator) but **DO NOT normalize fullness values**

# Save the cleaned dataset
cleaned_file_path = "cleaned_bin_data.csv"
df.to_csv(cleaned_file_path, index=False)

# Display success message
print(f"Cleaned dataset saved as: {cleaned_file_path}")

# Display the first few rows
print(df.head())
