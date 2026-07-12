import pandas as pd

# Load your data
df = pd.read_csv("synthetic_mumbai_waste_data.csv")

# Convert date column to datetime to group correctly
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")


# Sort to ensure proper order of time for each bin per day
df.sort_values(by=["Bin_ID", "date", "time"], inplace=True)


# Optional: Save the cleaned data
df.to_csv("cleaned_waste_data.csv", index=False)
