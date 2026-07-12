import random
import json
import csv
import datetime
import pandas as pd
import numpy as np

# Load the clusters of Mumbai dataset


def load_bin_data(csv_path):
    df = pd.read_csv(csv_path)
    if "religion_majority" not in df.columns:
        religions = ["hindu", "muslim"]
        df["religion_majority"] = [random.choice(religions) for _ in range(len(df))]
    return df


# Ensure diversity in religion majority per cluster


def ensure_religion_diversity(df):
    updated_df = df.copy()
    for cluster in df["knn_cluster"].unique():
        sub_df = df[df["knn_cluster"] == cluster]
        hindu_bins = sub_df[sub_df["religion_majority"] == "hindu"]
        muslim_bins = sub_df[sub_df["religion_majority"] == "muslim"]

        if len(hindu_bins) == 0:
            index = sub_df.sample(1).index[0]
            updated_df.at[index, "religion_majority"] = "hindu"
        if len(muslim_bins) == 0:
            index = sub_df.sample(1).index[0]
            updated_df.at[index, "religion_majority"] = "muslim"

    return updated_df


# Define filling rate categories


def categorize_bins(df):
    clusters = df["knn_cluster"].unique()
    bin_categories = {}

    for cluster in clusters:
        cluster_df = df[df["knn_cluster"] == cluster]
        cluster_size = len(cluster_df)
        cluster_bin_ids = cluster_df["Bin Id's"].tolist()
        random.shuffle(cluster_bin_ids)

        if cluster_size >= 3:
            num_high = max(1, round(cluster_size * 0.25))
            num_medium = max(1, round(cluster_size * 0.4))
            num_low = cluster_size - num_high - num_medium
        else:
            num_high, num_medium, num_low = 1, 1, max(0, cluster_size - 2)

        counts = {"high": num_high, "medium": num_medium, "low": num_low}
        i = 0

        for bin_id in cluster_bin_ids:
            if counts["high"] > 0:
                bin_categories[bin_id] = "high"
                counts["high"] -= 1
            elif counts["medium"] > 0:
                bin_categories[bin_id] = "medium"
                counts["medium"] -= 1
            else:
                bin_categories[bin_id] = "low"
                counts["low"] -= 1

    return bin_categories


# Define holidays


def get_holidays():
    hindu_holidays = [
        "2023-08-15",
        "2023-10-24",
        "2023-11-12",
        *[f"2023-09-{day}" for day in range(19, 29)],
        *[f"2023-10-{day}" for day in range(15, 26)],
        "2023-11-27",
        "2024-01-14",
        "2024-01-15",
        "2024-01-26",
        "2024-03-08",
        "2024-03-25",
        "2024-04-09",
        "2024-04-17",
        "2024-05-20",
        "2024-08-19",
        "2024-08-26",
        *[f"2024-09-{day:02d}" for day in range(7, 17)],
        "2024-10-02",
        *[f"2024-10-{day}" for day in range(12, 23)],
        "2024-10-12",
        "2024-10-31",
        "2024-11-01",
        "2024-11-15",
        "2024-12-25",
        "2025-01-14",
        "2025-01-15",
        "2025-01-26",
        "2025-02-26",
        "2025-03-14",
        "2025-03-30",
        "2025-04-06",
        "2025-04-13",
        "2025-04-19",
    ]
    ramadan_2023_march = [f"2023-03-{day:02d}" for day in range(23, 32)]
    ramadan_2023_april = [f"2023-04-{day:02d}" for day in range(1, 13)]
    islamic_holidays = [
        "2023-04-22",
        "2023-06-28",
        "2023-07-19",
        "2023-09-27",
        *ramadan_2023_march,
        *ramadan_2023_april,
        "2024-01-10",
        "2024-03-11",
        *[f"2024-03-{day:02d}" for day in range(11, 32)],
        *[f"2024-04-{day:02d}" for day in range(1, 10)],
        "2024-04-10",
        "2024-06-17",
        "2024-07-06",
        "2024-09-15",
        "2025-01-01",
        "2025-03-01",
        *[f"2025-03-{day:02d}" for day in range(1, 32)],
        *[f"2025-04-{day:02d}" for day in range(1, 26)],
    ]
    return hindu_holidays, islamic_holidays


def is_holiday(date_str, holidays):
    return date_str in holidays


def generate_fill_pattern(
    bin_category,
    is_hindu_holiday,
    is_islamic_holiday,
    location_cluster,
    day_of_week,
    prev_day_fill,
):
    hours = 12
    fill_pattern = []

    if bin_category == "high":
        base_increment = random.uniform(0.9, 1.5)
        emptying_time_range = (4, 5)
    elif bin_category == "medium":
        base_increment = random.uniform(0.5, 0.9)
        emptying_time_range = (5, 6)
    else:
        base_increment = random.uniform(0.2, 0.5)
        emptying_time_range = (6, 7)

    holiday_multiplier = (
        1.5 if bin_category == "high" else 1.3 if bin_category == "medium" else 1.2
    )
    weekend_multiplier = 1.4 if day_of_week in ["Saturday", "Sunday"] else 1.0

    if location_cluster in [0, 3] and is_islamic_holiday:
        holiday_multiplier *= 1.2
    elif location_cluster in [1, 2] and is_hindu_holiday:
        holiday_multiplier *= 1.2

    actual_increment = base_increment
    if is_hindu_holiday or is_islamic_holiday:
        actual_increment *= holiday_multiplier
    actual_increment *= weekend_multiplier

    current_fill = prev_day_fill
    empty_hour = random.randint(*emptying_time_range)

    for i in range(hours):
        time_of_day = i * 2
        if i < 4:
            fill_pattern.append(min(10, max(0, round(prev_day_fill))))
        elif i == empty_hour:
            current_fill = 0
            fill_pattern.append(0)
        else:
            hour_increment = actual_increment * random.uniform(0.8, 1.2)
            if time_of_day in [12, 14, 18, 20]:
                hour_increment *= 1.2
            if day_of_week in ["Saturday", "Sunday"] and time_of_day in [
                12,
                14,
                18,
                20,
            ]:
                hour_increment *= 1.2
            current_fill += hour_increment
            fill_pattern.append(min(10, max(0, round(current_fill))))

    return fill_pattern


def generate_waste_data(csv_path, start_date, end_date, output_file):
    bin_df = load_bin_data(csv_path)
    bin_df = ensure_religion_diversity(bin_df)
    bin_categories = categorize_bins(bin_df)
    hindu_holidays, islamic_holidays = get_holidays()

    bin_info = {}
    for _, row in bin_df.iterrows():
        bin_info[row["Bin Id's"]] = {
            "location": row["Location"],
            "cluster": row["knn_cluster"],
            "religion": row["religion_majority"],
        }

    data = []
    current_date = start_date
    entry_id = 1

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        day_of_week = current_date.strftime("%A")
        is_hindu_holiday = is_holiday(date_str, hindu_holidays)
        is_islamic_holiday = is_holiday(date_str, islamic_holidays)

        for _, row in bin_df.iterrows():
            bin_id = row["Bin Id's"]
            location = row["Location"]
            cluster = row["knn_cluster"]
            bin_category = bin_categories[bin_id]
            prev_fill = bin_info[bin_id].get("last_fill", 0)

            fill_pattern = generate_fill_pattern(
                bin_category,
                is_hindu_holiday,
                is_islamic_holiday,
                cluster,
                day_of_week,
                prev_fill,
            )

            bin_info[bin_id]["last_fill"] = fill_pattern[-1]

            for hour_index in range(12):
                hour_time = hour_index * 2
                time_str = f"{hour_time:02d}:00"
                data.append(
                    {
                        "entry_ID": entry_id,
                        "dustbin_id": int(bin_id),
                        "location": location,
                        "filled_capacity": fill_pattern[hour_index],
                        "date": date_str,
                        "time": time_str,
                        "day_of_week": day_of_week,
                        "fill_category": bin_category,
                    }
                )
                entry_id += 1

        current_date += datetime.timedelta(days=1)

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Data generation complete. {len(data)} entries saved to '{output_file}'.")

    print("\nBin Category Distribution by Cluster:")
    cluster_stats = {
        cluster: {"high": 0, "medium": 0, "low": 0}
        for cluster in bin_df["knn_cluster"].unique()
    }
    for bin_id, category in bin_categories.items():
        cluster = bin_info[bin_id]["cluster"]
        cluster_stats[cluster][category] += 1
    for cluster, stats in cluster_stats.items():
        print(
            f"Cluster {cluster}: High={stats['high']}, Medium={stats['medium']}, Low={stats['low']}"
        )


if __name__ == "__main__":
    start_date = datetime.date(2023, 7, 1)
    end_date = datetime.date(2025, 4, 25)
    generate_waste_data(
        csv_path="clusters_of_mumbai_dataset.csv",
        start_date=start_date,
        end_date=end_date,
        output_file="synthetic_mumbai_waste_data.csv",
    )
