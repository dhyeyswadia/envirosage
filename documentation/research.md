#  Research and Literature Foundation

## Introduction

The exponential increase in municipal solid waste (MSW) production, especially in urban centers, has created substantial challenges for city planners and waste management authorities. Traditional waste collection mechanisms follow static schedules without regard to the real-time status of bins, often resulting in overflowing bins or unnecessary trips. With the advent of IoT and advances in machine learning, the opportunity to design predictive systems has emerged. These systems can accurately estimate the fill levels of waste bins, allowing for optimized collection schedules.

This research focuses on the development and evaluation of predictive models—both traditional and advanced neural architectures—to forecast bin fill levels. Specifically, it investigates ARIMA, SARIMA, Exponential Smoothing (ES), and LSTM models, as well as multiple hybrid variations. These are implemented, tested, and benchmarked using real-world datasets from Australia and synthetic datasets modeled to simulate Indian urban conditions.

---

## Problem Statement

Despite significant technological advancements, current municipal waste management systems remain largely reactive. Most rely on static collection schedules and limited data analytics, leading to:

* Overflowing bins due to missed pickups
* Unnecessary collection rounds, consuming fuel and time
* Lack of adaptation to cultural or weather-driven waste surges (e.g., Diwali, monsoon season)

Additionally, most research models lack generalizability across geographies and are often limited to a single modeling paradigm (statistical or neural). This project aims to address those limitations by integrating real-time IoT sensor data with hybrid machine learning forecasting systems.

---

## Objectives

1. Design a forecasting framework that leverages hybrid time series models.
2. Integrate IoT-based bin fullness sensors to generate real-time data.
3. Evaluate the system on real and synthetic datasets.
4. Benchmark forecasting accuracy using RMSE, MAE, and MAPE.
5. Generate synthetic datasets to replicate Indian urban dynamics using Python.
6. Enable optimization of collection schedules based on predicted bin fullness.

---

## Dataset Description

### 1. Australian Real-World Dataset

* **Source**: Wyndham City Council (Melbourne, Australia) via data.gov.au
* **Time Period**: July 2018 to May 2021
* **Features**: Bin ID, LatestFullness, Timestamp, Geographic Coordinates, Thresholds, Alerts
* **Size**: 31,427 records across 32 bins
* **Fullness Encoding**: Values from 0 (empty) to 8 (full)

### 2. Synthetic Indian Dataset

Due to the lack of structured Indian MSW datasets, a synthetic generator script was created in [generate.py](../research/Generation_Cleaning/Indian/generate.py). The script allows for:

* Tuning waste levels based on cultural events, holidays, and day-of-week
* Assigning intensity levels (e.g., 1 for Hindu festivals, 4 for weekends)
* Realistic timestamp generation


The generated dataset mimics the behavior of bins in areas with different population densities and collection frequencies, enabling robust model testing for local Indian scenarios.

---

## Exploratory Data Analysis (EDA)

Prior to model building, the datasets were decomposed to assess:

* **Trend**: Gradual increase/decrease in waste over time
* **Seasonality**: Periodic fluctuations (weekly/monthly cycles)
* **Noise**: Random residuals not captured by model


Key findings:

* 35-day seasonality was strongest (Seasonal Strength > 0.5)
* Non-stationary series
* High variance across bins (some bins almost never filled; others filled weekly)


---

## Literature Review and Model Insights

The research is grounded in prior works across classical forecasting and deep learning:

| Model Type | Researcher    | Dataset          | Notes                                    |
| ---------- | ------------- | ---------------- | ---------------------------------------- |
| LSTM       | Ishii et al.  | Landfill data    | Best for complex non-linear data         |
| ARIMA      | Rashmi G.     | Bengaluru MSW    | Good for stationary short-term forecasts |
| ETSX       | Fokker et al. | Netherlands bins | Ensemble method using external variables |
| CNN vs RF  | Rajani & Devi | Waste images     | RF beat CNN in image classification      |

The literature supports the development of hybrid models to better capture complex patterns that are not fully addressed by individual statistical or deep learning models.

---

## Model Training and Testing

### Preprocessing Pipeline

* Handled missing values using forward-fill
* Converted timestamps to `datetime` objects
* Resampled data for daily and hourly granularity
* Normalized `LatestFullness` for training using MinMaxScaler


---

### Models Implemented

#### 1. ARIMA

* Used for linear, univariate forecasting.
* Assumes stationary data.
* Implemented with statsmodels.
* Struggled with long-term and seasonal variation.

#### 2. SARIMA

* Seasonal ARIMA variant.
* Captured weekly and monthly waste spikes.
* Modelled with parameters (p,d,q)(P,D,Q,s) determined through grid search.

#### 3. Exponential Smoothing (ES)

* Weighted average of past values.
* Simple and stable, but lacks seasonality modeling.
* Used as baseline.

#### 4. LSTM

* Recurrent neural network.
* Captured long-term dependencies and non-linear interactions.

---

## Hybrid Models

To overcome the shortcomings of individual models, hybrid systems were tested:

### SARIMA + LSTM

* SARIMA models the trend + seasonality
* Residuals are fed into LSTM to learn unexplained variations

### ES + LSTM

* Smooths data before sending to neural network

### ES + SARIMA + LSTM

* Final hybrid stack
* Resulted in best RMSE: **0.42** and MAE: **0.33**

Evaluation code:

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = mean_squared_error(y_true, y_pred, squared=False)
mae = mean_absolute_error(y_true, y_pred)
```

---

## Results Summary

| Model                  | RMSE     | MAE      |
| ---------------------- | -------- | -------- |
| ARIMA                  | 0.64     | 0.49     |
| SARIMA                 | 0.58     | 0.44     |
| LSTM                   | 0.50     | 0.39     |
| **ES + SARIMA + LSTM** | **0.42** | **0.33** |

---

## Contribution to Literature

This project contributes to the domain by:

* Combining time series decomposition with IoT
* Demonstrating the superiority of hybrid models on both sparse and dense data
* Generating a flexible, reusable synthetic dataset for Indian conditions
* Publishing a conference paper (IC-ICN2025) on results

Paper: ["Optimizing Waste Bin Management using Time Series Prediction Models"](https://drive.google.com/file/d/1mxvJYfuszuP4fVpKGxoeNyA285Pr_oQN/view?usp=sharing)

---

## Conclusion

The research lays the groundwork for intelligent waste bin forecasting systems that use hybrid models to optimize garbage collection schedules. Our ensemble approach combines the interpretability of statistical models with the flexibility of neural networks, offering a scalable and accurate forecasting solution applicable across cities and seasons.

Next steps include integrating real-time routing algorithms and extending the system to multi-bin zones with variable waste types.
