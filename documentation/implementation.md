#  Implementation and System Architecture

## Overview

The implementation phase of the EnviroSage project combines software engineering, machine learning, and embedded systems to build an end-to-end smart waste bin management solution. This solution is designed to predict the fill levels of waste bins in real time using hybrid time series forecasting models and control municipal operations accordingly. The implementation integrates sensor-based hardware with predictive modeling pipelines and prepares the foundation for eventual dashboard or application deployment.

The architecture includes three core components:

1. IoT-based smart bin hardware for data collection.
2. Data processing and model training layer for prediction.
3. Output interpretation for decision-making and potential visualization.

---

## Hardware Architecture

The smart bin prototype includes low-cost, open-source microcontrollers and sensors capable of measuring bin fill levels. The key components are:

* **NodeMCU ESP8266**: A Wi-Fi-enabled microcontroller used to interface with the sensor and transmit data to a central server.
* **HC-SR04 Ultrasonic Sensor**: Measures the distance from the top of the bin to the trash. This value is used to infer how full the bin is.
* **Breadboard and Jumper Wires**: For prototyping and circuit construction.
* **Power Supply**: USB power bank or direct micro-USB input.

The NodeMCU is programmed using the Arduino IDE and connected to the HC-SR04 sensor to collect distance measurements periodically. This sensor calculates the time taken for sound to reflect from the surface of the trash and computes the distance, which is then converted into fill level percentage.

The firmware deployed to the NodeMCU collects readings every 10 minutes and sends data over Wi-Fi to the server or logs it locally depending on connectivity. This is implemented and documented in [src/Hardware/README.md](../src/Hardware/README.md).

---

## Data Collection & Preprocessing

### Real-Time Sensor Data

Data from the NodeMCU is structured with the following attributes:

* Timestamp
* Bin ID
* Distance (in cm)
* Fill Level (%)
* Location ID or Zone Code

The server stores this data and processes it for analysis. Data is logged in CSV format or ingested via an HTTP API endpoint.

### Synthetic Dataset Simulation

In addition to real-time data, a synthetic dataset is generated using [generate.py](../research/Generation_Cleaning/Indian/generate.py). This dataset simulates:

* Indian festivals and holidays
* Weekend collection intensity
* Varying bin sizes and collection patterns

These datasets are vital in testing forecasting models when real-time data volume is insufficient.

### Preprocessing Steps

* Time-series resampling to hourly or daily intervals
* Forward-fill for missing values
* Feature scaling (MinMax normalization)
* One-hot encoding for categorical features like location/zone (if used)

---

## Prediction Pipeline

The core of the implementation lies in time series forecasting using multiple models, primarily LSTM (deep learning) and SARIMA (statistical).

### Steps:

1. Preprocessed data is loaded from CSV files.
2. The data is divided into training, validation, and testing sets.
3. Time series models are initialized and trained.
4. Each model predicts bin fullness over a chosen future time horizon (e.g., next 24 or 48 hours).
5. The forecasts are compared to actual values to compute performance metrics.

All of this is implemented in python file such as [LSTM.py](../research/Basic_Model_Research/Indian/LSTM.py), [SARIMA.py](../research/Basic_Model_Research/Indian/SARIMA.py), and [ARIMA.py](../research/Basic_Model_Research/Indian/ARIMA.py).

---

## Hybrid Model Implementation

Based on experimental evaluations, a hybrid model stack was implemented that uses:

* **Exponential Smoothing (ES)**: To smooth recent trends.
* **SARIMA**: To model seasonality and trend.
* **LSTM**: To learn residual patterns not captured by ES or SARIMA.

The hybrid model workflow is as follows:

1. Apply ES smoothing on raw data.
2. Apply SARIMA on smoothed data to extract forecast trend.
3. Feed residuals into the LSTM model for final correction.

This multi-model ensemble provides more robust and lower-error forecasts across both synthetic and real datasets.

---

## Model Evaluation

The predictions are evaluated using:

* **RMSE (Root Mean Square Error)**
* **MAE (Mean Absolute Error)**

All models are benchmarked against each other. Results are stored for plotting and comparison.

---

## Integration for Smart Bin Monitoring

While this prototype focuses on forecasting, the architecture supports future extensions:

* Trigger alerts when predicted fill levels exceed thresholds
* Dashboard view of predicted vs actual bin status
* Recommendation engine for optimized collection routes
* Flag bins needing attention due to high error or signal loss

---

## Conclusion

The implementation of the EnviroSage system successfully integrates both software and hardware to form a modular and intelligent waste monitoring framework. The sensor-driven hardware gathers real-world data, while hybrid time series models provide accurate predictions of waste bin status. This implementation lays the foundation for developing a scalable smart city waste management solution.

The complete hardware setup and connection diagrams are available in `src/Hardware/README.md`, and the model implementation logic is modularly organized inside the `research/` folder.

The next phase of development will involve real-time dashboard integration and potential deployment using cloud platforms or edge AI devices for on-device prediction.
