#  EnviroSage: Intelligent Waste Bin Management using Hybrid Time Series Forecasting

> Final Year Research Project | Department of Information Technology, TCET  
> Presented at: [IC-ICN 2025] | Developed by: Dhyey Swadia  

---

##  Project Summary

**EnviroSage** is a smart waste management solution that combines **IoT-based bin monitoring** with advanced **time series forecasting models** to predict bin fill levels and optimize waste collection routes. Using real-time sensor data and synthetic datasets, our hybrid modeling system delivers **highly accurate waste level predictions**, reducing overflow risks and operational costs in urban environments.

---

##  Tech Stack

- **Hardware**: NodeMCU (ESP8266), HC-SR04 Ultrasonic Sensor, Breadboard, Power supply
- **Languages & Libraries**: Python, TensorFlow/Keras, Pandas, Scikit-learn, Statsmodels
- **Models**: ARIMA, SARIMA, LSTM, Hybrid (ES + SARIMA + LSTM)
- **Tools**: Google Colab, Jupyter, matplotlib/seaborn for visualization
- **Deployment (future-ready)**: Flask/FastAPI + Cloud Hosting (GCP/AWS)

---

##  Key Features

-  **Time Series Forecasting** with hybrid models for short- and long-term waste prediction  
-  **IoT Integration** with smart bins (NodeMCU + HC-SR04)  
-  **Synthetic Data Generator** simulating real-world waste behavior across holidays, locations, and population densities  
-  **Real-Time Dashboard Ready** architecture  
-  **Optimized Collection Routing** based on bin priority and fill thresholds  

---

##  Folder Structure

```bash
envirosage/
├── Datasets/              # Generated and cleaned synthetic datasets
├── documentation/         # Full technical documentation (overview, methodology, etc.)
├── research/              # Indian datasets and comparative research scripts
├── src/                   # Source code for model training, sensor integration, APIs
└── README.md             # Git ignore rules
```

---

##  Documentation

All project documentation is organized in the [`documentation/`](documentation/) folder, including:

-  `overview.md`: Background, stakeholders, and system motivation  
-  `literature_review.md`: Review of ARIMA, SARIMA, LSTM, hybrid models  
-  `methodology.md`: Data generation, hardware, model training, and validation  
-  `implementation.md`: Architecture, block diagrams, and hardware details  
   See the [Hardware README](../src/Hardware/README.md) for circuit and setup steps  
-  `results_and_discussion.md`: Graphs, comparisons, RMSE/MAE insights  
-  `conclusion.md`: Final remarks, challenges, and future scope

---

##  Performance Snapshot

| Model              | RMSE   | MAE    |
|-------------------|--------|--------|
| ARIMA             | 0.64   | 0.49   |
| SARIMA            | 0.58   | 0.44   |
| LSTM              | 0.50   | 0.39   |
| **ES + SARIMA + LSTM** | **0.42** | **0.33** |

✔️ Ensemble model shows **up to 35% improvement in RMSE**  
✔️ Handles seasonal waste spikes (festivals, weekends, etc.)

---

##  Research Paper

> [Optimizing Waste Bin Management: A Novel Solution using Time Series Prediction Models and Machine Learning Techniques](https://drive.google.com/file/d/1mxvJYfuszuP4fVpKGxoeNyA285Pr_oQN/view?usp=sharing)  
Accepted at IC-ICN 2025 Conference

---

##  License

This project is academic in nature and may be reused with attribution.
