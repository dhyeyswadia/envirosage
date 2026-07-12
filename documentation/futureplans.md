
# Future Plans and TODO List

The following tasks are specifically designed to extend the capabilities of the EnviroSage project. These ideas are based on the existing implementation, datasets, and research work, and are organized as actionable goals.

---

## Immediate Project-Specific Tasks

* [ ] Deploy Dashboard Interface
  Develop a web-based interface or dashboard for visualizing real-time bin status, predictions, and alert notifications.


* [ ] Notification System
  Implement a threshold-based alerting mechanism (e.g., email/SMS) when bin fullness is predicted to exceed safe limits.

* [ ] Prediction Logging
  Store all prediction outputs with timestamps for ongoing accuracy tracking and retraining.

* [ ] Incorporate External Influences in Dataset
  Expand synthetic dataset generator to include features like weather, holidays, population density, or market zones.

* [ ] Integrate Collection Intensity Feedback
  Use daily markdown intensity input to dynamically scale synthetic generation and adjust forecasting parameters.

---

## Mid-Term Enhancements

* [ ] Geo-Zonal Prediction System
  Build separate models or prediction flows for specific wards, neighborhoods, or zones.

* [ ] Real-Time Dashboard for Monitoring
  Include predicted vs actual fill status, alert logs, and confidence scores for each bin.

* [ ] Synthetic Data Robustness
  Add anomalies and noise simulations to test model resilience.

* [ ] REST API for Central Aggregation
  Build a simple REST API to accept data from bins and expose forecast results.

* [ ] Lightweight Edge Deployment
  Evaluate if hybrid or LSTM models can be pruned and deployed directly on ESP32 or similar edge hardware.

---

## Research Continuation and Reporting

* [ ] Collaborate with Municipality
  Validate model predictions against real data from local waste management authorities.

* [ ] Add Anomaly Detection Module
  Use classification or clustering to detect hardware faults or bin tampering.

* [ ] Maintain Research Logs
  Add a changelog file for recording major updates, evaluations, and experiments over time.

This roadmap will evolve with implementation progress and feedback from simulated and field deployments.
