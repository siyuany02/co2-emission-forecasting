# CO₂ Emission Forecasting: A Decision-Oriented Time Series Analysis

## Overview

This project develops a structured forecasting pipeline to analyze long-term CO₂ emission trends across five Nordic countries (1860–2021).

Rather than focusing solely on predictive accuracy, the goal is to evaluate **model reliability across different forecast horizons** and generate insights relevant for long-term environmental planning and policy discussions.

The project applies ARIMA modeling with systematic parameter selection and performance diagnostics to understand both short-term stability and long-term uncertainty.

---

## Data

- Source: Our World in Data  
- Metric: Fossil CO₂ emissions (tonnes per capita)  
- Countries: Denmark, Finland, Iceland, Norway, Sweden  
- Time Span: 1860–2021  

The long historical window allows examination of structural shifts, industrial transitions, and post-2000 decarbonization patterns.

---

## Methodology

### 1. Stationarity & Differencing
- Tested for non-stationarity in emission trends  
- Applied differencing to stabilize variance and remove trend components  

### 2. Model Selection
- ARIMA(p, d, q) models evaluated  
- Parameters selected using AIC and BIC criteria  
- Residual diagnostics performed to ensure white-noise behavior  

### 3. Evaluation Metrics
Model performance assessed using:

- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  

Evaluation considered both:
- Short-term predictive accuracy  
- Forecast degradation over longer horizons  

### 4. Forecast Horizon
Generated forecasts for 2022–2030 to examine near-term emission trajectory patterns.

---

## Key Insights

- ARIMA models provide strong short-term predictive stability.
- Forecast uncertainty increases significantly over longer horizons.
- Different countries exhibit distinct post-2000 structural behavior.
- Classical time series models may underperform during regime shifts or policy-driven transitions.

These findings highlight the importance of combining statistical forecasting with structural or policy-aware modeling approaches.

---

## Practical Implications

- Forecast models should be evaluated differently for short-term vs long-term planning.
- Policy decisions relying on long-horizon forecasts must incorporate uncertainty bounds.
- Historical emission stability does not guarantee future structural continuity.

---

## Technical Stack

- Python  
- statsmodels (ARIMA implementation)  
- NumPy / Pandas  
- Matplotlib  

---

## Repository Structure

- `/notebooks/arima_forecasting.ipynb` – Full forecasting workflow  
- `/docs/report.pdf` – Project summary and analysis write-up  

---

## Future Work

- Multivariate forecasting incorporating GDP and energy intensity  
- Regime-switching models  
- Neural sequence models (LSTM/GRU) for nonlinear temporal dynamics  

---

## Notes

This repository emphasizes structured forecasting methodology and interpretability rather than complex model architectures.  
The focus is on building reliable analytical pipelines that can support long-term decision-making under uncertainty.
