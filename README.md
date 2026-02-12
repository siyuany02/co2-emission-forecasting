# CO2 Emission Forecasting: ARIMA vs Deep Learning

Time series forecasting of per-capita CO2 emissions for five Nordic countries (1860–2021), comparing classical statistical modeling (ARIMA) with modern sequence models.

## Overview

This project investigates long-term CO2 emission trends using time series forecasting methods.

The baseline approach applies ARIMA with AIC/BIC-based parameter selection. 
Future extensions incorporate LSTM/GRU neural networks for nonlinear temporal modeling.

## Dataset

Source: Our World in Data  
Metric: Fossil CO2 emissions (tonnes per capita)  
Countries: Denmark, Finland, Iceland, Norway, Sweden  
Period: 1860–2021

## Methodology

### 1. Stationarity Testing
- Differencing to determine order d
- White noise detection

### 2. Model Selection
- ARIMA(p,d,q)
- Parameter selection via AIC & BIC

### 3. Evaluation
- MAE
- RMSE
- Short-term vs long-term prediction accuracy

### 4. Forecast Horizon
Predictions generated for 2022–2030.

## Key Findings

- ARIMA achieved strong short-term accuracy.
- Prediction stability declines with longer horizons.
- Some countries exhibit divergent future emission trends.

## Future Work

- Multivariate forecasting
- LSTM/GRU sequence modeling
- Hybrid ARIMA + Neural Networks

## Tech Stack

Python · statsmodels · NumPy · Pandas · Matplotlib
