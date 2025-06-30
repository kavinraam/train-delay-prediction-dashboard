# train-delay-prediction-dashboard
# Vaigai Express Train Delay Analysis & Prediction

This project was developed as part of a summer internship at **Centre for Railway Information Systems (CRIS)** under the **Control Office Application (COA)** department. The goal of the project was to analyze train delay patterns and build a predictive model using machine learning, integrated into an interactive Streamlit dashboard.

## Project Overview

This application allows users to:
- Explore historical train delay patterns
- Analyze route performance through interactive visualizations
- Predict arrival delays using a trained Random Forest model

The train studied is the **Vaigai Express (Train No. 12635)** which runs between **Chennai Egmore (MS)** and **Madurai (MDU)**.

---

## Features

### 1. **Train Delay Analysis**
- Line chart of delays at the final station (MDU)
- Interactive date filter
- Delay summary table

### 2. **Route Performance Analysis**
- Boxplot for delay distribution at each station
- Bar chart showing average delay per station

### 3. **Delay Prediction**
- Predicts arrival delay using a pre-trained Random Forest model
- Inputs include:
  - Day of week
  - Hour of departure
  - Weekend flag
  - Halt duration
  - Departure delay
  - Distance (auto-filled by station)

---

## Machine Learning Model

- Model: `RandomForestRegressor`
- Hyperparameter tuning: `RandomizedSearchCV`
- Evaluation Metric: R² Score
- Final R² Score achieved: **0.83**

The trained model is saved as:
-vaigai_delay_model.pkl

## License

This project was developed as part of a internship at CRIS (Centre for Railway Information Systems). Not an official CRIS product, This project is intended for educational and research purposes only.

## Author

Kavin Raam M | LinkedIn: linkedin.com/in/kavinraam
