# -*- coding: utf-8 -*-
"""
Created on Jun 2025

@author: kavin
"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np

st.set_page_config(page_title="Vaigai Express Delay Dashboard", layout="wide")

st.title("Train Delay Analysis & Prediction Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("merged_vaigai_df.csv", parse_dates=["TRAINDATE", "ARVL_TIME", "SCHED_ARVL_TIME"])
    return df

df = load_data()
df.columns = df.columns.str.strip()
df["STTN_CODE"] = df["STTN_CODE"].astype(str).str.strip()

# Sidebar Navigation
section = st.sidebar.radio("Select Section", ["Train Delay Analysis", "Route Performance", "Delay Prediction"])

if section == "Train Delay Analysis":
    st.header("Arrival Delay at Final Station (MDU)")
    min_date = df["TRAINDATE"].min()
    max_date = df["TRAINDATE"].max()

    date_range = st.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
    filtered_df = df[(df["TRAINDATE"] >= pd.to_datetime(date_range[0])) & (df["TRAINDATE"] <= pd.to_datetime(date_range[1]))]

    mdu_df = filtered_df[filtered_df["STTN_CODE"] == "MDU"].dropna(subset=["ARVL_DELAY_MIN"])
    if not mdu_df.empty:
        st.line_chart(mdu_df.set_index("TRAINDATE")["ARVL_DELAY_MIN"])
        st.dataframe(mdu_df[["TRAINDATE", "ARVL_DELAY_MIN"]].rename(columns={"ARVL_DELAY_MIN": "Delay at MDU (min)"}))
    else:
        st.warning("No MDU delay data found for selected dates.")

elif section == "Route Performance":
    st.header("Route-wise Delay Distribution")
    route_df = df.dropna(subset=["ARVL_DELAY_MIN"])
    if not route_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=route_df, x="STTN_CODE", y="ARVL_DELAY_MIN", ax=ax)
        ax.set_title("Arrival Delay per Station")
        ax.set_xlabel("Station Code")
        ax.set_ylabel("Arrival Delay (minutes)")
        st.pyplot(fig)

        mean_delay = route_df.groupby("STTN_CODE")["ARVL_DELAY_MIN"].mean().reset_index()
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        sns.barplot(data=mean_delay, x="STTN_CODE", y="ARVL_DELAY_MIN", palette="coolwarm", ax=ax2)
        ax2.set_title("Average Delay per Station")
        st.pyplot(fig2)
    else:
        st.warning("No delay data available.")

elif section == "Delay Prediction":
    st.header("Predict Arrival Delay Using ML Model")

    with st.expander("Input Features"):
        day_of_week = st.selectbox("Day of Week (0=Mon, ..., 6=Sun)", list(range(7)))
        hour_of_day = st.slider("Hour of Departure", 0, 23, 14)
        is_weekend = st.radio("Is Weekend?", [0, 1])
        halt_min = st.number_input("Halt Duration (minutes)", min_value=0.0, max_value=15.0, value=2.0)
        dep_delay = st.number_input("Departure Delay (min)", min_value=-30.0, max_value=300.0, value=5.0)
        distance_km = st.number_input("Distance from Start (km)", min_value=0.0, max_value=600.0, value=250.0)

    if st.button("Predict Arrival Delay"):
        model = joblib.load("vaigai_delay_model.pkl")
        sample = pd.DataFrame([{
            "DAY_OF_WEEK": day_of_week,
            "HOUR_OF_DAY": hour_of_day,
            "IS_WEEKEND": is_weekend,
            "HALT_MIN": halt_min,
            "DEP_DELAY_MIN": dep_delay,
            "DISTANCE_KM": distance_km
        }])
        prediction = model.predict(sample)[0]
        st.success(f"Predicted Arrival Delay: {round(prediction, 2)} minutes")
