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
from PIL import Image

st.set_page_config(page_title="Vaigai Express Delay Dashboard", layout="wide")

cris_logo = Image.open("logo_cris.png")
st.image(cris_logo, width=150)

st.title("12635 : Vaigai Train Delay Analysis & Prediction Dashboard")
st.header("[15-06-2025 to 24-06-2025]")

@st.cache_data
def load_data():
    df = pd.read_csv("merged_vaigai_df.csv", parse_dates=["TRAINDATE", "ARVL_TIME", "SCHED_ARVL_TIME"])
    return df

df = load_data()
df.columns = df.columns.str.strip()
df["STTN_CODE"] = df["STTN_CODE"].astype(str).str.strip()

section = st.sidebar.radio("Select Section", ["Train Delay Analysis", "Route Performance", "Delay Prediction"])

if section == "Train Delay Analysis":
    st.header("Arrival Delay at Final Station : MDU")

    start_date = pd.to_datetime("15-06-2025")
    end_date = pd.to_datetime("24-06-2025")
    st.info(f"Showing delays from **{start_date.date()}** to **{end_date.date()}**")

    filtered_df = df[(df["TRAINDATE"] >= start_date) & (df["TRAINDATE"] <= end_date)]

    if not filtered_df.empty:
        mdu_df = filtered_df[filtered_df["STTN_CODE"] == "MDU"].dropna(subset=["ARVL_DELAY_MIN"])
        if not mdu_df.empty:
            mdu_df["DATE_ONLY"] = mdu_df["TRAINDATE"].dt.date
            st.line_chart(mdu_df.set_index("DATE_ONLY")["ARVL_DELAY_MIN"])
            df_display = (mdu_df[["TRAINDATE", "ARVL_DELAY_MIN"]]
                          .rename(columns={"ARVL_DELAY_MIN": "DELAY AT MDU (min)"})
                          .reset_index(drop=True))
            df_display["TRAINDATE"] = df_display["TRAINDATE"].dt.date 
            df_display.index += 1
            df_display.index.name = "S. No"
            st.dataframe(df_display)
        else:
            st.warning("No MDU delay data found for selected dates.")
    else:
        st.warning("No data found for the selected date range.")

elif section == "Route Performance":
    st.header("Route-wise Delay Analysis")

    route_df = df.dropna(subset=["ARVL_DELAY_MIN", "TRAINDATE"])
    route_df["TRAINDATE"] = pd.to_datetime(route_df["TRAINDATE"]).dt.date

    if not route_df.empty:
        mean_delay = route_df.groupby("STTN_CODE")["ARVL_DELAY_MIN"].mean().reset_index()
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        sns.barplot(data=mean_delay, x="STTN_CODE", y="ARVL_DELAY_MIN", palette="coolwarm", ax=ax1)
        ax1.set_title("Average Arrival Delay per Station")
        ax1.set_xlabel("Station")
        ax1.set_ylabel("Avg Delay (min)")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(14, 6))
        sns.lineplot(data=route_df, x="TRAINDATE", y="ARVL_DELAY_MIN", hue="STTN_CODE", marker="o", ax=ax2)
        ax2.set_title("Arrival Delay Trend Over Days (Per Station)")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Delay (min)")
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True)
        st.pyplot(fig2)

    else:
        st.warning("No delay data available.")

elif section == "Delay Prediction":
    st.header("Predict Arrival Delay at Madurai (MDU) Using ML Model")

    distance_km = 493.0  # Distance for Madurai is fixed

    with st.expander("Enter Journey Details"):
        st.info(f"Prediction is only for final station **MDU** ({distance_km} km from MS)")

        day_of_week = st.selectbox("Day of Week (0=Mon, ..., 6=Sun)", list(range(7)))
        hour_of_day = st.slider("Hour of Departure", 0, 23, 14)
        is_weekend = st.radio("Is Weekend?", [0, 1])
        prev_station_delay = st.number_input("Previous Station Departure Delay (min)", min_value=-30.0, max_value=300.0, value=5.0)

    if st.button("Predict Arrival Delay"):
        try:
            model = joblib.load("mdu_delay_model.pkl")
            scaler = joblib.load("mdu_scaler.pkl")
            feature_names = joblib.load("mdu_features.pkl")

            time_of_day = "Night"
            if 5 <= hour_of_day < 12:
                time_of_day = "Morning"
            elif 12 <= hour_of_day < 17:
                time_of_day = "Afternoon"
            elif 17 <= hour_of_day < 21:
                time_of_day = "Evening"
              
            input_dict = {feature: 0 for feature in feature_names}
            input_dict.update({
                "DAY_OF_WEEK": day_of_week,
                "HOUR_OF_DAY": hour_of_day,
                "IS_WEEKEND": is_weekend,
                "DISTANCE_KM": distance_km,
                "PREV_STATION_DEP_DELAY_MIN": prev_station_delay,
                f"TIME_OF_DAY_{time_of_day}": 1
            })

            input_df = pd.DataFrame([input_dict])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            st.success(f"Predicted Arrival Delay at MDU: {round(prediction, 2)} minutes")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
