# app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# Function to convert time string to minutes
def time_to_minutes(time_str):
    try:
        h, m = map(int, str(time_str).split(':'))
        return h * 60 + m
    except:
        return np.nan

# -----------------------------
# Load model and encoders
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['le_origin'], data['le_dest']

model, le_origin, le_dest = load_model()

# -----------------------------
# Streamlit UI
st.title("✈️ Flight Delay Prediction App")
st.write("Enter flight information to predict if the flight will be **Delayed** or **On-Time**.")

origin = st.text_input("Origin Airport Code (e.g., JFK, LAX):")
destination = st.text_input("Destination Airport Code (e.g., ATL, SFO):")
sched_dep = st.text_input("Scheduled Departure Time (HH:MM):")
sched_arr = st.text_input("Scheduled Arrival Time (HH:MM):")

if st.button("Predict"):
    try:
        input_data = {
            'origin_enc': le_origin.transform([origin])[0],
            'destination_enc': le_dest.transform([destination])[0],
            'sched_dep_min': time_to_minutes(sched_dep),
            'sched_arr_min': time_to_minutes(sched_arr),
        }

        input_df = pd.DataFrame([input_data])

        if input_df.isnull().any().any():
            st.error("⚠️ Invalid time format. Please use HH:MM.")
        else:
            pred = model.predict(input_df)[0]
            result = "🟥 Delayed" if pred == 1 else "🟩 On-Time"
            st.success(f"Prediction: **{result}**")
    except Exception as e:
        st.error(f"Error: {str(e)}")
