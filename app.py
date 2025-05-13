import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
def time_to_minutes(time_str):
    try:
        h, m = map(int, str(time_str).split(':'))
        return h * 60 + m
    except:
        return np.nan

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['le_origin'], data['le_dest'], data['le_carrier']

# Load model and encoders
model, le_origin, le_dest, le_carrier = load_model()

# -----------------------------
st.title("✈️ Flight Delay Prediction App")
st.write("Enter flight details to predict if it will be **Delayed** or **On-Time**.")

origin = st.text_input("Origin Airport Code (e.g., JFK, LAX):")
destination = st.text_input("Destination Airport Code (e.g., ATL, SFO):")
carrier = st.text_input("Airline Carrier (e.g., Delta, United):")
year = st.number_input("Year of Flight (e.g., 2023)", min_value=2000, max_value=2100, value=2023)
sched_dep = st.text_input("Scheduled Departure Time (HH:MM):")
sched_arr = st.text_input("Scheduled Arrival Time (HH:MM):")

if st.button("Predict"):
    try:
        input_data = {
            'origin_enc': le_origin.transform([origin])[0],
            'destination_enc': le_dest.transform([destination])[0],
            'carrier_enc': le_carrier.transform([carrier])[0],
            'year': year,
            'sched_dep_min': time_to_minutes(sched_dep),
            'sched_arr_min': time_to_minutes(sched_arr),
        }

        input_df = pd.DataFrame([input_data])

        if input_df.isnull().any().any():
            st.error("⚠️ Invalid time format or missing input.")
        else:
            pred = model.predict(input_df)[0]
            result = "🟥 Delayed" if pred == 1 else "🟩 On-Time"
            st.success(f"Prediction: **{result}**")
    except Exception as e:
        st.error(f"Error: {str(e)}")
