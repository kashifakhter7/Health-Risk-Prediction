# app.py
try:
    import streamlit as st
except ModuleNotFoundError:
    raise ImportError("Streamlit is not installed. Please run 'pip install streamlit' to use this app.")

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from utils.nutrition_lookup import get_nutrition

# Load model and scaler
try:
    model = joblib.load("model/disease_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
except FileNotFoundError as e:
    raise FileNotFoundError("Model or scaler file not found. Ensure 'model/disease_model.pkl' and 'model/scaler.pkl' exist.") from e

st.title("Nutrition Tracker with Disease Risk Alert")

# User Inputs
name = st.text_input("Name")
age = st.number_input("Age", min_value=1)
gender = st.selectbox("Gender", ["Male", "Female"])
weight = st.number_input("Weight (kg)")
height = st.number_input("Height (cm)")
activity = st.selectbox("Activity Level", ["Sedentary", "Light", "Moderate", "Active"])
diseases = st.multiselect("Existing Diseases", ["Diabetes", "Hypertension", "Heart Issues", "None"])

meal_input = st.text_area("Enter your meals (e.g., 2 chapatis, dal, rice)")

if st.button("Analyze"):
    nutrients = get_nutrition(meal_input)

    # Prediction using ML model
    try:
        input_data = scaler.transform([[nutrients['calories'], nutrients['protein'],
                                         nutrients['fat'], nutrients['carbs'],
                                         nutrients['sugar'], nutrients['sodium']]])
        risk = model.predict(input_data)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        risk = None

    # Visualization
    st.subheader("Nutrient Breakdown")
    st.write(nutrients)

    labels = list(nutrients.keys())
    values = list(nutrients.values())
    fig, ax = plt.subplots()
    ax.bar(labels, values, color='skyblue')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Risk Alert
    st.subheader("Health Alert")
    if risk is not None:
        if risk == 1:
            st.warning("⚠️ Potential health risk detected based on your meal intake!")
        else:
            st.success("✅ No major health risks detected based on your meals.")

    # Recommendations
    st.subheader("Recommendations")
    if nutrients['sugar'] > 50:
        st.info("Consider reducing sugar intake to lower diabetes risk.")
    if nutrients['sodium'] > 2300:
        st.info("High sodium intake! Cut back on salty foods.")
    if nutrients['calories'] > 2500:
        st.info("High calorie intake. Add more activity to your routine.")
