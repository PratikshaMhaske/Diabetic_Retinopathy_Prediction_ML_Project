# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 13:57:51 2025

@author: Datta
"""

import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("clf.pkl", "rb"))
sc = pickle.load(open("sc.pkl", "rb"))

# Prediction function
def predict(age, systolic_bp, diastolic_bp, cholesterol):

    # Convert input to array
    input_data = np.array([[age, systolic_bp, diastolic_bp, cholesterol]])

    # Scale input
    scaled_data = sc.transform(input_data)

    # Predict
    prediction = model.predict(scaled_data)

    return prediction[0]     # return the value instead of array

# Streamlit User Interface
def main():
    st.title("Prediction of Diabetic Retinopathy Model")

    age = st.number_input("Enter Age:", min_value=0.00)
    systolic_bp = st.number_input("Enter Systolic Blood Pressure:", min_value=0.00)
    diastolic_bp = st.number_input("Enter Diastolic Blood Pressure:", min_value=0.00)
    cholesterol = st.number_input("Enter Cholesterol:", min_value=0.00)

    if st.button("Predict"):
        result = predict(age, systolic_bp, diastolic_bp, cholesterol)

        if result == 0:
            st.success("No Diabetic Retinopathy (Negative)")
        else:
            st.error("Diabetic Retinopathy Detected (Positive)")
        st.warning("⚠️ Please consult a doctor. Do not rely completely on this prediction.")

if __name__ == "__main__":
    main()