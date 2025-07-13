import streamlit as st
import pickle
import numpy as np

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💓 Heart Disease Prediction")

age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
bp = st.number_input("Resting BP", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
ekg = st.selectbox("EKG results", [0, 1, 2])
hr = st.number_input("Max Heart Rate", 60, 220, 150)
angina = st.selectbox("Exercise Induced Angina", [0, 1])
st_depression = st.number_input("ST Depression", 0.0, 5.0, 1.0)
slope = st.selectbox("Slope of ST", [1, 2, 3])
vessels = st.selectbox("Number of vessels (0-3)", [0, 1, 2, 3])
thallium = st.selectbox("Thallium Test", [3, 6, 7])

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, bp, chol, fbs, ekg, hr, angina,
                            st_depression, slope, vessels, thallium]])
    result = model.predict(input_data)
    if result[0] == 1:
        st.error("⚠️ Likely Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")
