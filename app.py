import streamlit as st
import numpy as np
import joblib

# Load scaler and models
scaler = joblib.load("scaler.pkl")
best_lr = joblib.load("best_logistic_regression.pkl")
best_rf = joblib.load("best_random_forest.pkl")
best_svm = joblib.load("svm_best_model.pkl")  

st.title("🩺 Diabetes Prediction App")

# Collect user input (raw, unscaled)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
skinthickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=85)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=28.5)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)


model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "SVM"])

if st.button("Predict"):
    features = np.array([[glucose, skinthickness, insulin, bmi, dpf, age]])
    
    # Scale input with the SAME scaler used in training
    features_scaled = scaler.transform(features)

    if model_choice == "Logistic Regression":
        prediction = best_lr.predict(features_scaled)[0]
    elif model_choice == "Random Forest":
        prediction = best_rf.predict(features_scaled)[0]
    else:  
        prediction = best_svm.predict(features_scaled)[0]


    if prediction == 1:
        st.error("⚠️ The model predicts: DIABETIC")
    else:
        st.success("✅ The model predicts: NOT DIABETIC")
