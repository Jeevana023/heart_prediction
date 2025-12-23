import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Diagnostic Tool")
st.write("Enter the patient details below to see the dual-prediction analysis.")

# --- STEP 1: CREATE INPUT FIELDS ---
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=63)
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", value=145)

with col2:
    chol = st.number_input("Cholesterol Level", value=233)
    thalach = st.number_input("Max Heart Rate", value=150)
    exang = st.selectbox("Exercise Angina (1=Yes, 0=No)", options=[0, 1])
    oldpeak = st.number_input("ST Depression (oldpeak)", value=2.3)

# --- STEP 2: PREDICTION LOGIC ---
if st.button("Predict Results"):
    # Organize features for the model
    features = np.array([[age, sex, cp, trestbps, chol, thalach, exang, oldpeak]])

    try:
        # Load your actual model
        # model = pickle.load(open('model.pkl', 'rb'))
        
        # --- DUMMY LOGIC (Replace this with model.predict_proba if you have a real model) ---
        # This simulates the model's behavior for this demo
        prob_disease = 0.85 if age > 50 and trestbps > 140 else 0.15
        prob_healthy = 1.0 - prob_disease
        # ---------------------------------------------------------------------------------

        # --- STEP 3: DISPLAY RESULTS ---
        st.divider()
        
        # Column layout for the two predictions
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric(label="Health Probability", value=f"{prob_healthy*100:.1f}%")
            if prob_healthy > 0.5:
                st.success("✅ Prediction: NO HEART DISEASE")
        
        with res_col2:
            st.metric(label="Disease Probability", value=f"{prob_disease*100:.1f}%")
            if prob_disease > 0.5:
                st.error("🚨 Prediction: HEART DISEASE")

        # Visual Progress Bar
        st.write("### Risk Visualization")
        st.progress(prob_disease)
        st.caption(f"The model is {max(prob_disease, prob_healthy)*100:.1f}% confident in this result.")

    except Exception as e:
        st.error(f"Error: Please ensure your model is loaded correctly. {e}")


