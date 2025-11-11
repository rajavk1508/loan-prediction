import streamlit as st
import pandas as pd
import joblib
import os

BASE = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE, 'models', 'best_pipeline.joblib')

st.set_page_config(page_title="Loan Sanction Amount Predictor", layout="centered")

st.title("Loan Sanction Amount Predictor")
st.markdown("Enter applicant details below")

@st.cache_resource
def load_model(path=MODEL_PATH):
    return joblib.load(path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model. Run training first. Error: {e}")
    st.stop()

with st.form("predict_form"):
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    property_price = st.number_input("Property Price (INR)", min_value=0, value=5000000, step=10000)
    loan_requested = st.number_input("Loan Amount Requested (INR)", min_value=0, value=2000000, step=10000)
    num_defaults = st.number_input("Number of Defaults", min_value=0, value=0)
    income = st.number_input("Applicant Annual Income (INR)", min_value=0, value=600000, step=10000)
    property_location = st.selectbox("Property Location", ["Urban", "Semi-Urban", "Rural"])
    applicant_location = st.selectbox("Applicant Location", ["CityA", "CityB", "CityC", "CityD"])
    submitted = st.form_submit_button("Predict")

if submitted:
    X_new = pd.DataFrame([{
        'Credit_Score': credit_score,
        'Property_Price': property_price,
        'Loan_Amount_Requested': loan_requested,
        'Number_of_Defaults': num_defaults,
        'Income': income,
        'Property_Location': property_location,
        'Applicant_Location': applicant_location
    }])
    pred = model.predict(X_new)[0]
    st.success(f"Predicted Loan Sanction Amount: ₹{int(pred):,}")
