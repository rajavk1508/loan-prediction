# src/app_streamlit.py
"""
Streamlit app for Loan Sanction Amount Predictor.

Behavior:
- If models/best_pipeline.joblib is missing or can't be loaded, this app will:
  1) run src/train.py as a subprocess,
  2) show training stdout/stderr in the UI,
  3) attempt to load the saved model again.
- Shows full error tracebacks inside Streamlit for easy debugging.
"""

import streamlit as st
import pandas as pd
import joblib
import os
import sys
import subprocess
import traceback
from pathlib import Path
from time import sleep

# Paths
BASE = Path(__file__).resolve().parents[1]  # project root
MODEL_PATH = BASE / "models" / "best_pipeline.joblib"
TRAIN_SCRIPT = BASE / "src" / "train.py"
DATA_PATH = BASE / "data" / "loan_data.csv"

st.set_page_config(page_title="Loan Sanction Amount Predictor", layout="centered")
st.title("🏦 Loan Sanction Amount Predictor")
st.markdown("Enter applicant details below. If needed, the app will train the model automatically (first run only).")

st.sidebar.header("Environment")
st.sidebar.write({
    "project_root": str(BASE),
    "model_path": str(MODEL_PATH),
    "data_path": str(DATA_PATH),
    "python": sys.executable
})

def run_training_and_capture():
    """Run the training script as a subprocess and return (returncode, stdout, stderr)."""
    if not TRAIN_SCRIPT.exists():
        return (1, "", f"Training script not found at: {TRAIN_SCRIPT}")
    # Use same python executable to run training
    cmd = [sys.executable, str(TRAIN_SCRIPT)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return (proc.returncode, proc.stdout, proc.stderr)
    except Exception as exc:
        return (1, "", f"Exception when launching training subprocess: {exc}\n{traceback.format_exc()}")

@st.cache_resource(show_spinner=False)
def load_model_cached():
    """Load model without retraining. Use caching only for successful loads."""
    return joblib.load(str(MODEL_PATH))

def try_load_model():
    """Try to load model; returns (model, None) on success or (None, error_message) on failure."""
    if not MODEL_PATH.exists():
        return None, f"Model file not found at {MODEL_PATH}"
    try:
        model = joblib.load(str(MODEL_PATH))
        return model, None
    except Exception as e:
        return None, f"Error loading model: {e}\n\nTraceback:\n{traceback.format_exc()}"

# 1) Try to load model
model, err = try_load_model()

if model is None:
    st.warning("Model not available or failed to load. Training will be started now (this may take a while).")
    with st.expander("Why training?", expanded=True):
        st.write(
            "The app couldn't find or load `models/best_pipeline.joblib`. "
            "We will run `src/train.py` to create the model. Training prints logs here; please wait until it finishes."
        )

    with st.spinner("Running training (src/train.py)..."):
        retcode, out, errout = run_training_and_capture()

    st.subheader("Training logs (stdout)")
    if out:
        st.code(out)
    else:
        st.write("_(no stdout)_")

    st.subheader("Training logs (stderr)")
    if errout:
        st.code(errout)
    else:
        st.write("_(no stderr)_")

    if retcode != 0:
        st.error(f"Training process exited with code {retcode}. See logs above for details.")
        st.stop()
    else:
        st.success("Training process finished. Attempting to load the saved model...")
        # small pause to ensure file system consistency
        sleep(0.5)
        model, err = try_load_model()
        if model is None:
            st.error("Model still could not be loaded after training. Full error:")
            st.code(err)
            st.stop()
        else:
            st.success("Model loaded successfully after training.")

else:
    st.success("Model loaded successfully.")

# At this point, `model` should be a loaded pipeline.
st.markdown("---")
st.subheader("Predict a loan sanction amount")

with st.form("predict_form"):
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    property_price = st.number_input("Property Price (INR)", min_value=0, value=5_000_000, step=10_000)
    loan_requested = st.number_input("Loan Amount Requested (INR)", min_value=0, value=2_000_000, step=10_000)
    num_defaults = st.number_input("Number of Defaults", min_value=0, value=0)
    income = st.number_input("Applicant Annual Income (INR)", min_value=0, value=600_000, step=10_000)
    property_location = st.selectbox("Property Location", ["Urban", "Semi-Urban", "Rural"])
    applicant_location = st.selectbox("Applicant Location", ["CityA", "CityB", "CityC", "CityD"])
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        X_new = pd.DataFrame([{
            'Credit_Score': int(credit_score),
            'Property_Price': int(property_price),
            'Loan_Amount_Requested': int(loan_requested),
            'Number_of_Defaults': int(num_defaults),
            'Income': int(income),
            'Property_Location': property_location,
            'Applicant_Location': applicant_location
        }])
        # debug: show the row passed to model
        st.write("Input data:")
        st.dataframe(X_new)

        pred = model.predict(X_new)[0]
        st.success(f"💰 Predicted Loan Sanction Amount: ₹{int(pred):,}")
    except Exception as e:
        st.error("Error during prediction. Full traceback follows:")
        st.code(traceback.format_exc())

