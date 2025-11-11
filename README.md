# Loan Amount Prediction App

Small ML project to predict loan sanction amount using Linear Regression, Decision Tree, and Random Forest.

## Steps to Run
1. Clone the repo.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Generate data:
   ```bash
   python data/generate_sample_data.py
   ```
4. Train model:
   ```bash
   python src/train.py
   ```
5. Run Streamlit app:
   ```bash
   streamlit run src/app_streamlit.py
   ```
