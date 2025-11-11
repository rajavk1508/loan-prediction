import pandas as pd
import numpy as np
import os

os.makedirs(os.path.dirname(__file__), exist_ok=True)
n = 2000
np.random.seed(1)

credit_score = np.clip((np.random.normal(650, 70, n)).astype(int), 300, 850)
property_price = np.round(np.random.normal(5000000, 1500000, n)).astype(int)
loan_requested = np.round(property_price * np.random.uniform(0.1, 0.8, n)).astype(int)
num_defaults = np.random.poisson(0.2, n)
income = np.round(np.random.normal(600000, 200000, n)).astype(int)
property_locations = np.random.choice(['Urban', 'Semi-Urban', 'Rural'], n, p=[0.5,0.3,0.2])
applicant_locations = np.random.choice(['CityA','CityB','CityC','CityD'], n)

loan_sanction = (0.2*property_price + 0.4*loan_requested*0.8 + income*0.5 + (credit_score-600)*1000 - num_defaults*50000)
loan_sanction = (loan_sanction * np.random.uniform(0.85, 1.05, n)).astype(int)
loan_sanction = np.clip(loan_sanction, 20000, property_price)

df = pd.DataFrame({
    'Credit_Score': credit_score,
    'Property_Price': property_price,
    'Loan_Amount_Requested': loan_requested,
    'Number_of_Defaults': num_defaults,
    'Income': income,
    'Property_Location': property_locations,
    'Applicant_Location': applicant_locations,
    'Loan_Sanction_Amount': loan_sanction
})

df.to_csv(os.path.join(os.path.dirname(__file__), 'loan_data.csv'), index=False)
print("Saved sample dataset to data/loan_data.csv")
