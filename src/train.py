import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

BASE = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE, 'data', 'loan_data.csv')
MODEL_DIR = os.path.join(BASE, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

TARGET = 'Loan_Sanction_Amount'
FEATURES = [
    'Credit_Score', 'Property_Price', 'Loan_Amount_Requested',
    'Number_of_Defaults', 'Income', 'Property_Location', 'Applicant_Location'
]

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['Credit_Score', 'Property_Price', 'Loan_Amount_Requested', 'Number_of_Defaults', 'Income']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['Property_Location', 'Applicant_Location']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(max_depth=6, min_samples_leaf=13, random_state=1),
    'RandomForest': RandomForestRegressor(n_estimators=48, random_state=1, n_jobs=-1)
}

best_r2 = -np.inf
best_pipeline = None
best_name = None

for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"{name} -> R2: {r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    if r2 > best_r2:
        best_r2 = r2
        best_pipeline = pipe
        best_name = name

model_path = os.path.join(MODEL_DIR, 'best_pipeline.joblib')
joblib.dump(best_pipeline, model_path)
print(f"Best model: {best_name} with R2={best_r2:.4f}. Saved to {model_path}")
