import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os   
import pickle

import streamlit as st
import pandas as pd
import pickle

# --- Define the pipeline as you already have ---
num_mean = ['years_experience', 'bonus_percentage', 'overtime_hours']
num_median = ['age', 'weekly_hours', 'performance_score']
cat_ord = ['education_level']
cat_ohe = ['gender', 'city', 'department', 'marital_status']

edu_map = [['SMA', 'D3', 'S1', 'S2']]

num_mean_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

num_median_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_ord_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('ordinal', OrdinalEncoder(categories=edu_map))
])

cat_ohe_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num_mean', num_mean_pipe, num_mean),
    ('num_median', num_median_pipe, num_median),
    ('cat_ord', cat_ord_pipe, cat_ord),
    ('cat_onehot', cat_ohe_pipe, cat_ohe)
])

# Example trained model (replace with your actual trained model)
best_model = Pipeline([
    ('preprocessing', preprocessor),
    ('model', LinearRegression())
])

# --- Streamlit App ---
st.title("Monthly Income Prediction")

st.header("Input Employee Data:")

years_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
bonus_percentage = st.number_input("Bonus Percentage", min_value=0, max_value=100, value=10)
overtime_hours = st.number_input("Overtime Hours", min_value=0, max_value=100, value=12)
age = st.number_input("Age", min_value=18, max_value=70, value=30)
weekly_hours = st.number_input("Weekly Working Hours", min_value=0, max_value=80, value=40)
performance_score = st.number_input("Performance Score", min_value=0, max_value=100, value=85)

education_level = st.selectbox("Education Level", ['SMA', 'D3', 'S1', 'S2'])
gender = st.selectbox("Gender", ['Male', 'Female'])
city = st.text_input("City", value='Jakarta')
department = st.text_input("Department", value='Sales')
marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced', 'Widower', 'Facto Union', 'Legally Separated'])

if st.button("Predict Monthly Income"):
    input_df = pd.DataFrame([{
        'years_experience': years_experience,
        'bonus_percentage': bonus_percentage,
        'overtime_hours': overtime_hours,
        'age': age,
        'weekly_hours': weekly_hours,
        'performance_score': performance_score,
        'education_level': education_level,
        'gender': gender,
        'city': city,
        'department': department,
        'marital_status': marital_status
    }])

    # Prediction
    predicted_income = best_model.predict(input_df)
    st.success(f"Predicted Monthly Income: ${predicted_income[0]:,.2f}")
