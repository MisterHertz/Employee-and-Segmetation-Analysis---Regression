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
import joblib

# Load the pre-trained model
best_model = joblib.load('best_model.pkl')

# Streamlit app title
st.title("Employee Monthly Income Prediction")

# User input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
years_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
weekly_hours = st.number_input("Weekly Hours", min_value=0, max_value=80, value=40)
overtime_hours = st.number_input("Overtime Hours", min_value=0, max_value=100, value=5)
bonus_percentage = st.number_input("Bonus Percentage", min_value=0, max_value=100, value=10)
performance_score = st.number_input("Performance Score", min_value=0, max_value=10, value=7)
education_level = st.selectbox("Education Level", ['SMA', 'D3', 'S1', 'S2'])
gender = st.selectbox("Gender", ['Male', 'Female'])
city = st.text_input("City", value='Jakarta')
department = st.selectbox("Department", ['Marketing', 'IT', 'Sales', 'Finance'])
marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])

# Collect user inputs into a DataFrame
input_data = pd.DataFrame([{
    'age': age,
    'years_experience': years_experience,
    'weekly_hours': weekly_hours,
    'overtime_hours': overtime_hours,
    'bonus_percentage': bonus_percentage,
    'performance_score': performance_score,
    'education_level': education_level,
    'gender': gender,
    'city': city,
    'department': department,
    'marital_status': marital_status
}])

# Predict button
if st.button("Predict Monthly Income"):
    prediction = best_model.predict(input_data)
    st.write(f"Predicted Monthly Income: ${prediction[0]:,.2f}")
