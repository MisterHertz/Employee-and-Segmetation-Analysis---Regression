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

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Streamlit UI components
st.title("Employee Income Prediction")
st.write("Enter the details below to predict monthly income:")

# Input fields
years_experience = st.number_input("Years of Experience", min_value=0)
age = st.number_input("Age", min_value=18)
weekly_hours = st.number_input("Weekly Hours Worked", min_value=0)
performance_score = st.number_input("Performance Score", min_value=1, max_value=5)
education_level = st.selectbox("Education Level", ['SMA', 'D3', 'S1', 'S2'])

# Prepare input data
input_data = np.array([[years_experience, age, weekly_hours, performance_score, education_level]])
input_df = pd.DataFrame(input_data, columns=['years_experience', 'age', 'weekly_hours', 'performance_score', 'education_level'])

# Preprocess the input data
input_scaled = scaler.transform(input_df)

# Make prediction
if st.button("Predict Income"):
    prediction = model.predict(input_scaled)
    st.write(f"Predicted Monthly Income: ${prediction[0]:,.2f}")
