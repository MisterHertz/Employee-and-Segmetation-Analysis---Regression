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

# Load the entire pipeline (preprocessing + model)
with open('model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

st.title("Employee Income Prediction")

# Collect input from user
years_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
age = st.number_input("Age", min_value=18, max_value=70, value=25)
education = st.selectbox("Education Level", ["SMA", "Diploma", "Sarjana", "Magister", "Doktor"])
job_role = st.selectbox("Job Role", ["Staff", "Supervisor", "Manager", "Director", "Other"])
# Add more inputs as required by your model...

# Prepare input DataFrame
input_data = pd.DataFrame(
    [[years_experience, age, education, job_role]],
    columns=['years_experience', 'age', 'education', 'job_role']  # match exactly your training columns
)

# Make prediction
if st.button("Predict Income"):
    prediction = pipeline.predict(input_data)
    st.success(f"Predicted Income: ${prediction[0]:,.2f}")
