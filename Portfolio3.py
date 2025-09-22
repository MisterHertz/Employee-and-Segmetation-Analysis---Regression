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

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(page_title="Employee Segmentation Analysis", layout="wide")
st.title("Employee Segmentation Analysis and Regression")

# -------------------------------
# Load dataset
# -------------------------------
st.header("Data Overview")
try:
    df = pd.read_csv("employee_for_ML.csv")
    st.write(df.head())
except FileNotFoundError:
    st.error("Dataset not found. Ensure 'employee_for_ML.csv' is in the directory.")
    st.stop()

# -------------------------------
# Data Preprocessing
# -------------------------------
st.header("Data Preprocessing")

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
    ('cat_ohe', cat_ohe_pipe, cat_ohe)
])

# -------------------------------
# Split data
# -------------------------------
st.header("Train-Test Split")
X = df.drop(columns=['monthly_income'])
y = df['monthly_income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.write(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# -------------------------------
# Model Training
# -------------------------------
st.header("Model Training")

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X_train, y_train)
st.success("Linear Regression model trained successfully!")

# -------------------------------
# Model Evaluation
# -------------------------------
st.header("Model Evaluation")
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'MSE', 'R2'],
    'Value': [rmse, mae, mse, r2]
})
st.dataframe(metrics)

# -------------------------------
# Feature Importance (Linear Regression Coefficients)
# -------------------------------
st.header("Feature Importance")
# Extract feature names after preprocessing
feature_names = list(preprocessor.get_feature_names_out())
coefficients = model.named_steps['regressor'].coef_
feat_imp = pd.Series(coefficients, index=feature_names).sort_values(key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(12,6))
feat_imp.plot(kind='barh', ax=ax)
ax.invert_yaxis()
plt.title("Feature Importance (Linear Regression Coefficients)")
plt.xlabel("Coefficient Value")
st.pyplot(fig)

# -------------------------------
# Conclusion
# -------------------------------
st.header("Conclusion")
st.markdown("""
### Model Performance
- Linear Regression was trained and evaluated on the employee dataset.
- RMSE, MAE, MSE, and R² metrics are displayed above.
### Feature Drivers
- Most impactful features can be seen in the feature importance chart.
- `years_experience`, `weekly_hours`, `overtime_hours`, and `education_level` are major contributors.
""")
