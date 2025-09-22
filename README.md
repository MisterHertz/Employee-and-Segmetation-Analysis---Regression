# 📊 Employee Income & Segmentation Analysis
**MAJU JAYA DIGITAL TECHNOLOGY – Data Science Portfolio Project**

## 🔹 Project Overview
This project explores factors affecting employee income, builds predictive models to estimate **monthly income**, and provides insights for HR decision-making. The dataset contains employee demographic, work-related, and performance attributes.  

Main goals:
- Perform **Exploratory Data Analysis (EDA)** to understand data distribution and relationships.  
- Build a **regression model** to predict employee `monthly_income`.  
- Provide insights into which features are most influential for income prediction.  
- (Optional) Perform clustering for employee segmentation.  

---

## 🔹 Dataset
Columns included in the dataset:  

| Column              | Description |
|---------------------|-------------|
| `employee_id`       | Unique employee ID |
| `age`               | Employee age |
| `gender`            | Gender |
| `marital_status`    | Marital status |
| `city`              | City of workplace |
| `education_level`   | Highest education level |
| `years_experience`  | Years of work experience |
| `weekly_hours`      | Weekly working hours |
| `department`        | Department of employment |
| `bonus_percentage`  | Annual bonus percentage |
| `performance_score` | Performance score |
| `overtime_hours`    | Monthly overtime hours |
| `monthly_income`    | Monthly income (regression target) |
| `income_class`      | Income category (classification target) |

---

## 🔹 Methodology

### 1. Business Understanding
- **Background:** The company wants to understand factors that influence employee income and leverage insights for recruitment, compensation, and retention strategies.  
- **Objective:** Predict monthly income using regression models and provide insights into key influencing factors.  
- **Modeling Selection:** Regression task with algorithms such as **Linear Regression**, **Random Forest Regressor**, and **XGBoost**.  

### 2. Data Preparation
- Handle missing values with **mean/median imputation** for numerical features and **mode imputation** for categorical features.  
- Encode categorical variables using **One-Hot Encoding**.  
- Scale numerical features (optional, depending on model).  

### 3. Modeling
- Train regression models: Linear Regression, Random Forest, and XGBoost.  
- Evaluate using **MAE, MSE, RMSE, and R²**.  

### 4. Insights
- **Years of Experience** has the strongest positive correlation with income.  
- **Bonus Percentage** and **Overtime Hours** show negligible correlation.  
- **Weekly hours** have only a weak positive correlation.  

---

## 🔹 Installation & Usage

### Requirements
Install dependencies:
```bash
pip install -r requirements.txt
