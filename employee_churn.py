# Import necessary libraries
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

print("Libraries imported successfully...")


# Load the data, inspect data

df = pd.read_csv('employees_c.csv')
print("First few rows of the data:")
print(df.head())
print("\nLast few rows of the data:")
print(df.tail())
print("\nData information:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())
print("\nShape of the DataFrame:")
print(df.shape)
print("\nMissing values")
print(df.isnull().sum())

print("Loaded data successfully...")


# Data Cleaning - Review missing values, incomplete data, duplicates
print("\nCategorical values and counts")
print(df['age'].value_counts())
print(df['gender'].value_counts())
print(df['department'].value_counts())
print(df['has_401k'].value_counts())

print("Reviewed successfully...")


# Exploratory Data Analysis (EDA) - Counts
df['gender'].value_counts().plot(kind='bar')
plt.title('Count of Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

df['state'].value_counts().head(10).plot(kind='bar')
plt.title('Count of Top States')
plt.xlabel('State')
plt.ylabel('Count')
plt.show()

df['department'].value_counts().plot(kind='bar')
plt.title('Count of Department')
plt.xlabel('Department')
plt.ylabel('Count')
plt.show()

df['has_401k'].value_counts().plot(kind='bar')
plt.title('Count of 401k')
plt.xlabel('Has_401k')
plt.ylabel('Count')
plt.show()

df['churned'].value_counts().plot(kind='bar')
plt.title('Count of Churned')
plt.xlabel('Churned')
plt.ylabel('Count')
plt.show()

print("Created counts successfully...")


# Exploratory Data Analysis (EDA) - Churn Rates 

# --- Churn Rate by Gender ---
churn_counts_gender = df.groupby('gender', observed=True)['churned'].sum()
total_counts_gender = df['gender'].value_counts()
churn_rate_gender = (churn_counts_gender / total_counts_gender) * 100

plt.figure(figsize=(8, 6))
churn_rate_gender.plot(kind='bar', color=['blue', 'orange'])
plt.xlabel('Gender')
plt.ylabel('Churn Rate (%)')
plt.title('Churn Rate by Gender')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

# --- Churn Rate by Age Group ---
bins_age = list(range(0, int(df['age'].max()) + 11, 10))
labels_age = [f"{i}-{i+10}" for i in bins_age[:-1]]
df['age_group'] = pd.cut(df['age'], bins=bins_age, right=False, labels=labels_age)
churn_counts_age = df.groupby('age_group', observed=True)['churned'].sum()
total_counts_age = df['age_group'].value_counts()
churn_rate_age = (churn_counts_age / total_counts_age) * 100

plt.figure(figsize=(10, 6))
churn_rate_age.plot(kind='bar', color='blue')
plt.xlabel('Age Group')
plt.ylabel('Churn Rate (%)')
plt.title('Churn Rate by Age Group')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

# --- Churn Rate by State ---
churn_counts_state = df.groupby('state', observed=True)['churned'].sum()
total_counts_state = df['state'].value_counts()
churn_rate_state = (churn_counts_state / total_counts_state) * 100
top_churn_states = churn_rate_state.nlargest(5)

plt.figure(figsize=(12, 6))
top_churn_states.plot(kind='bar', color='blue')
plt.xlabel('State')
plt.ylabel('Churn Rate (%)')
plt.title('Churn Rate by Top States')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

# --- Churn Rate by Department ---
churn_counts_dept = df.groupby('department', observed=True)['churned'].sum()
total_counts_dept = df['department'].value_counts()
churn_rate_dept = (churn_counts_dept / total_counts_dept) * 100
sorted_churn_rate_dept = churn_rate_dept.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sorted_churn_rate_dept.plot(kind='bar', color='blue')
plt.xlabel('Department')
plt.ylabel('Churn Rate (%)')
plt.title('Churn Rate by Department')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

# --- Churn Rate by Salary Group ---
bins_salary = list(range(0, int(df['salary'].max()) + 50001, 50000))
labels_salary = [f"{i}-{i+50000}" for i in bins_salary[:-1]]
df['salary_group'] = pd.cut(df['salary'], bins=bins_salary, right=False, labels=labels_salary)
churn_counts_salary = df.groupby('salary_group', observed=True)['churned'].sum()
total_counts_salary = df['salary_group'].value_counts()
churn_rate_salary = (churn_counts_salary / total_counts_salary) * 100

plt.figure(figsize=(12, 6))
churn_rate_salary.plot(kind='bar', color='blue')
plt.xlabel('Salary Group')
plt.ylabel('Churn Rate (%)')
plt.title('Churn Rate by Salary Group')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

# --- Churn Rate by 401k Participation ---
churn_counts_401k = df.groupby('has_401k', observed=True)['churned'].sum()
total_counts_401k = df['has_401k'].value_counts()
churn_rate_401k = (churn_counts_401k / total_counts_401k) * 100
sorted_churn_rate_401k = churn_rate_401k.sort_values(ascending=False)

plt.figure(figsize=(6, 6))
sorted_churn_rate_401k.plot(kind='bar', color=['blue', 'orange'])
plt.xlabel('Has 401k')
plt.ylabel('Churn Rate (%)')
plt.title('Churn Rate by 401k Participation')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

# --- Churn Rate by Tenure ---
# Define tenure bins
bins_tenure = list(range(0, int(df['tenure'].max()) + 1, 1))
labels_tenure = [f"{i}-{i+1}" for i in bins_tenure[:-1]]
if len(bins_tenure) > 1:  # Handle case where max tenure is 0
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins_tenure, right=False, labels=labels_tenure)
    churn_counts_tenure = df.groupby('tenure_group', observed=True)['churned'].sum()
    total_counts_tenure = df['tenure_group'].value_counts()
    churn_rate_tenure = (churn_counts_tenure / total_counts_tenure) * 100

    # Create the bar chart for Tenure vs. Churn Rate
    plt.figure(figsize=(12, 6))  # Adjust figsize for tenure bins
    churn_rate_tenure.plot(kind='bar', color='blue')
    plt.xlabel('Tenure (Years)')
    plt.ylabel('Churn Rate (%)')
    plt.title('Churn Rate by Tenure')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()
else:
    print("Cannot create tenure chart as there is no tenure data or the maximum tenure is 0.")


# --- Tenure vs. Age Scatter Plot ---
plt.figure(figsize=(10, 6))
plt.scatter(df['age'], df['tenure'], alpha=0.6, color=blue)
plt.xlabel('Age (Years)')
plt.ylabel('Tenure (Years)')
plt.title('Employee Tenure vs. Age')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


print("Created churn rates successfully...")
