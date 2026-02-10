# =====================================
# Exploratory Data Analysis: Indian Rainfall
# Agriculture-Oriented
# =====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# -------------------------------------
# Load Dataset
# -------------------------------------
df = pd.read_csv("data/rainfall_india.csv")

# -------------------------------------
# Basic Inspection
# -------------------------------------
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# -------------------------------------
# Monthly Columns
# -------------------------------------
monthly_cols = [
    'JAN','FEB','MAR','APR','MAY','JUN',
    'JUL','AUG','SEP','OCT','NOV','DEC'
]

# -------------------------------------
# Long Format Conversion
# -------------------------------------
rain_long = df.melt(
    id_vars=['STATE','YEAR'],
    value_vars=monthly_cols,
    var_name='MONTH',
    value_name='RAINFALL'
)

# -------------------------------------
# Annual Rainfall Trend (India Average)
# -------------------------------------
annual_trend = df.groupby('YEAR')['ANNUAL'].mean()

plt.figure(figsize=(10,5))
plt.plot(annual_trend.index, annual_trend.values)
plt.xlabel("Year")
plt.ylabel("Rainfall (mm)")
plt.title("Average Annual Rainfall Trend in India")
plt.tight_layout()
plt.show()

# -------------------------------------
# Seasonal Rainfall (Kharif & Rabi)
# -------------------------------------
kharif_months = ['JUN','JUL','AUG','SEP']
rabi_months = ['OCT','NOV','DEC','JAN','FEB']

df['KHARIF_RAIN'] = df[kharif_months].sum(axis=1)
df['RABI_RAIN'] = df[rabi_months].sum(axis=1)

print(df[['KHARIF_RAIN','RABI_RAIN']].describe())

# -------------------------------------
# State-wise Average Rainfall
# -------------------------------------
state_avg = df.groupby('STATE')['ANNUAL'].mean().sort_values()

plt.figure(figsize=(8,10))
state_avg.plot(kind='barh')
plt.xlabel("Rainfall (mm)")
plt.ylabel("State")
plt.title("Average Annual Rainfall by State")
plt.tight_layout()
plt.show()

# -------------------------------------
# Drought / Normal / Excess Classification
# -------------------------------------
mean_rain = df['ANNUAL'].mean()
std_rain = df['ANNUAL'].std()

df['RAIN_CLASS'] = np.where(
    df['ANNUAL'] < mean_rain - std_rain, 'Drought',
    np.where(df['ANNUAL'] > mean_rain + std_rain, 'Excess', 'Normal')
)

print(df['RAIN_CLASS'].value_counts())

# -------------------------------------
# Monthly Rainfall Correlation
# -------------------------------------
plt.figure(figsize=(10,6))
sns.heatmap(
    df[monthly_cols].corr(),
    annot=True,
    cmap='coolwarm',
    linewidths=0.5
)
plt.title("Monthly Rainfall Correlation Matrix")
plt.tight_layout()
plt.show()

# -------------------------------------
# Year-wise Rainfall Variability
# -------------------------------------
year_std = df.groupby('YEAR')['ANNUAL'].std()

plt.figure(figsize=(10,5))
plt.plot(year_std.index, year_std.values)
plt.xlabel("Year")
plt.ylabel("Standard Deviation")
plt.title("Year-wise Rainfall Variability")
plt.tight_layout()
plt.show()

# -------------------------------------
# Save Processed Data
# -------------------------------------
df.to_csv("outputs/processed_rainfall_data.csv", index=False)
rain_long.to_csv("outputs/rainfall_long_format.csv", index=False)
