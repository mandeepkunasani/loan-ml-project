# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print("STARTING PROGRAM...")

# ==============================
# 2. LOAD DATASET
# ==============================
url = "https://raw.githubusercontent.com/SR1608/Datasets/main/covid-data.csv"
df = pd.read_csv(url)

# ==============================
# 3. HIGH LEVEL DATA UNDERSTANDING
# ==============================

# a. rows & columns
print("Shape of dataset:", df.shape)

# b. data types
print("\nData Types:\n", df.dtypes)

# c. info
print("\nInfo:\n")
df.info()

# d. describe
print("\nDescription:\n", df.describe())

# ==============================
# 4. LOW LEVEL DATA UNDERSTANDING
# ==============================

# a. unique locations
print("\nUnique locations:", df['location'].nunique())

# b. continent max frequency
print("\nContinent frequency:\n", df['continent'].value_counts())
print("Max frequency continent:", df['continent'].value_counts().idxmax())

# c. max & mean total_cases
print("\nMax total cases:", df['total_cases'].max())
print("Mean total cases:", df['total_cases'].mean())

# d. quartiles total_deaths
print("\nQuartiles (total_deaths):\n",
      df['total_deaths'].quantile([0.25, 0.5, 0.75]))

# e. continent max HDI
print("\nContinent with max HDI:",
      df.groupby('continent')['human_development_index'].max().idxmax())

# f. continent min GDP
print("\nContinent with min GDP:",
      df.groupby('continent')['gdp_per_capita'].min().idxmin())

# ==============================
# 5. FILTER COLUMNS
# ==============================
df = df[['continent','location','date','total_cases',
         'total_deaths','gdp_per_capita','human_development_index']]

# ==============================
# 6. DATA CLEANING
# ==============================

# a. remove duplicates
df.drop_duplicates(inplace=True)

# b. missing values
print("\nMissing values:\n", df.isnull().sum())

# c. remove rows with missing continent
df.dropna(subset=['continent'], inplace=True)

# d. fill missing values
df.fillna(0, inplace=True)

# ==============================
# 7. DATE TIME FORMAT (FIXED)
# ==============================
df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)

# create month column
df['month'] = df['date'].dt.month

# ==============================
# 8. DATA AGGREGATION
# ==============================
df_groupby = df.groupby('continent').max().reset_index()

print("\nGrouped Data:\n", df_groupby.head())

# ==============================
# 9. FEATURE ENGINEERING
# ==============================
df_groupby['total_deaths_to_total_cases'] = (
    df_groupby['total_deaths'] / df_groupby['total_cases']
)

# ==============================
# 10. DATA VISUALIZATION
# ==============================

# a. histogram
plt.figure()
sns.histplot(df['gdp_per_capita'], kde=True)
plt.title("GDP per Capita Distribution")
plt.show()

# b. scatter plot
plt.figure()
sns.scatterplot(x='total_cases', y='gdp_per_capita', data=df)
plt.title("Total Cases vs GDP per Capita")
plt.show()

# c. pairplot (IMPORTANT but optimized)
sample_df = df_groupby.sample(min(5, len(df_groupby)))
sns.pairplot(sample_df)
plt.show()

# d. bar plot
sns.catplot(x='continent', y='total_cases', kind='bar', data=df_groupby)
plt.title("Total Cases by Continent")
plt.show()

# ==============================
# 11. SAVE DATA
# ==============================
df_groupby.to_csv("df_groupby.csv", index=False)

print("\n✅ FULL PROJECT COMPLETED SUCCESSFULLY!")