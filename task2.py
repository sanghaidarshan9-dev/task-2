# ==============================
# Titanic Dataset: Data Cleaning & EDA
# ==============================

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")

# 2. Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

print("First 5 rows:")
print(df.head())

# 3. Basic understanding
print("\nDataset Shape:", df.shape)
print("\nDataset Info:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# 4. Data Cleaning

# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column due to too many missing values
df.drop(columns=['Cabin'], inplace=True)

# Remove duplicate rows if any
df.drop_duplicates(inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# 5. Feature Engineering

# Create FamilySize feature
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Convert categorical variables to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# 6. Exploratory Data Analysis (EDA)

# Survival count
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Survival by gender
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender (0 = Male, 1 = Female)")
plt.show()

# Survival by passenger class
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")
plt.show()

# Age distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Survival vs Age
plt.figure(figsize=(6,4))
sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Survival vs Age")
plt.show()

# Family size vs survival
plt.figure(figsize=(8,5))
sns.barplot(x='FamilySize', y='Survived', data=df)
plt.title("Family Size vs Survival Rate")
plt.show()

# Fare vs survival
plt.figure(figsize=(8,5))
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title("Survival vs Fare")
plt.show()

# 7. Correlation Analysis
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 8. Final cleaned dataset preview
print("\nCleaned Dataset Preview:")
print(df.head())

print("\nEDA and Data Cleaning Completed Successfully.")
