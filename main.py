import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
df = pd.read_csv("train.csv")

# Display first 5 rows
print("First 5 Rows")
print(df.head())

# Dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# Survival Count
plt.figure(figsize=(5,4))
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Passenger Class Count
plt.figure(figsize=(5,4))
sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class")
plt.show()

# Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(df['Age'].dropna(), bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.show()

# Fare Distribution
plt.figure(figsize=(6,4))
sns.histplot(df['Fare'], bins=20)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.show()

# -----------------------------
# Part B: Identify Attributes
# -----------------------------

nominal_attributes = ["Name", "Sex", "Ticket", "Cabin", "Embarked"]
numeric_attributes = ["PassengerId", "Pclass", "Age", "SibSp", "Parch", "Fare"]
binary_attributes = ["Survived"]

print("\nNominal Attributes:")
print(nominal_attributes)

print("\nNumeric Attributes:")
print(numeric_attributes)

print("\nBinary Attributes:")
print(binary_attributes)
