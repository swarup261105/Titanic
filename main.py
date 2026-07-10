import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Read the dataset
df = pd.read_csv("train.csv")

print("=" * 50)
print("PART A : EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 50)

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


# Dataset Information
print("\nDataset Information:")
df.info()

# Last 5 Rows
print("\nLast 5 Rows:")
print(df.tail())

# Duplicate Rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

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

# Correlation Heatmap
plt.figure(figsize=(8,6))

# Select only numeric columns
numeric_df = df.select_dtypes(include='number')

# Create heatmap
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# Part B: Identify Attributes
# -----------------------------
print("\n" + "=" * 50)
print("PART B : IDENTIFY ATTRIBUTES")
print("=" * 50)

nominal_attributes = ["Name", "Sex", "Ticket", "Cabin", "Embarked"]
numeric_attributes = ["PassengerId", "Pclass", "Age", "SibSp", "Parch", "Fare"]
binary_attributes = ["Survived"]

print("\nNominal Attributes:")
print(nominal_attributes)

print("\nNumeric Attributes:")
print(numeric_attributes)

print("\nBinary Attributes:")
print(binary_attributes)

# -----------------------------
# Part C: Nominal Attribute Dissimilarity
# -----------------------------
print("\n" + "=" * 50)
print("PART C : NOMINAL ATTRIBUTE DISSIMILARITY")
print("=" * 50)

# Nominal attributes to compare
nominal_cols = ["Sex", "Embarked"]

# Remove missing values
nominal_data = df[nominal_cols].dropna().reset_index(drop=True)

# Select first two passengers
p1 = nominal_data.iloc[0]
p2 = nominal_data.iloc[1]

print("\nPassenger 1")
print(p1)

print("\nPassenger 2")
print(p2)

# Calculate Simple Matching Dissimilarity
matches = 0
total = len(nominal_cols)

for col in nominal_cols:
    if p1[col] == p2[col]:
        print(f"{col}: Same")
        matches += 1
    else:
        print(f"{col}: Different")

dissimilarity = 1 - (matches / total)

print(f"\nSimple Matching Dissimilarity = {dissimilarity:.2f}")

# -----------------------------
# Part D: Numeric Attribute Dissimilarity
# -----------------------------

print("\n" + "=" * 50)
print("PART D : NUMERIC ATTRIBUTE DISSIMILARITY")
print("=" * 50)

# Numeric attributes to compare
numeric_cols = ["Age", "Fare"]

# Remove missing values
numeric_data = df[numeric_cols].dropna().reset_index(drop=True)

# Select first two passengers
p1 = numeric_data.iloc[0]
p2 = numeric_data.iloc[1]

print("\nPassenger 1")
print(p1)

print("\nPassenger 2")
print(p2)

# Calculate Euclidean Distance
distance = math.sqrt(
    (p1["Age"] - p2["Age"])**2 +
    (p1["Fare"] - p2["Fare"])**2
)

print(f"\nEuclidean Distance = {distance:.2f}")