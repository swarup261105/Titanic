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

# -----------------------------
# Part C: Nominal Attribute Dissimilarity
# -----------------------------

print("\n==============================")
print("Part C: Nominal Attribute Dissimilarity")
print("==============================")

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