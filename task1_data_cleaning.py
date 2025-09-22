import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

sns.set_style("ticks")
os.makedirs("images", exist_ok=True)

data = pd.read_csv("titanic.csv")
print(data.info())
print(data.isnull().sum())

fig, axes = plt.subplots(1, 2, figsize=(14,6))
sns.histplot(data['Age'], bins=30, kde=True, color="lightblue", ax=axes[0])
axes[0].set_title("Age Distribution (Raw Data)")
axes[0].set_xlabel("Age")
sns.boxplot(x=data['Fare'], color="lightcoral", ax=axes[1])
axes[1].set_title("Fare Boxplot (Raw Data)")
axes[1].set_xlabel("Fare")
plt.tight_layout()
plt.savefig("images/raw_data_visuals.png")
plt.close()

data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop(columns=['Cabin'], inplace=True)

encoder = LabelEncoder()
for col in ['Sex', 'Embarked']:
    data[col] = encoder.fit_transform(data[col])

scaler = StandardScaler()
data[['Age','Fare']] = scaler.fit_transform(data[['Age','Fare']])

plt.figure(figsize=(10,5))
sns.boxplot(x=data['Fare'], color="lightgreen")
plt.title("Fare Distribution (Standardized, Before Outlier Removal)")
plt.xlabel("Fare (Standardized)")
plt.savefig("images/fare_before_outlier_removal.png")
plt.close()

fare_q1 = data['Fare'].quantile(0.25)
fare_q3 = data['Fare'].quantile(0.75)
fare_iqr = fare_q3 - fare_q1
data = data[(data['Fare'] >= fare_q1 - 1.5*fare_iqr) & (data['Fare'] <= fare_q3 + 1.5*fare_iqr)]

fig, axes = plt.subplots(1, 2, figsize=(14,6))
sns.histplot(data['Age'], bins=30, kde=True, color="gold", ax=axes[0])
axes[0].set_title("Age Distribution (Cleaned & Scaled)")
axes[0].set_xlabel("Age (Standardized)")
sns.boxplot(x=data['Fare'], color="mediumpurple", ax=axes[1])
axes[1].set_title("Fare Distribution (After Outlier Removal)")
axes[1].set_xlabel("Fare (Standardized)")
plt.tight_layout()
plt.savefig("images/cleaned_data_visuals.png")
plt.close()

plt.figure(figsize=(8,5))
sns.countplot(x="Sex", hue="Survived", data=data, palette="coolwarm")
plt.title("Survival by Sex")
plt.xlabel("Sex (0=Female, 1=Male)")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.savefig("images/survival_by_sex_cleaned.png")
plt.close()

print(data.info())
print(data.head())