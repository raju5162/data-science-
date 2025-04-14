data-science project
analysis of dataset
import pandas as p
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
//
# Load data
df = pd.read_csv("C:\\Users\\sraju\\OneDrive\\Documents\\50000 HRA Records.csv")
df.columns = df.columns.str.strip()

# Dataset Overview
print("📋 Dataset Overview:\n")
print(df.head())
print("\n🔢 Dataset Info:\n")
print(df.info())
print("\n📈 Statistical Summary:\n")
print(df.describe())
print("\n🔍 Missing Values:\n")
print(df.isnull().sum())

# Style Settings
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# 1. Attrition Rate
attrition_counts = df['Attrition'].value_counts()
attrition_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Attrition Rate")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# 2. Age Distribution
plt.hist(df['Age'], bins=20, color='lightgreen', edgecolor='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# 3. Attrition by Department
sns.countplot(x='Department', hue='Attrition', data=df, palette='pastel')
plt.title("Attrition by Department")
plt.xticks(rotation=45)
plt.show()

# 4. Job Satisfaction vs Attrition fixed
sns.boxplot(x='Attrition', y='JobSatisfaction', hue='Attrition', data=df, palette='Set2', legend=False)
plt.title("Job Satisfaction vs Attrition")
plt.show()

# 5. Years Since Last Promotion vs Attrition (fixed)
sns.violinplot(x='Attrition', y='YearsSinceLastPromotion', hue='Attrition', data=df, palette='muted', legend=False)
plt.title("Years Since Last Promotion vs Attrition")
plt.show()

# 6. Correlation Heatmap (Color Only)
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, cmap="coolwarm", cbar=True, square=True, linewidths=0.5)
plt.title("Correlation Heatmap (Color Only)")
plt.show()

# 7. Boxplots for Outlier Detection
outlier_cols = ['Age', 'MonthlyIncome', 'DistanceFromHome', 'YearsAtCompany']
for col in outlier_cols:
    plt.figure()
    sns.boxplot(data=df[col], color='lightcoral')
    plt.title(f"Boxplot for Outlier Detection: {col}")
    plt.show()

# 8. Scatter Plots
plt.figure()
sns.scatterplot(x='Age', y='MonthlyIncome', hue='Attrition', data=df, palette='cool')
plt.title("Age vs Monthly Income by Attrition")
plt.show()

plt.figure()
sns.scatterplot(x='TotalWorkingYears', y='YearsAtCompany', hue='Attrition', data=df, palette='coolwarm')
plt.title("Total Working Years vs Years at Company")
plt.show()

# 9. HourlyRate vs Department
plt.figure(figsize=(10, 6))
sns.boxplot(x='Department', y='HourlyRate', data=df, palette='pastel')
plt.title("HourlyRate by Department")
plt.xticks(rotation=45)
plt.show()
