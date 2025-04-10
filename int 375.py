import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and clean dataset
df = pd.read_csv("C:\\Users\\sraju\\OneDrive\\Documents\\new dataset for python.csv", low_memory=False)
df = df.dropna(subset=["Attrition"])
df = df.ffill()

# Display basic info
print("✅ Dataset Shape:", df.shape)
print("📋 Columns:", df.columns.tolist())
print("\n🧾 Dataset Info:")
print(df.info())

# Convert categorical values to numbers for plotting
df_viz = df.copy()
for col in df_viz.select_dtypes(include='object').columns:
    df_viz[col], _ = pd.factorize(df_viz[col])

# Refactor Attrition to binary labels: 0=No, 1=Yes
df_viz['Attrition'], _ = pd.factorize(df_viz['Attrition'])

# ------------------------ 🎯 OBJECTIVES ------------------------
print("\n🎯 Project Objectives:")
objectives = [
    "1. Predict employee attrition using key features.",
    "2. Identify patterns that lead to employee exits.",
    "3. Visualize job satisfaction, income, work-life balance.",
    "4. Analyze without machine learning – pure data insights.",
    "5. Estimate feature importance visually.",
    "6. Help HR develop effective retention strategies.",
    "7. Simulate real-world attrition scenarios.",
    "8. Understand risk by department, role, or age."
]
for obj in objectives:
    print(obj)

# 1. Attrition Count Bar Chart
plt.figure(figsize=(6, 4))
counts = df_viz['Attrition'].value_counts()
plt.bar(['No Attrition', 'Attrition'], counts, color=['skyblue', 'salmon'])
plt.title("1. Employee Attrition Count")
plt.ylabel("Number of Employees")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 2. Age Distribution Histogram by Attrition
plt.figure(figsize=(6, 4))
for val in [0, 1]:
    plt.hist(df_viz[df_viz['Attrition'] == val]['Age'], bins=20, alpha=0.6, label=f"Attrition = {val}")
plt.legend()
plt.title("2. Age Distribution by Attrition")
plt.xlabel("Age")
plt.ylabel("Number of Employees")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Boxplot: Monthly Income vs Attrition
plt.figure(figsize=(6, 4))
data = [df_viz[df_viz['Attrition'] == val]['MonthlyIncome'] for val in [0, 1]]
plt.boxplot(data, tick_labels=["No Attrition", "Attrition"])
plt.title("3. Monthly Income Comparison")
plt.ylabel("Monthly Income")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Bar Chart: Work-Life Balance by Attrition
avg_wlb = df_viz.groupby('Attrition')['WorkLifeBalance'].mean()
plt.figure(figsize=(6, 4))
plt.bar(['No Attrition', 'Attrition'], avg_wlb, color=['lightgreen', 'orange'])
plt.title("4. Work-Life Balance Average")
plt.ylabel("Work-Life Balance Score")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 5. Bar Chart: Job Satisfaction vs Attrition
plt.figure(figsize=(6, 4))
for level in sorted(df_viz['JobSatisfaction'].unique()):
    counts = df_viz[df_viz['JobSatisfaction'] == level]['Attrition'].value_counts(normalize=True)
    plt.bar(level - 0.2, counts.get(0, 0), width=0.4, color='lightblue', label='No Attrition' if level == 1 else "")
    plt.bar(level + 0.2, counts.get(1, 0), width=0.4, color='coral', label='Attrition' if level == 1 else "")
plt.title("5. Attrition by Job Satisfaction Level")
plt.xlabel("Satisfaction Level (1 = Low, 4 = High)")
plt.ylabel("Proportion")
plt.xticks([1, 2, 3, 4])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Scatter: Age vs Monthly Income
plt.figure(figsize=(7, 5))
scatter = plt.scatter(df_viz["Age"], df_viz["MonthlyIncome"], c=df_viz["Attrition"],
                      cmap="coolwarm", alpha=0.6)
plt.xlabel("Age")
plt.ylabel("Monthly Income")
plt.title("6. Age vs Income Colored by Attrition")
plt.colorbar(scatter, label="Attrition (0 = No, 1 = Yes)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Scatter: Years at Company vs Total Experience
plt.figure(figsize=(7, 5))
scatter = plt.scatter(df_viz["YearsAtCompany"], df_viz["TotalWorkingYears"], c=df_viz["Attrition"],
                      cmap="viridis", alpha=0.6)
plt.xlabel("Years at Company")
plt.ylabel("Total Working Years")
plt.title("7. Company Tenure vs Experience by Attrition")
plt.colorbar(scatter, label="Attrition (0 = No, 1 = Yes)")
plt.grid(True)
plt.tight_layout()
plt.show()
