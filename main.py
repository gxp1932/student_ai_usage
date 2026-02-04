##Student AI Usage Data Analysis

#Importing libraries
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import numpy as np
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Importing Dataset
url = 'Dataset.csv'
names = ['age', 'education_level', 'study_hours_per_day','uses_ai','ai_tools_used','purpose_of_ai','grades_before_ai', 'grades_after_ai', 'daily_screen_time_hours']
df = pd.read_csv(url, names=names, encoding='latin-1', skiprows=1)

#Visualize size, head and tail of data
print(f"\nShape: {df.shape}")
print("\nDataset head:")
print(df.head())
print("\nDataset tail:")
print(df.tail())

##Data cleaning & normalization

#Normalize Columns
df.columns = df.columns.str.lower().str.replace(' ', '_')

#Noralize numeric columns
for col in ["age", "study_hours_per_day", "grades_before_ai", "grades_after_ai", "daily_screen_time_hours"]:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)
        )
        df[col]=pd.to_numeric(df[col], errors='coerce')


#print(df["purpose_of_ai"].value_counts())

##Feature Engineering

#Grade Improvement with AI use (percent)
df["improvement"] = (((df["grades_after_ai"] - df["grades_before_ai"])/df["grades_before_ai"])*100)
#print(df.head())

##Correlation Analysis
correlation_matrix = df.corr(numeric_only=True)
plt.figure(figsize = (10,10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap", fontsize=16, fontweight='bold', pad=14)
plt.tight_layout()
plt.show()

## Visualizations

# Scatter - ai vs no ai
yes_uses_ai_df = df[df["uses_ai"]=="Yes"]
no_uses_ai_df = df[df["uses_ai"]=="No"]

plt.figure(figsize = (12,4), dpi=120)
plt.scatter(yes_uses_ai_df["study_hours_per_day"], yes_uses_ai_df["improvement"], color="blue")
plt.scatter(no_uses_ai_df["study_hours_per_day"], no_uses_ai_df["improvement"], color="red")
plt.xlabel("Study Hours Per Day")
plt.ylabel("Improvement")
plt.title("Uses AI (Blue) vs No AI (Red)")
plt.show()

# Bar Plot - Average improvement per AI tool
avg_improvement_tool = df.groupby("ai_tools_used")["improvement"].mean()

plt.figure(figsize=(10,5), dpi=120)
avg_improvement_tool.plot(kind="bar", color=["blue","red","green"])

plt.xlabel("AI Tool Used")
plt.ylabel("Average Grade Improvement")
plt.title("Average Grade Improvement by AI Tool")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

# Bar Plot - Average improvement per Purpose
avg_improvement_purpose = df.groupby("purpose_of_ai")["improvement"].mean()

plt.figure(figsize=(10,5), dpi=120)
avg_improvement_purpose.plot(kind="bar", color=["blue","red","green"])

plt.xlabel("Purpose of AI Tool")
plt.ylabel("Average Grade Improvement")
plt.title("Average Grade Improvement per Purpose")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

# Conclusion
print("\nConclusion:")
print("\n - Students who use AI tools to study see a grade improvement compared to students who do not use AI to study.")
print("\n - Students who used Gemini had a higher grade improvement compared to Copilot or ChatGPT.")
print("\n - Students who used AI for research purposes have a slightly higher grade improvement compared to students who use AI for Coding or Homework.")