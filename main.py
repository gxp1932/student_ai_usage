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
plt.figure(figsize = (12,4), dpi=120)

# Scatter - study hours vs age
sns.regplot(
    x="study_hours_per_day", y="improvement", data=df,
    scatter_kws={
        "color": "red",
        "alpha": 0.4,
        "s":35
    }
)
plt.show()

#Bar plot - Age vs Study hours per day
plt.figure(figsize=[14, 8])
sns.barplot(x="age", y="study_hours_per_day", data=df,)
plt.xlabel('Age')
plt.ylabel('Study hours per day')
plt.title('Age vs Study hours per day')
plt.xticks(rotation=0, ha='right')
plt.tight_layout()
plt.show()