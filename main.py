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


print(df["age"].value_counts())