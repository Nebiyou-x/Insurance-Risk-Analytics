# src/eda.py
import pandas as pd
import numpy as np

# path to CSV - adjust
df = pd.read_csv("data/insurance.csv", parse_dates=["TransactionMonth"], dayfirst=True, infer_datetime_format=True)

# quick overview
print(df.shape)
print(df.dtypes)
display(df.head())

# Standardize column names (lowercase, underscores)
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

# compute Loss Ratio
df['loss_ratio'] = df['totalclaims'] / df['totalpremium']
df['margin'] = df['totalpremium'] - df['totalclaims']

# missing values
missing = df.isnull().sum().sort_values(ascending=False)
print(missing[missing>0])

# for numeric features, summary stats
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(df[num_cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T)
