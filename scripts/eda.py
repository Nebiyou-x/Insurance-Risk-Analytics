
import pandas as pd
import numpy as np


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

overall_loss_ratio = df['totalclaims'].sum() / df['totalpremium'].sum()
print("Overall loss ratio:", overall_loss_ratio)

# by province
loss_by_province = df.groupby('province').agg(
    total_premium=('totalpremium','sum'),
    total_claims=('totalclaims','sum'),
).assign(loss_ratio=lambda x: x.total_claims/x.total_premium,
         count=lambda x: df.groupby('province').size())
loss_by_province = loss_by_province.sort_values('loss_ratio', ascending=False)
display(loss_by_province.head(20))

# by vehicle type
loss_by_vehicletype = df.groupby('vehicletype').agg(
    total_premium=('totalpremium','sum'),
    total_claims=('totalclaims','sum')
).assign(loss_ratio=lambda x: x.total_claims/x.total_premium).sort_values('loss_ratio', ascending=False)
display(loss_by_vehicletype.head(20))

# by gender
loss_by_gender = df.groupby('gender').agg(total_premium=('totalpremium','sum'), total_claims=('totalclaims','sum')).assign(loss_ratio=lambda x: x.total_claims/x.total_premium)
display(loss_by_gender)


df['month'] = df['transactionmonth'].dt.to_period('M')
monthly = df.groupby('month').agg(total_premium=('totalpremium','sum'), total_claims=('totalclaims','sum'), policies=('policyid','nunique')).reset_index()
monthly['loss_ratio'] = monthly['total_claims'] / monthly['total_premium']
display(monthly)


def find_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5*iqr
    high = q3 + 1.5*iqr
    return series[(series < low) | (series > high)]

outlier_total_claims = find_outliers(df['totalclaims'])
print("Outlier count in TotalClaims:", len(outlier_total_claims))
