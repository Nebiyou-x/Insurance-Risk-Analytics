

## Overview

This project analyses historical car insurance data to optimize marketing strategy, identify low-risk segments, and predict optimal insurance premiums for new clients in South Africa.

The analysis includes:

* Exploratory Data Analysis (EDA) to understand risk and profitability patterns.
* Visualization of trends, distributions, and outliers.
* Hypothesis testing (A/B tests) for provinces, zip codes, and gender.
* Linear regression models to predict total claims per zipcode.
* Machine learning models to predict optimal premium values based on car, owner, and location features.

The goal is to provide actionable insights for AlphaCare Insurance Solutions to design tailored insurance products and improve profitability.

---

## 🔹 Project Structure

```
project/
│
├── data/
│   └── insurance.csv            # Historical insurance data
│
├── eda.py                       # Full exploratory data analysis script
├── visuals.py                   # Visualizations and trend analysis
├── ab_tests.py                   # A/B hypothesis tests
├── linear_models.py              # Linear regression per zipcode
├── ml_model.py                   # Machine learning model to predict premiums
├── predict.py                    # Example script to predict premiums for new data
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation (this file)
└── .gitignore                    # Ignored files and folders
```



## 🔹 Key Analyses & Scripts

### 1️⃣ EDA (`eda.py`)

* Overview of dataset shape, missing values, and data types
* Standardized column names
* Computed `loss_ratio` and `margin`
* Descriptive statistics for numeric features
* Aggregated loss ratios by **Province**, **Vehicle Type**, and **Gender**
* Monthly trends and outlier detection

### 2️⃣ Visualizations (`visuals.py`)

* Bar charts for loss ratio by province
* Distribution plots for total claims
* Monthly trends of claims and premiums

### 3️⃣ A/B Testing (`ab_tests.py`)

* Hypothesis testing for:

  * Risk differences across provinces
  * Risk differences between zip codes
  * Margin differences across zip codes
  * Risk differences between men and women
* Uses **t-tests** to accept/reject null hypotheses

### 4️⃣ Linear Regression (`linear_models.py`)

* Fits a separate **linear regression model per zipcode** to predict `TotalClaims`
* Outputs model summary and R² scores
* Helps identify high/low-risk zipcodes

### 5️⃣ Machine Learning Model (`ml_model.py`)

* Predicts **optimal premium** using Random Forest
* Input features: `SumInsured`, `CubicCapacity`, `Kilowatts`, `VehicleType`, `Gender`, `Province`
* Preprocessing: One-hot encoding of categorical features
* Saves model as `premium_model.joblib` for future predictions

### 6️⃣ Prediction (`predict.py`)

* Loads trained model
* Predicts premium for new hypothetical clients

---

## 🔹 Setup Instructions

1. **Clone Repository**

```bash
git clone https://github.com/Nebiyou-x/Insurance-Risk-Analytics.git
cd Insurance-Risk-Analytics
```

2. **Create virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run scripts**

```bash
python eda.py
python visuals.py
python ab_tests.py
python linear_models.py
python ml_model.py
python predict.py
```

---

## 🔹 Methodology

1. **Data Understanding**: Check missing values, data types, and distributions
2. **EDA**: Identify patterns, high-risk segments, and temporal trends
3. **Statistical Testing**: Use hypothesis testing to confirm differences in risk/margin across categories
4. **Predictive Modeling**:

   * Linear regression for zipcode-specific risk
   * Random Forest regression for premium optimization
5. **Insights & Recommendations**:

   * Highlight low-risk targets for premium reduction
   * Inform marketing and product design decisions

---

## 🔹 Key Metrics

* **Loss Ratio** = TotalClaims / TotalPremium
* **Margin** = TotalPremium − TotalClaims
* **R²** for linear regression per zipcode
* **Train/Test Score** for ML premium prediction

---

## 🔹 Dependencies

* Python ≥ 3.9
* pandas, numpy, matplotlib, seaborn
* scikit-learn, statsmodels, joblib

---

## 🔹 Version Control & Reproducibility

* **Git** used for code versioning
* **DVC** used for dataset versioning (data stored in `data/` and tracked via `.dvc` files)
* CI/CD can be implemented via **GitHub Actions** for automated testing

