# 📦 Imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 📥 Load dataset
data = pd.read_csv("/model/Online Retail.csv")

data.dropna(subset=["CustomerID"],inplace=True)
# Remove canceled orders (those with InvoiceNo starting with "C")
data = data[~data.InvoiceNo.astype(str).str.startswith('C')]

# Convert InvoiceDate to datetime
data.InvoiceDate = pd.to_datetime(data.InvoiceDate)

# Create TotalPrice
data["TotalPrice"] = data.Quantity* data.UnitPrice

# Remove negative values (for refunds or errors)
data = data[data["TotalPrice"] > 0]
cutoff_date = pd.to_datetime("2011-06-01")
calibration_df = data[data['InvoiceDate'] < cutoff_date]
holdout_df = data[data['InvoiceDate'] >= cutoff_date]

snapshot_date = calibration_df['InvoiceDate'].max()

features = calibration_df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()

features.columns = ['CustomerID', 'recency_days', 'frequency', 'monetary']

# Optional: add avg order value
features["avg_order_value"] = pd.to_numeric(features['monetary']) / pd.to_numeric(features['frequency'])
# Check the type first (debug step)
print("features type:", type(features))

# Ensure it's a DataFrame
if not isinstance(features, pd.DataFrame):
    raise TypeError("❌ 'features' is not a DataFrame — check your previous steps or variable assignments.")
## manupulating the data and adding some new columns
# 1. Heuristic CLV Calculation
purchase_frequency = features['frequency'].sum() / features.shape[0]
repeat_rate = features[features['frequency'] > 1].shape[0] / features.shape[0]
churn_rate = 1 - repeat_rate
profit_margin = 0.10

features['CLV'] = features['avg_order_value'] * purchase_frequency * (1 / churn_rate) * profit_margin

# 2. First Purchase Date (Customer Tenure)
first_purchase = data.groupby('CustomerID')['InvoiceDate'].min().reset_index()
first_purchase.columns = ['CustomerID', 'first_purchase_date']
features = pd.merge(features, first_purchase, on='CustomerID', how='left')

max_date = data['InvoiceDate'].max()
features['customer_age_days'] = (max_date - features['first_purchase_date']).dt.days
features.drop(columns='first_purchase_date', inplace=True)

# 3. Avg Days Between Orders
data_sorted = data.sort_values(by=['CustomerID', 'InvoiceDate'])
data_sorted['order_diff'] = data_sorted.groupby('CustomerID')['InvoiceDate'].diff().dt.days

avg_days = data_sorted.groupby('CustomerID')['order_diff'].mean().reset_index()
avg_days.columns = ['CustomerID', 'avg_days_between_orders']
features = pd.merge(features, avg_days, on='CustomerID', how='left')

# 4. Actual Future CLV from holdout_df
clv_target = holdout_df.groupby('CustomerID')['TotalPrice'].sum().reset_index()
clv_target.columns = ['CustomerID', 'FutureCLV']

# 5. Merge with features
final_df = pd.merge(features, clv_target, on='CustomerID', how='inner')


#%%
# Step 1: Filter recent data (last 30 days)
recent_cutoff = data['InvoiceDate'].max() - pd.Timedelta(days=30)
recent_data = data[data['InvoiceDate'] >= recent_cutoff]

# Step 2: Compute recent frequency
recent_freq = recent_data.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
recent_freq.columns = ['CustomerID', 'recent_frequency']

# Step 3: Compute recent monetary value
recent_monetary = recent_data.groupby('CustomerID')['TotalPrice'].sum().reset_index()
recent_monetary.columns = ['CustomerID', 'recent_monetary']

# 🔍 Check before merge
print("recent_freq sample:\n", recent_freq.head())
print("recent_monetary sample:\n", recent_monetary.head())
print("features columns before merge:\n", features.columns)

# Step 4: Merge both recent metrics into features
features = pd.merge(features, recent_freq, on='CustomerID', how='left')
features = pd.merge(features, recent_monetary, on='CustomerID', how='left')

# 🔍 Check after merge
print("features columns after merge:\n", features.columns)

# Step 5: Handle NaN values (✅ no FutureWarning)
features['recent_frequency'] = features['recent_frequency'].fillna(0)
features['recent_monetary'] = features['recent_monetary'].fillna(0)
clv_target = holdout_df.groupby('CustomerID')['TotalPrice'].sum().reset_index()
clv_target.columns = ['CustomerID', 'FutureCLV']
X = features.drop(['CustomerID'], axis=1)

#%%
# Calculate avg_order_value if not already there
features['avg_order_value'] = features['monetary'] / features['frequency']

# Calculate order_per_day (with handling for zero division)
features['order_per_day'] = features.apply(
    lambda row: row['frequency'] / row['customer_age_days'] if row['customer_age_days'] > 0 else 0,
    axis=1
)

# Define the number of future days you want to project CLV for
future_days = 90  # You can change this to 180, 365, etc.

# Compute Future CLV
features['FutureCLV'] = features['avg_order_value'] * features['order_per_day'] * future_days
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- Features & Target ---
X = features[['recency_days', 'frequency', 'monetary', 'avg_order_value',
              'customer_age_days', 'order_per_day', 'recent_frequency', 'recent_monetary']]
y = features['FutureCLV']

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Models ---
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# --- Train and Evaluate ---
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n🔹 {name}")
    print("R2 Score:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
joblib.dump(models["Linear Regression"], "/home/big_fat_penguin/PycharmProjects/projects/Customer lifetime value prediction/model/best_clv_model.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")