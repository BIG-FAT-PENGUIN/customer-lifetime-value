# Customer-lifetime-value Prediction--
## link to the project--([https://customer-lifetime-value-mqmdldbfep9gy4trk7pwk4.streamlit.app/])
           # ML-based CLV prediction with Streamlit dashboard


# Project overview--
This dashboard is part of a complete machine learning pipeline that predicts **Customer Lifetime Value (CLV)** using real retail transaction data.

    ### 🧾 Step 1: Data Source
    - Dataset: `Online Retail.csv`
    - Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online+retail)
    - Contains invoices, customers, product details, dates, quantity, unit price

    ### 🧪 Step 2: Preprocessing (Jupyter Notebook)
    - Removed:
      - Canceled orders (InvoiceNo starting with 'C')
      - Missing `CustomerID` values
      - Negative values (refunds or errors)
    - Parsed `InvoiceDate` into datetime
    - Calculated `TotalPrice = Quantity * UnitPrice`

    ### 🏗️ Step 3: Feature Engineering
    From the cleaned data, we created these features:
    - `recency_days` – Days since last purchase
    - `frequency` – Number of unique invoices
    - `monetary` – Total amount spent
    - `avg_order_value` – Monetary / frequency
    - `customer_age_days` – Customer lifespan
    - `order_per_day` – Frequency normalized by age
    - `recent_frequency`, `recent_monetary` – Past 30-day activity

    ### 🤖 Step 4: Model Training
    - Target: Predicted **CLV for next 90 days**
    - Models trained and evaluated:
        - Linear Regression ✅
        - Ridge Regression
        - Random Forest Regressor
    - Metric used: **R² Score**, **MSE**
    - Best Model: **Linear Regression** (saved using `joblib`)

    ### 📦 Step 5: Assets Saved for Deployment
    - `best_clv_model.pkl`: Trained model
    - `feature_columns.pkl`: Required features
    - `sample_clv_data.csv`: Sample feature dataset
    - `final_features_dataset.csv`: Full engineered dataset (optional)

    ### 🌐 Step 6: Streamlit Dashboard
    This app allows:
    - 📝 Manual Input prediction
    - 📁 CSV Upload for batch prediction
    - 🧪 Testing with sample data
    - 📊 Visualization of predicted CLV distributions and customer value
    - 🧠 All predictions are made using the **trained model**, not retrained live

    ---
    🧠 **Goal:** Help businesses understand which customers are likely to bring more value and guide marketing/retention strategies accordingly.
    """)

