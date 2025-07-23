import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Load model and features
# -------------------------
model = joblib.load("model/best_clv_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# -------------------------
# Streamlit Setup
# -------------------------
st.set_page_config(page_title="CLV Predictor", layout="wide")
st.title("🔮 Customer Lifetime Value Predictor")
st.markdown("Predict and visualize customer value using real transactional features.")

# -------------------------
# Session State
# -------------------------
if 'predicted_df' not in st.session_state:
    st.session_state.predicted_df = None
if 'last_prediction_hash' not in st.session_state:
    st.session_state.last_prediction_hash = None

# -------------------------
# Sidebar Selection
# -------------------------
mode = st.sidebar.radio("Choose Mode", ["Manual Input", "Upload CSV", "Use Sample Data"])

# -------------------------
# Manual Input Mode
# -------------------------
if mode == "Manual Input":
    st.sidebar.subheader("Enter Feature Values")
    input_data = {}

    for col in feature_columns:
        if "days" in col or "frequency" in col:
            input_data[col] = st.sidebar.number_input(col, min_value=0, value=10)
        else:
            input_data[col] = st.sidebar.number_input(col, min_value=0.0, value=100.0)

    if st.sidebar.button("🔮 Predict CLV"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.metric(label="Predicted CLV (90 Days)", value=f"£{prediction:,.2f}")
        st.session_state.predicted_df = input_df.assign(Predicted_CLV=prediction)
        st.session_state.last_prediction_hash = "manual"

# -------------------------
# CSV Upload Mode
# -------------------------
elif mode == "Upload CSV":
    st.header("📁 Upload CSV for CLV Prediction")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        user_df = pd.read_csv(uploaded_file)
        file_hash = hash(uploaded_file.getvalue())

        if all(col in user_df.columns for col in feature_columns):
            if st.button("🔮 Predict CLV"):
                user_df['Predicted_CLV'] = model.predict(user_df[feature_columns])
                st.session_state.predicted_df = user_df
                st.session_state.last_prediction_hash = file_hash
                st.success("✅ Predictions completed!")
                st.dataframe(user_df)

                csv = user_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Predictions CSV", data=csv, file_name="predicted_clv.csv", mime='text/csv')
        else:
            st.error("❌ Required feature columns missing!")

# -------------------------
# Sample Data Mode
# -------------------------
elif mode == "Use Sample Data":
    st.header("🧪 Predict Using Sample Data")
    try:
        sample_df = pd.read_csv("model/final_features_dataset.csv")
        if st.button("🔮 Predict Sample CLV"):
            sample_df['Predicted_CLV'] = model.predict(sample_df[feature_columns])
            st.session_state.predicted_df = sample_df
            st.session_state.last_prediction_hash = "sample"
            st.success("✅ Sample predictions complete!")
            st.dataframe(sample_df)

            csv = sample_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Sample Predictions", data=csv, file_name="sample_predictions.csv", mime='text/csv')
    except:
        st.error("⚠️ sample_clv_data.csv not found in /model folder.")

# -------------------------
# 📊 Visualization Section
# -------------------------
st.markdown("---")
if st.button("📊 Visualize Predicted CLV"):
    if st.session_state.predicted_df is not None and st.session_state.last_prediction_hash is not None:
        df = st.session_state.predicted_df

        # 📈 Histogram
        st.subheader("📈 Distribution of Predicted CLV")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['Predicted_CLV'], bins=30, kde=True, ax=ax1)
        st.pyplot(fig1)
        with st.expander("ℹ️ What this shows"):
            st.markdown("""
            - This histogram displays the **distribution of predicted customer lifetime value (CLV)**.
            - You can identify common value ranges and detect high-value outliers.
            """)

        # 🏆 Top 10 Customers
        if 'CustomerID' in df.columns:
            st.subheader("🏆 Top 10 Customers by CLV")
            top = df[['CustomerID', 'Predicted_CLV']].sort_values(by="Predicted_CLV", ascending=False).head(10)
            fig2, ax2 = plt.subplots()
            ax2.hlines(y=top['CustomerID'], xmin=0, xmax=top['Predicted_CLV'], color='skyblue')
            ax2.plot(top['Predicted_CLV'], top['CustomerID'], "o")
            ax2.set_xlabel("Predicted CLV")
            ax2.set_ylabel("Customer ID")
            st.pyplot(fig2)
            with st.expander("ℹ️ What this shows"):
                st.markdown("""
                - Shows the **most valuable customers** as per predicted CLV.
                - Helps you **target** the top 10 with retention offers or loyalty programs.
                """)
        else:
            st.info("CustomerID column not found — skipping Top 10 plot.")

        # 🎯 Scatter: CLV vs Recency
        if 'recency_days' in df.columns:
            st.subheader("🎯 CLV vs Recency")
            fig3, ax3 = plt.subplots()
            sns.scatterplot(data=df, x="recency_days", y="Predicted_CLV", hue="frequency", size="monetary", ax=ax3)
            ax3.set_title("CLV vs Recency (color=frequency, size=monetary)")
            st.pyplot(fig3)
            with st.expander("ℹ️ What this shows"):
                st.markdown("""
                - Visualizes how **recent activity affects predicted CLV**.
                - Larger bubbles = higher monetary spend, more orders = darker shade.
                - Look for clusters of **valuable recent customers**.
                """)
    else:
        st.warning("⚠️ Please make a prediction first before using visualization.")

# -------------------------
# General Info
# -------------------------
with st.expander("📘 What is CLV?"):
    st.markdown("""
    **Customer Lifetime Value (CLV)** estimates the total revenue a business can expect from a single customer over time.

    This app:
    - Uses features like **recency, frequency, monetary, order gaps, and customer age**
    - Predicts 90-day CLV using a trained machine learning model
    - Lets you upload data, test manually, and visualize insights
    """)
# -------------------------
# 📚 Project Overview Section
# -------------------------
with st.expander("📚 Project Workflow Overview"):
    st.markdown("""
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
