import sys
import os
import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Fix path (important for imports)
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# -----------------------------
# Load pipeline
# -----------------------------
@st.cache_resource
def load_pipeline():
    with open(r"D:\Doccuments\GitHub\End-to-End-MLOps-Pipeline\streamlit_app\pipeline.pkl", "rb") as f:
        return pickle.load(f)

pipe = load_pipeline()

st.title("💳 Fraud Detection System")
st.write("Predict fraudulent transactions using ML model")

# =============================
# OPTION 1 - CSV UPLOAD
# =============================
st.header("Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("### Preview")
    st.dataframe(data.head())

    preds = pipe.predict(data)
    probs = pipe.predict_proba(data)[:, 1]

    data["prediction"] = preds
    data["fraud_probability"] = probs

    st.write("### Results")
    st.dataframe(data)

    st.write(f"Total Fraud Cases: {(preds == 1).sum()}")

# =============================
# OPTION 2 - SINGLE INPUT FORM
# =============================
st.write("---")
st.header("Single Transaction Prediction")

with st.form("single_input"):

    amt = st.number_input("Transaction Amount", min_value=0.0)
    city_pop = st.number_input("City Population", min_value=0)
    log_amt = st.number_input("Log Amount", value=0.0)
    amt_user_mean = st.number_input("User Avg Amount", value=0.0)
    amt_user_std = st.number_input("User Std Amount", value=0.0)
    amt_user_zscore = st.number_input("User Z-Score", value=0.0)
    user_txn_count = st.number_input("User Transaction Count", min_value=0)
    user_merchant_count = st.number_input("User Merchant Count", min_value=0)

    hour = st.number_input("Hour", min_value=0, max_value=23)
    day = st.number_input("Day", min_value=1, max_value=31)
    month = st.number_input("Month", min_value=1, max_value=12)
    weekday = st.number_input("Weekday", min_value=0, max_value=6)
    is_weekend = st.selectbox("Is Weekend", [0, 1])

    age = st.number_input("Age", min_value=0)
    time_diff = st.number_input("Time Difference", min_value=0)

    merchant_freq = st.number_input("Merchant Frequency", min_value=0)
    job_freq = st.number_input("Job Frequency", min_value=0)
    city_freq = st.number_input("City Frequency", min_value=0)

    gender_M = st.selectbox("Gender (Male=1, Female=0)", [0, 1])

    # simplified category/state examples (you can expand later)
    category_food_dining = st.selectbox("Food Dining", [0, 1])
    category_gas_transport = st.selectbox("Gas Transport", [0, 1])

    state_CA = st.selectbox("State CA", [0, 1])
    state_TX = st.selectbox("State TX", [0, 1])

    txn_last_1h = st.number_input("Transactions Last 1 Hour", min_value=0)

    submitted = st.form_submit_button("Predict Fraud")
    feature_cols = pipe.feature_names_in_

    if submitted:

        input_data = pd.DataFrame([[
    amt, city_pop, log_amt, amt_user_mean, amt_user_std,
    amt_user_zscore, user_txn_count, user_merchant_count,
    hour, day, month, weekday, is_weekend,
    age, time_diff, merchant_freq, job_freq, city_freq,
    gender_M,
    category_food_dining,
    category_gas_transport,
    state_CA,
    state_TX,
    txn_last_1h
]], columns=pipe.feature_names_in_)

        pred = pipe.predict(input_data)[0]
        prob = pipe.predict_proba(input_data)[0][1]

        st.write("### Result")

        if pred == 1:
            st.error(f"🚨 Fraud Detected (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Legit Transaction (Probability: {prob:.2f})")