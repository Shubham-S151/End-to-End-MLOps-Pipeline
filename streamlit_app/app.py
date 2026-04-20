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
    with open(os.path.join(BASE_DIR,"streamlit_app","pipeline.pkl"), "rb") as f:
        return pickle.load(f)

pipe = load_pipeline()

st.title("Credit Card Fraud Detection System")
st.write("Predict fraudulent transactions using ML model")

# OPTION 1 - CSV UPLOAD
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

# OPTION 2 - SINGLE INPUT FORM
# OPTION 2 - SINGLE INPUT FORM
st.write("---")
st.header("Single Transaction Prediction")

with st.form("single_input"):
    trans_date_trans_time = st.text_input("Transaction DateTime (YYYY-MM-DD HH:MM:SS)")
    cc_num = st.text_input("Credit Card Number")
    merchant = st.text_input("Merchant")
    category = st.text_input("Category")
    amt = st.number_input("Amount", min_value=0.0)
    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    gender = st.selectbox("Gender", ["M", "F"])
    street = st.text_input("Street")
    city = st.text_input("City")
    state = st.text_input("State")
    zip_code = st.text_input("ZIP Code")
    lat = st.number_input("Latitude")
    long = st.number_input("Longitude")
    city_pop = st.number_input("City Population", min_value=0)
    job = st.text_input("Job")
    dob = st.text_input("Date of Birth (YYYY-MM-DD)")
    trans_num = st.text_input("Transaction Number")
    unix_time = st.number_input("Unix Time", min_value=0)
    merch_lat = st.number_input("Merchant Latitude")
    merch_long = st.number_input("Merchant Longitude")

    submitted = st.form_submit_button("Predict Fraud")

    if submitted:
        input_df = pd.DataFrame([{
            'trans_date_trans_time': trans_date_trans_time,
            'cc_num': cc_num,
            'merchant': merchant,
            'category': category,
            'amt': amt,
            'first': first_name,
            'last': last_name,
            'gender': gender,
            'street': street,
            'city': city,
            'state': state,
            'zip': zip_code,
            'lat': lat,
            'long': long,
            'city_pop': city_pop,
            'job': job,
            'dob': dob,
            'trans_num': trans_num,
            'unix_time': unix_time,
            'merch_lat': merch_lat,
            'merch_long': merch_long
        }])

        pred = pipe.predict(input_df)[0]
        prob = pipe.predict_proba(input_df)[0][1]

        st.write("### Result")

        if pred == 1:
            st.error(f"Fraud Detected (Probability: {prob:.2f})")
        else:
            st.success(f"Legit Transaction (Probability: {prob:.2f})")