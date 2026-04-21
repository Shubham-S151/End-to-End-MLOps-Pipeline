import os
import sys
import json
import zipfile
import pandas as pd
import streamlit as st

# D:\Doccuments\GitHub\End-to-End-MLOps-Pipeline\data\interim\churn_data
ROOT_DIR = os.path.abspath(os.path.dirname("End-to-End-MLOps-Pipeline"))
sys.path.append(ROOT_DIR)

# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..."))
# BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(BASE_DIR)

print(ROOT_DIR)

zip_data_path = os.path.join(ROOT_DIR,"data","raw_for_webuse","Data.zip")

with zipfile.ZipFile(zip_data_path, 'r') as z:
    csv_files = [f for f in z.namelist() if f.endswith('.csv')]
    dfs = {f: pd.read_csv(z.open(f)) for f in csv_files}

# Access your dataframes
data = dfs.get("fraud test.csv").head()
data = data[data.columns[1:]]

# Streamlit Dashboard Making
st.title("Credit Card Fraud Analysis Dashboard")
st.write("#### A comprehensive study of patterns of frauds in data")
st.write("---")
st.write("### Data Preview")
st.dataframe(data.head())

# 