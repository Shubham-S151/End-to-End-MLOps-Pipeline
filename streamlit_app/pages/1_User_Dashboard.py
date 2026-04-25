# It is still slow and lagging needs *Improovement
import os
import zipfile
import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import calendar


# =========================================================
# PAGE CONFIG (FAST UI LOAD)
# =========================================================
st.set_page_config(
    page_title="Fraud EDA Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD DATA (CACHED)
# =========================================================
ROOT_DIR = os.path.abspath(os.path.dirname("End-to-End-MLOps-Pipeline"))
zip_data_path = os.path.join(ROOT_DIR, "data", "raw_for_webuse", "Data.zip")

@st.cache_data
def load_data(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        dfs = {f: pd.read_csv(z.open(f)) for f in csv_files}
    return dfs

dfs = load_data(zip_data_path)
data = dfs.get("fraud test.csv")

if data is None:
    st.error("Dataset not found")
    st.stop()

data = data[data.columns[1:]]

# datetime conversion (safe)
if "trans_date_trans_time" in data.columns:
    data["trans_date_trans_time"] = pd.to_datetime(
        data["trans_date_trans_time"], errors="coerce"
    )

# =========================================================
# CACHE DERIVED COMPUTATIONS (IMPORTANT SPEED BOOST)
# =========================================================
@st.cache_data
def get_monthly_fraud(df):
    return df.groupby(df["trans_date_trans_time"].dt.month)["is_fraud"].sum()

@st.cache_data
def get_category_split(df):
    return df.groupby(["is_fraud", "category"]).size().unstack().fillna(0)

@st.cache_data
def get_merchant_stats(df):
    return df.groupby("merchant").agg(
        total_txn=("is_fraud", "count"),
        fraud_count=("is_fraud", "sum"),
        fraud_rate=("is_fraud", "mean")
    ).reset_index()

# =========================================================
# TITLE
# =========================================================
st.title("💳 Credit Card Fraud EDA Dashboard")
st.write("Optimized exploratory analysis of fraud transactions")
st.write("---")

# =========================================================
# SIDEBAR FILTERS (OPTIMIZED)
# =========================================================
st.sidebar.header("Filters")

def apply_filters(df):
    filtered = df.copy()

    if "category" in df.columns:
        cats = df["category"].dropna().unique()
        selected_cat = st.sidebar.multiselect("Category", cats, default=cats)
        filtered = filtered[filtered["category"].isin(selected_cat)]

    if "amt" in df.columns:
        min_amt, max_amt = float(df["amt"].min()), float(df["amt"].max())
        amt_range = st.sidebar.slider("Amount Range", min_amt, max_amt, (min_amt, max_amt))
        filtered = filtered[(filtered["amt"] >= amt_range[0]) & (filtered["amt"] <= amt_range[1])]

    if "is_fraud" in df.columns:
        fraud_filter = st.sidebar.radio("Fraud Filter", ["All", "Fraud Only", "Non-Fraud Only"])
        if fraud_filter == "Fraud Only":
            filtered = filtered[filtered["is_fraud"] == 1]
        elif fraud_filter == "Non-Fraud Only":
            filtered = filtered[filtered["is_fraud"] == 0]

    return filtered

data = apply_filters(data)

# =========================================================
# KPI METRICS
# =========================================================
st.write("## Key Metrics")

c1, c2, c3 = st.columns(3)

c1.metric("Total Transactions", len(data))
c2.metric("Fraud Cases", int(data["is_fraud"].sum()))
c3.metric("Fraud Rate", round(data["is_fraud"].mean(), 5))

st.write("---")

# =========================================================
# UNIVARIATE ANALYSIS
# =========================================================
st.write("## Univariate Analysis")

fig = px.histogram(data, x="is_fraud", color="is_fraud")
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown("Severe class imbalance is present in the dataset.")

if "amt" in data.columns:
    fig = px.histogram(data, x="amt", nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Transaction amounts are highly skewed with extreme outliers.")

# =========================================================
# PIE CHART: GLOBAL FRAUD DISTRIBUTION
# =========================================================
st.write("## Fraud Distribution")

fraud_dist = data["is_fraud"].value_counts()

fig = px.pie(
    names=["Non-Fraud", "Fraud"],
    values=fraud_dist.values,
    hole=0.5
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("Fraud cases form a very small portion of total transactions.")

# =========================================================
# PIE CHART: FRAUD BY CATEGORY
# =========================================================
st.write("## Fraud by Category")

if "category" in data.columns:

    fraud_category = data[data["is_fraud"] == 1]["category"].value_counts()

    fig = px.pie(
        names=fraud_category.index,
        values=fraud_category.values,
        hole=0.5
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Fraud is concentrated in specific categories.")

# =========================================================
# CATEGORY SPLIT (FAST + INTERACTIVE)
# =========================================================
st.write("## Category-wise Fraud Split (Interactive)")

if "category" in data.columns:

    cat_split = data.groupby(["is_fraud", "category"]).size().unstack().fillna(0)

    # ---------------- NON-FRAUD PIE ----------------
    fig_non_fraud = px.pie(
        names=cat_split.columns,
        values=cat_split.loc[0],
        hole=0.5,
        title="Non-Fraud Transactions by Category"
    )

    # ---------------- FRAUD PIE ----------------
    fig_fraud = px.pie(
        names=cat_split.columns,
        values=cat_split.loc[1],
        hole=0.5,
        title="Fraud Transactions by Category"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_non_fraud, use_container_width=True)

    with col2:
        st.plotly_chart(fig_fraud, use_container_width=True)

    st.markdown("""
    ### Insight:
    - Non-fraud transactions are widely distributed across categories.
    - Fraud transactions are concentrated in specific categories.
    - This highlights category-level fraud risk concentration.
    """)

# =========================================================
# BIVARIATE ANALYSIS
# =========================================================
st.write("## Bivariate Analysis")

if "amt" in data.columns:
    fig = px.box(data, x="is_fraud", y="amt", color="is_fraud")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("No clear separation between fraud and non-fraud transaction amounts.")

if "city_pop" in data.columns:
    fig = px.scatter(data, x="amt", y="city_pop", color="is_fraud")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("No strong relationship between city population and fraud.")

# =========================================================
# CORRELATION HEATMAP (STATIC, FAST)
# =========================================================
st.write("## Correlation Heatmap")

num_cols = data.select_dtypes(include=["int64", "float64"]).columns

fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(data[num_cols].corr(), cmap="RdBu", vmin=-1, annot=True, ax=ax)

st.pyplot(fig)

st.markdown("Weak correlations indicate non-linear fraud patterns.")

# =========================================================
# CATEGORY ANALYSIS
# =========================================================
st.write("## Category Risk Analysis")

if "category" in data.columns:

    fraud_rate = data.groupby("category")["is_fraud"].mean().sort_values(ascending=False)

    fig = px.bar(
        x=fraud_rate.head(10).index,
        y=fraud_rate.head(10).values,
        color=fraud_rate.head(10).values
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Certain categories are significantly more fraud-prone.")

# =========================================================
# MERCHANT ANALYSIS (OPTIMIZED)
# =========================================================
st.write("## Merchant Risk Analysis")

if "merchant" in data.columns:

    merchant_stats = get_merchant_stats(data)

    top_risky = merchant_stats.sort_values("fraud_rate", ascending=False).head(10)

    fig = px.bar(
        top_risky,
        x="merchant",
        y="fraud_rate",
        color="fraud_rate"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Fraud rate is a better indicator than raw counts.")

# =========================================================
# TIME SERIES (OPTIMIZED)
# =========================================================
st.write("## Time Series Analysis")

if "trans_date_trans_time" in data.columns:

    monthly = get_monthly_fraud(data)

    monthly.index = [calendar.month_abbr[m] for m in monthly.index]

    fig = px.line(x=monthly.index, y=monthly.values, markers=True)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Fraud shows seasonal variation (limited dataset window).")

# =========================================================
# FINAL INSIGHTS
# =========================================================
st.write("## Final Insights")

st.markdown("""
- Strong class imbalance exists
- Transaction amounts are heavily skewed
- Weak linear relationships among features
- Fraud depends on category and merchant behavior
- Temporal patterns exist but are limited
- Problem requires non-linear ML models
""")