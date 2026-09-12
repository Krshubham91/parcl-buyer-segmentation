import os
import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="Parcl | Buyer Segmentation & Market Intelligence",
    page_icon="🏢",
    layout="wide"
)

# -------------------------------------------------------------
# Data Loading
# -------------------------------------------------------------
@st.cache_data
def load_data():
    file_path = os.path.join("data", "processed", "segmented_buyer_profiles.csv")
    if not os.path.exists(file_path):
        st.error(f"File '{file_path}' nahi mili! Kripya pehle Phase 2 run karein.")
        st.stop()
    data = pd.read_csv(file_path)
    return data

df = load_data()

# -------------------------------------------------------------
# Sidebar: User Controls & Filters
# -------------------------------------------------------------
st.sidebar.title("🔍 Buyer Filters")
st.sidebar.markdown("Filter property buyers across segments:")

# 1. Country Filter
all_countries = ["All"] + sorted(df["country"].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Select Country", all_countries)

# 2. Region Filter (Dynamically filtered by country)
if selected_country != "All":
    available_regions = ["All"] + sorted(df[df["country"] == selected_country]["region"].dropna().unique().tolist())
else:
    available_regions = ["All"] + sorted(df["region"].dropna().unique().tolist())
selected_region = st.sidebar.selectbox("Select Region", available_regions)

# 3. Acquisition Purpose Filter
all_purposes = ["All"] + sorted(df["acquisition_purpose"].dropna().unique().tolist())
selected_purpose = st.sidebar.selectbox("Acquisition Purpose", all_purposes)

# 4. Client Type Filter
all_types = ["All"] + sorted(df["client_type"].dropna().unique().tolist())
selected_type = st.sidebar.selectbox("Client Type", all_types)

# Apply Filters
filtered_df = df.copy()

if selected_country != "All":
    filtered_df = filtered_df[filtered_df["country"] == selected_country]

if selected_region != "All":
    filtered_df = filtered_df[filtered_df["region"] == selected_region]

if selected_purpose != "All":
    filtered_df = filtered_df[filtered_df["acquisition_purpose"] == selected_purpose]

if selected_type != "All":
    filtered_df = filtered_df[filtered_df["client_type"] == selected_type]

# -------------------------------------------------------------
# Main Dashboard Header
# -------------------------------------------------------------
st.title("🏢 Parcl Real Estate Buyer Intelligence")
st.markdown("AI-driven Buyer Segmentation & Investment Profiling Dashboard")
st.markdown("---")

# KPI Summary Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Buyers", f"{len(filtered_df):,}")
col2.metric("Total Capital Invested", f"${filtered_df['total_invested'].sum():,.0f}")
col3.metric("Avg Portfolio Size", f"{filtered_df['properties_bought'].mean():.1f} Units")
col4.metric("Avg Satisfaction", f"{filtered_df['satisfaction_score'].mean():.2f} / 5.0")

st.markdown("---")

# -------------------------------------------------------------
# Section 1: Segmentation Overview & Investor Behavior
# -------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Segmentation Overview", 
    "💰 Investment Behavior", 
    "🌍 Geographic Analysis", 
    "🔍 Segment Profiler & Client Lookup"
])

with tab1:
    st.subheader("Buyer Segment Distribution")
    col_chart1, col_chart2 = st.columns([1.2, 1])

    with col_chart1:
        segment_counts = filtered_df["segment_name"].value_counts().reset_index()
        segment_counts.columns = ["Segment", "Buyers Count"]
        fig_pie = px.pie(
            segment_counts, 
            names="Segment", 
            values="Buyers Count", 
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="Proportion of Identified Clusters"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        fig_bar = px.bar(
            segment_counts,
            x="Segment",
            y="Buyers Count",
            color="Segment",
            text="Buyers Count",
            title="Buyer Volume per Segment"
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.subheader("Investment Patterns & Financial Profile")
    col_inv1, col_inv2 = st.columns(2)

    with col_inv1:
        fig_box = px.box(
            filtered_df,
            x="segment_name",
            y="total_invested",
            color="segment_name",
            title="Capital Invested by Segment ($)",
            labels={"segment_name": "Segment", "total_invested": "Total Invested ($)"}
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    with col_inv2:
        loan_dist = filtered_df.groupby(["segment_name", "loan_applied"]).size().reset_index(name="count")
        fig_loan = px.bar(
            loan_dist,
            x="segment_name",
            y="count",
            color="loan_applied",
            barmode="group",
            title="Financing & Mortgage Usage by Segment",
            labels={"segment_name": "Segment", "count": "Buyers Count", "loan_applied": "Loan Applied"}
        )
        st.plotly_chart(fig_loan, use_container_width=True)

    # Referral channel breakdown
    fig_channel = px.histogram(
        filtered_df,
        x="referral_channel",
        color="segment_name",
        barmode="stack",
        title="Customer Acquisition Source (Referral Channel) by Segment"
    )
    st.plotly_chart(fig_channel, use_container_width=True)

with tab3:
    st.subheader("Geographic Intelligence")
    col_geo1, col_geo2 = st.columns([1, 1])

    with col_geo1:
        country_agg = filtered_df["country"].value_counts().head(10).reset_index()
        country_agg.columns = ["Country", "Buyers"]
        fig_country = px.bar(
            country_agg,
            x="Buyers",
            y="Country",
            orientation="h",
            title="Top Buyer Origin Countries",
            color="Buyers",
            color_continuous_scale="Blues"
        )
        fig_country.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_country, use_container_width=True)

    with col_geo2:
        region_agg = filtered_df["region"].value_counts().head(10).reset_index()
        region_agg.columns = ["Region", "Buyers"]
        fig_region = px.bar(
            region_agg,
            x="Region",
            y="Buyers",
            title="Top Active Regions",
            color="Buyers",
            color_continuous_scale="Teal"
        )
        st.plotly_chart(fig_region, use_container_width=True)

with tab4:
    st.subheader("Descriptive Profile per Segment")
    profile_summary = filtered_df.groupby("segment_name").agg(
        Buyers_Count=("client_id", "count"),
        Avg_Age=("age", "mean"),
        Avg_Investment=("total_invested", "mean"),
        Avg_Units=("properties_bought", "mean"),
        Avg_Unit_Price=("avg_unit_price", "mean"),
        Avg_Satisfaction=("satisfaction_score", "mean")
    ).reset_index()

    st.dataframe(profile_summary.style.format({
        "Avg_Age": "{:.1f} yrs",
        "Avg_Investment": "${:,.0f}",
        "Avg_Units": "{:.1f}",
        "Avg_Unit_Price": "${:,.0f}",
        "Avg_Satisfaction": "{:.2f} / 5"
    }), use_container_width=True)

    st.markdown("---")
    st.subheader("🔎 Individual Client Look-Up")
    selected_client_id = st.selectbox("Search Client by ID", filtered_df["client_id"].unique())
    client_row = filtered_df[filtered_df["client_id"] == selected_client_id]
    
    st.table(client_row[[
        "client_id", "client_type", "country", "region", "age",
        "segment_name", "acquisition_purpose", "loan_applied",
        "properties_bought", "total_invested", "satisfaction_score"
    ]])