import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium


st.set_page_config(layout="wide")

# ---------- LOAD DATA ----------
df = pd.read_csv("smart_mobility_dataset.csv")

# Convert timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Year"] = df["Timestamp"].dt.year
df["Month"] = df["Timestamp"].dt.month
df["Hour"] = df["Timestamp"].dt.hour

# ---------- SIDEBAR FILTER ----------
st.sidebar.header("🔍 Filters")

year_filter = st.sidebar.multiselect("Select Year", df["Year"].unique(), default=df["Year"].unique())
weather_filter = st.sidebar.multiselect("Weather", df["Weather_Condition"].unique(), default=df["Weather_Condition"].unique())
traffic_filter = st.sidebar.multiselect("Traffic Condition", df["Traffic_Condition"].unique(), default=df["Traffic_Condition"].unique())

df_filtered = df[
    (df["Year"].isin(year_filter)) &
    (df["Weather_Condition"].isin(weather_filter)) &
    (df["Traffic_Condition"].isin(traffic_filter))
]

# ---------- TITLE ----------
st.title("🚦 Smart Traffic Intelligence Dashboard")

# ---------- KPI ----------
col1, col2, col3, col4 = st.columns(4)

col1.metric("🚗 Total Vehicles", int(df_filtered["Vehicle_Count"].sum()))
col2.metric("⚡ Avg Energy", round(df_filtered["Energy_Consumption_L_h"].mean(), 2))
col3.metric("💨 Avg Emission", round(df_filtered["Emission_Levels_g_km"].mean(), 2))
col4.metric("🚨 Total Accidents", int(df_filtered["Accident_Report"].sum()))

# ---------- LEAFLET MAP ----------
st.subheader("📍 Traffic Map (Leaflet)")

# Create map
m = folium.Map(
    location=[df["latitude"].mean(), df["longitude"].mean()],
    zoom_start=10
)

# Sample data for performance
sample_df = df_filtered.sample(min(500, len(df_filtered)))

# Add markers
for _, row in sample_df.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=4,
        color="red" if row["Accident_Report"] == 1 else "green",
        fill=True,
        fill_opacity=0.7
    ).add_to(m)

# Display in Streamlit
st_folium(m, width=1200, height=400)

# ---------- ACCIDENT ANALYSIS ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📅 Accidents per Month")
    monthly = df_filtered.groupby("Month")["Accident_Report"].sum()
    st.bar_chart(monthly)

with col2:
    st.subheader("📆 Accidents per Year")
    yearly = df_filtered.groupby("Year")["Accident_Report"].sum()
    st.bar_chart(yearly)

# ---------- TIME ANALYSIS ----------
st.subheader("⏰ Accidents by Hour")
hourly = df_filtered.groupby("Hour")["Accident_Report"].sum()
st.line_chart(hourly)

# ---------- NUMERIC ANALYSIS ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚡ Energy Consumption Trend")
    st.line_chart(df_filtered["Energy_Consumption_L_h"][:20])

with col2:
    st.subheader("🚗 Speed Distribution")
    fig, ax = plt.subplots()
    ax.hist(df_filtered["Traffic_Speed_kmh"], bins=20)
    st.pyplot(fig)

# ---------- CATEGORICAL VISUALIZATION ----------
st.subheader("📊 Categorical Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("Traffic Condition")
    st.bar_chart(df_filtered["Traffic_Condition"].value_counts())

with col2:
    st.subheader("🌦 Traffic Condition vs Weather Condition")

    # Create cross-tab
    cross_tab = pd.crosstab(
        df_filtered["Weather_Condition"],
        df_filtered["Traffic_Condition"]
    )

    # Plot heatmap
    fig, ax = plt.subplots()
    sns.heatmap(cross_tab, annot=True, fmt="d", cmap="coolwarm", ax=ax)

    ax.set_xlabel("Traffic Condition")
    ax.set_ylabel("Weather Condition")

    st.pyplot(fig)

with col3:
    st.write("Traffic Light State")
    st.bar_chart(df_filtered["Traffic_Light_State"].value_counts())

# ---------- HEATMAP (PRO STYLE) ----------
#st.subheader("🔥 Correlation Heatmap")

# corr = df_filtered.select_dtypes(include="number").corr()

# fig, ax = plt.subplots()
# sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
# st.pyplot(fig)

# ---------- RAW DATA ----------
with st.expander("📄 Show Raw Data"):
    st.dataframe(df_filtered)
