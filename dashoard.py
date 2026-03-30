import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import folium
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
st.set_page_config(layout="wide")

# Load data
df = pd.read_csv("smart_mobility_dataset.csv")
# Convert timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Year"] = df["Timestamp"].dt.year
df["Month"] = df["Timestamp"].dt.month
df["Hour"] = df["Timestamp"].dt.hour
# ---------- TITLE ----------
st.title("🚦 Smart Traffic Dashboard")

# ---------- KPI ROW ----------
col1, col2, col3, col4 = st.columns(4)

col1.metric("🚗 Total Vehicles", int(df["Vehicle_Count"].sum()))
col2.metric("⚡️ Avg Energy (L/h)", round(df["Energy_Consumption_L_h"].mean(), 2))
col3.metric("💨 Avg Emission", round(df["Emission_Levels_g_km"].mean(), 2))
current_accidents = df["Accident_Report"].sum()
previous_accidents = df[df["Year"] == df["Year"].min()]["Accident_Report"].sum()

delta = current_accidents - previous_accidents

col4.metric("🚨 Accidents", int(current_accidents), delta=int(delta))


# ---------- MAP + TRAFFIC ----------
col1, col2,col3 = st.columns(3)

with col1:
    st.subheader("📍 Traffic Locations")
    st.map(df[["latitude", "longitude"]])

with col2:
    st.subheader("🚦 Traffic Condition Distribution")
    #st.bar_chart(df["Traffic_Condition"].value_counts())
    fig = px.bar(
        df,
        x="Traffic_Condition",
        color="Traffic_Condition",
        title="Traffic Condition Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

with col3:
    st.subheader("📍 Cluster Traffic Map")

    m = folium.Map(
        location=[df["latitude"].mean(), df["longitude"].mean()],
        zoom_start=10
    )

    cluster = MarkerCluster().add_to(m)

    sample_df = df.sample(min(500, len(df)))  # ✅ FIXED

    for _, row in sample_df.iterrows():
        
        popup = f"""
        🚗 {row['Vehicle_Count']} vehicles<br>
        ⚡ {row['Traffic_Speed_kmh']} km/h<br>
        🌦 {row['Weather_Condition']}<br>
        🚨 Accident: {row['Accident_Report']}
        """

        folium.Marker(
            location=[row["latitude"], row["longitude"]],  # ✅ FIXED
            popup=popup
        ).add_to(cluster)

    st_folium(m, width=1200, height=400)


# ---------- ACCIDENTS BY TIME ----------
st.subheader("⏰ Accidents by Hour")

accidents_by_hour = df.groupby("Hour")["Accident_Report"].sum()

st.line_chart(accidents_by_hour)

# ---------- ENERGY + SPEED ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚡️ Energy Consumption Trend")
    st.line_chart(df["Energy_Consumption_L_h"][:100])

with col2:
    st.subheader("🚗 Traffic Speed Distribution")
    st.hist_chart = plt.figure()
    plt.hist(df["Traffic_Speed_kmh"], bins=20)
    st.pyplot(st.hist_chart)

# ---------- WEATHER IMPACT ----------
st.subheader("🌦 Weather vs Traffic Speed")

weather_speed = df.groupby("Weather_Condition")["Traffic_Speed_kmh"].mean()
st.bar_chart(weather_speed)

# ---------- RAW DATA ----------
with st.expander("📄 Show Raw Data"):
    st.dataframe(df)

# Select features
features = [
    "Vehicle_Count",
    "Traffic_Speed_kmh",
    "Road_Occupancy_%",
    "Sentiment_Score",
    "Ride_Sharing_Demand",
    "Parking_Availability"
]

X = df[features]
y = df["Accident_Report"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)
st.sidebar.header("🤖 Predict Accident")

vehicle = st.sidebar.slider("Vehicle Count", 0, 500, 100)
speed = st.sidebar.slider("Speed", 0, 120, 40)
occupancy = st.sidebar.slider("Road Occupancy", 0, 100, 50)
sentiment = st.sidebar.slider("Sentiment", -1.0, 1.0, 0.0)
ride = st.sidebar.slider("Ride Demand", 0, 100, 10)
parking = st.sidebar.slider("Parking", 0, 100, 50)

input_data = [[vehicle, speed, occupancy, sentiment, ride, parking]]

prediction = model.predict(input_data)

st.sidebar.write("Prediction:", "🚨 Accident Likely" if prediction[0] == 1 else "✅ Safe")
