import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from sklearn.datasets import fetch_california_housing

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("models/house_price_model.pkl")

# -----------------------------
# Load Dataset
# -----------------------------
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# -----------------------------
# Title
# -----------------------------
st.title("🏠 California House Price Prediction")
st.markdown("Predict California house prices using a trained Machine Learning model.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Enter House Details")

MedInc = st.sidebar.slider(
    "Median Income",
    float(df.MedInc.min()),
    float(df.MedInc.max()),
    3.5
)

HouseAge = st.sidebar.slider(
    "House Age",
    float(df.HouseAge.min()),
    float(df.HouseAge.max()),
    20.0
)

AveRooms = st.sidebar.slider(
    "Average Rooms",
    float(df.AveRooms.min()),
    20.0,
    5.5
)

AveBedrms = st.sidebar.slider(
    "Average Bedrooms",
    float(df.AveBedrms.min()),
    10.0,
    1.0
)

Population = st.sidebar.slider(
    "Population",
    float(df.Population.min()),
    float(df.Population.max()),
    1000.0
)

AveOccup = st.sidebar.slider(
    "Average Occupancy",
    float(df.AveOccup.min()),
    20.0,
    3.0
)

Latitude = st.sidebar.slider(
    "Latitude",
    float(df.Latitude.min()),
    float(df.Latitude.max()),
    34.0
)

Longitude = st.sidebar.slider(
    "Longitude",
    float(df.Longitude.min()),
    float(df.Longitude.max()),
    -118.0
)

# -----------------------------
# Input DataFrame
# -----------------------------
input_data = pd.DataFrame({
    "MedInc":[MedInc],
    "HouseAge":[HouseAge],
    "AveRooms":[AveRooms],
    "AveBedrms":[AveBedrms],
    "Population":[Population],
    "AveOccup":[AveOccup],
    "Latitude":[Latitude],
    "Longitude":[Longitude]
})

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict House Price"):

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted House Value: ${prediction*100000:.2f}"
    )

# -----------------------------
# Show Inputs
# -----------------------------
st.subheader("Input Features")

st.dataframe(input_data, use_container_width=True)

# -----------------------------
# Dataset Preview
# -----------------------------
with st.expander("Dataset Preview"):

    st.dataframe(df.head())

# -----------------------------
# Statistics
# -----------------------------
with st.expander("Dataset Statistics"):

    st.write(df.describe())

# -----------------------------
# Distribution Plot
# -----------------------------
st.subheader("House Price Distribution")

fig = px.histogram(
    df,
    x="MedHouseVal",
    nbins=50,
    title="Distribution of House Prices"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Correlation Heatmap
# -----------------------------
st.subheader("Correlation Matrix")

corr = df.corr()

fig = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="Viridis"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed by Muhammad Tayyab | Machine Learning Project")