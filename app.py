import pandas as pd
import streamlit as st

from src.config import OCEAN_PROXIMITY_VALUES
from src.predict import load_artifacts, models_exist, predict

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")


@st.cache_resource
def get_model():
    return load_artifacts()


st.title("House Price Predictor")
st.write("Enter the details of a California district to predict its median house value.")

if not models_exist():
    st.error("Model not found. Run `python main.py` first to train and save the model.")
    st.stop()

model, pipeline = get_model()

with st.form("house_form"):
    col1, col2 = st.columns(2)

    longitude = col1.number_input("Longitude", value=-122.23, format="%.2f")
    latitude = col2.number_input("Latitude", value=37.88, format="%.2f")

    housing_median_age = col1.number_input("Housing median age", min_value=0.0, value=30.0)
    median_income = col2.number_input("Median income (in $10,000s)", min_value=0.0, value=4.5)

    total_rooms = col1.number_input("Total rooms", min_value=0.0, value=2000.0)
    total_bedrooms = col2.number_input("Total bedrooms", min_value=0.0, value=400.0)

    population = col1.number_input("Population", min_value=0.0, value=800.0)
    households = col2.number_input("Households", min_value=0.0, value=300.0)

    ocean_proximity = st.selectbox("Ocean proximity", OCEAN_PROXIMITY_VALUES)

    submitted = st.form_submit_button("Predict price", type="primary")

if submitted:
    house = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }])

    prediction = float(predict(model, pipeline, house)[0])
    st.success(f"Predicted median house value: **${prediction:,.2f}**")
