import streamlit as st
import pandas as pd
import numpy as np
from preprocess import preprocess_data
from model import load_models, predict, predict_price_carbon_for_demand, load_feature_cols
from rl_agent import EnergyOptimizationRLAgent  # <-- RL agent import

RETAIL_PRICE_MARKUP = 4.5
st.title("California Energy Optimization System")

@st.cache_data(show_spinner=True)
def load_data():
    return preprocess_data()

df = load_data()

# Define feature columns excluding targets, timestamp, and object cols
exclude_cols = ["timestamp", "carbonIntensity", "price", "powerDemand"]
exclude_cols += df.select_dtypes(include=["object"]).columns.tolist()
feature_cols_all = [col for col in df.columns if col not in exclude_cols]

@st.cache_resource
def get_models_and_features():
    target_cols = ["carbonIntensity", "price", "powerDemand"]
    models = load_models(target_cols)
    feature_cols_per_model = load_feature_cols()
    return models, feature_cols_per_model

models, feature_cols_per_model = get_models_and_features()

def scale_predictions(df_pred):
    household_power_scaling_factor = 1.25 / 20000  # MW to kWh approx.
    eur_to_usd_rate = 1.08
    price_adjustment_factor = 0.003
    epsilon = 1e-6

    df_pred = df_pred.copy()
    df_pred["predicted_powerDemand"] = df_pred["predicted_powerDemand"] * household_power_scaling_factor

    df_pred["predicted_price"] = (
        (df_pred["predicted_price"] / 1000) * eur_to_usd_rate * price_adjustment_factor * 100
    )

    df_pred["predicted_carbonIntensity"] = (
        df_pred["predicted_carbonIntensity"] / (df_pred["predicted_powerDemand"] + epsilon)
    )
    df_pred["predicted_carbonIntensity"] = df_pred["predicted_carbonIntensity"].clip(lower=0, upper=2000)

    return df_pred

if st.button("Show data preview"):
    st.dataframe(df.head())

if models:
    st.write("### Predictions for next 24 hours")
    forecast_df = df.tail(24).copy()

    # For each model, use correct features from feature_cols_per_model
    forecast_df = predict(models, forecast_df, feature_cols_per_model)
    forecast_df = scale_predictions(forecast_df)

    # Format for display
    forecast_df['predicted_powerDemand'] = forecast_df['predicted_powerDemand'].map(lambda x: f"{x:.4f} kWh")
    forecast_df['predicted_price'] = forecast_df['predicted_price'].map(lambda x: f"${x:.6f} / kWh")
    forecast_df['predicted_carbonIntensity'] = forecast_df['predicted_carbonIntensity'].map(lambda x: f"{x:.4f} gCO₂/kWh")

    st.dataframe(forecast_df[["timestamp", "predicted_powerDemand", "predicted_price", "predicted_carbonIntensity"]])

    # RL Agent section
    raw_forecast_df = df.tail(24).copy()
    raw_forecast_df = predict(models, raw_forecast_df, feature_cols_per_model)
    raw_forecast_df = scale_predictions(raw_forecast_df)

    rl_agent = EnergyOptimizationRLAgent()
    rl_agent.train(raw_forecast_df, episodes=500)

    recommended_hour = rl_agent.recommend_hour(raw_forecast_df)
    rec_time = raw_forecast_df.iloc[recommended_hour]['timestamp']
    rec_power = raw_forecast_df.iloc[recommended_hour]['predicted_powerDemand']
    rec_price = raw_forecast_df.iloc[recommended_hour]['predicted_price']
    rec_carbon = raw_forecast_df.iloc[recommended_hour]['predicted_carbonIntensity']

    st.write("### RL Agent Recommendation")
    st.write(f"**Recommended hour to use electricity:** {rec_time}")
    st.write(f"Expected usage: {rec_power:.4f} kWh")
    st.write(f"Expected price: ${rec_price:.6f} / kWh")
    st.write(f"Expected carbon intensity: {rec_carbon:.4f} gCO₂/kWh")

else:
    st.warning("No models loaded. Please train models first.")

st.write("---")
st.header("Predict price and carbon intensity for your own power demand")

user_power_demand = st.number_input(
    "Enter your power demand (kWh):",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=0.1
)

if st.button("Predict price and CO2"):
    if models and 'price' in models and 'carbonIntensity' in models and feature_cols_per_model:
        base_features = df.tail(1).copy()  # last row as context

        try:
            preds = predict_price_carbon_for_demand(models, user_power_demand, base_features, feature_cols_per_model)

            # Apply retail markup to predicted price here
            retail_price = preds['predicted_price'] * RETAIL_PRICE_MARKUP

            # Show the scaled power demand used internally for prediction
            household_power_scaling_factor = 1.25 / 20000
            scaled_kwh = user_power_demand * household_power_scaling_factor

            st.write(f"**Input power demand:** {user_power_demand:.2f} kWh")
            st.write(f"**Scaled power demand (used internally):** {scaled_kwh:.6f} kWh")
            st.write(f"**Predicted Price:** ${retail_price:.6f} per kWh")
            st.write(f"**Predicted Carbon Intensity:** {preds['predicted_carbonIntensity']:.2f} gCO₂ per kWh")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.error("Models or feature columns not loaded correctly.")