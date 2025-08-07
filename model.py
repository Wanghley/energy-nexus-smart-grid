import os
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from preprocess import preprocess_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES_FILE = os.path.join(MODEL_DIR, "feature_cols_per_model.joblib")

def save_feature_cols(feature_cols_per_model: dict):
    joblib.dump(feature_cols_per_model, FEATURES_FILE)
    logger.info(f"Saved feature columns per model to {FEATURES_FILE}")

def load_feature_cols() -> dict:
    if os.path.exists(FEATURES_FILE):
        return joblib.load(FEATURES_FILE)
    else:
        logger.warning(f"Feature columns file not found at {FEATURES_FILE}")
        return {}

def train_models(df: pd.DataFrame, target_cols: list, hours_ahead: int = 24):
    logger.info("Training XGBoost models...")
    models = {}
    feature_cols_per_model = {}

    for target in target_cols:
        if target == "powerDemand":
            # Exclude all targets for powerDemand model
            feature_cols = [col for col in df.columns if col not in ["timestamp"] + target_cols]
        else:
            # For other targets, include powerDemand but exclude the other targets
            excluded_targets = [t for t in target_cols if t != target]
            feature_cols = [col for col in df.columns if col not in ["timestamp", target] + excluded_targets]
            if "powerDemand" not in feature_cols and "powerDemand" in df.columns:
                feature_cols.append("powerDemand")

        feature_cols_per_model[target] = feature_cols
        logger.info(f"Training model for target '{target}' using features: {feature_cols}")

        X = df[feature_cols]
        y = df[target]
        X_train = X.iloc[:-hours_ahead]
        y_train = y.iloc[:-hours_ahead]

        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6,
                                 random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        models[target] = model

        joblib.dump(model, os.path.join(MODEL_DIR, f"xgb_model_{target}.joblib"))
        logger.info(f"Model for '{target}' trained and saved.")

    save_feature_cols(feature_cols_per_model)
    return models, feature_cols_per_model

def load_models(target_cols: list) -> dict:
    models = {}
    for target in target_cols:
        path = os.path.join(MODEL_DIR, f"xgb_model_{target}.joblib")
        if os.path.exists(path):
            models[target] = joblib.load(path)
            logger.info(f"Loaded model for '{target}' from disk.")
        else:
            logger.warning(f"Model file for '{target}' not found at {path}")
    return models

def predict(models: dict, df: pd.DataFrame, feature_cols_per_model: dict) -> pd.DataFrame:
    logger.info("Making predictions with trained models...")
    for target, model in models.items():
        feature_cols = feature_cols_per_model.get(target, [])
        if not feature_cols:
            logger.warning(f"No feature columns found for target '{target}', skipping prediction.")
            continue

        X = df[feature_cols].copy()

        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                X[col] = pd.to_datetime(X[col]).astype(np.int64) // 10**9

        df[f"predicted_{target}"] = model.predict(X)
        logger.info(f"Predictions for '{target}' generated.")

    return df

def find_optimal_energy_window(df_predictions: pd.DataFrame, window_hours: int = 24) -> dict:
    df_window = df_predictions.tail(window_hours).copy()
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    df_window[["normalized_demand", "normalized_price", "normalized_carbon"]] = scaler.fit_transform(
        df_window[["predicted_powerDemand", "predicted_price", "predicted_carbonIntensity"]]
    )

    df_window["score"] = (
        0.5 * df_window["normalized_demand"] -
        0.25 * df_window["normalized_price"] -
        0.25 * df_window["normalized_carbon"]
    )
    best_row = df_window.loc[df_window["score"].idxmax()]
    return {
        "timestamp": best_row["timestamp"].isoformat(),
        "recommended_demand_kWh": round(best_row["predicted_powerDemand"], 2),
        "recommended_price_cents_kWh": round(best_row["predicted_price"], 2),
        "recommended_carbon_gCO2_kWh": round(best_row["predicted_carbonIntensity"], 2),
        "score": round(best_row["score"], 4)
    }
RETAIL_PRICE_MARKUP = 4.5

def predict_price_carbon_for_demand(models: dict, user_power_demand: float, other_features: pd.DataFrame, feature_cols_per_model: dict) -> dict:
    # The factor you use to scale from grid MW to household kWh in training/prediction outputs
    household_power_scaling_factor = 1.25 / 20000  # ~0.0000625

    # Scale user input (household kWh) UP to grid scale expected by the model
    scaled_power_demand_for_model = user_power_demand / household_power_scaling_factor

    input_row = other_features.copy()
    input_row['powerDemand'] = scaled_power_demand_for_model

    # Get features used by price and carbonIntensity models
    price_features = feature_cols_per_model.get('price', [])
    carbon_features = feature_cols_per_model.get('carbonIntensity', [])

    # Prepare input DataFrames for both models
    X_price = input_row[price_features].copy().fillna(0)
    X_carbon = input_row[carbon_features].copy().fillna(0)

    # Convert datetime columns to int timestamps if any
    for df_features in [X_price, X_carbon]:
        for col in df_features.columns:
            if pd.api.types.is_datetime64_any_dtype(df_features[col]):
                df_features[col] = pd.to_datetime(df_features[col]).astype(np.int64) // 10**9

    # Predict on grid scale features
    pred_price_grid = models['price'].predict(X_price)[0]
    pred_carbon_grid = models['carbonIntensity'].predict(X_carbon)[0]

    # Convert price to household scale cents/kWh (same scaling as your training code)
    eur_to_usd_rate = 1.08
    price_adjustment_factor = 0.02
    pred_price_household_cents = (pred_price_grid / 1000) * eur_to_usd_rate * price_adjustment_factor * 100

    pred_price_household_cents *= RETAIL_PRICE_MARKUP
    # Carbon intensity in gCO2/kWh = predicted total emissions divided by predicted power (household scale)
    # Use the user_power_demand (household scale) for denominator
    epsilon = 1e-6
    carbon_intensity = pred_carbon_grid / (user_power_demand + epsilon)

    # Clip carbon intensity to realistic max
    carbon_intensity = min(carbon_intensity, 2000)

    return {
        'predicted_price': pred_price_household_cents,
        'predicted_carbonIntensity': carbon_intensity
    }


if __name__ == "__main__":
    logger.info("Starting full data processing and model training pipeline...")
    df_raw = preprocess_data()
    df_numeric = df_raw.select_dtypes(include=np.number).copy()
    df_numeric["timestamp"] = df_raw["timestamp"]
    TARGET_VARS = ["powerDemand", "price", "carbonIntensity"]

    models, feature_cols_per_model = train_models(df_numeric, TARGET_VARS)

    df_predictions = predict(models, df_numeric, feature_cols_per_model)

    # Scale predictions to household level
    household_power_scaling_factor = 1.25 / 20000
    df_predictions["predicted_powerDemand"] *= household_power_scaling_factor
    logger.info(f"Scaled power predictions by factor {household_power_scaling_factor}")

    eur_to_usd_rate = 1.08
    price_adjustment_factor = 0.003
    df_predictions["predicted_price"] = (df_predictions["predicted_price"] / 1000) * eur_to_usd_rate * price_adjustment_factor * 100

    epsilon = 1e-6
    df_predictions["predicted_carbonIntensity"] = (
        df_predictions["predicted_carbonIntensity"] / (df_predictions["predicted_powerDemand"] + epsilon)
    )
    df_predictions["predicted_price"] *= RETAIL_PRICE_MARKUP
    df_predictions["predicted_carbonIntensity"] = df_predictions["predicted_carbonIntensity"].clip(lower=0, upper=2000)

    logger.info("Calculated carbon intensity as gCO2/kWh")

    optimal_time = find_optimal_energy_window(df_predictions)
    logger.info(f"Optimal usage window: {optimal_time}")

    logger.info("Displaying last 24 hours of predictions:")
    print(df_predictions[["timestamp"] + [f"predicted_{t}" for t in TARGET_VARS]].tail(24))

    logger.info("Pipeline complete.")










