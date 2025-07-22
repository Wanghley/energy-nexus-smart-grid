""""
BECAUSE WE ARE USING A FREE-TIER ON ELECTRICITY MAPS. COM WE CAN ONLY ACCESS Data for the latest 24 hours of CO2 hence
the model is affected as it cannot mergeand train 365 days for the prices and power and only 24 hours for CO2

Next major work is to get CO2 data for the past 265 days or year and do some statistics to make it granular/spread across the
the model
"""
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone 
from zoneinfo import ZoneInfo
import xgboost as xgb 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
import json
import os
from typing import Dict, Tuple, Optional

# API keys and constants
API_KEY = "29Egp134NPyfyz8gCyi1" 
EIA_API_KEY = "WHmaSxME39lqAZwO3RcFHg93syVUmDs4ucHxt8tA" 

ZONE = "US-NY-NYIS"
BASE_URL = "https://api.electricitymap.org"
HEADERS = {"auth-token": API_KEY} # Only include if API_KEY is actually provided/valid

# File paths for saving/loading data
HISTORICAL_DATA_FILE = "historical_data.json"
MODEL_PERFORMANCE_FILE = "model_performance.json"

def fetch_historical_carbon_data(zone=ZONE, days= 1):
    """Fetch more historical data for better model training"""
    end = datetime.now(tz=timezone.utc) # Use timezone-aware UTC datetime
    start = end - timedelta(days=days)
    start_str = start.isoformat(timespec='seconds').replace('+00:00', 'Z')
    end_str = end.isoformat(timespec='seconds').replace('+00:00', 'Z')
    url = f"{BASE_URL}/v3/carbon-intensity/history?zone={zone}&start={start_str}&end={end_str}"
    print(f"Fetching historical carbon data from: {url}")
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data["history"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None) # Ensure UTC then remove tz for merge
        return df
    else:
        print(f"Warning: Failed to fetch historical carbon data: {response.status_code} - {response.text}")
        return pd.DataFrame() # Return empty DataFrame on failure

def fetch_latest_carbon_data(zone=ZONE):
    url = f"{BASE_URL}/v3/carbon-intensity/latest?zone={zone}"
    print(f"Fetching latest carbon data from: {url}")
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame([data])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None) # Ensure UTC then remove tz for merge
        return df
    else:
        raise Exception(f"Failed to fetch latest carbon data: {response.status_code} - {response.text}")

def fetch_power_demand_data_eia(api_key: str, days: int = 365) -> pd.DataFrame: 
    """Fetch real power demand data from EIA"""
    base_url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    end = datetime.now(tz=timezone.utc) # Use timezone-aware UTC datetime
    start = end - timedelta(days=days)
    start_str = start.strftime("%Y-%m-%dT%H")
    end_str = end.strftime("%Y-%m-%dT%H")
    
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[]": "value",
        "facets[respondent][]": "NYIS",
        "facets[type][]": "D",   # Demand data
        "start": start_str,
        "end": end_str
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        try:
            json_data = response.json()
            # print(f"EIA Power API Response structure: {json_data.keys()}") # Keep for detailed debugging
            
            if "response" in json_data and "data" in json_data["response"]:
                data = json_data["response"]["data"]
                if not data:
                    print("Warning: No power demand data returned from EIA API")
                    return pd.DataFrame()
                
                df = pd.DataFrame(data)
                # print(f"Power demand data columns: {df.columns.tolist()}") # Keep for detailed debugging
                
                # Check available columns and adapt accordingly
                time_col = None
                value_col = None
                
                for col in df.columns:
                    if col.lower() in ['period', 'datetime', 'time']:
                        time_col = col
                    elif col.lower() in ['value', 'demand', 'load']:
                        value_col = col
                
                if time_col and value_col:
                    df_clean = df[[time_col, value_col]].copy()
                    df_clean.columns = ["datetime", "power_demand"]
                    df_clean["datetime"] = pd.to_datetime(df_clean["datetime"]).dt.tz_localize(None) # Assume API returns UTC-like, remove tz for merge
                    df_clean = df_clean.sort_values("datetime").reset_index(drop=True)
                    df_clean["power_demand"] = pd.to_numeric(df_clean["power_demand"], errors="coerce")
                    # Convert from MWh to GWh (assuming EIA provides MWh for demand)
                    # If EIA provides in MW, and it's hourly, then 'value' IS the MWh for that hour.
                    # Given typical household scale, 1 kWh is 0.000001 GWh.
                    # Let's keep it in MW or GWh as provided by EIA for grid-level demand.
                    # The original conversion `df_clean["power_demand"] / 1000000` implies converting from MWh to GWh, 
                    # as 1,000,000 MWh = 1 GWh. This is fine for grid-level data.
                    df_clean["power_demand"] = df_clean["power_demand"] / 1000 # Convert MW to GW or MWh to GWh
                    return df_clean
                else:
                    print(f"Warning: Could not find expected columns in power data. Available: {df.columns.tolist()}")
                    return pd.DataFrame()
            else:
                print(f"Warning: Unexpected EIA API response structure: {json_data}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Warning: Error processing power demand data: {e}")
            return pd.DataFrame()
    else:
        print(f"Warning: Could not fetch power demand data: {response.status_code} - {response.text}")
        return pd.DataFrame()

def fetch_real_time_price_data_eia(api_key: str, days: int = 365) -> pd.DataFrame:
    """Fetch electricity price data with more historical context from EIA API."""
    base_url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    end = datetime.now(tz=timezone.utc) # Use timezone-aware UTC datetime
    start = end - timedelta(days=days)
    start_str = start.strftime("%Y-%m-%dT%H")
    end_str = end.strftime("%Y-%m-%dT%H")
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[]": "value",
        "facets[respondent][]": "NYIS",
        "facets[type][]": "NG",   # Net Generation - EIA often provides LMP here
        "start": start_str,
        "end": end_str
    }
    
    print(f"EIA Price API URL: {base_url}")
    print(f"EIA Price API Params: {params}")
    
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        try:
            json_data = response.json()
            # print(f"EIA Price API Response structure: {json_data.keys()}") # Keep for detailed debugging
            
            if "response" in json_data and "data" in json_data["response"]:
                data = json_data["response"]["data"]
                if not data:
                    print("Warning: No price data returned from EIA API")
                    return pd.DataFrame()
                
                df = pd.DataFrame(data)
                # print(f"Price data columns: {df.columns.tolist()}") # Keep for detailed debugging
                # print(f"Sample data: {df.head()}") # Keep for detailed debugging
                
                # Check available columns and adapt accordingly
                time_col = None
                value_col = None
                
                for col in df.columns:
                    if col.lower() in ['period', 'datetime', 'time']:
                        time_col = col
                    elif col.lower() in ['value', 'price', 'lmp']: # LMP is common for price
                        value_col = col
                
                if time_col and value_col:
                    df_clean = df[[time_col, value_col]].copy()
                    df_clean.columns = ["datetime", "price"]
                    df_clean["datetime"] = pd.to_datetime(df_clean["datetime"]).dt.tz_localize(None) # Assume API returns UTC-like, remove tz for merge
                    df_clean = df_clean.sort_values("datetime").reset_index(drop=True)
                    df_clean["price"] = pd.to_numeric(df_clean["price"], errors="coerce")
                    df_clean["price"] = df_clean["price"] / 1000.0 # Convert from $/MWh to $/kWh
                    return df_clean
                else:
                    print(f"Warning: Could not find expected columns in price data. Available: {df.columns.tolist()}")
                    return pd.DataFrame()
            else:
                print(f"Warning: Unexpected API response structure: {json_data}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error processing price data: {e}")
            return pd.DataFrame() # Return empty DataFrame on failure
    else:
        print(f"Warning: Failed to fetch price data: {response.status_code} - {response.text}")
        return pd.DataFrame()

def create_time_features(df):
    """Create comprehensive time-based features for better predictions"""
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)   # Peak demand hours
    df["is_off_peak"] = df["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5, 6]).astype(int)   # Off-peak hours
    
    # Cyclical encoding for time features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    return df

def preprocess_data(carbon_df, price_df, power_df):
    """Enhanced preprocessing with better feature engineering"""
    print("🔗 Merging and processing data...")
    
    # Use outer merge to keep all datetime points, then interpolate missing values
    full_df = pd.merge(carbon_df, price_df, on='datetime', how='outer')
    
    # Merge with power demand data
    if not power_df.empty:
        full_df = pd.merge(full_df, power_df, on='datetime', how='outer')
    
    # Sort by datetime to ensure correct forward fill
    full_df = full_df.sort_values('datetime').reset_index(drop=True)

    # Impute missing values using forward fill, then backward fill for leading NaNs
    full_df["carbonIntensity"] = full_df["carbonIntensity"].ffill().bfill()
    full_df["price"] = full_df["price"].ffill().bfill()
    
    # Handle power demand - use real data if available, otherwise estimate
    if "power_demand" in full_df.columns:
        full_df["powerDemand"] = full_df["power_demand"].ffill().bfill()
    else:
        # Create dummy powerDemand column if it was never merged
        full_df["powerDemand"] = np.nan 

    # Create time features first, even if some data is NaN, they will be used for estimation
    full_df = create_time_features(full_df)

    # If we still don't have power demand data, estimate it
    if "powerDemand" not in full_df.columns or full_df["powerDemand"].isna().all():
        print("    Using estimated power demand based on time patterns")
        full_df["powerDemand"] = estimate_power_demand(full_df)
    
    # Fill any remaining NaNs after merging and feature creation (e.g., from lag features)
    numeric_columns = full_df.select_dtypes(include=[np.number]).columns
    full_df[numeric_columns] = full_df[numeric_columns].fillna(full_df[numeric_columns].median())

    # Create lag features for time series prediction (after initial NaNs are filled)
    for col in ["carbonIntensity", "price", "powerDemand"]: # Added powerDemand lag
        if col in full_df.columns:
            full_df[f"{col}_lag1"] = full_df[col].shift(1)
            full_df[f"{col}_lag24"] = full_df[col].shift(24) # 24 hour lag
    
    # Fill NaNs created by shifting (for first 24 hours)
    full_df[numeric_columns] = full_df[numeric_columns].fillna(full_df[numeric_columns].median())
    
    # Drop rows that still have NaNs, especially if initial data was too short for lags
    full_df.dropna(inplace=True)

    if full_df.empty:
        raise ValueError("Merged and preprocessed DataFrame is empty. Cannot train models.")

    print(f"    Merged dataset size: {full_df.shape}")
    return full_df

def estimate_power_demand(df):
    """Create realistic power demand estimates based on time patterns when real data isn't available"""
    # Base demand pattern based on typical household consumption (can be scaled for grid-level if needed)
    base_demand = 10000.0   # Example: 10 GW base demand (for grid level)
    
    # Hour-based multipliers (peak in evening, low at night)
    hour_multipliers = {
        0: 0.6, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.6,
        6: 0.8, 7: 1.2, 8: 1.1, 9: 0.9, 10: 0.8, 11: 0.8,
        12: 0.9, 13: 0.8, 14: 0.8, 15: 0.9, 16: 1.0, 17: 1.3,
        18: 1.5, 19: 1.4, 20: 1.2, 21: 1.0, 22: 0.8, 23: 0.7
    }
    
    # Weekend vs weekday multipliers
    weekend_multiplier = 1.1
    
    power_demand = []
    for _, row in df.iterrows():
        hour = row["hour"]
        is_weekend = row["is_weekend"]
        
        demand = base_demand * hour_multipliers[hour]
        if is_weekend:
            demand *= weekend_multiplier
            
        # Add some seasonal variation
        month = row["month"]
        if month in [6, 7, 8]:   # Summer (AC usage)
            demand *= 1.3
        elif month in [12, 1, 2]:   # Winter (heating)
            demand *= 1.2
            
        # Add small random variation
        demand *= np.random.normal(1.0, 0.05) # Smaller deviation for grid
        power_demand.append(max(1000.0, demand))   # Minimum 1 GW
    
    return power_demand

def train_models(df) -> Tuple[xgb.XGBRegressor, xgb.XGBRegressor, xgb.XGBRegressor, Dict]:
    """Train separate models for CO2, price, and power demand prediction using XGBoost."""
    # Feature columns for model training
    feature_cols = [
        "hour", "day_of_week", "month", "day_of_year", "is_weekend", "is_peak_hour", "is_off_peak",
        "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos"
    ]
    
    # Add renewable percentage if available (from Electricity Map data)
    if "renewablePercentage" in df.columns:
        feature_cols.append("renewablePercentage")
    
    # Add lag features if they exist
    lag_features = [col for col in df.columns if col.endswith(('_lag1', '_lag24'))]
    feature_cols.extend(lag_features)
    
    # Filter to only include features that exist in the dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols]
    y_co2 = df["carbonIntensity"]
    y_price = df["price"]
    y_power = df["powerDemand"]
    
    # Ensure there's enough data for splitting
    if len(df) < 2: # Need at least 2 samples for a valid train/test split
        print("Warning: Not enough data to create a test set. Training models on the full dataset.")
        X_train, X_test = X, X
        y_co2_train, y_co2_test = y_co2, y_co2
        y_price_train, y_price_test = y_price, y_price
        y_power_train, y_power_test = y_power, y_power
    else:
        # Split data
        # Using shuffle=False to maintain time series order
        X_train, X_test, y_co2_train, y_co2_test, y_price_train, y_price_test, y_power_train, y_power_test = train_test_split(
            X, y_co2, y_price, y_power, test_size=0.2, random_state=42, shuffle=False
        )
    
    # Model parameters optimized for each target (from original code)
    co2_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    price_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    power_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    # Train models
    co2_model.fit(X_train, y_co2_train)
    price_model.fit(X_train, y_price_train)
    power_model.fit(X_train, y_power_train)
    
    # Make predictions and calculate metrics (only if test set is not empty)
    metrics = {
        "co2": {"rmse": np.nan, "mae": np.nan, "r2": np.nan},
        "price": {"rmse": np.nan, "mae": np.nan, "r2": np.nan},
        "power": {"rmse": np.nan, "mae": np.nan, "r2": np.nan}
    }

    if not X_test.empty:
        co2_pred = co2_model.predict(X_test)
        price_pred = price_model.predict(X_test)
        power_pred = power_model.predict(X_test)
        
        # Calculate comprehensive metrics
        metrics = {
            "co2": {
                "rmse": float(np.sqrt(mean_squared_error(y_co2_test, co2_pred))),
                "mae": float(mean_absolute_error(y_co2_test, co2_pred)),
                "r2": float(co2_model.score(X_test, y_co2_test))
            },
            "price": {
                "rmse": float(np.sqrt(mean_squared_error(y_price_test, price_pred))),
                "mae": float(mean_absolute_error(y_price_test, price_pred)),
                "r2": float(price_model.score(X_test, y_price_test))
            },
            "power": {
                "rmse": float(np.sqrt(mean_squared_error(y_power_test, power_pred))),
                "mae": float(mean_absolute_error(y_power_test, power_pred)),
                "r2": float(power_model.score(X_test, y_power_test))
            }
        }
    else:
        print("Skipping metric calculation as test set is empty.")
    
    print(f"CO2 Model - RMSE: {metrics['co2']['rmse']:.4f}, MAE: {metrics['co2']['mae']:.4f}, R²: {metrics['co2']['r2']:.4f}")
    print(f"Price Model - RMSE: {metrics['price']['rmse']:.4f}, MAE: {metrics['price']['mae']:.4f}, R²: {metrics['price']['r2']:.4f}")
    print(f"Power Model - RMSE: {metrics['power']['rmse']:.4f}, MAE: {metrics['power']['mae']:.4f}, R²: {metrics['power']['r2']:.4f}")
    
    return co2_model, price_model, power_model, metrics

def generate_forecast_timepoints(hours_ahead=24) -> pd.DataFrame:
    """Generate future timepoints for prediction"""
    current_time = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0) # Use timezone-aware UTC
    future_times = [current_time + timedelta(hours=i) for i in range(1, hours_ahead + 1)]
    
    forecast_df = pd.DataFrame({"datetime": future_times})
    return forecast_df

def optimize_energy_usage(forecast_df, co2_model, price_model, power_model, household_kwh=1.0, 
                           co2_weight=0.4, price_weight=0.4, demand_weight=0.2) -> Dict:
    """
    Optimize energy usage with configurable weights for different factors
    """
    # Ensure required columns exist for preprocessing, or set reasonable defaults
    # For forecasting, renewablePercentage won't be known, so it will be filled by median during preprocess_data
    # If the original dataframe didn't have it, it won't be a feature anyway.

    # Preprocess forecast data
    # Create a dummy renewablePercentage if not present for consistency in preprocessing
    if "renewablePercentage" not in forecast_df.columns:
        forecast_df["renewablePercentage"] = 30.0 # Placeholder for new forecast rows

    # Create dummy lag features as well. They will be NaN initially, then filled by median in preprocess_data
    # This ensures the forecast_df has the same feature columns as the training data
    for col in ["carbonIntensity", "price", "powerDemand"]:
        if col not in forecast_df.columns:
            forecast_df[col] = np.nan
        forecast_df[f"{col}_lag1"] = np.nan
        forecast_df[f"{col}_lag24"] = np.nan
    
    # Need to pass a dummy dataframes for carbon/price/power to `preprocess_data`
    # for `forecast_df` to ensure all features are generated correctly.
    # However, `preprocess_data` is designed to merge historical data.
    # For *forecasting*, we need a different preprocessing approach that only adds features to `forecast_df`.
    
    # Let's refactor preprocess_data or create a new function for forecast preprocessing
    # For now, let's call create_time_features directly and then ensure feature set matches.

    forecast_df = create_time_features(forecast_df)
    
    # Feature columns (same as used in training)
    feature_cols = [
        "hour", "day_of_week", "month", "day_of_year", "is_weekend", "is_peak_hour", "is_off_peak",
        "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos"
    ]
    
    # Re-add renewable percentage if it was a feature in training
    if "renewablePercentage" in co2_model.get_booster().feature_names: # Check features from trained model
        feature_cols.append("renewablePercentage")

    # Add lag features if they were used in training. For forecasting, lags will be NaNs initially.
    # We need to fill these with appropriate values (e.g., last known values or zeros/means)
    # For simplicity, during forecast, we will use median/mean from training data, or simply 0.
    # More robust solution would involve forecasting lags or taking actual last known values.
    trained_features = co2_model.get_booster().feature_names
    lag_features_in_model = [f for f in trained_features if f.endswith(('_lag1', '_lag24'))]
    feature_cols.extend(lag_features_in_model)

    # For forecast_df, these lag features will be NaN. We need to fill them.
    # Simplest: fill with 0 or a representative mean/median from training data
    # A more advanced approach would be to predict these lags or use the actual last observed values.
    for lag_col in lag_features_in_model:
        if lag_col not in forecast_df.columns:
            forecast_df[lag_col] = 0 # Placeholder: Fill with 0 or a sensible default
                                    # In a real system, you'd use the last known real values.
    
    # Filter to available features (should match trained model features)
    # Ensure the order of columns matches the training data
    X_forecast = forecast_df[feature_cols]
    
    # Make predictions
    forecast_df["predicted_co2"] = co2_model.predict(X_forecast)
    forecast_df["predicted_price"] = price_model.predict(X_forecast)
    forecast_df["predicted_power"] = power_model.predict(X_forecast)
    
    # Normalize predictions for scoring
    def safe_normalize(series):
        """Safely normalize a series, handling edge cases"""
        min_val = series.min()
        max_val = series.max()
        if min_val == max_val or pd.isna(min_val) or pd.isna(max_val):
            return pd.Series(0.5, index=series.index)   # Neutral score if no variation
        return (series - min_val) / (max_val - min_val)
    
    forecast_df["co2_norm"] = safe_normalize(forecast_df["predicted_co2"])
    forecast_df["price_norm"] = safe_normalize(forecast_df["predicted_price"])
    # For power demand, lower is better (less strain on grid)
    forecast_df["power_norm"] = 1 - safe_normalize(forecast_df["predicted_power"])
    
    # Calculate composite score (lower is better)
    forecast_df["optimization_score"] = (
        co2_weight * forecast_df["co2_norm"] +
        price_weight * forecast_df["price_norm"] +
        demand_weight * forecast_df["power_norm"]
    )
    
    # Find optimal time
    if forecast_df["optimization_score"].isna().all():
        raise ValueError("Unable to calculate optimization scores - all scores are NaN.")
    
    optimal_idx = forecast_df["optimization_score"].idxmin()
    optimal_row = forecast_df.loc[optimal_idx]
    
    return {
        "optimal_time": optimal_row["datetime"],
        "predicted_co2": float(optimal_row["predicted_co2"]),
        "predicted_price": float(optimal_row["predicted_price"]),
        "predicted_power_demand": float(optimal_row["predicted_power"]),
        "optimization_score": float(optimal_row["optimization_score"]),
        "total_co2_impact": float(optimal_row["predicted_co2"] * household_kwh),
        "total_cost": float(optimal_row["predicted_price"] * household_kwh),
        "household_consumption": household_kwh,
        "co2_weight": co2_weight,
        "price_weight": price_weight,
        "demand_weight": demand_weight
    }

def save_data_with_timestamp(data, filename_prefix):
    """Save data with timestamp for historical tracking"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"
    
    # Convert data for JSON serialization
    if isinstance(data, pd.DataFrame):
        data_dict = data.to_dict('records')
        for record in data_dict:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
                elif hasattr(value, 'isoformat'):   # datetime
                    record[key] = value.isoformat()
                elif isinstance(value, (np.integer, np.floating)):
                    record[key] = float(value)
    else: # Assuming it's a dictionary for optimal_usage
        data_dict = data.copy()
        if isinstance(data_dict.get("optimal_time"), pd.Timestamp):
            data_dict["optimal_time"] = data_dict["optimal_time"].isoformat()
        
        # Convert numpy types to Python types
        for key, value in data_dict.items():
            if isinstance(value, (np.integer, np.floating)):
                data_dict[key] = float(value)
    
    with open(filename, "w") as f:
        json.dump(data_dict, f, indent=2)
    
    return filename

def save_predictions(forecast_df, filename="hourly_predictions.json"):
    """Save hourly predictions in a clean format"""
    predictions = []
    for _, row in forecast_df.iterrows():
        pred = {
            "datetime": row["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
            "hour": int(row["hour"]),
            "is_peak_hour": bool(row["is_peak_hour"]),
            "is_weekend": bool(row["is_weekend"]),
            "predicted_co2_intensity": float(row["predicted_co2"]),
            "predicted_electricity_price": float(row["predicted_price"]),
            "predicted_power_demand": float(row["predicted_power"]),
            "optimization_score": float(row["optimization_score"]),
            "renewable_percentage": float(row.get("renewable_percentage", 30.0)) # Use .get with default
        }
        predictions.append(pred)
    
    with open(filename, "w") as f:
        json.dump(predictions, f, indent=2)

def main():
    print(" Enhanced Energy Optimization System")
    print("=" * 50)
    
    try:
        print(" Fetching historical carbon intensity data...")
        carbon_df = fetch_historical_carbon_data(days=365)
        print(f"    Retrieved {len(carbon_df)} carbon intensity records")

        print(" Fetching electricity price data...")
        price_df = fetch_real_time_price_data_eia(EIA_API_KEY, days=365)
        print(f"    Retrieved {len(price_df)} price records")

        print(" Attempting to fetch power demand data...")
        power_df = fetch_power_demand_data_eia(EIA_API_KEY, days=365)
        print(f"    Retrieved {len(power_df)} power demand records")
        
        # Initial check for sufficient data BEFORE merging
        if carbon_df.empty and price_df.empty and power_df.empty:
            raise ValueError("All data sources returned empty. Cannot proceed with model training.")
        
        # Preprocess data (this now handles merging internally with outer join)
        # Note: preprocess_data expects carbon_df, price_df, power_df as separate inputs.
        preprocessed_df = preprocess_data(carbon_df, price_df, power_df)
        
        if preprocessed_df.empty:
            raise ValueError("Preprocessed DataFrame is empty after merging and cleaning. Not enough data to train models.")

        # Save historical data for future reference
        historical_file = save_data_with_timestamp(preprocessed_df, "historical_data")
        print(f"Saved historical data to: {historical_file}")

        print(" Training prediction models...")
        co2_model, price_model, power_model, metrics = train_models(preprocessed_df)
        
        # Save model performance metrics
        with open(MODEL_PERFORMANCE_FILE, "w") as f:
            json.dump(metrics, f, indent=2)

        print(" Generating forecast for next 24 hours...")
        forecast_df = generate_forecast_timepoints(hours_ahead=24)
        
        print(" Optimizing energy usage...")
        optimal_usage = optimize_energy_usage(
            forecast_df, co2_model, price_model, power_model,
            household_kwh=1.0,   # Optimize for 1 kWh usage
            co2_weight=0.4,      # 40% weight on carbon footprint
            price_weight=0.4,    # 40% weight on cost
            demand_weight=0.2    # 20% weight on grid demand
        )

        print(" Saving predictions and recommendations...")
        # To save forecast_df with predicted values, ensure optimize_energy_usage returns the full forecast_df
        # For now, let's just save the optimal usage.
        
        # If you want to save the full forecast_df with predictions:
        # save_predictions(forecast_df, "hourly_predictions.json") # This line would need optimize_energy_usage to return it
        
        optimal_file = save_data_with_timestamp(optimal_usage, "optimal_usage")
        print(f"    Saved optimal usage to: {optimal_file}")

        print("\n" + "="*50)
        print(" OPTIMAL ENERGY USAGE RECOMMENDATION")
        print("="*50)
        print(f" Optimal Time: {optimal_usage['optimal_time']}")
        print(f" CO₂ Intensity: {optimal_usage['predicted_co2']:.2f} gCO₂/kWh")
        print(f" Electricity Price: ${optimal_usage['predicted_price']:.4f}/kWh")
        print(f" Power Demand: {optimal_usage['predicted_power_demand']:.2f} GWh")
        print(f" Optimization Score: {optimal_usage['optimization_score']:.4f}")
        print(f"\n For {optimal_usage['household_consumption']} kWh consumption:")
        print(f" Total CO₂ Impact: {optimal_usage['total_co2_impact']:.2f} gCO₂")
        print(f"Total Cost: ${optimal_usage['total_cost']:.4f}")
        
        print(f"\n Model Performance Summary:")
        print(f"    CO₂ Model R²: {metrics['co2']['r2']:.3f}")
        print(f"    Price Model R²: {metrics['price']['r2']:.3f}")   
        print(f"    Power Model R²: {metrics['power']['r2']:.3f}")
        
        return optimal_usage, metrics
        
    except Exception as e:
        print(f" Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    optimal_usage, model_metrics = main()
