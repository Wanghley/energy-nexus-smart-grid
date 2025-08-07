import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "data/processed"

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    return df

def preprocess_data() -> pd.DataFrame:
    logger.info("Loading and merging CSV data...")

    # Paths
    lmp_price_path = os.path.join(DATA_DIR, "ca_lmp_prices.csv")
    demand_path = os.path.join(DATA_DIR, "ca_demand.csv")
    carbon_path = os.path.join(DATA_DIR, "us_carbon_intensity.csv")

    # Load data
    lmp_df = pd.read_csv(lmp_price_path)
    demand_df = pd.read_csv(demand_path)
    carbon_df = pd.read_csv(carbon_path)

    # Convert timestamps and remove timezone
    for df in [lmp_df, demand_df, carbon_df]:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    # Merge all on timestamp
    df = lmp_df.merge(demand_df, on="timestamp", how="outer")
    df = df.merge(carbon_df, on="timestamp", how="outer")

    # Rename columns to consistent names
    df.rename(columns={
        "lmp_price": "price",
        "power_demand": "powerDemand",
        "carbon_intensity_direct": "carbonIntensity"
    }, inplace=True)

    # Sort and reindex hourly
    df = df.sort_values("timestamp").set_index("timestamp").asfreq("h")

    # Fill missing values by time interpolation
    df.interpolate(method="time", limit_direction="both", inplace=True)

    # Fill remaining missing numeric values with median
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    df.reset_index(inplace=True)

    # Create time features
    df = create_time_features(df)

    # Create lag features for main targets
    for col in ["carbonIntensity", "price", "powerDemand"]:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag24"] = df[col].shift(24)

    df.dropna(inplace=True)

    logger.info(f"Preprocessed data shape: {df.shape}")

    return df

if __name__ == "__main__":
    df = preprocess_data()

