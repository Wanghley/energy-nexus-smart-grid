
import os
import logging
from datetime import datetime, timedelta, timezone
import pandas as pd
from gridstatus import CAISO, Markets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "data/processed"
RAW_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_lmp_data(days: int = 90) -> None:
    logger.info("Fetching CAISO LMP data...")
    caiso = CAISO()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    try:
        df = caiso.get_lmp(market=Markets.DAY_AHEAD_HOURLY, start=start, end=end)
        df = df[['Time', 'LMP']]
        df.columns = ['timestamp', 'lmp_price']
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        df = df.drop_duplicates(subset='timestamp').reset_index(drop=True)
        df.to_csv(os.path.join(DATA_DIR, "ca_lmp_prices.csv"), index=False)
        logger.info(f"CAISO LMP data saved with {len(df)} records.")
    except Exception as e:
        logger.error("Error fetching CAISO LMP data", exc_info=True)

def fetch_real_ca_demand(days: int = 90) -> None:
    logger.info(f"Fetching real CAISO demand data for past {days} days...")
    caiso = CAISO()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    try:
        df = caiso.get_load(start=start, end=end)
        df = df[['Time', 'Load']]
        df.columns = ['timestamp', 'power_demand']
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        df = df.drop_duplicates(subset='timestamp').reset_index(drop=True)
        df.to_csv(os.path.join(DATA_DIR, "ca_demand.csv"), index=False)
        logger.info(f"Real CA demand data saved with {len(df)} records.")
    except Exception as e:
        logger.error("Error fetching CAISO demand data", exc_info=True)

def fetch_us_carbon_intensity(csv_path=os.path.join(RAW_DIR, "us_hourly.csv")) -> None:
    logger.info("Loading US CAISO carbon intensity data from local CSV...")
    try:
        df = pd.read_csv(csv_path)

        df.rename(columns={
            "Datetime (UTC)": "timestamp",
            "Carbon intensity gCO₂eq/kWh (direct)": "carbon_intensity_direct",
            "Carbon intensity gCO₂eq/kWh (Life cycle)": "carbon_intensity_lifecycle",
            "Carbon-free energy percentage (CFE%)": "carbon_free_pct",
            "Renewable energy percentage (RE%)": "renewable_pct"
        }, inplace=True)

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)

        df = df[[
            "carbon_intensity_direct",
            "carbon_intensity_lifecycle",
            "carbon_free_pct",
            "renewable_pct"
        ]]

        df = df.ffill().bfill()
        df.reset_index(inplace=True)
        df.to_csv(os.path.join(DATA_DIR, "us_carbon_intensity.csv"), index=False)

        logger.info(f"US hourly carbon intensity data saved with {len(df)} records.")
    except Exception as e:
        logger.error("Error processing us_hourly.csv", exc_info=True)

if __name__ == "__main__":
    fetch_us_carbon_intensity()
    fetch_lmp_data()
    fetch_real_ca_demand()



