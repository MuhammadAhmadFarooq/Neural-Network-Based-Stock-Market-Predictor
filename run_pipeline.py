# run_pipeline.py

import yaml
import sys
import logging
from pathlib import Path

# Ingestion
from src.ingestion.historical_data import fetch_and_save_bars

# Preprocessing step 1: clean timestamps
from src.preprocessing.run_preprocessing_1 import main as run_preprocessing_1

# Feature engineering step 2: add indicators
from src.preprocessing.run_feature_engineering_2 import main as run_feature_engineering_2

# Filter step 3: drop initial NaNs
from src.preprocessing.run_filter_initial_3 import main as run_filter_initial_3

# Feature engineering step 4: drop non‑numeric
from src.preprocessing.run_feature_engineering_4 import main as run_feature_engineering_4

# Dataset splitting, modeling, backtest, report (if desired)
from src.dataset.run_split       import main as run_split
from src.modeling.run_training   import main as run_training
from src.backtest.run_backtest   import main as run_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def load_settings(path="configs/settings.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    cfg      = load_settings()
    symbols  = cfg["symbols"]
    interval = cfg["interval"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]

    # optional: pass one of ingest, prep1, feat2, filt3, feat4, split, train, backtest, report
    step = sys.argv[1] if len(sys.argv)>1 else "all"

    if step in ("all","ingest"):
        logging.info("→ INGESTION")
        for sym in symbols:
            fetch_and_save_bars(sym, interval, start, end)

    if step in ("all","prep1"):
        logging.info("→ PREPROCESSING 1: clean_time_index")
        run_preprocessing_1()

    if step in ("all","feat2"):
        logging.info("→ FEATURE ENGINEERING 2: add_technical_indicators")
        run_feature_engineering_2()

    if step in ("all","filt3"):
        logging.info("→ FILTER INITIAL NaNs")
        run_filter_initial_3()

    if step in ("all","feat4"):
        logging.info("→ FEATURE ENGINEERING 4: drop_non_numeric")
        run_feature_engineering_4()

    if step in ("all","split"):
        logging.info("→ SPLITTING DATA")
        run_split()

    if step in ("all","train"):
        logging.info("→ TRAINING MODELS")
        run_training()

    if step in ("all","backtest"):
        logging.info("→ BACKTESTING")
        run_backtest()

    logging.info("✅ pipeline complete")
    logging.info("→ all done!")