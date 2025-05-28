# src/backtest/run_backtest.py

from pathlib import Path
from src.backtest.backtest_models import backtest_model
import pickle

def main():
    split_dir  = Path("data/splits")
    model_dir  = Path("models")
    scaler_dir = model_dir / "scalers"
    results_dir = Path("data/backtest")

    # iterate over every .keras model file
    for model_file in model_dir.glob("*.keras"):
        # e.g. model_file.stem == "AAPL_cnn1d"
        stem_parts = model_file.stem.split("_", 1)
        if len(stem_parts) != 2:
            print(f"Skipping unrecognized model file name: {model_file.name}")
            continue

        symbol, arch = stem_parts
        test_csv = split_dir / f"{symbol}_test.csv"
        scaler_pkl = scaler_dir / f"{symbol}_scaler.pkl"

        if not test_csv.exists():
            print(f"Missing test CSV for {symbol}, skipping.")
            continue
        if not scaler_pkl.exists():
            print(f"Missing scaler for {symbol}, skipping.")
            continue

        # load scaler
        with open(scaler_pkl, "rb") as f:
            scaler = pickle.load(f)

        # backtest
        backtest_model(
            model_path  = str(model_file),
            scaler      = scaler,
            test_csv    = str(test_csv),
            window_size = 10,
            results_dir = str(results_dir)
        )

if __name__ == "__main__":
    main()
