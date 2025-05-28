# src/modeling/run_training.py

from pathlib import Path
from src.modeling.train_models import train_and_save
from src.modeling import models_lstm, models_cnn, models_transformer

def main():
    split_dir = Path("data/splits")

    # Loop over each train split
    for train_path in split_dir.glob("*_train.csv"):      # e.g. AAPL_train.csv
        # Derive symbol as text before first underscore
        symbol = train_path.stem.split("_")[0]           # "AAPL_train" → "AAPL" :contentReference[oaicite:1]{index=1}

        # Reconstruct the validation split path
        val_path = split_dir / f"{symbol}_val.csv"       # e.g. data/splits/AAPL_val.csv

        # Sanity check
        if not val_path.exists():
            print(f"Validation file not found for {symbol}, skipping.")
            continue

        print(f"Training models for {symbol}...")

        # train each architecture, naming outputs by symbol
        train_and_save(
            models_lstm.build_lstm,
            str(train_path), str(val_path),
            f"{symbol}_lstm"
        )
        train_and_save(
            models_cnn.build_cnn1d,
            str(train_path), str(val_path),
            f"{symbol}_cnn1d"
        )
        train_and_save(
            models_transformer.build_transformer,
            str(train_path), str(val_path),
            f"{symbol}_transformer"
        )

if __name__ == "__main__":
    main()
