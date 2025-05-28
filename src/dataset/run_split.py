# src/dataset/run_split.py

from pathlib import Path
from src.dataset.split_dataset import split_time_series


def main():
    numeric_dir = Path("data/processed/numeric_bars")
    split_dir   = Path("data/splits")
    split_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in numeric_dir.glob("*.csv"):
        split_time_series(
            in_path  = str(csv_file),
            out_dir  = str(split_dir),
            train_frac=0.7,
            val_frac  =0.15
        )

if __name__ == "__main__":
    main()
