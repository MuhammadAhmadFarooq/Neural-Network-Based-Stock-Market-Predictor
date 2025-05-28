# src/preprocessing/run_filter_initial.py

from pathlib import Path
from src.preprocessing.filter_initial import drop_initial_nan


def main():
    raw_cleaned = Path("data/features/bars")
    filtered_dir = Path("data/processed/filtered_bars")
    filtered_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in raw_cleaned.glob("*_cleaned_features.csv"):
        out_file = filtered_dir / f"{csv_file.stem}_filtered.csv"
        drop_initial_nan(str(csv_file), str(out_file))

if __name__ == "__main__":
    main()
