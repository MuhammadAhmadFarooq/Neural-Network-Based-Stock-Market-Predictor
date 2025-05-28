# src/preprocessing/run_preprocessing.py

from pathlib import Path
from src.preprocessing.clean_and_feature import clean_time_index


def main():
    raw_dir = Path("data/raw/bars")
    clean_dir = Path("data/processed/bars")
    clean_dir.mkdir(parents=True, exist_ok=True)

    # loop over every raw CSV and produce a cleaned version
    for csv_file in raw_dir.glob("*_day.csv"):
        out_file = clean_dir / f"{csv_file.stem}_cleaned.csv"
        clean_time_index(str(csv_file), str(out_file))

if __name__ == "__main__":
    main()
