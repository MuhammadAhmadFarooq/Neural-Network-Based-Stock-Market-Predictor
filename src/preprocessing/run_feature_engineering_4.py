#src/preprocessing/run_feature_engineering.py

from pathlib import Path
from src.preprocessing.remove_non_numeric import drop_non_numeric


def main():
    raw_cleaned = Path("data/processed/filtered_bars")
    numeric_dir = Path("data/processed/numeric_bars")
    numeric_dir.mkdir(parents=True, exist_ok=True)

    # process each cleaned file
    for csv_file in raw_cleaned.glob("*_cleaned_features_filtered.csv"):
        out_file = numeric_dir / f"{csv_file.stem}_numeric.csv"
        drop_non_numeric(str(csv_file), str(out_file))

if __name__ == "__main__":
    main()
