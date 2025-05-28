from pathlib import Path
# src/preprocessing/run_feature_engineering_2.py
from src.preprocessing.feature_engineering import add_technical_indicators


def main():
    raw_dir = Path("data/processed/bars")
    feat_dir = Path("data/features/bars")
    feat_dir.mkdir(parents=True, exist_ok=True)

    # Process each cleaned CSV
    for csv_file in raw_dir.glob("*_cleaned.csv"):
        out_file = feat_dir / f"{csv_file.stem}_features.csv"
        add_technical_indicators(str(csv_file), str(out_file))

if __name__ == "__main__":
    main()
