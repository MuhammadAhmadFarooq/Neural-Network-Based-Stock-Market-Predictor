# src/preprocessing/clean_and_feature.py

import os
import pandas as pd

def clean_time_index(
    in_path: str,
    out_path: str = None
) -> str:
    """
    1. Load raw bars CSV, parsing 't' as datetime and setting as index.
    2. Rename the index to 'datetime'.
    3. Save cleaned data to CSV (or Parquet).
    """
    # 1. Read CSV, parse 't' into datetime, set as index
    df = pd.read_csv(
        in_path,
        parse_dates=['t'],        # convert ‘t’ to datetime64 :contentReference[oaicite:3]{index=3}
        index_col='t',            # make it the DataFrame index :contentReference[oaicite:4]{index=4}
         # speed up parsing if format is consistent :contentReference[oaicite:5]{index=5}
    )

    # 2. Rename index for clarity (optional)
    df.index.rename('datetime', inplace=True)  # rename index axis :contentReference[oaicite:6]{index=6}

    # 3. Save cleaned data
    if out_path is None:
        base, ext = os.path.splitext(in_path)
        out_path = f"{base}_cleaned.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path)
    print(f"Cleaned data saved to {out_path}")
    return out_path
