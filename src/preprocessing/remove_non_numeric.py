# src/preprocessing/remove_non_numeric.py

import os
import pandas as pd
import numpy as np

def drop_non_numeric(
    in_path: str,
    out_path: str = None
) -> str:
    """
    1. Load cleaned CSV (datetime index, numeric + non‑numeric columns).
    2. Keep only numeric columns (drops object / string dtypes).
    3. Save to new CSV for modeling.
    """
    # parse datetime index and load all columns
    df = pd.read_csv(
        in_path,
        parse_dates=['datetime'],    # ensure datetime index :contentReference[oaicite:4]{index=4}
        index_col='datetime',        # set as index :contentReference[oaicite:5]{index=5}
        infer_datetime_format=True    # speed up parsing :contentReference[oaicite:6]{index=6}
    )

    # select only numeric dtypes (float, int) :contentReference[oaicite:7]{index=7}
    numeric_df = df.select_dtypes(include=[np.number])

    # prepare output path
    if out_path is None:
        base, _ = os.path.splitext(in_path)
        out_path = f"{base}_numeric.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # save numeric‑only data
    numeric_df.to_csv(out_path)
    print(f"Numeric data saved to {out_path}")
    return out_path
