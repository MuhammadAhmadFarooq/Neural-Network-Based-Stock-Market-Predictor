# src/preprocessing/filter_initial.py

import pandas as pd

def drop_initial_nan(
    in_path: str,
    out_path: str = None,
    max_window: int = 20
) -> str:
    """
    Drops the first (max_window - 1) rows, which will contain NaNs
    for any rolling/window-based features.
    """
    df = pd.read_csv(
        in_path,
        parse_dates=['datetime'],
        index_col='datetime',
        infer_datetime_format=True
    )

    # 1. Drop all rows where ANY column is NaN
    df_clean = df.dropna(axis=0, how='any')  # :contentReference[oaicite:4]{index=4}

    if out_path is None:
        base, ext = in_path.rsplit('.', 1)
        out_path = f"{base}_filtered.csv"

    df_clean.to_csv(out_path)
    print(f"Dropped initial NaNs; saved to {out_path}")
    return out_path
