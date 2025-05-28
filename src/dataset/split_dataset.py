import os
import pandas as pd
from typing import Tuple

def split_time_series(
    in_path: str,
    out_dir: str,
    train_frac: float = 0.7,
    val_frac: float = 0.15
) -> Tuple[str, str, str]:
    """
    1. Load a numeric CSV with datetime index.
    2. Split into train/val/test by chronological fractions.
    3. Save three CSVs to out_dir named symbol_train.csv, symbol_val.csv, symbol_test.csv.
    """
    # Load data with datetime index
    df = pd.read_csv(
        in_path,
        parse_dates=['datetime'],
        index_col='datetime',
        infer_datetime_format=True
    )

    # Compute split indices
    n = len(df)
    train_end = int(n * train_frac)
    val_end   = train_end + int(n * val_frac)

    # Slice data
    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    # Derive symbol (prefix before any underscore or extension)
    filename = os.path.basename(in_path)
    symbol = filename.split('_')[0]

    # Prepare output directory
    os.makedirs(out_dir, exist_ok=True)

    # Define clean output names
    train_path = os.path.join(out_dir, f"{symbol}_train.csv")
    val_path   = os.path.join(out_dir, f"{symbol}_val.csv")
    test_path  = os.path.join(out_dir, f"{symbol}_test.csv")

    # Save splits
    train_df.to_csv(train_path)
    val_df.to_csv(val_path)
    test_df.to_csv(test_path)

    print(f"Split {symbol}: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    return train_path, val_path, test_path
