import pandas as pd
import pandas_ta as ta

def add_technical_indicators(
    in_path: str,
    out_path: str = None
) -> str:
    """
    1. Load cleaned CSV with datetime index.
    2. Add TA indicators: MA, EMA, RSI, MACD, Bollinger Bands.
    3. Add lagged returns and rolling stats.
    4. Save enriched DataFrame.
    """
    df = pd.read_csv(
        in_path,
        parse_dates=['datetime'],
        index_col='datetime'
    )

    # 1. Moving averages
    df['ma_10'] = ta.sma(df['c'], length=10)    # 10‑day simple MA
    df['ema_10'] = ta.ema(df['c'], length=10)   # 10‑day exponential MA

    # 2. Momentum
    df['rsi_14'] = ta.rsi(df['c'], length=14)   # 14‑day RSI
    df['macd'], df['macd_signal'], _ = ta.macd(df['c'])

    # 3. Volatility
    bb = ta.bbands(df['c'], length=20)
    df = df.join(bb)

    # 4. Lagged returns & rolling stats
    df['ret_1'] = df['c'].pct_change(1)
    df['roll_std_10'] = df['ret_1'].rolling(window=10).std()
    df['roll_mean_10'] = df['ret_1'].rolling(window=10).mean()

    # Save
    if out_path is None:
        base, _ = in_path.rsplit('.', 1)
        out_path = f"{base}_features.csv"
    df.to_csv(out_path)
    print(f"Features saved to {out_path}")
    return out_path
