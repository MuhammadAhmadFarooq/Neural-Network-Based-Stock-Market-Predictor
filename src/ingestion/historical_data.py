# src/ingestion/historical_data.py

import os
import yaml
import time
import pandas as pd
from polygon import RESTClient
from datetime import datetime
from typing import Any

def load_api_keys(path="configs/api_keys.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def fetch_and_save_bars(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    out_dir: str = "data/raw/bars"
) -> str | None:
    keys   = load_api_keys()
    client = RESTClient(keys["polygon"]["api_key"])
    timespan = interval if interval in {"minute","hour","day"} else interval

    print(f"Requesting {symbol} from {start} to {end} @ {timespan}")

    def safe_aggs(**kwargs):
        backoff = 1
        while True:
            try:
                return client.get_aggs(**kwargs)
            except (TypeError, AttributeError) as e:
                if "unexpected keyword" in str(e):
                    return client.stocks_equities_aggregates(**kwargs)
                raise
            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    print(f"Rate limit hit; sleeping {backoff}s…")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                else:
                    raise

    try:
        aggs = safe_aggs(
            ticker=symbol,
            multiplier=1,
            timespan=timespan,
            from_=start,
            to=end
        )
        if not aggs:
            print("→ No data returned")
            return None

        records: list[dict[str, Any]] = []
        for a in aggs:
            if isinstance(a, dict):
                t = a.get("t") or a.get("timestamp")
                o = a.get("o") or a.get("open")
                h = a.get("h") or a.get("high")
                l = a.get("l") or a.get("low")
                c = a.get("c") or a.get("close")
                v = a.get("v") or a.get("volume")
            else:
                t = getattr(a, "timestamp", None)
                o = getattr(a, "open",      None)
                h = getattr(a, "high",      None)
                l = getattr(a, "low",       None)
                c = getattr(a, "close",     None)
                v = getattr(a, "volume",    None)

            records.append({"t": t, "o": o, "h": h, "l": l, "c": c, "v": v})

        df = pd.DataFrame(records)
        df["t"] = pd.to_datetime(df["t"], unit="ms")
        df.set_index("t", inplace=True)

        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{symbol}_{interval}.csv")
        df.to_csv(path)
        print(f"✓ Saved: {path}")
        return path

    except Exception as e:
        print(f"Fetch failed for {symbol}: {e}")
        return None
