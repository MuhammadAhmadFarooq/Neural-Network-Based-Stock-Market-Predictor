# src/backtest/backtest_models.py

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

def directional_accuracy(y_true, y_pred):
    actual_change = np.sign(np.diff(y_true, prepend=y_true[0]))
    pred_change   = np.sign(np.diff(y_pred, prepend=y_pred[0]))
    return np.mean(actual_change == pred_change)

def make_sequences(data: np.ndarray, window_size: int):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size, :-1])
        y.append(data[i + window_size - 1, 3])  # close price at index 3
    return np.array(X), np.array(y)

def backtest_model(
    model_path: str,
    scaler,
    test_csv: str,
    window_size: int = 10,
    results_dir: str = "data/backtest"
):
    # Load test data
    df_test = pd.read_csv(
        test_csv,
        parse_dates=['datetime'],
        index_col='datetime',
        infer_datetime_format=True
    )
    values = df_test.values

    # Scale features
    data_scaled = scaler.transform(values)

    # Build sequences
    X_test, y_test = make_sequences(data_scaled, window_size)

    # Load model and predict
    model = load_model(model_path)
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Invert scaling for the predicted and true close price
    dummy = np.zeros((len(y_pred_scaled), values.shape[1]))
    dummy[:, 3] = y_pred_scaled.flatten()
    inv_pred = scaler.inverse_transform(dummy)[:, 3]
    dummy[:, 3] = y_test
    inv_true = scaler.inverse_transform(dummy)[:, 3]

    # Compute metrics
    mse = mean_squared_error(inv_true, inv_pred)
    mae = mean_absolute_error(inv_true, inv_pred)
    da  = directional_accuracy(inv_true, inv_pred)

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(model_path))[0]
    pred_df = pd.DataFrame({
        'true': inv_true,
        'pred': inv_pred
    }, index=df_test.index[window_size:])
    pred_df.to_csv(f"{results_dir}/{base}_predictions.csv")

    with open(f"{results_dir}/{base}_metrics.txt", 'w') as f:
        f.write(f"MSE: {mse:.6f}\nMAE: {mae:.6f}\nDirAcc: {da:.4f}\n")

    print(f"Backtested {base}: MSE={mse:.4f}, MAE={mae:.4f}, DirAcc={da:.4f}")
    return mse, mae, da
