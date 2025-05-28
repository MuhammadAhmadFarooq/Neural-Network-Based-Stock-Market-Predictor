import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from src.modeling.models_lstm      import build_lstm
from src.modeling.models_cnn       import build_cnn1d
from src.modeling.models_transformer import build_transformer
import pickle
    


def load_data(path):
    df = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    return df.values

def scale_data(train, val):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)  # fit on train :contentReference[oaicite:7]{index=7}
    val_scaled = scaler.transform(val)          # transform val/test :contentReference[oaicite:8]{index=8}
    return train_scaled, val_scaled, scaler

def make_sequences(data, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, :-1])    # all features except last col? adjust as needed
        y.append(data[i+window_size-1, 3])      # e.g. closing price index=3
    return np.array(X), np.array(y)

def train_and_save(model_fn, train_path, val_path, name, out_dir="models"):
    # load & scale
    train = load_data(train_path)
    val   = load_data(val_path)
    train_s, val_s, scaler = scale_data(train, val)
    # windowed sequences
    X_train, y_train = make_sequences(train_s)
    X_val,   y_val   = make_sequences(val_s)
    # build model
    model = model_fn(input_shape=X_train.shape[1:])
    # callbacks
    ckpt_path = f"{out_dir}/{name}.keras"
    os.makedirs(out_dir, exist_ok=True)
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # :contentReference[oaicite:9]{index=9}
    cp = ModelCheckpoint(ckpt_path, save_best_only=True)                         # :contentReference[oaicite:10]{index=10}
    # train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100, callbacks=[es, cp], batch_size=32, verbose=2
    )

    
    os.makedirs("models/scalers", exist_ok=True)
    # save the scaler for later backtest
    scaler_path = f"models/scalers/{name.split('_')[0]}_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"✓ scaler saved to {scaler_path}")


    # return history for later plotting/backtest
    return history, scaler
