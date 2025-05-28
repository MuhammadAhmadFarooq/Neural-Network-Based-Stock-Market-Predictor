from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, input_shape=input_shape),     # 50 units :contentReference[oaicite:0]{index=0}
        Dense(1)                                # single output for next‑step price :contentReference[oaicite:1]{index=1}
    ])
    model.compile(loss='mse', optimizer='adam')  # MSE loss, Adam optimizer :contentReference[oaicite:2]{index=2}
    return model
