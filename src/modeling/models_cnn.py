from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense

def build_cnn1d(input_shape):
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),  # 1D conv :contentReference[oaicite:3]{index=3}
        GlobalAveragePooling1D(),                                                        # pool features :contentReference[oaicite:4]{index=4}
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    return model
