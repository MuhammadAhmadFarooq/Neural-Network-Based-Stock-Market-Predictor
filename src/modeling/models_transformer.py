import tensorflow as tf

def build_transformer(input_shape, head_size=64, num_heads=2, ff_dim=128):
    inputs = tf.keras.Input(shape=input_shape)
    # positional embedding
    x = tf.keras.layers.Dense(head_size)(inputs)               # project features :contentReference[oaicite:5]{index=5}
    # multi‑head self‑attention
    x = tf.keras.layers.MultiHeadAttention(num_heads, head_size)(x, x)  # attention :contentReference[oaicite:6]{index=6}
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(ff_dim, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(loss='mse', optimizer='adam')
    return model
