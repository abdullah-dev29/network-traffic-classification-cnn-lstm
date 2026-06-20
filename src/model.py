"""
CNN-LSTM hybrid model for binary Darknet-vs-Benign classification.
Architecture: faithful to Figure 7 of Mandela et al. (2025).

No Embedding layer — input is tabular scaled floats reshaped to (n_features, 1),
not character-token indices. The "hybrid" is the Conv1D front-end + stacked LSTMs.
"""
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Dropout, LSTM, Dense,
)


def build_cnn_lstm(n_features: int, n_channels: int = 1) -> tf.keras.Model:
    """
    Build, compile, and return the CNN-LSTM model.

    Expected output shapes with n_features=62 (after dropping ~15 constant cols):
        Input          : (None, 62, 1)
        Conv1D-128     : (None, 62, 128)   padding="same" keeps length
        MaxPool1D      : (None, 31, 128)
        Dropout        : (None, 31, 128)
        Conv1D-64      : (None, 31, 64)
        MaxPool1D      : (None, 15, 64)    floor(31/2)=15
        LSTM-100 (seq) : (None, 15, 100)
        Dropout        : (None, 15, 100)
        LSTM-50        : (None, 50)
        Dense-1 sigmoid: (None, 1)

    Parameters
    ----------
    n_features : int
        Number of feature columns after preprocessing (typically ~62).
    n_channels : int
        Channel dimension after reshape; always 1 for our tabular input.

    Returns
    -------
    tf.keras.Model — compiled and ready to fit.
    """
    import config as cfg

    inputs = Input(shape=(n_features, n_channels), name="input")

    x = Conv1D(cfg.CONV1_FILTERS, cfg.CONV1_KERNEL,
               activation="relu", padding=cfg.CONV_PADDING, name="conv1")(inputs)
    x = MaxPooling1D(pool_size=cfg.POOL_SIZE, name="pool1")(x)
    x = Dropout(cfg.DROPOUT_RATE, name="drop1")(x)

    x = Conv1D(cfg.CONV2_FILTERS, cfg.CONV2_KERNEL,
               activation="relu", padding=cfg.CONV_PADDING, name="conv2")(x)
    x = MaxPooling1D(pool_size=cfg.POOL_SIZE, name="pool2")(x)

    x = LSTM(cfg.LSTM1_UNITS, return_sequences=True, name="lstm1")(x)
    x = Dropout(cfg.DROPOUT_RATE, name="drop2")(x)

    x = LSTM(cfg.LSTM2_UNITS, return_sequences=False, name="lstm2")(x)

    outputs = Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="CNN_LSTM_Darknet")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    model.summary()
    return model


if __name__ == "__main__":
    # Build with the expected post-preprocessing feature count and print summary.
    # Does NOT fit the model — shapes only.
    model = build_cnn_lstm(n_features=62)
    print("\nModel built successfully. Parameter count:", model.count_params())
