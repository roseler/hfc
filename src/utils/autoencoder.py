import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF info/warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN ops

# ⬇️ Now import everything else
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam


def autoencoder_pipeline(X, encoding_dim=8, epochs=20, batch_size=16):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    input_dim = X_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='linear')(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)

    autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    autoencoder.fit(X_scaled, X_scaled, epochs=epochs, batch_size=batch_size, verbose=0)

    X_encoded = encoder.predict(X_scaled)
    encoded_df = pd.DataFrame(X_encoded, columns=[f"ae_feat_{i}" for i in range(encoding_dim)])
    return encoded_df