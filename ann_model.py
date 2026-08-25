import os
import numpy as np
import pandas as pd
import joblib


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "cleaned_air_quality.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

ANN_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "ann_model.keras"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "ann_scaler.pkl"
)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# =====================================================
# FEATURES + TARGET
# =====================================================

features = [
    "temp_c",
    "humidity",
    "pressure_mb",
    "windspeed_kph",
    "pm2_5",
    "pm10",
    "co",
    "no2"
]

target = "aqi_index"

X = df[features]
y = df[target]


# =====================================================
# TRAIN / TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =====================================================
# FEATURE SCALING
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =====================================================
# ANN MODEL
# =====================================================

model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.2),

    Dense(32, activation="relu"),
    Dropout(0.2),

    Dense(16, activation="relu"),

    Dense(1)
])


# =====================================================
# COMPILE
# =====================================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


# =====================================================
# EARLY STOPPING
# =====================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# =====================================================
# TRAIN
# =====================================================

print("\nTraining ANN model...")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)


# =====================================================
# PREDICTION
# =====================================================

y_pred = model.predict(X_test).flatten()


# =====================================================
# EVALUATION
# =====================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\n====================================")
print("ANN MODEL PERFORMANCE")
print("====================================")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")
print("====================================")


# =====================================================
# SAVE MODEL + SCALER
# =====================================================

model.save(ANN_MODEL_PATH)

joblib.dump(scaler, SCALER_PATH)

print("\nANN model saved successfully!")
print("File:", ANN_MODEL_PATH)

print("\nANN scaler saved successfully!")
print("Scaler:", SCALER_PATH)