import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_air_quality.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# Features
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

# Target
target = "aqi_index"

X = df[features]
y = df[target]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# Train
print("\nTraining model...")
model.fit(X_train, y_train)

print("Model training completed!")


# Prediction
y_pred = model.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)


print("\nModel Performance")
print("-----------------------")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# Save model
joblib.dump(model, "model/aqi_model.pkl")

print("\nModel saved successfully!")
print("File: model/aqi_model.pkl")