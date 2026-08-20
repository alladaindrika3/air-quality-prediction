import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_air_quality.csv")

# Features
features = [
    "lat",
    "lon",
    "temp_c",
    "humidity",
    "pressure_mb",
    "windspeed_kph",
    "pm2_5",
    "pm10",
    "co",
    "no2"
]

X = df[features]
y = df["aqi_index"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression Model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("ML Model: Linear Regression")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save model
joblib.dump(model, "air_quality_model.pkl")

print("Model saved successfully!")