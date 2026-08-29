import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_air_quality.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# Features - kept the same 8 inputs used by the Streamlit app and ANN
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


# Linear Regression model
model = LinearRegression()


# Train
print("\nTraining Linear Regression model...")
model.fit(X_train, y_train)

print("Model training completed!")


# Prediction
y_pred = model.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)


print("\n====================================")
print("LINEAR REGRESSION MODEL PERFORMANCE")
print("====================================")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)
print("====================================")


# Save model
joblib.dump(model, "model/aqi_model.pkl")

print("\nLinear Regression model saved successfully!")
print("File: model/aqi_model.pkl")
