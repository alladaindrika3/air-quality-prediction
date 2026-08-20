import pandas as pd

# Load dataset
df = pd.read_csv("dataset/air_quality.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Data types
print("\nData Types:")
print(df.dtypes)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ---------------------------------------
# Date and Time Preprocessing
# ---------------------------------------

# Combine date and time
df["datetime"] = pd.to_datetime(
    df["date_ist"] + " " + df["time_ist"],
    format="%d/%m/%Y %H:%M"
)

# Check the converted datetime
print("\nDatetime after conversion:")
print(df["datetime"].head())

# Check datatype
print("\nDatetime Data Type:")
print(df["datetime"].dtype)

# ---------------------------------------
# Step 4: Check Invalid Values
# ---------------------------------------

# Columns that should not normally have negative values
non_negative_columns = [
    "humidity",
    "pressure_mb",
    "windspeed_kph",
    "aqi_index",
    "pm2_5",
    "pm10",
    "co",
    "no2"
]

print("\nNegative Values Check:")

for column in non_negative_columns:
    negative_count = (df[column] < 0).sum()
    print(f"{column}: {negative_count}")


# ---------------------------------------
# Check basic statistics
# ---------------------------------------

print("\nStatistical Summary:")
print(df[
    [
        "temp_c",
        "humidity",
        "pressure_mb",
        "windspeed_kph",
        "aqi_index",
        "pm2_5",
        "pm10",
        "co",
        "no2"
    ]
].describe())

# ---------------------------------------
# Step 5: Outlier Detection using IQR
# ---------------------------------------

numerical_columns = [
    "temp_c",
    "humidity",
    "pressure_mb",
    "windspeed_kph",
    "aqi_index",
    "pm2_5",
    "pm10",
    "co",
    "no2"
]

print("\nOutlier Count using IQR:")

for column in numerical_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = (
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ).sum()

    print(f"{column}: {outliers}")

    # ---------------------------------------
# Step 6: Prepare Final Cleaned Dataset
# ---------------------------------------

# Remove unnecessary original date/time columns
df = df.drop(columns=["date_ist", "time_ist"])

# Save cleaned dataset
output_file = "dataset/cleaned_air_quality.csv"
df.to_csv(output_file, index=False)

print("\nCleaned dataset saved successfully!")
print(f"File: {output_file}")

# Final dataset information
print("\nFinal Dataset Shape:")
print(df.shape)

print("\nFinal Columns:")
print(df.columns.tolist())