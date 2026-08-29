

````markdown
# 🌍 Air Quality Prediction & Pollution Risk Analysis

An end-to-end **Machine Learning and Deep Learning application** that predicts the **Air Quality Index (AQI)** and analyzes pollution risk based on air-quality and environmental parameters.

The project combines **Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Joblib, and Streamlit** to provide an interactive and easy-to-use air-quality prediction dashboard.

---

## 🚀 Live Demo

🔗 **Streamlit App:**  
https://air-quality-prediction-bu2djbk25qhcstsgecauyd.streamlit.app/

🔗 **GitHub Repository:**  
https://github.com/alladaindrika3/air-quality-prediction

---

## 📌 Project Overview

Air pollution is an important environmental concern. This project uses Machine Learning and Deep Learning techniques to analyze air-quality parameters and predict the **Air Quality Index (AQI)**.

The Streamlit application provides an interactive dashboard where users can enter air-quality parameters and obtain an AQI prediction along with a corresponding pollution-risk category.

### 🎯 Objectives

- Predict the Air Quality Index using Machine Learning.
- Analyze pollution levels from air-quality parameters.
- Categorize AQI into meaningful pollution-risk levels.
- Use Deep Learning with an Artificial Neural Network (ANN).
- Provide an interactive web interface using Streamlit.
- Make AQI prediction simple and accessible.
- Present prediction results through a visually interactive dashboard.

---

## ✨ Features

- 🌍 **AQI Prediction**
- 📊 **Pollution Risk Analysis**
- 🧮 Interactive air-quality input fields
- 📈 Linear Regression-based AQI prediction
- 🧠 ANN-based Deep Learning prediction
- 📊 AQI category classification
- 🌤️ Environmental parameter analysis
- 🏙️ Pollution visualization
- 📈 Data visualization
- 🤖 AI model comparison
- ⚡ Fast prediction
- 📱 Simple and user-friendly interface
- ☁️ Streamlit Cloud deployment

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| 🐍 Python | Core programming language |
| 🐼 Pandas | Data processing and analysis |
| 🔢 NumPy | Numerical operations |
| 📈 Scikit-learn | Machine Learning and model evaluation |
| 🧠 TensorFlow / Keras | Artificial Neural Network (ANN) |
| 💾 Joblib | Model saving and loading |
| 🎨 Streamlit | Web application and dashboard |
| 📁 Git & GitHub | Version control and project hosting |
| ☁️ Streamlit Community Cloud | Application deployment |

---

## 📂 Project Structure

```text
Air_Quality_Project/
│
├── app/
│   └── streamlit_app.py
│
├── model/
│   ├── aqi_model.pkl
│   ├── ann_model.keras
│   └── ann_scaler.pkl
│
├── screenshots/
│   ├── home.png
│   ├── inputs.png
│   ├── predict.png
│   ├── aqi_results.png
│   ├── category.png
│   ├── analysis.png
│   ├── visualization.png
│   └── final_dashboard.png
│
├── dataset/
│   └── cleaned_air_quality.csv
│
├── cleaned_air_quality.csv
├── cleaned_air_quality.zip
├── model.py
├── preprocessing.py
├── train_model.py
├── requirements.txt
└── README.md
````

---

## 🔄 Project Workflow

```text
Air Quality Dataset
        ↓
Data Collection
        ↓
Data Preprocessing
        ↓
Data Cleaning
        ↓
Feature Selection
        ↓
Train-Test Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Save Trained Models
        ↓
Streamlit Application
        ↓
User Inputs
        ↓
AQI Prediction
        ↓
Pollution Risk Category
        ↓
Analysis & Visualization
        ↓
Cloud Deployment
```

---

## 📊 Input Parameters

The system uses the following parameters for AQI prediction:

| Parameter       | Description                              |
| --------------- | ---------------------------------------- |
| 🌡️ Temperature | Temperature of the environment           |
| 💧 Humidity     | Amount of moisture present in the air    |
| 🌬️ Pressure    | Atmospheric pressure                     |
| 💨 Wind Speed   | Speed of wind in the environment         |
| PM2.5           | Fine particulate matter                  |
| PM10            | Particulate matter with larger particles |
| CO              | Carbon Monoxide concentration            |
| NO₂             | Nitrogen Dioxide concentration           |

The model uses these parameters to predict the numerical AQI value.

---

## 📊 AQI Categories

| AQI Range | Category        |
| --------: | --------------- |
|    0 – 50 | 🟢 Good         |
|  51 – 100 | 🟡 Satisfactory |
| 101 – 200 | 🟠 Moderate     |
| 201 – 300 | 🔴 Poor         |
| 301 – 400 | 🟣 Very Poor    |
| 401 – 500 | 🟤 Severe       |

---

# 🧠 Machine Learning & Deep Learning Models

The project uses both **Machine Learning** and **Deep Learning** approaches for AQI prediction.

---

## 📈 Machine Learning – Linear Regression

**Linear Regression** is used as the primary Machine Learning model for predicting the numerical AQI value.

The model learns the relationship between the environmental and pollutant parameters and the AQI value.

The trained model is saved as:

```text
model/aqi_model.pkl
```

The model is loaded in the Streamlit application using **Joblib**.

---

## 🧠 Deep Learning – Artificial Neural Network (ANN)

We also use an **Artificial Neural Network (ANN)** as our Deep Learning model.

The ANN learns the relationship between multiple environmental and pollutant parameters and the AQI value.

The ANN model is saved as:

```text
model/ann_model.keras
```

The scaler used for ANN input preprocessing is saved as:

```text
model/ann_scaler.pkl
```

---

# 📈 Model Performance

## Linear Regression Performance

| Metric                         |  Score |
| ------------------------------ | -----: |
| R² Score                       | 0.7089 |
| R² Percentage                  |  70.9% |
| Mean Absolute Error (MAE)      |  95.37 |
| Root Mean Squared Error (RMSE) | 176.64 |

### Interpretation

The Linear Regression model achieved an **R² score of approximately 70.9%**, indicating that the model explains a significant portion of the variation in AQI values within the evaluation dataset.

**MAE** and **RMSE** are used to measure the prediction error of the regression model.

---

## 🧠 ANN Performance

The ANN model is used as the Deep Learning approach for AQI prediction.

The ANN model was evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

The trained ANN model is stored in:

```text
model/ann_model.keras
```

---

# 📊 Data Processing

The dataset is cleaned and prepared before being used for model training.

The preprocessing stage includes:

* Handling missing values
* Cleaning the dataset
* Selecting relevant features
* Preparing data for Machine Learning
* Splitting data into training and testing sets
* Training the prediction model
* Saving the processed dataset

The processed dataset is stored as:

```text
cleaned_air_quality.csv
```

---

# 🖥️ Application Screenshots

## 🏠 1. Home Page

The home page provides an overview of the Air Quality Prediction application.

<img src="https://raw.githubusercontent.com/alladaindrika3/air-quality-prediction/main/screenshots/home.png" width="800">

---

## 📝 2. Input Interface

Users can enter the required air-quality parameters through the interactive input interface.

<img src="https://raw.githubusercontent.com/alladaindrika3/air-quality-prediction/main/screenshots/inputs.png" width="800">

---

## 🔮 3. AQI Prediction

The prediction section processes the entered air-quality parameters and generates the predicted AQI.

<img src="https://raw.githubusercontent.com/alladaindrika3/air-quality-prediction/main/screenshots/predict.png" width="800">

---

## 📊 4. Data Visualization

The dashboard provides visual representations of air-quality and AQI-related information.

<img src="https://raw.githubusercontent.com/alladaindrika3/air-quality-prediction/main/screenshots/visualization.png" width="800">

---

## 📈 5. AQI Results

The application displays the predicted AQI value along with the corresponding result.

<img src="https://raw.githubusercontent.com/alladaindrika3/air-quality-prediction/main/screenshots/aqi_results.png" width="800">

---

## 🔍 6. Pollution Risk Analysis

The analysis section provides additional insights into the predicted pollution level.

<img src="https://raw.githubusercontent.com/alladaindrika3/air-quality-prediction/main/screenshots/analysis.png" width="800">

---

## 🚦 7. Pollution Category

The predicted AQI is classified into an appropriate pollution category.

<img src="https://raw.githubusercontent.com/alladaindrika3/air-quality-prediction/main/screenshots/category.png" width="800">

---

## 🖥️ 8. Final Dashboard

The final dashboard combines AQI prediction, pollution analysis, category information, and visualization into an interactive interface.

<img src="https://raw.githubusercontent.com/alladaindrika3/air-quality-prediction/main/screenshots/final_dashboard.png" width="800">

---

# ⚙️ Installation & Local Setup

## 1. Clone the repository

```bash
git clone https://github.com/alladaindrika3/air-quality-prediction.git
```

## 2. Navigate to the project directory

```bash
cd air-quality-prediction
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Streamlit application

```bash
streamlit run app/streamlit_app.py
```

The application will open in your browser.

---

# 📦 Requirements

The project requires the following main Python packages:

```text
streamlit
pandas
numpy
scikit-learn
joblib
tensorflow
```

The Python version should be compatible with the TensorFlow version specified in `requirements.txt`.

---

# ☁️ Deployment

The application is deployed using **Streamlit Community Cloud**.

### Live Application

[https://air-quality-prediction-bu2djbk25qhcstsgecauyd.streamlit.app/](https://air-quality-prediction-bu2djbk25qhcstsgecauyd.streamlit.app/)

### GitHub Repository

[https://github.com/alladaindrika3/air-quality-prediction](https://github.com/alladaindrika3/air-quality-prediction)

The Streamlit application uses:

```text
app/streamlit_app.py
```

The dependency configuration is provided through:

```text
requirements.txt
```

---

# 🔮 Future Enhancements

* 📍 Location-based AQI prediction
* 🌦️ Real-time air-quality data integration
* 📊 Interactive historical AQI charts
* 🗺️ AQI visualization using maps
* 🔔 Pollution-level alerts
* 🤖 Improved Machine Learning and Deep Learning models
* 📈 Advanced prediction analysis
* 📱 Improved mobile-responsive interface
* 🌐 Integration with live environmental datasets
* 📡 Integration with IoT-based air-quality sensors

---

# 👩‍💻 Team

### Team Members

* **Indrika Allada** – Data Collection & Preprocessing, ML/DL
* **Nikitha** – Machine Learning Modeling
* **Gayatri** – User Interface
* **Revathi** – Deployment

---

# 📜 Project Summary

**Air Quality Prediction & Pollution Risk Analysis** is a Machine Learning and Deep Learning-based application designed to predict AQI values and provide an understandable analysis of pollution levels.

The project combines **Data Processing, Machine Learning, Deep Learning, Model Evaluation, and Streamlit** to build an interactive air-quality prediction system.

The overall workflow is:

```text
Air Quality Dataset
        ↓
Data Preprocessing
        ↓
Feature Selection
        ↓
Machine Learning Model
        ↓
Deep Learning Model
        ↓
Model Evaluation
        ↓
AQI Prediction
        ↓
Pollution Risk Analysis
        ↓
Interactive Streamlit Dashboard
        ↓
Cloud Deployment
```

The project demonstrates the practical integration of **Machine Learning and Deep Learning models** with an interactive web application for air-quality prediction and pollution-risk analysis.

---

# 🔗 Project Links

**GitHub Repository:**
[https://github.com/alladaindrika3/air-quality-prediction](https://github.com/alladaindrika3/air-quality-prediction)

**Live Streamlit Application:**
[https://air-quality-prediction-bu2djbk25qhcstsgecauyd.streamlit.app/](https://air-quality-prediction-bu2djbk25qhcstsgecauyd.streamlit.app/)

---

# ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub!

````



