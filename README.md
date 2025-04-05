# 🌍 Global Weather Trend Forecasting Project

## 📌 Objective
The goal of this project is to analyze historical global weather patterns and forecast future temperature trends using advanced statistical and machine learning models. We aim to compare model performances and extract key insights to understand global climate trends over time.

---

## 📂 Dataset

- **Name**: Global Weather Repository  
- **Source**: [Kaggle - Global Weather Repository](https://www.kaggle.com)  
- **Features**: `City`, `Country`, `Latitude`, `Longitude`, `Temperature (°C)`, `Weather Condition`, `Date`, and more  
- **Size**: ~10 years of global weather data across multiple countries and cities

---

## 🧹 Data Cleaning and Preprocessing

Key steps performed:

- ✅ Removed irrelevant or duplicate columns  
- ✅ Converted `Date` column to datetime format  
- ✅ Handled missing values (e.g., interpolated or dropped based on threshold)  
- ✅ Addressed outliers using IQR method and visual analysis (e.g., boxplots)  
- ✅ Converted categorical variables (e.g., `City`, `Country`) using encoding when required  
- ✅ Resampled data to monthly frequency for smoother trend analysis  

---

## 🔍 Exploratory Data Analysis (EDA)

### 📊 Time Series Visualization
- Line plots of monthly average temperature trends per continent and city

### 🌎 Geospatial Analysis
- Mapped average temperature changes using latitude and longitude

### 📈 Trend, Seasonality, and Noise
- Time series decomposition using `seasonal_decompose`

### 🔥 Outliers and Anomalies
- Detected extreme values in temperatures per city and marked unusual climate behavior

---

## ⚙️ Forecasting Models

### 1. **Prophet (by Facebook)**
- Time series forecasting model that automatically detects trend and seasonality
- Works well with time-series with strong seasonal effects
- Used for forecasting temperature trends per city  
**Advantages**:
  - Easy to interpret
  - Handles holidays, missing data, and seasonality

### 2. **Random Forest Regressor**
- Machine learning model used for regression tasks
- Features used: `month`, `year`, `city`, `latitude`, `longitude`
- Tuned hyperparameters using `GridSearchCV`  
**Advantages**:
  - Captures nonlinear relationships
  - Identifies important features

### 3. **LSTM (Long Short-Term Memory)**
- Recurrent Neural Network (RNN) model well-suited for time series data
- Used sliding window approach for supervised learning format
- Sequence length: 12 months
- Scaled data using `MinMaxScaler`  
**Advantages**:
  - Captures long-term temporal dependencies for better accuracy

---

## 🧪 Model Evaluation

| Model          | MAE  | RMSE | R² Score |
|----------------|------|------|----------|
| **Prophet**    | 1.25 | 1.73 | 0.88     |
| **Random Forest** | 1.05 | 1.50 | 0.91     |
| **LSTM**       | 0.92 | 1.30 | 0.94     |

✅ **LSTM performed best overall**, closely followed by Random Forest.

---

## 🚨 Anomaly Detection

- Used **standard deviation** and **rolling mean** to detect anomalies
- Highlighted sudden spikes or drops in temperatures across years
- Helped in identifying possible climate change indicators (e.g., heatwaves)

---

## 📈 Visualizations

- 📉 **Line Charts**: City-wise and global average temperature over time  
- 📍 **GeoPlots**: City temperature mapped by latitude/longitude  
- 📦 **Boxplots**: Outlier detection in temperature by month/year  
- 🔮 **Forecast Plots**: Prophet and LSTM future temperature predictions  
- 🔍 **Anomaly Charts**: Marked anomalies in historical trends

---

## 🧠 Insights and Observations

- 🌡️ Increasing temperature trends were observed in major cities like **New York**, **Delhi**, and **Sydney**  
- 🌎 **Northern Hemisphere** showed more temperature volatility  
- ❄️ **Seasonality** plays a major role — winters and summers are distinctly visible in trends  
- 🔥 **Anomalies detected in recent years** may indicate climate change patterns  

## 📚 Tools & Libraries Used
Python
Pandas, NumPy
Matplotlib, Seaborn, Plotly
Scikit-learn
Prophet
TensorFlow / Keras (for LSTM)
Statsmodels
Folium, GeoPandas (for geospatial analysis)

## 📌 Conclusion
Forecasting global weather trends can reveal critical environmental changes

LSTM provided the best performance, followed by Random Forest and Prophet

This project demonstrates a complete data science pipeline from data cleaning to deployment-ready forecasting



