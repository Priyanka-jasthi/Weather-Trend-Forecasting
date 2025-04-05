#Importing Required Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from prophet import Prophet
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("D:\Weather Trend Forecasting\GlobalWeatherRepository.csv")

df.isnull().sum() #No missing values found in the dataset
df.dtypes
df.columns.tolist()

#Checking for unusual values
#1. Temperature: Unusually low or high temperatures
unusual_temps = df[(df['temperature_celsius'] < -80) | (df['temperature_celsius'] > 60)]

# 2. Humidity: Should be between 0 and 100
unusual_humidity = df[(df['humidity'] < 0) | (df['humidity'] > 100)]

# 3. Wind Speed: Negative values don't make sense
unusual_wind_speed = df[df['wind_kph'] < 0]

# 4. UV Index: Typically between 0 and 11+ (above 15 is rare)
unusual_uv = df[(df['uv_index'] < 0) | (df['uv_index'] > 15)]

# 5. Pressure: Usually between 870 hPa to 1085 hPa
unusual_pressure = df[(df['pressure_mb'] < 870) | (df['pressure_mb'] > 1085)]

# 6. Precipitation: Negative precipitation is invalid
unusual_precip = df[df['precip_mm'] < 0]

print("Unusual Temperature Values:", unusual_temps.shape[0])
print("Unusual Humidity Values:", unusual_humidity.shape[0])
print("Unusual Wind Speed Values:", unusual_wind_speed.shape[0])
print("Unusual UV Index Values:", unusual_uv.shape[0])
print("Unusual Pressure Values:", unusual_pressure.shape[0])
print("Unusual Precipitation Values:", unusual_precip.shape[0])


## Exploratory Data Analysis

df.describe()

#Convert the date column
df['last_updated'] = pd.to_datetime(df['last_updated'], errors='coerce')
df['month'] = df['last_updated'].dt.month
df['year'] = df['last_updated'].dt.year


# Line plot: Avg temperature by month
plt.figure(figsize=(10, 6))
monthly_avg_temp = df.groupby('month')['temperature_celsius'].mean()
monthly_avg_temp.plot(marker='o', color='tomato')
plt.title("Average Temperature by Month")
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.show()

# Heatmap: Correlation between features
plt.figure(figsize=(10, 6))
numeric_cols = df.select_dtypes(include='number')
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# Boxplot: Temperature by month
plt.figure(figsize=(10, 6))
sns.boxplot(x='month', y='temperature_celsius', data=df, palette="Set3")
plt.title("Temperature Distribution by Month")
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.show()

# Histogram: Humidity
plt.figure(figsize=(10, 6))
sns.histplot(df['humidity'], bins=30, kde=True, color='skyblue')
plt.title("Humidity Distribution")
plt.xlabel("Humidity (%)")
plt.tight_layout()
plt.show()

# Temperature trend over time (worldwide)
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='last_updated', y='temperature_celsius', color='red')
plt.title('Temperature Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.show()

# Choose top 3 cities with the most data
top_cities = df['location_name'].value_counts().head(3).index.tolist()

plt.figure(figsize=(14, 6))

for city in top_cities:
    df_city = df[df['location_name'] == city].sort_values('last_updated')
    df_city['rolling_temp'] = df_city['temperature_celsius'].rolling(window=30).mean()
    plt.plot(df_city['last_updated'], df_city['rolling_temp'], label=city)

plt.title("Temperature Trends in Top 3 Cities (30-Day Rolling Avg)")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Geographical Patterns( Visualize temperature variations across regions)
# Load world map
world = gpd.read_file(r"C:\Users\pc\Downloads\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp")
print(world.columns) 
# Merge with temperature data
geo_data = df.groupby('country').mean()[['temperature_celsius']].reset_index()
world = world[['ADMIN', 'geometry']].rename(columns={'ADMIN': 'name'})
world = world.merge(geo_data, left_on="name", right_on="country", how="left")

# Plot
fig, ax = plt.subplots(figsize=(15, 10))
world.plot(column='temperature_celsius', cmap='coolwarm', linewidth=0.8, edgecolor='black', legend=True, ax=ax)
plt.title("Global Temperature Variations")
plt.show()


# Anomoly Detection
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Select relevant numerical features for anomaly detection
features_for_anomaly = [
    'temperature_celsius',
    'humidity',
    'cloud',
    'wind_kph',
    'pressure_mb',
    'air_quality_PM2.5',
    'air_quality_PM10'
]

anomaly_data = df[features_for_anomaly].dropna()

# Step 2: Fit Isolation Forest
iso = IsolationForest(contamination=0.02, random_state=42)  # 2% expected outliers
anomaly_data['anomaly'] = iso.fit_predict(anomaly_data)

# Step 3: Tag anomalies
df['is_anomaly'] = -1  # Default: Not anomaly
df.loc[anomaly_data.index, 'is_anomaly'] = anomaly_data['anomaly']

# Step 4: Visualize anomalies
plt.figure(figsize=(10, 5))
sns.scatterplot(data=anomaly_data, x='temperature_celsius', y='humidity', hue='anomaly', palette={1: "green", -1: "red"})
plt.title("Anomaly Detection using Isolation Forest")
plt.xlabel("Temperature (Celsius)")
plt.ylabel("Humidity")
plt.legend(title="Anomaly")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 5: Count anomalies
outlier_count = (anomaly_data['anomaly'] == -1).sum()
print(f" Total anomalies detected: {outlier_count}")

df['last_updated'] = pd.to_datetime(df['last_updated'])

# Filter anomalies
anomalies = df[df['is_anomaly'] == -1]

plt.figure(figsize=(12, 5))
plt.plot(df['last_updated'], df['temperature_celsius'], label='Temperature', alpha=0.5)
plt.scatter(anomalies['last_updated'], anomalies['temperature_celsius'], color='red', label='Anomalies')
plt.title('Temperature Anomalies Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

df[df['is_anomaly'] == -1]['location_name'].value_counts().head(10).plot(kind='bar', color='red')
plt.title('Top Cities with Most Anomalies')
plt.xlabel('City')
plt.ylabel('Number of Anomalies')
plt.tight_layout()
plt.show()

# Correlation between air quality and weather
env_features = ['temperature_celsius', 'humidity', 'wind_kph', 'pressure_mb', 'cloud', 'air_quality_PM2.5', 'air_quality_PM10']
env_data = df[env_features].dropna()

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(env_data.corr(), annot=True, cmap='BrBG', fmt='.2f')
plt.title("Correlation between Weather and Air Quality")
plt.tight_layout()
plt.show()

# Scatter plot: PM2.5 vs Temperature
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='air_quality_PM2.5', y='temperature_celsius', hue='humidity', palette='coolwarm')
plt.title("PM2.5 vs Temperature Colored by Humidity")
plt.xlabel("Air Quality PM2.5")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.show()


#Temperature distribution accross global coordinates
if 'latitude' in df.columns and 'longitude' in df.columns:
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='longitude', y='latitude', hue='temperature_celsius', data=df,
                    palette='coolwarm', alpha=0.6, edgecolor=None)
    plt.title("Temperature Distribution Across Global Coordinates")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(title="Temperature")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#### Model Building #########

# Load and prepare the data
df_prophet = df[['last_updated', 'temperature_celsius']].rename(columns={'last_updated': 'ds', 'temperature_celsius': 'y'})

# Ensure datetime format
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

# Initialize and fit the model
model = Prophet()
model.fit(df_prophet)

# Set forecast horizon
forecast_period = 100  # number of future time steps you want to predict

# Create future dataframe
future = model.make_future_dataframe(periods=forecast_period)

# Predict
forecast = model.predict(future)

# Merge actual and predicted for evaluation
historical_forecast = forecast[forecast['ds'].isin(df_prophet['ds'])].copy()
historical_forecast = historical_forecast.set_index('ds').join(df_prophet.set_index('ds'), how='left')
historical_forecast = historical_forecast.dropna(subset=['y'])

# Evaluation metrics on historical data
mae = mean_absolute_error(historical_forecast['y'], historical_forecast['yhat'])
mse = mean_squared_error(historical_forecast['y'], historical_forecast['yhat'])
rmse = np.sqrt(mse)
r2 = r2_score(historical_forecast['y'], historical_forecast['yhat'])

# Print evaluation results
print(f"Evaluation on Historical Data:")
print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(df_prophet['ds'], df_prophet['y'], label='Actual')
plt.plot(forecast['ds'], forecast['yhat'], color='red', label='Prophet Forecast')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Prophet Forecast: Future Temperature Prediction')
plt.legend()
plt.grid(True)
plt.show()

'''Evaluation on Historical Data:
MAE:  6.92
MSE:  74.54
RMSE: 8.63
R² Score: 0.19
'''


# Random Forest

X = df.select_dtypes(include=[np.number]).drop(columns=['temperature_celsius'])
y = df['temperature_celsius']

redundant_features = [
    'feels_like_celsius',
    'feels_like_fahrenheit',
    'temperature_fahrenheit'
]

X = X.drop(columns=redundant_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)

importances = model_rf.feature_importances_
feature_names = df.select_dtypes(include=[np.number]).drop(columns=['temperature_celsius']).columns
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[sorted_idx], y=np.array(feature_names)[sorted_idx])
plt.title("Feature Importance from Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

''' Evaluation metrics Of Random Forest:
MAE: 1.2057713395140666
RMSE: 1.7179049238667583
R² Score: 0.9689840440062487
'''


import shap

# Use TreeExplainer for Random Forest
explainer = shap.TreeExplainer(model_rf)

# Calculate SHAP values on test set
shap_values = explainer.shap_values(X_test)

# Visualize summary of feature impacts
shap.summary_plot(shap_values, X_test)


#### Long-Short Term Memory

# Select features
features = [
    'temperature_celsius',
    'humidity',
    'cloud',
    'wind_kph',
    'pressure_mb',
    'air_quality_PM2.5',
    'air_quality_PM10'
]

data = df[features].dropna()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, lookback=7):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, 0])  # temperature is the target
    return np.array(X), np.array(y)

lookback = 14
X, y = create_sequences(scaled_data, lookback)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=50, batch_size=16, callbacks=[early_stop], verbose=1)

# Predict
y_pred = model.predict(X_test)

# Inverse transform temp predictions
temp_scaler = MinMaxScaler()
temp_scaler.fit(data[['temperature_celsius']])
y_test_inv = temp_scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = temp_scaler.inverse_transform(y_pred)

# Metrics
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
r2 = r2_score(y_test_inv, y_pred_inv)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Plot results
plt.figure(figsize=(12, 5))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.title('LSTM Temperature Forecasting')
plt.xlabel('Time Steps')
plt.ylabel('Temperature (C)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

''' Evaluation metrics of LSTM
MAE: 3.49
RMSE: 4.68
R² Score: 0.7957
'''


#Model Comparison Plot
plt.figure(figsize=(14, 6))
plt.plot(y_test_inv[:100], label='Actual', color='dodgerblue')
plt.plot(y_pred_inv[:100], label='LSTM', color='orange')
plt.plot(y_pred_inv[:100], label='Random Forest', color='green')
plt.plot(forecast['yhat'].values[:100], label='Prophet', color='crimson')

plt.title('Model Comparison: Temperature Forecasting', fontsize=14)
plt.xlabel('Time Step')
plt.ylabel('Temperature (°C)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()






