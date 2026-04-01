import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
df = pd.read_csv("dataset/train.csv")

# Convert DateTime
df["DateTime"] = pd.to_datetime(df["DateTime"])

# Sort by time
df = df.sort_values("DateTime")

# Use DateTime as index
df.set_index("DateTime", inplace=True)

# Traffic series
series = df["Vehicles"]

# Train ARIMA model
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit()

# Predict next 24 time steps
forecast = model_fit.forecast(steps=24)

print(forecast)

# Plot results
plt.figure(figsize=(10,5))
plt.plot(series[-100:], label="Actual Traffic")
plt.plot(forecast, label="Forecast Traffic", color="red")
plt.title("Traffic Forecast using ARIMA")
plt.legend()
plt.show()