# ----------------------------------------
# 📦 Import Libraries
# ----------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------
# 1️⃣ Create / Load Dataset
# ----------------------------------------
# Example monthly sales data (you can replace this with a CSV file)
data = {
    'Month': pd.date_range(start='2022-01-01', periods=24, freq='M'),
    'Sales': [200, 220, 250, 270, 260, 300, 320, 310, 330, 360, 400, 420,
              380, 400, 450, 470, 480, 500, 520, 550, 600, 620, 640, 660]
}

df = pd.DataFrame(data)
df.set_index('Month', inplace=True)

print("📊 Sample Sales Data:")
print(df.head())

# ----------------------------------------
# 2️⃣ Visualize Sales Trend
# ----------------------------------------
plt.figure(figsize=(8,4))
plt.plot(df.index, df['Sales'], marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.show()

# ----------------------------------------
# 3️⃣ Train-Test Split
# ----------------------------------------
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# ----------------------------------------
# 4️⃣ Fit ARIMA Model
# ----------------------------------------
# (p, d, q) values can be tuned using AIC/BIC or auto_arima
model = ARIMA(train['Sales'], order=(1, 1, 1))
model_fit = model.fit()

# ----------------------------------------
# 5️⃣ Forecast Future Sales
# ----------------------------------------
forecast = model_fit.forecast(steps=len(test))
forecast = pd.Series(forecast, index=test.index)

# ----------------------------------------
# 6️⃣ Evaluate Model Accuracy
# ----------------------------------------
mape = mean_absolute_percentage_error(test['Sales'], forecast) * 100
rmse = np.sqrt(mean_squared_error(test['Sales'], forecast))

print(f"\n📈 Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"📉 Root Mean Squared Error (RMSE): {rmse:.2f}")

# ----------------------------------------
# 7️⃣ Plot Actual vs Predicted
# ----------------------------------------
plt.figure(figsize=(8,4))
plt.plot(train.index, train['Sales'], label='Train')
plt.plot(test.index, test['Sales'], label='Actual', color='blue')
plt.plot(test.index, forecast, label='Predicted', color='red', linestyle='--')
plt.title("Sales Forecasting using ARIMA")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.show()

# ----------------------------------------
# 8️⃣ Forecast Future (Next 6 Months)
# ----------------------------------------
future_forecast = model_fit.forecast(steps=6)
future_dates = pd.date_range(start=df.index[-1] + pd.offsets.MonthEnd(), periods=6, freq='M')
future_df = pd.DataFrame({'Month': future_dates, 'Forecasted_Sales': future_forecast})
future_df.set_index('Month', inplace=True)

print("\n🔮 Future 6-Month Sales Forecast:")
print(future_df)
