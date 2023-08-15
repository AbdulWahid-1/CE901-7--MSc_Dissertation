# Have been getting an error while applying the fbProphet

import numpy as np
import pandas as pd
import yfinance as yf
from fbprophet import Prophet
import matplotlib.pyplot as plt

# Specifyig apple as the stock symbol and time range
stock_symbol = 'AAPL' 
start_date = '2010-01-01'
end_date = '2021-12-31'

# Downloading the stock market data from Yahoo Finance
data = yf.download(stock_symbol, start=start_date, end=end_date)

# Preprocessing the dataset
df = data[['Close']].reset_index()
df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

# Creating and fitting the Prophet model with advanced settings
model = Prophet(
    growth='linear',
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    changepoint_range=0.8
)
model.add_seasonality(
    name='monthly',
    period=30.5,
    fourier_order=5
)

# Adding additional regressors
# Example: Adding 'Volume' as an additional regressor
df['Volume'] = data['Volume']
model.add_regressor('Volume')

# Splitting the data into training and testing sets
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]

# Fitting the Prophet model
model.fit(train_data)

# Making future predictions
future_dates = model.make_future_dataframe(periods=len(test_data), freq='D')
future_dates = pd.merge(future_dates, test_data[['ds']], on='ds', how='inner')
forecast = model.predict(future_dates)

# Plotting the predictions
fig, ax = plt.subplots(figsize=(12, 6))
model.plot(forecast, ax=ax)
plt.plot(df['ds'], df['y'], label='Actual', color='blue')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Market Prediction using Prophet')
plt.xticks(rotation=45)
plt.show()
