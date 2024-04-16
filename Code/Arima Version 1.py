import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

def main():
    # Specify the stock symbol and time range
    stock_symbol = 'F'  
    start_date = '2010-01-01'
    end_date = '2023-06-30'

    # Downloading the stock market data from Yahoo Finance
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Preprocessing the dataset
    df = data[['Close']].reset_index()
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

    # Splitting the data into training and testing sets
    train_size = int(len(df) * 0.8)
    train_data = df[:train_size]
    test_data = df[train_size:]

    # Fitting the ARIMA model
    model = ARIMA(train_data['y'], order=(2, 1, 2))
    model_fit = model.fit()

    # Making predictions
    train_predictions = model_fit.predict(start=train_data.index[0], end=train_data.index[-1], typ='levels')
    test_predictions = model_fit.predict(start=test_data.index[0], end=test_data.index[-1], typ='levels')

    # Calculate metrics for training predictions
    train_mse = mean_squared_error(train_data['y'], train_predictions)
    train_rmse = sqrt(train_mse)

    # Calculate metrics for testing predictions
    test_mse = mean_squared_error(test_data['y'], test_predictions)
    test_rmse = sqrt(test_mse)

    # Print the calculated metrics
    print(f"Training MSE: {train_mse}")
    print(f"Training RMSE: {train_rmse}")
    print(f"Testing MSE: {test_mse}")
    print(f"Testing RMSE: {test_rmse}")

    # Plotting the predictions
    plt.figure(figsize=(12, 6))
    plt.plot(train_data['ds'], train_data['y'], label='Actual (Train)')
    plt.plot(train_data['ds'], train_predictions, label='Predicted (Train)', linestyle='dashed')
    plt.plot(test_data['ds'], test_data['y'], label='Actual (Test)')
    plt.plot(test_data['ds'], test_predictions, label='Predicted (Test)', linestyle='dashed')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Market Prediction using ARIMA version 1 on Ford')
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    main()
