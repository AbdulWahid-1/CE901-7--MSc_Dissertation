import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

# Function to download stock market data from Yahoo Finance
def download_stock_data(stock_symbol, start_date, end_date):
    return yf.download(stock_symbol, start=start_date, end=end_date)

# Function to preprocess the data
def preprocess_data(data):
    stock_df = data[['Close']].reset_index()
    stock_df = stock_df.rename(columns={'Date': 'ds', 'Close': 'y'})
    return stock_df

# Function to split data into training and testing sets
def split_data(stock_df, train_ratio=0.8):
    train_size = int(len(stock_df) * train_ratio)
    train_data = stock_df[:train_size]
    test_data = stock_df[train_size:]
    return train_data, test_data

# Function for evaluating ARIMA model with given order
def evaluate_arima_model(data, order):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    predictions = model_fit.predict(start=data.index[0], end=data.index[-1], typ='levels')
    mse = mean_squared_error(data, predictions)
    rmse = sqrt(mse)
    return mse, rmse, model_fit

# Function for selecting the best ARIMA order
def select_best_arima_order(data, p_values, d_values, q_values):
    best_mse = float('inf')
    best_order = None
    best_model_fit = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse, _, model_fit = evaluate_arima_model(data, order)
                    if mse < best_mse:
                        best_mse = mse
                        best_order = order
                        best_model_fit = model_fit
                except:
                    continue

    return best_order, best_model_fit

# Function for making predictions using the fitted ARIMA model
def make_arima_predictions(data, best_order, model_fit):
    predictions = model_fit.predict(start=data.index[0], end=data.index[-1], typ='levels')
    return predictions

# Function for plotting the predictions
def plot_arima_results(train_data, train_predictions, test_data, test_predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(train_data['ds'], train_data['y'], label='Actual (Train)')
    plt.plot(train_data['ds'], train_predictions, label='Predicted (Train)', linestyle='dashed')
    plt.plot(test_data['ds'], test_data['y'], label='Actual (Test)')
    plt.plot(test_data['ds'], test_predictions, label='Predicted (Test)', linestyle='dashed')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Market Prediction using ARIMA for Amazon')
    plt.xticks(rotation=45)
    plt.show()

def stock_market_prediction_arima(stock_symbol, start_date, end_date, p_values=range(0, 3), d_values=range(0, 2), q_values=range(0, 3)):
    data = download_stock_data(stock_symbol, start_date, end_date)
    df = preprocess_data(data)
    train_data, test_data = split_data(df)
    best_order, model_fit = select_best_arima_order(train_data['y'], p_values, d_values, q_values)
    train_predictions = make_arima_predictions(train_data, best_order, model_fit)
    test_predictions = make_arima_predictions(test_data, best_order, model_fit)

    train_mse, train_rmse, _ = evaluate_arima_model(train_data['y'], best_order)
    test_mse, test_rmse, _ = evaluate_arima_model(test_data['y'], best_order)

    print(f"Train MSE: {train_mse}")
    print(f"Train RMSE: {train_rmse}")
    print(f"Test MSE: {test_mse}")
    print(f"Test RMSE: {test_rmse}")

    plot_arima_results(train_data, train_predictions, test_data, test_predictions)

if __name__ == "__main__":
    # Specify the stock symbol and time range
    stock_symbol = 'AMZN'
    start_date = '2010-01-01'
    end_date = '2023-06-30'

    # Perform stock market prediction using ARIMA
    stock_market_prediction_arima(stock_symbol, start_date, end_date)
