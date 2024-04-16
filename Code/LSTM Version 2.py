import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

def download_stock_data(stock_symbol, start_date, end_date):
    return yf.download(stock_symbol, start=start_date, end=end_date)

def preprocess_data(data):
    df = data[['Close']].reset_index()
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    dataset = df['y'].values.reshape(-1, 1)
    return df, dataset

def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler, scaler.fit_transform(data)

def split_data(scaled_data, train_size):
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    return train_data, test_data

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length + 1):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length-1])
    return np.array(X), np.array(y)

def build_lstm_model(sequence_length):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs, batch_size):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

def make_predictions(model, X_train, X_test):
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    return train_predictions, test_predictions

def inverse_scale_data(scaler, train_predictions, y_train, test_predictions, y_test):
    train_predictions = scaler.inverse_transform(train_predictions)
    y_train = scaler.inverse_transform(y_train)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_test = scaler.inverse_transform(y_test)
    return train_predictions, y_train, test_predictions, y_test

def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = sqrt(mse)
    return mse, rmse

def plot_predictions(df, sequence_length, train_dates, test_dates, y_train, train_predictions, y_test, test_predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(train_dates, y_train, label='Actual (Train)')
    plt.plot(train_dates, train_predictions, label='Predicted (Train)', linestyle='dashed')
    plt.plot(test_dates, y_test, label='Actual (Test)')
    plt.plot(test_dates, test_predictions, label='Predicted (Test)', linestyle='dashed')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Market Prediction using LSTM Version 2 for Toyota Motors')
    plt.xticks(rotation=45)
    plt.show()

def main():
    # Specify the stock symbol and time range
    stock_symbol = 'TM'
    start_date = '2010-01-01'
    end_date = '2021-12-31'
    sequence_length = 10
    epochs = 50
    batch_size = 32

    # Downloading the stock market data from Yahoo Finance
    data = download_stock_data(stock_symbol, start_date, end_date)

    # Preprocessing the dataset
    df, dataset = preprocess_data(data)

    # Scaling the data
    scaler, scaled_data = scale_data(dataset)

    # Splitting the data into training and testing sets
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = split_data(scaled_data, train_size)

    # Create input sequences for the LSTM model
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)

    # Creating and training the LSTM model with multiple layers
    model = build_lstm_model(sequence_length)
    model = train_model(model, X_train, y_train, epochs, batch_size)

    # Making predictions
    train_predictions, test_predictions = make_predictions(model, X_train, X_test)

    # Inverse scaling the predictions
    train_predictions, y_train, test_predictions, y_test = inverse_scale_data(scaler, train_predictions, y_train, test_predictions, y_test)

    # Prepare the x-axis for plotting
    train_dates = df['ds'][sequence_length-1:train_size].values
    test_dates = df['ds'][train_size+sequence_length-1:].values

    # Calculate metrics for training and testing predictions
    train_mse, train_rmse = calculate_metrics(y_train, train_predictions)
    test_mse, test_rmse = calculate_metrics(y_test, test_predictions)

    # Print the calculated metrics
    print(f"Train MSE: {train_mse}")
    print(f"Train RMSE: {train_rmse}")
    print(f"Test MSE: {test_mse}")
    print(f"Test RMSE: {test_rmse}")

    # Plotting the predictions
    plot_predictions(df, sequence_length, train_dates, test_dates, y_train, train_predictions, y_test, test_predictions)

if __name__ == "__main__":
    main()
