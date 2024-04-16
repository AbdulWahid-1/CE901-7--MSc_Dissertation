import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

def main():
    # Downloading the stock market data from Yahoo Finance
    stock_symbol = 'F' 
    start_date = '2010-01-01'
    end_date = '2023-06-30'
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Preprocessing the dataset
    df = data[['Close']].reset_index()
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    dataset = df['y'].values.reshape(-1, 1)

    # Scaling the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Splitting the data into training and testing sets
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    # Function creating input sequences for the LSTM model
    def create_sequences(data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length + 1):
            X.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length-1])
        return np.array(X), np.array(y)

    # Define the sequence length
    sequence_length = 10

    # Create input sequences for the LSTM model
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)

    # Creating and training the LSTM model with multiple layers
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Making predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Inverse scaling the predictions
    train_predictions = scaler.inverse_transform(train_predictions)
    y_train = scaler.inverse_transform(y_train)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_test = scaler.inverse_transform(y_test)

    # Calculate metrics for training and testing predictions
    train_mse = mean_squared_error(y_train, train_predictions)
    train_rmse = sqrt(train_mse)
    test_mse = mean_squared_error(y_test, test_predictions)
    test_rmse = sqrt(test_mse)

    # Print the calculated metrics
    print(f"Train MSE: {train_mse}")
    print(f"Train RMSE: {train_rmse}")
    print(f"Test MSE: {test_mse}")
    print(f"Test RMSE: {test_rmse}")

    # Prepare the x-axis for plotting
    train_dates = df['ds'][sequence_length-1:train_size].values
    test_dates = df['ds'][train_size+sequence_length-1:].values

    # Plotting the predictions
    plt.figure(figsize=(12, 6))
    plt.plot(train_dates, y_train, label='Actual (Train)')
    plt.plot(train_dates, train_predictions, label='Predicted (Train)', linestyle='dashed')
    plt.plot(test_dates, y_test, label='Actual (Test)')
    plt.plot(test_dates, test_predictions, label='Predicted (Test)', linestyle='dashed')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Market Prediction using LSTM Version 1 for Ford')
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    main()
