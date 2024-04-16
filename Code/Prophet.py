import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
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

    # Creating and fitting the Prophet model
    model = Prophet()
    model.fit(df)

    # Making future predictions
    future = model.make_future_dataframe(periods=365)  # making predictions for 365 days, can change it depending on the need
    forecast = model.predict(future)

    # Extracting actual and predicted values
    actual_values = df['y'].values
    predicted_values = forecast['yhat'][:-365].values  # Exclude the future predictions for calculation

    # Calculate metrics for training predictions
    train_actual_values = actual_values[:-365]  # Exclude the future predictions
    train_predicted_values = predicted_values[:len(train_actual_values)]
    train_mse = mean_squared_error(train_actual_values, train_predicted_values)
    train_rmse = sqrt(train_mse)

    # Calculate metrics for testing predictions
    test_actual_values = actual_values[-365:]  # Only include the future predictions
    test_predicted_values = predicted_values[len(train_actual_values):]
    test_mse = mean_squared_error(test_actual_values, test_predicted_values)
    test_rmse = sqrt(test_mse)

    # Print the calculated metrics
    print(f"Training MSE: {train_mse}")
    print(f"Training RMSE: {train_rmse}")
    print(f"Testing MSE: {test_mse}")
    print(f"Testing RMSE: {test_rmse}")

    # Plotting the actual and predicted values with years
    plt.figure(figsize=(12, 6))
    plt.plot(df['ds'][:-365], train_actual_values, label='Actual (Train)')
    plt.plot(df['ds'][:-365], train_predicted_values, label='Predicted (Train)', linestyle='dotted')
    plt.plot(df['ds'][-365:], test_actual_values, label='Actual (Test)')
    plt.plot(df['ds'][-365:], test_predicted_values, label='Predicted (Test)', linestyle='dotted')
    plt.xlabel('Year')
    plt.ylabel('Stock Price')
    plt.title('Stock Market Prediction using Prophet For Ford')
    plt.legend()
    plt.xticks(rotation=45)
    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()
