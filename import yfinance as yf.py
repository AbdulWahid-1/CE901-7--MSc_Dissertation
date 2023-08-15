import yfinance as yf

def load_yahoo_finance_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Replace these variables with your desired parameters
ticker_symbol = "AAPL"   # Ticker symbol of the stock you want to retrieve
start_date = "2022-01-01"  # Start date for the data
end_date = "2022-12-31"    # End date for the data

# Load the data
stock_data = load_yahoo_finance_data(ticker_symbol, start_date, end_date)
print(stock_data)