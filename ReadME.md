**Deep Learning for Arbitrary Stock Forecasting (MSc Thesis)**



**Why I chose this research topic**



Navigating the stock market is notoriously difficult due to how noisy and unpredictable financial data can be. I chose this topic for my Master's dissertation at the University of Essex because I wanted to see if modern deep learning architectures could actually outperform traditional statistical methods (like ARIMA) in predicting these volatile trends. Specifically, I wanted to investigate how different models balance short-term market noise with long-term financial patterns.







**What this repository contains**



This repository contains the results, data outputs, and comparative graphs from my research. I tested several models—ARIMA, CNN, RNN, LSTM, and PROPHET—using historical data from the Yahoo Finance dataset.







**Key Findings**



My research validated that deep learning models significantly outperform traditional machine learning on time-series data. The standout model was the LSTM. Because of its unique gating mechanisms (input, output, and forget gates), the LSTM successfully bypassed the vanishing gradient problem that usually breaks standard RNNs. It proved to be the most highly effective model for capturing complex, long-term temporal dependencies in the stock market.

