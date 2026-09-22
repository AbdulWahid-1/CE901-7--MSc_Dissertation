[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22904795.svg)](https://doi.org/10.5281/zenodo.22904795)

**ORCID:** [Your ORCID Number Here](https://orcid.org/0000-0000-0000-0000)

**LinkedIn:** [Your Name](https://linkedin.com/in/your-profile)

# Deep Learning for Arbitrary Stock Forecasting (MSc Thesis)

## Why I Chose This Research Topic
Navigating the stock market is notoriously difficult due to how noisy, chaotic, and unpredictable financial time-series data can be. I chose this topic for my Master's dissertation at the University of Essex because I wanted to move away from toy projects and test a hard real-world question: Can modern deep learning architectures genuinely outperform traditional statistical methods when trying to predict volatile market trends?

Specifically, I wanted to investigate how different model families handle the friction between short-term market noise and long-term financial patterns. Traditional tools like ARIMA are great for linear trends, but they usually fall apart when things get non-linear. I wanted to build a rigorous testing ground to see where classical statistics end and deep learning takes over.

## The Models I Tested
Building a fair comparison meant testing multiple generations of time-series forecasting tools across the exact same market data. I structured the repository to evaluate six distinct approaches:

*   **ARIMA (Autoregressive Integrated Moving Average):** The traditional statistical baseline used for decades in econometrics.
*   **Prophet:** Facebook’s open-source forecasting tool designed for handling strong seasonal effects and missing data.
*   **Convolutional Neural Networks (CNN):** Used to see if spatial feature extraction over rolling window matrices could capture local trend shapes.
*   **Recurrent Neural Networks (RNN):** Basic sequential modeling, which ultimately suffered from memory loss over long time horizons.
*   **Gated Recurrent Units (GRU):** A streamlined version of LSTMs designed to process sequential data faster with fewer parameters.
*   **Long Short-Term Memory Networks (LSTM):** The core deep learning architecture of the study, built specifically to retain long-range temporal dependencies.

## Dataset & Preprocessing
To keep the testing environment consistent and reproducible, I pulled historical financial data directly from the Yahoo Finance dataset.

Financial data is full of missing values, trading holidays, and sudden pricing spikes that can break a neural network during training. I wrote custom preprocessing scripts to clean the data, normalize pricing scales, and structure sliding time-window arrays so the sequential models could look back across historical trading sessions to predict future movement.

## Key Findings & Performance Analysis
My research validated that deep learning models significantly outperform traditional statistical methods on complex time-series forecasting.

The standout architecture across all testing cycles was the LSTM. Standard RNNs usually fail on long financial sequences because of the vanishing gradient problem—where the math forgets what happened fifty days ago by the time it reaches day one. The LSTM's internal gating mechanisms (specifically the input, output, and forget gates) completely solved this hurdle. It allowed the network to selectively remember critical long-term market shifts while ignoring day-to-day noise.

## Repository Structure & Contents
This repository contains all the experimental code, data outputs, evaluation logs, and comparative graphs generated during my MSc research:

*   `data/`: Raw and preprocessed historical Yahoo Finance records.
*   `models/`: Implementation scripts for ARIMA, Prophet, CNN, RNN, GRU, and LSTM architectures.
*   `output/`: Comparative loss curves, error metrics, and visual forecast plots.
*   `train.py`: The main training and evaluation loop.

## How to Run the Code
If you want to replicate my experiments or inspect the model evaluations locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AbdulWahid-1/CE901-7--MSc_Dissertation.git
   cd CE901-7--MSc_Dissertation
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the evaluation script** to generate the comparative forecast outputs:
   ```bash
   python evaluate.py
   ```
