# Sample Code: GBM + Monte Carlo Simulation (Placeholder)
# Note: This base version was copied from ChatGPT for testing and learning purposes.
# I will test, modify, and personalize this for actual NSE tickers later.
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Step 1: Fetch data
ticker = 'INFY.NS'  # Replace with any NSE stock
data = yf.download(ticker, start='2022-01-01', end='2024-12-31')
close_prices = data['Close']
log_returns = np.log(1 + close_prices.pct_change().dropna())

# Step 2: Calculate drift and volatility
mu = log_returns.mean()
sigma = log_returns.std()

# Step 3: Monte Carlo parameters
S0 = close_prices[-1]  # Last known price
T = 1  # 1 year
N = 252  # trading days
dt = T / N
iterations = 1000

# Step 4: Simulate paths
price_paths = np.zeros((N, iterations))
price_paths[0] = S0

for t in range(1, N):
    Z = np.random.standard_normal(iterations)
    price_paths[t] = price_paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

# Step 5: Plotting
plt.figure(figsize=(10,6))
plt.plot(price_paths[:, :10])  # Plot 10 paths
plt.title(f"Monte Carlo Simulation of {ticker} using GBM")
plt.xlabel("Days")
plt.ylabel("Price")
plt.grid(True)
plt.show()

# Final distribution
plt.hist(price_paths[-1], bins=50)
plt.title(f"Distribution of Simulated Prices for {ticker}")
plt.xlabel("Price after 1 year")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
