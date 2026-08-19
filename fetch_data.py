import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os

def fetch_options_data(tickers=["SPY", "AAPL", "QQQ", "MSFT"]):
    # Get risk free rate approximation (e.g., 10-year treasury yield)
    tnx = yf.Ticker("^TNX")
    tnx_hist = tnx.history(period="1d")
    if not tnx_hist.empty:
        r = tnx_hist['Close'].iloc[-1] / 100.0
    else:
        r = 0.05

    os.makedirs('data', exist_ok=True)
    
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        
        # Get current price
        hist = stock.history(period="1d")
        if hist.empty:
            print(f"No history found for {ticker}.")
            continue
        current_price = hist['Close'].iloc[-1]
        
        # Get options expirations
        expirations = stock.options
        if not expirations:
            print(f"No options found for {ticker}.")
            continue
            
        # We will pick the first expiration that is at least 30 days out for good liquidity
        target_exp = expirations[0]
        for exp in expirations:
            exp_date = datetime.strptime(exp, '%Y-%m-%d')
            days_to_exp = (exp_date - datetime.now()).days
            if days_to_exp >= 30:
                target_exp = exp
                break
                
        opt = stock.option_chain(target_exp)
        calls = opt.calls
        puts = opt.puts
        
        exp_date = datetime.strptime(target_exp, '%Y-%m-%d')
        t = (exp_date - datetime.now()).days / 365.25
        if t <= 0:
            t = 0.01 # prevent div by zero
            
        # Combine calls and puts
        calls['Type'] = 'Call'
        puts['Type'] = 'Put'
        
        df = pd.concat([calls, puts])
        
        # Select specific columns
        result = pd.DataFrame()
        result['Ticker'] = ticker
        result['Type'] = df['Type']
        result['Strike'] = df['strike']
        result['Expiry_Years'] = t
        result['Underlying_Price'] = current_price
        result['Bid'] = df['bid']
        result['Ask'] = df['ask']
        result['Implied_Volatility'] = df['impliedVolatility']
        result['Risk_Free_Rate'] = r
        
        # Filter out highly illiquid or broken data
        result = result[(result['Bid'] > 0) & (result['Ask'] > 0) & (result['Implied_Volatility'] > 0)]
        
        filename = f"data/{ticker}.csv"
        result.to_csv(filename, index=False)
        print(f"Saved {len(result)} options for {ticker} to {filename}")

if __name__ == "__main__":
    fetch_options_data(["SPY", "AAPL", "QQQ", "MSFT"])
