import yfinance as yf
import pandas as pd
import numpy as np
import math
from datetime import datetime
import os

def N(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_price(type_opt, S, K, T, r, q, v):
    if v <= 0 or T <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * v * v) * T) / (v * math.sqrt(T))
    d2 = d1 - v * math.sqrt(T)
    if type_opt == 'Call':
        return S * math.exp(-q * T) * N(d1) - K * math.exp(-r * T) * N(d2)
    else:
        return K * math.exp(-r * T) * N(-d2) - S * math.exp(-q * T) * N(-d1)

def calibrate_iv(type_opt, S, K, T, r, q, market_mid):
    if market_mid <= 0.001 or T <= 0 or S <= 0 or K <= 0:
        return 0.1
    # Check intrinsic value
    intrinsic = max(0.0, (S * math.exp(-q * T) - K * math.exp(-r * T)) if type_opt == 'Call' else (K * math.exp(-r * T) - S * math.exp(-q * T)))
    if market_mid <= intrinsic:
        return 0.01
    
    v_low, v_high = 0.001, 3.0
    for _ in range(50):
        v_mid = (v_low + v_high) / 2.0
        p = bs_price(type_opt, S, K, T, r, q, v_mid)
        if p < market_mid:
            v_low = v_mid
        else:
            v_high = v_mid
    return v_mid

def fetch_options_data(tickers=["SPY", "AAPL", "QQQ", "MSFT"]):
    # Use 13-week Treasury Bill (^IRX) for short-term risk free rate
    irx = yf.Ticker("^IRX")
    irx_hist = irx.history(period="1d")
    if not irx_hist.empty:
        r = irx_hist['Close'].iloc[-1] / 100.0
    else:
        r = 0.037

    os.makedirs('data', exist_ok=True)
    
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        
        # Get dividend yield (convert percentage e.g. 0.35% -> 0.0035)
        info = stock.info
        raw_div = info.get('dividendYield') or info.get('trailingAnnualDividendYield') or 0.0
        if raw_div > 0.03: # if in percent e.g. 0.35% or 1.01%
            q = raw_div / 100.0
        else:
            q = raw_div
        
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
            
        # Pick the first expiration that is at least 30 days out
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
            t = 0.01
            
        calls['Type'] = 'Call'
        puts['Type'] = 'Put'
        
        df = pd.concat([calls, puts])
        
        # Filter valid bid/ask
        df = df[(df['bid'] > 0) & (df['ask'] > 0)]
        
        result = pd.DataFrame()
        result['Ticker'] = ticker
        result['Type'] = df['Type']
        result['Strike'] = df['strike']
        result['Expiry_Years'] = t
        result['Underlying_Price'] = current_price
        result['Bid'] = df['bid']
        result['Ask'] = df['ask']
        result['Risk_Free_Rate'] = r
        result['Dividend_Yield'] = q
        
        # Calibrate clean IV matching market mid price with (r, q)
        calibrated_ivs = []
        for _, row in result.iterrows():
            mid = (row['Bid'] + row['Ask']) / 2.0
            iv = calibrate_iv(row['Type'], current_price, row['Strike'], t, r, q, mid)
            calibrated_ivs.append(iv)
        
        result['Implied_Volatility'] = calibrated_ivs
        
        filename = f"data/{ticker}.csv"
        result.to_csv(filename, index=False)
        print(f"Saved {len(result)} options for {ticker} (q={q:.4f}, r={r:.4f}) to {filename}")

if __name__ == "__main__":
    fetch_options_data(["SPY", "AAPL", "QQQ", "MSFT"])
