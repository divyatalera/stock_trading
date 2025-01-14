import streamlit as st
from setup import *
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import time
import schedule

today_date = datetime.today().strftime('%Y-%m-%d')

# Upstox API credentials
api_key = apiKey
api_secret = secretKey
redirect_uri = rurl
access_token = access_token

# Set up the authorization header
headers = {
    'Authorization': f'Bearer {access_token}',
    'Content-Type': 'application/json'
}

# Get all Contracts from NSE
def fetch_stock_contracts():
    url = 'https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz'
    symboldf = pd.read_csv(url)
    symboldf['expiry'] = pd.to_datetime(symboldf['expiry']).apply(lambda x: x.date())
    
    # Filter the instruments for NSE exchange
    df_stock = symboldf[symboldf['exchange'].isin(['NSE_EQ', 'BSE_EQ'])]
    df_stock = df_stock[(df_stock['last_price'] != 0) & (df_stock['last_price'] >= 20)]
    df_stock = df_stock.drop(columns=['expiry', 'strike', 'tick_size','option_type','instrument_type'])
    return df_stock

def CheckBalance():
    url = 'https://api-v2.upstox.com/user/get-funds-and-margin'
    headers = {
        'accept': 'application/json',
        'Api-Version': '2.0',
        'Authorization': f'Bearer {access_token}'
    }
    params = {
        'segment': 'SEC'  # 'Equity'
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        json_response = response.json()
        available_margin = json_response.get('data', {}).get('equity', {}).get('available_margin', 0.0)
        return available_margin
    else:
        return None

# Function to know whether today is a holiday or not
def isholiday():
    url = f"https://api.upstox.com/v2/market/holidays/{today_date}"
    
    headers = {
        'Accept': 'application/json'
    }

    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        holiday_data = response.json()
        if holiday_data.get('isHoliday'):
            return True
        else:
            return False
    else:
        return None
    

def fetch_intraday_candle(instrument_key, interval, access_token):
    url = f"https://api.upstox.com/v2/historical-candle/intraday/{instrument_key}/{interval}"

    payload={}
    headers = {
    'Accept': 'application/json'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    if response.status_code == 200:
        try:
            data = response.json().get('data')
            if not data:
                return None

            candles = data.get('candles', [])
            if not candles:
                return None

            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'other'])
            df = df.drop(columns=['other'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=False)
            return df
        except KeyError as e:
            return None
    else:
        return None  


# Function to fetch historical data 
def fetch_historical_candle(instrument_key, interval, to_date, from_date, access_token):
    url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"
    
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        try:
            data = response.json().get('data')
            if not data:
                return None

            candles = data.get('candles', [])
            if not candles:
                return None

            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'other'])
            df = df.drop(columns=['other'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=False)
            return df
        except KeyError as e:
            return None
    else:
        return None  

#get symbol name from instrument key 
def get_instrument_name_by_key(instrument_key):
    with open('stock_contracts.csv', 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)  # Skip the header row

        # Loop through each row to find the matching key
        for row in csvreader:
            if row[0] == instrument_key:  # Assuming the key is in the first column
                return row[2]  # Assuming the name is in the second column
    return None

def volume_based_trading(df, symbol):
    latest_volume = df['volume'].iloc[0]
    previous_volume = df['volume'].iloc[1]
    latest_close = df['close'].iloc[0]
    previous_close = df['close'].iloc[1]
    
    # If volume is increasing with an upward price move (Buy signal)
    if latest_volume > previous_volume and latest_close > previous_close:
        return f"Volume increasing with price: Consider buying {symbol} at {latest_close}"
        
    # If volume is increasing with a downward price move (Sell signal)
    elif latest_volume > previous_volume and latest_close < previous_close:
        return f"Volume increasing with price drop: Consider selling {symbol} at {latest_close}"
    
    # If volume is decreasing during a price uptrend (Caution)
    elif latest_volume < previous_volume and latest_close > previous_close:
        return f"Volume decreasing with price uptrend: Caution for {symbol}, trend may weaken"
    
    # If volume is decreasing during a price downtrend (Potential reversal)
    elif latest_volume < previous_volume and latest_close < previous_close:
        return f"Volume decreasing with price downtrend: Potential reversal for {symbol}"

# RSI strategy
def calculate_rsi(df, period=14):
    if len(df) < period + 1:
        df['RSI'] = np.nan
        return df

    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    return df

def check_rsi(df, symbol, buy_threshold=30, sell_threshold=70):
    latest_rsi = df['RSI'].iloc[-1].astype(int)
    latest_close = df['close'].iloc[0]        
    if pd.notna(latest_rsi):
        if latest_rsi < buy_threshold:
            return f"RSI Strategy result: Buying {symbol} at {latest_close} (RSI: {latest_rsi})"
        elif latest_rsi > sell_threshold:
            return f"RSI Strategy result: Selling {symbol} at {latest_close} (RSI: {latest_rsi})"
        else:
            return 'The stock is not over bought or over-sold'
            
# MACD strategy
def calculate_ema(df, period, column='close'):
    return df[column].ewm(span=period, adjust=False).mean()

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    df['EMA_fast'] = calculate_ema(df, fast_period)
    df['EMA_slow'] = calculate_ema(df, slow_period)
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    return df

def check_macd_and_trade(df, symbol):
    latest_macd = df['MACD'].iloc[-1]
    latest_signal = df['Signal_Line'].iloc[-1]
    latest_close = df['close'].iloc[0]
    if latest_macd > latest_signal:
        return f"MACD Strategy result: Buying {symbol} at {latest_close}"
    elif latest_macd < latest_signal:
        return f"MACD Strategy result: Selling {symbol} at {latest_close}"
    else:
        return 'No clear MACD signal for trading'
    
#Ichimoku Cloud
# Function to calculate Ichimoku Cloud components
def calculate_ichimoku(df):
    # Calculate Tenkan-sen (Conversion Line) - 9 period high/low average
    df['tenkan_sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2

    # Calculate Kijun-sen (Base Line) - 26 period high/low average
    df['kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2

    # Calculate Senkou Span A (Leading Span A) - average of Tenkan-sen and Kijun-sen, shifted 26 periods ahead
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

    # Calculate Senkou Span B (Leading Span B) - 52 period high/low average, shifted 26 periods ahead
    df['senkou_span_b'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)

    # Calculate Chikou Span (Lagging Span) - current close, shifted 26 periods back
    df['chikou_span'] = df['close'].shift(-26)

    return df

# Function to check for Ichimoku Cloud trading signals
def check_ichimoku_and_trade(df, symbol):
    latest_close = df['close'].iloc[0]
    latest_span_a = df['senkou_span_a'].iloc[-1]
    latest_span_b = df['senkou_span_b'].iloc[-1]
    # Check if price is above the cloud (bullish signal)
    if latest_close > max(latest_span_a, latest_span_b):
        return f"Ichimoku Strategy result: Buying {symbol} at {latest_close}"
    # Check if price is below the cloud (bearish signal)
    elif latest_close < min(latest_span_a, latest_span_b):
        return f"Ichimoku Strategy result: Selling {symbol} at {latest_close}"
    else:
        return f"No clear Ichimoku signal for {symbol}"
  

# Streamlit app
st.title("Stock Trading Strategies")

# Function to run trading strategies during the specified time

def run_trading_strategies():
    stock_contracts = fetch_stock_contracts()
    if stock_contracts is not None:
        stock_contracts.to_csv('stock_contracts.csv', index=False) # Save to CSV
        
    stock_list = ['NSE_EQ|INE481G01011','NSE_EQ|INE040A01034','NSE_EQ|INE387A01021','NSE_EQ|INE758T01015']
    instrument_key = st.selectbox("Select Stock Instrument Key", stock_list)
    instrument_name = get_instrument_name_by_key(instrument_key)

    interval = '30minute'
    from_date = '2023-09-23'
    to_date = today_date

    df1 = fetch_intraday_candle(instrument_key, interval, access_token) 
    df2 = fetch_historical_candle(instrument_key, interval, to_date, from_date, access_token)

    if df1 is not None and df2 is not None:
        historical_data = pd.concat([df1, df2], ignore_index=True)
        
        # Perform and display analysis
        volume_signal = volume_based_trading(historical_data, instrument_name)
        rsi_data = calculate_rsi(historical_data)
        rsi_signal = check_rsi(rsi_data, instrument_name)
        macd_data = calculate_macd(historical_data)
        macd_signal = check_macd_and_trade(macd_data, instrument_name)
        ichimoku_data = calculate_ichimoku(historical_data)
        ichimoku_signal = check_ichimoku_and_trade(ichimoku_data, instrument_name)

        st.write(volume_signal)
        st.write(rsi_signal)
        st.write(macd_signal)
        st.write(ichimoku_signal)
    else:
        st.write("No Historical Data was fetched.")

# Scheduler setup
schedule.every().monday.to(schedule.friday).at("08:45").do(run_trading_strategies)
schedule.every().monday.to(schedule.friday).at("15:45").do(lambda: st.write("Trading session ended for today."))

# Run scheduled tasks
while True:
    schedule.run_pending()
    time.sleep(1)
