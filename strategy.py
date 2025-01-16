import streamlit as st, requests,pandas as pd,numpy as np
from setup import *
from datetime import datetime
import csv

today_date = datetime.today().strftime('%Y-%m-%d')
# Upstox API credentials
api_key = apiKey
api_secret = secretKey
redirect_uri = rurl
access_token = access_token
headers = {
    'Authorization': f'Bearer {access_token}',
    'Accept': 'application/json'
}

# Function to fetch stock contracts from CSV file
def fetch_stock_contracts_from_csv():
    name_to_key_mapping = []
    with open('stock_contracts.csv', 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            name_to_key_mapping.append({'name': row['name'], 'instrument_key': row['instrument_key']})
    name_to_key = {item['name']: item['instrument_key'] for item in name_to_key_mapping}
    return name_to_key

# Function to fetch intraday data 
def fetch_intraday_candle(instrument_key, interval, access_token):
    url = f"https://api.upstox.com/v2/historical-candle/intraday/{instrument_key}/{interval}"
    payload={}
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
            print(f"KeyError occurred: {e}")
            return None
    else:
        print(f"Error fetching data: {response.status_code} - {response.text}")
        return None  
 
# Function to fetch historical data 
def fetch_historical_candle(instrument_key, interval, to_date, from_date, access_token):
    url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"
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
            print(f"KeyError occurred: {e}")
            return None
    else:
        print(f"Error fetching data: {response.status_code} - {response.text}")
        return None  
 

#get symbol name from instrument key 
def get_instrument_name_by_key(instrument_key):
    with open('stock_contracts.csv', 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)  # Skip the header row
        for row in csvreader:
            if row[0] == instrument_key:  # Assuming the key is in the first column
                return row[2]  # Assuming the name is in the second column
    return None

# Fetch stock data for a given instrument key (both intraday and historical)
def fetch_stock_data(instrument_key, interval, from_date, to_date,access_token):
    df1 = fetch_intraday_candle(instrument_key, interval,access_token)
    df2 = fetch_historical_candle(instrument_key, interval, to_date, from_date,access_token)
    if df1 is not None and df2 is not None:
        historical_data = pd.concat([df1, df2], ignore_index=True)
        return historical_data
    return None

# Strategy calculation functions
def volume_based_trading(df, symbol):
    latest_volume = df['volume'].iloc[0]
    previous_volume = df['volume'].iloc[1]
    latest_close = df['close'].iloc[0]
    previous_close = df['close'].iloc[1]    
    if latest_volume > previous_volume and latest_close > previous_close:
        return f"Volume increasing with price: Consider buying at {latest_close}"
    elif latest_volume > previous_volume and latest_close < previous_close:
        return f"Volume increasing with price drop: Consider selling : {latest_close}"
    elif latest_volume < previous_volume and latest_close > previous_close:
        return f"Volume decreasing with price uptrend: Caution for {symbol}, trend may weaken"
    elif latest_volume < previous_volume and latest_close < previous_close:
        return f"Volume decreasing with price downtrend: Potential reversal for {symbol}"

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
            return f"RSI Strategy result: Buy at {latest_close} (RSI: {latest_rsi})"
        elif latest_rsi > sell_threshold:
            return f"RSI Strategy result: Sell at {latest_close} (RSI: {latest_rsi})"
        else:
            return 'The stock is not overbought or oversold'
        
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
        return f"MACD Strategy result: Buy at {latest_close}"
    elif latest_macd < latest_signal:
        return f"MACD Strategy result: Sell at {latest_close}"
    else:
        return 'No clear MACD signal for trading'
    
def calculate_ichimoku(df):
    df['tenkan_sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
    df['kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    df['senkou_span_b'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
    df['chikou_span'] = df['close'].shift(-26)
    return df

def check_ichimoku_and_trade(df, symbol):
    latest_close = df['close'].iloc[0]
    latest_span_a = df['senkou_span_a'].iloc[-1]
    latest_span_b = df['senkou_span_b'].iloc[-1]
    if latest_close > max(latest_span_a, latest_span_b):
        return f"Ichimoku Strategy result: Buy at {latest_close}"
    elif latest_close < min(latest_span_a, latest_span_b):
        return f"Ichimoku Strategy result: Sell at {latest_close}"
    else:
        return f"No clear Ichimoku signal"

# Streamlit app
st.title("Stock Trading Strategies")

# Select stocks from the CSV
name_to_key = fetch_stock_contracts_from_csv()
selected_names = st.multiselect("Select stocks", options=list(name_to_key.keys()))
instrument_keys = [name_to_key[name] for name in selected_names]

if selected_names:
    st.write("Selected stocks:", selected_names)

    # Set the timeframe and date range
    access_token = access_token
    interval = '30minute'
    from_date = '2023-09-23'
    to_date = today_date

    # Fetch stock data for each selected stock and evaluate strategies
    for instrument_key in instrument_keys:
        historical_data = fetch_stock_data(instrument_key, interval, from_date, to_date,access_token)
        
        if historical_data is not None:
            # Perform analysis
            symbol = get_instrument_name_by_key(instrument_key)
            volume_signal = volume_based_trading(historical_data, symbol)
            rsi_data = calculate_rsi(historical_data)
            rsi_signal = check_rsi(rsi_data, symbol)
            macd_data = calculate_macd(historical_data)
            macd_signal = check_macd_and_trade(macd_data, symbol)
            ichimoku_data = calculate_ichimoku(historical_data)
            ichimoku_signal = check_ichimoku_and_trade(ichimoku_data, symbol)

            # Display individual strategy results
            st.markdown(f"Analysis for **{symbol}**:")
            st.write(volume_signal)
            st.write(rsi_signal)
            st.write(macd_signal)
            st.write(ichimoku_signal)

            # Composite decision logic
            composite_score = 0
            # Add scoring logic based on signals
            if volume_signal and "Buy" in volume_signal or "Buying" in (rsi_signal or '') or "Buying" in (macd_signal or '') or "Buying" in (ichimoku_signal or ''):
                composite_score += 1
            if volume_signal and "Sell" in volume_signal or "Selling" in (rsi_signal or '') or "Selling" in (macd_signal or '') or "Selling" in (ichimoku_signal or ''):
                composite_score -= 1

            # Display the final recommendation
            if composite_score > 0:
                st.markdown(f"**Recommended Action: Buy {symbol}**")
            elif composite_score < 0:
                st.markdown(f"**Recommended Action: Sell {symbol}**")
            else:
                st.markdown(f"**No clear recommendation for {symbol}**")
        else:
            st.markdown(f"**No data available for {symbol}.**")
else:
    st.write("Please select at least one stock to analyze.")

