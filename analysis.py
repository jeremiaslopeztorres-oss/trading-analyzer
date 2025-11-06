# analysis.py
import pandas as pd
import numpy as np

def sma(series: pd.Series, period: int):
    return series.rolling(period).mean()

def ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    # Use Wilder's method once we have initial average
    # convert to RSI
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # neutral until enough data

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def generate_signals(df: pd.DataFrame, params=None):
    """
    Returns df with indicators and a final 'signal' column: 'buy'/'sell'/'neutral'
    params: dict to customize indicator periods and thresholds
    """
    if params is None:
        params = {}
    fast = params.get('ema_fast', 12)
    slow = params.get('ema_slow', 26)
    rsi_period = params.get('rsi_period', 14)
    rsi_buy = params.get('rsi_buy', 30)
    rsi_sell = params.get('rsi_sell', 70)
    ma_period = params.get('ma_period', 50)

    close = df['Close'].astype(float)

    df = df.copy()
    df['EMA_fast'] = ema(close, fast)
    df['EMA_slow'] = ema(close, slow)
    df['SMA'] = sma(close, ma_period)
    df['RSI'] = rsi(close, rsi_period)
    df['MACD_line'], df['MACD_signal'], df['MACD_hist'] = macd(close, fast, slow, params.get('macd_signal',9))

    # simple rule-based signal
    def row_signal(row):
        # EMA crossover
        ema_cross = None
        if row['EMA_fast'] > row['EMA_slow']:
            ema_cross = 'bull'
        elif row['EMA_fast'] < row['EMA_slow']:
            ema_cross = 'bear'
        else:
            ema_cross = 'flat'

        # RSI zone
        if row['RSI'] < rsi_buy:
            rsi_zone = 'oversold'
        elif row['RSI'] > rsi_sell:
            rsi_zone = 'overbought'
        else:
            rsi_zone = 'neutral'

        # MACD histogram direction
        macd_hist = row['MACD_hist']
        if macd_hist > 0:
            macd_dir = 'bull'
        elif macd_hist < 0:
            macd_dir = 'bear'
        else:
            macd_dir = 'flat'

        # Combine rules: conservative approach
        buy_score = 0
        sell_score = 0

        if ema_cross == 'bull': buy_score += 1
        if macd_dir == 'bull': buy_score += 1
        if rsi_zone == 'oversold': buy_score += 1

        if ema_cross == 'bear': sell_score += 1
        if macd_dir == 'bear': sell_score += 1
        if rsi_zone == 'overbought': sell_score += 1

        if buy_score >= 2 and buy_score > sell_score:
            return 'buy'
        elif sell_score >= 2 and sell_score > buy_score:
            return 'sell'
        else:
            return 'neutral'

    df['signal'] = df.apply(row_signal, axis=1)

    return df

# Simple backtest helper (very naive)
def naive_backtest(df: pd.DataFrame, capital=1000.0):
    """
    Naive backtest: take 1 unit when 'buy' (at next open), sell when 'sell' (next open).
    This is illustrative only.
    """
    df = df.copy()
    position = 0
    cash = capital
    trades = []
    for i in range(1, len(df)):
        today = df.iloc[i]
        prev = df.iloc[i-1]
        # enter on prev signal at today's Open
        if prev['signal'] == 'buy' and position == 0:
            price = today['Open']
            position = cash / price  # all-in units
            cash = 0
            trades.append(('buy', df.index[i], price))
        elif prev['signal'] == 'sell' and position > 0:
            price = today['Open']
            cash = position * price
            position = 0
            trades.append(('sell', df.index[i], price))
    # close at last close
    final_value = cash + position * df['Close'].iloc[-1]
    return {'final_value': final_value, 'returns_pct': (final_value - capital)/capital*100, 'trades': trades}
