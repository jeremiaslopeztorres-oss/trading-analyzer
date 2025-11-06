# fetcher.py
import pandas as pd
import yfinance as yf
import ccxt
from datetime import datetime, timedelta

def fetch_yfinance(symbol: str, period: str = "1mo", interval: str = "1h"):
    """
    Fetch OHLCV with yfinance.
    symbol: e.g. 'AAPL' or 'GC=F' (gold futures) or 'EURUSD=X'
    period: '1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','max'
    interval: '1m','2m','5m','15m','30m','60m','90m','1h','1d','1wk','1mo'
    """
    data = yf.download(tickers=symbol, period=period, interval=interval, progress=False)
    if data.empty:
        raise ValueError(f"No data from yfinance for {symbol} with period={period} interval={interval}")
    data = data.rename(columns={"Adj Close": "Adj_Close"})
    data.index = pd.to_datetime(data.index)
    return data[['Open','High','Low','Close','Volume']].copy()

def fetch_ccxt(exchange_id: str, symbol: str, since_minutes: int = 60*24*30, timeframe: str = "1h", limit: int = 1000):
    """
    Fetch OHLCV from a spot exchange via ccxt.
    exchange_id: 'binance','kraken', etc.
    symbol: e.g. 'BTC/USDT'
    since_minutes: how far back (in minutes) - used to compute since timestamp
    timeframe: '1m','5m','15m','1h','4h','1d',...
    limit: max bars to request
    """
    exchange_cls = getattr(ccxt, exchange_id)
    exchange = exchange_cls({'enableRateLimit': True})
    now = exchange.milliseconds()
    since = now - since_minutes * 60 * 1000
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    if not ohlcv:
        raise ValueError("No OHLCV returned from exchange")
    df = pd.DataFrame(ohlcv, columns=['timestamp','Open','High','Low','Close','Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    return df

def fetch_data(symbol: str, market_type: str = 'stock', **kwargs):
    """
    Generic fetcher. market_type: 'stock' or 'crypto'
    For stock -> uses yfinance symbol.
    For crypto -> expects kwargs['exchange_id'] and symbol in ccxt format like 'BTC/USDT'
    """
    if market_type == 'stock':
        return fetch_yfinance(symbol, period=kwargs.get('period','1mo'), interval=kwargs.get('interval','1h'))
    elif market_type == 'crypto':
        exchange_id = kwargs.get('exchange_id','binance')
        return fetch_ccxt(exchange_id, symbol, since_minutes=kwargs.get('since_minutes',60*24*90),
                          timeframe=kwargs.get('timeframe','1h'), limit=kwargs.get('limit',1000))
    else:
        raise ValueError("market_type must be 'stock' or 'crypto'")
