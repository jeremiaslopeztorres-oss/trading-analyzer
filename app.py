# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from fetcher import fetch_data
from analysis import generate_signals, naive_backtest

st.set_page_config(page_title="Trading Bot Analyzer", layout="wide")

st.title("🔎 Bot de Trading — Analizador de señales (Python)")

with st.sidebar:
    st.header("Configuración")
    market_type = st.selectbox("Tipo de mercado", ['stock','crypto'])
    if market_type == 'stock':
        symbol = st.text_input("Símbolo (yfinance)", value="AAPL")
        period = st.selectbox("Período", ['5d','1mo','3mo','6mo','1y','2y'], index=1)
        interval = st.selectbox("Intervalo", ['1m','5m','15m','30m','60m','90m','1h','1d'], index=7)
        fetch_args = {'period': period, 'interval': interval}
    else:
        exchange_id = st.selectbox("Exchange (ccxt)", ['binance','kraken','ftx','coinbasepro'])
        symbol = st.text_input("Símbolo ccxt (ej. BTC/USDT)", value="BTC/USDT")
        timeframe = st.selectbox("Timeframe", ['1m','5m','15m','30m','1h','4h','1d'], index=4)
        fetch_args = {'exchange_id': exchange_id, 'timeframe': timeframe}

    btn = st.button("Cargar datos y analizar")

if btn:
    try:
        with st.spinner("Descargando datos..."):
            df = fetch_data(symbol, market_type=market_type, **fetch_args)
        st.success(f"Datos cargados: {len(df)} filas — rango {df.index[0]} a {df.index[-1]}")
    except Exception as e:
        st.error(f"Error al descargar datos: {e}")
        st.stop()

    st.sidebar.header("Parámetros del analizador")
    ema_fast = st.sidebar.number_input("EMA rápida", value=12)
    ema_slow = st.sidebar.number_input("EMA lenta", value=26)
    rsi_period = st.sidebar.number_input("RSI periodo", value=14)
    rsi_buy = st.sidebar.number_input("RSI umbral compra", value=30)
    rsi_sell = st.sidebar.number_input("RSI umbral venta", value=70)
    ma_period = st.sidebar.number_input("SMA periodo", value=50)
    params = {'ema_fast': int(ema_fast),'ema_slow': int(ema_slow),'rsi_period': int(rsi_period),
              'rsi_buy': int(rsi_buy),'rsi_sell': int(rsi_sell),'ma_period': int(ma_period)}

    df_sign = generate_signals(df, params=params)

    # Latest signal
    latest = df_sign.iloc[-1]
    st.markdown("### Señal actual")
    if latest['signal'] == 'buy':
        st.success(f"BUY — última vela {df_sign.index[-1].strftime('%Y-%m-%d %H:%M')} — RSI {latest['RSI']:.1f}")
    elif latest['signal'] == 'sell':
        st.error(f"SELL — última vela {df_sign.index[-1].strftime('%Y-%m-%d %H:%M')} — RSI {latest['RSI']:.1f}")
    else:
        st.info(f"NEUTRAL — última vela {df_sign.index[-1].strftime('%Y-%m-%d %H:%M')} — RSI {latest['RSI']:.1f}")

    # Price chart with EMA and signals
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_sign.index, open=df_sign['Open'], high=df_sign['High'], low=df_sign['Low'], close=df_sign['Close'],
        name='OHLC'
    ))
    fig.add_trace(go.Scatter(x=df_sign.index, y=df_sign['EMA_fast'], name=f"EMA {params['ema_fast']}", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df_sign.index, y=df_sign['EMA_slow'], name=f"EMA {params['ema_slow']}", line=dict(width=1)))
    # add buy/sell markers
    buys = df_sign[df_sign['signal']=='buy']
    sells = df_sign[df_sign['signal']=='sell']
    if not buys.empty:
        fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers', marker_symbol='triangle-up', marker_size=12, name='BUY signals', marker=dict(color='green')))
    if not sells.empty:
        fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers', marker_symbol='triangle-down', marker_size=12, name='SELL signals', marker=dict(color='red')))

    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # show indicators table tail
    st.subheader("Últimas columnas (indicadores)")
    st.dataframe(df_sign[['Close','EMA_fast','EMA_slow','SMA','RSI','MACD_line','MACD_signal','MACD_hist','signal']].tail(10))

    # optional naive backtest
    st.subheader("Backtest muy simple (naive)")
    initial_capital = st.number_input("Capital inicial (USD)", value=1000.0)
    bt = naive_backtest(df_sign, capital=float(initial_capital))
    st.write(f"Valor final: ${bt['final_value']:.2f} ({bt['returns_pct']:.2f}%)")
    st.write("Trades (tipo, timestamp, price):")
    st.write(bt['trades'])

    st.info("Recuerda: esto es orientativo — backtests simples no contemplan comisiones, slippage, ni gestión de riesgo.")
