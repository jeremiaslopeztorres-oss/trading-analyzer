# app.py
import streamlit as st
from PIL import Image
import io, re
import numpy as np
import pandas as pd
import yfinance as yf
import pytesseract
import datetime as dt
import requests

st.set_page_config(page_title="Mega Analizador (foto→precio, señal, SL/TP, news, sizing)", layout="wide")
st.title("📸→📈 Mega Analizador de Mercado (imagen) + SL/TP + Noticias + Position Sizing")

# ---------------- Sidebar: configuración ----------------
st.sidebar.header("Configuración del trader")
capital = st.sidebar.number_input("Capital disponible (USD)", value=20.0, step=1.0, format="%.2f")
leverage = st.sidebar.number_input("Apalancamiento", value=500, step=1)
risk_pct = st.sidebar.slider("Riesgo por operación (%)", 0.1, 10.0, 2.0)
newsapi_key = st.sidebar.text_input("NewsAPI key (opcional, mejora resultados)", type="password")
st.sidebar.markdown("---")
st.sidebar.markdown("Nota: si no pones NewsAPI, usaré noticias que entregue yfinance cuando esté disponible.")

# ---------------- Helpers ----------------
SYMBOL_MAPPING = {
    # nombres comunes -> Yahoo tickers
    "XAU": "GC=F",          # Gold futures (aprox para XAUUSD)
    "XAUUSD": "GC=F",
    "GOLD": "GC=F",
    "EURUSD": "EURUSD=X",
    "EUR/USD": "EURUSD=X",
    "GBPJPY": "GBPJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "BTCUSD": "BTC-USD",
    "BTC": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "ETH": "ETH-USD",
    "US30": "^DJI",
    "SPX": "^GSPC",
    "NAS100": "^IXIC"
}

def normalize_text(s: str):
    return re.sub(r'[^A-Za-z0-9\./\- ]', ' ', s).upper()

def find_ticker_in_text(text: str):
    # detectar símbolos comunes en el texto OCR
    txt = normalize_text(text)
    for k in SYMBOL_MAPPING.keys():
        if k in txt:
            return k, SYMBOL_MAPPING[k]
    # también buscar patrones como EURUSD, XAUUSD etc.
    tokens = re.findall(r'[A-Z]{3,6}\/?[A-Z]{0,3}', txt)
    for t in tokens:
        tt = t.replace("/", "")
        if tt in SYMBOL_MAPPING:
            return tt, SYMBOL_MAPPING[tt]
        if t in SYMBOL_MAPPING:
            return t, SYMBOL_MAPPING[t]
    return None, None

def extract_price_from_text(text: str):
    # busca números que parezcan precios (ej. 1.16345 o 1987.34)
    nums = re.findall(r'\d{1,5}[.,]\d{1,6}', text)
    if not nums:
        nums = re.findall(r'\d{1,6}', text)
    # convertimos y tomamos el más "razonable" (ej. si hay varios, preferimos el mayor para metales)
    cleaned = []
    for n in nums:
        n2 = n.replace(',', '.')
        try:
            cleaned.append(float(n2))
        except:
            pass
    if not cleaned:
        return None
    # heurística: el precio más grande si parece metal/índice, o el mediano para FX
    cleaned = sorted(cleaned)
    return cleaned[-1]  # regresamos el mayor como heurística simple

def get_history_and_atr(yticker, period='60d', interval='1d'):
    try:
        tk = yf.Ticker(yticker)
        df = tk.history(period=period, interval=interval)
        if df is None or df.empty:
            return None
        df = df[['Open','High','Low','Close']].dropna()
        # ATR (14)
        high = df['High']; low = df['Low']; close = df['Close']
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=1).mean()
        return df, atr
    except Exception as e:
        return None

def simple_signal_logic(df, atr, timeframe='1h'):
    # Lógica simplificada: tendencia por EMA + último cierre vs EMA + RSI-like quick
    close = df['Close']
    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    last = close.iloc[-1]
    if ema20.iloc[-1] > ema50.iloc[-1] and last > ema20.iloc[-1]:
        return "BUY"
    if ema20.iloc[-1] < ema50.iloc[-1] and last < ema20.iloc[-1]:
        return "SELL"
    return "NEUTRAL"

def suggest_sl_tp(last_price, atr_value, side):
    # sugerir SL y TP basados en ATR
    atr = float(max(atr_value, 0.0001))
    # parámetros: stop = 1.0 * ATR (ajustable según instrumento)
    stop_dist = atr * 1.0
    if side == "BUY":
        sl = last_price - stop_dist
        tp = last_price + stop_dist * 2  # RR = 1:2
    elif side == "SELL":
        sl = last_price + stop_dist
        tp = last_price - stop_dist * 2
    else:
        sl = last_price - atr
        tp = last_price + atr
    return round(float(sl), 6), round(float(tp), 6), round(float(stop_dist), 6)

def pip_value_estimate(instrument, lot=0.01, price=None):
    # Valores aproximados por pip para 0.01 lote — AJUSTA según tu broker.
    instrument = instrument.upper()
    if "XAU" in instrument or "GOLD" in instrument or instrument=="GC=F":
        # asumimos 0.01 lote en XAU da ~ $0.10 por pip (esto depende del broker)
        base = 0.10
    elif "BTC" in instrument:
        base = 0.01
    elif "JPY" in instrument:
        base = 1.0
    else:
        base = 0.10
    return base * (lot / 0.01)

def calc_position_size(capital, risk_pct, entry, stop, instrument):
    if entry == 0 or stop == 0:
        return {"error":"Entrada o stop faltantes"}
    risk_amount = capital * (risk_pct/100.0)
    pip_distance = abs(entry - stop)
    if pip_distance == 0:
        return {"error":"Stop y Entry iguales"}
    lot_01_value = pip_value_estimate(instrument, 0.01)
    pips = pip_distance
    lots = (risk_amount * 0.01) / (pips * lot_01_value)
    return {
        "risk_amount": round(risk_amount, 6),
        "pip_distance": round(pips, 6),
        "recommended_lots": round(lots, 6),
        "note": "Ajusta pip/value según tu broker. Cálculo aproximado."
    }

# ---------------- UI: subir imagen ----------------
st.header("1) Sube tu foto del gráfico (captura de TradingView, app, etc.)")
uploaded_file = st.file_uploader("PNG/JPG", type=["png","jpg","jpeg"])
col1, col2 = st.columns([1,1])

if uploaded_file is None:
    st.info("Sube una imagen para que la app intente detectar instrumento, precio, señal y busque noticias.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)
    # OCR
    st.markdown("**Extrayendo texto de la imagen (OCR)...**")
    try:
        # pytesseract.image_to_string requiere tesseract instalado en el sistema
        ocr_text = pytesseract.image_to_string(image)
    except Exception as e:
        st.error("Error con Tesseract OCR: asegúrate de tener 'tesseract' instalado en el servidor/PC.")
        ocr_text = ""
    st.text_area("Texto OCR", value=ocr_text, height=150)

    # ---------------- Detectar ticker y precio desde OCR ----------------
    detected_label, yahoo_ticker = find_ticker_in_text(ocr_text)
    ocr_price = extract_price_from_text(ocr_text)
    if detected_label is None:
        st.warning("No encontré un ticker claro en la imagen. Escoge manualmente si es incorrecto.")
    else:
        st.success(f"Ticker detectado (heurístico): {detected_label} → {yahoo_ticker}")

    # permitir selección manual / corrección
    manual_instrument = st.selectbox("Confirma o selecciona instrumento:", options=list(SYMBOL_MAPPING.keys()), index=0 if detected_label in SYMBOL_MAPPING else 0)
    chosen_symbol = SYMBOL_MAPPING.get(manual_instrument, SYMBOL_MAPPING.get(detected_label, "EURUSD=X"))
    # permitir usuario cambiar ticker si OCR detectó otro
    chosen_ticker_input = st.text_input("Ticker Yahoo (editable)", value=chosen_symbol)

    # si OCR extrajo precio, mostrarlo y permitir corregir
    if ocr_price:
        st.info(f"Precio detectado en la imagen (heurística): {ocr_price}")
        entry_price = st.number_input("Precio de referencia (entrada)", value=float(ocr_price), format="%.6f")
    else:
        entry_price = st.number_input("Precio de referencia (entrada)", value=0.0, format="%.6f")

    # ---------------- Obtener histórico y calcular señal ----------------
    st.markdown("**Obteniendo datos de mercado (histórico) para calcular ATR / señal...**")
    data_res = get_history_and_atr(chosen_ticker_input, period='90d', interval='1d')
    if data_res is None:
        st.error("No pude obtener histórico con yfinance para ese ticker. Revisa el ticker.")
        df_hist, atr = None, None
    else:
        df_hist, atr_series = data_res
        st.success(f"Histórico obtenido: {len(df_hist)} filas.")
        last_close = float(df_hist['Close'].iloc[-1])
        st.write(f"Último close histórico: {last_close}")
        # señal simple
        signal = simple_signal_logic(df_hist, atr_series)
        st.info(f"Señal técnica automática (EMA20/50 heuristic): **{signal}**")

        # SL / TP basado en ATR
        last_atr = atr_series.iloc[-1]
        sl, tp, stop_dist = suggest_sl_tp(last_close, last_atr, signal)
        st.write("**Sugerencia automática de SL / TP basada en ATR (14):**")
        st.write(f"- ATR(14) último: {round(float(last_atr),6)}")
        st.write(f"- Stop-loss sugerido: {sl}")
        st.write(f"- Take-profit sugerido: {tp}  (RR ≈ 1:2)")

        # Mostrar gráfico histórico simple
        st.markdown("Histórico (últimas 60 velas):")
        st.line_chart(df_hist['Close'].tail(60))

    # ---------------- Noticias ----------------
    st.markdown("**Buscando noticias relevantes del instrumento...**")
    news_items = []
    # 1) intentar news desde yfinance
    try:
        tk = yf.Ticker(chosen_ticker_input)
        yf_news = tk.news
        if yf_news:
            for n in yf_news[:5]:
                news_items.append({"title": n.get('title'), "link": n.get('link'), "provider": n.get('publisher')})
    except Exception:
        pass

    # 2) si NewsAPI key provista, usarla para buscar más resultados
    if newsapi_key:
        try:
            q = manual_instrument if manual_instrument else detected_label
            url = f"https://newsapi.org/v2/everything?q={q}&pageSize=5&apiKey={newsapi_key}"
            r = requests.get(url, timeout=8)
            j = r.json()
            if j.get("articles"):
                for a in j['articles']:
                    news_items.append({"title": a.get('title'), "link": a.get('url'), "provider": a.get('source', {}).get('name')})
        except Exception as e:
            st.warning("No se pudieron obtener noticias de NewsAPI: " + str(e))

    if len(news_items) == 0:
        st.info("No encontré noticias automáticamente. Puedes pegar enlaces o titulares manualmente.")
    else:
        for n in news_items:
            st.markdown(f"- **{n.get('title')}**  — _{n.get('provider')}_  \n  {n.get('link')}")

    # ---------------- Sugerencia final y SL/TP editables ----------------
    st.header("Resultado final (editable)")
    chosen_side = st.radio("Señal final (puedes ajustar):", options=["BUY","SELL","NEUTRAL"], index=0 if signal=="BUY" else (1 if signal=="SELL" else 2))
    sl_input = st.number_input("Stop-loss (editable)", value=float(sl) if 'sl' in locals() else 0.0, format="%.6f")
    tp_input = st.number_input("Take-profit (editable)", value=float(tp) if 'tp' in locals() else 0.0, format="%.6f")
    qty_lots_manual = st.text_input("Lotes manual (opcional) - deja vacío para calcular óptimo", "")

    # ---------------- Calculadora de tamaño (basada en SL) ----------------
    st.header("Calculadora de tamaño de posición (basado en SL elegido)")
    if st.button("Calcular tamaño óptimo (con SL actual)"):
        if entry_price == 0 or sl_input == 0:
            st.error("Necesitamos un precio de entrada y un stop válidos.")
        else:
            res = calc_position_size(capital, risk_pct, entry_price, sl_input, chosen_ticker_input)
            if "error" in res:
                st.error(res["error"])
            else:
                st.success(f"Riesgo permitido: ${res['risk_amount']}")
                st.info(f"Distancia stop-entry (unidades): {res['pip_distance']}")
                st.info(f"Lotes recomendados (aprox): {res['recommended_lots']}")
                margin_est = (entry_price * 100 * res['recommended_lots']) / max(1, leverage)
                st.write(f"Margen aproximado requerido: ${round(margin_est,2)} (aprox)")
                # mostrar PnL objetivo si TP alcanzado
                potential_profit = abs(tp_input - entry_price) * pip_value_estimate(chosen_ticker_input, res['recommended_lots'], entry_price if 'entry_price' in locals() else None)
                st.write(f"Ganancia aprox si TP alcanzado (estimación muy aproximada): ${round(potential_profit,2)}")

    # ---------------- Registro ----------------
    st.header("Registro (simulado) — añadir operación")
    name = st.text_input("Notas (ej. XAU scalp)", "")
    if st.button("Añadir operación al registro"):
        if "trades" not in st.session_state:
            st.session_state["trades"] = []
        st.session_state["trades"].append({
            "instrument": manual_instrument or detected_label,
            "yahoo_ticker": chosen_ticker_input,
            "entry": entry_price,
            "side": chosen_side,
            "sl": sl_input,
            "tp": tp_input,
            "lots": qty_lots_manual or "calc",
            "risk_pct": risk_pct,
            "timestamp": dt.datetime.now().isoformat()
        })
        st.success("Operación añadida al registro (simulado).")

    if "trades" in st.session_state and st.session_state["trades"]:
        st.table(pd.DataFrame(st.session_state["trades"]))
