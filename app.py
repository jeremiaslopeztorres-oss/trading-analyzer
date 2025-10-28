# app.py (v3.1) - Interfaz central, señal visible y SL/TP visuales
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Analizador Trading v3.1", layout="centered")
st.title("📊 Analizador de Gráficos (v3.1) — Señal visible + SL/TP")

st.write("Sube una captura del gráfico. Ajusta activo, inversión y riesgo aquí abajo. La app detecta tendencia, propone SL/TP y marca las líneas en la imagen. Solo educativo.")

# --------- Helpers (simplified versions) ---------
def load_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img), img

def detect_plot_area(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape[:2]
    best = (0,0,w,h)
    best_area = 0
    for c in contours:
        x,y,ww,hh = cv2.boundingRect(c)
        area = ww*hh
        if area > best_area and 0.12*w*h < area < 0.95*w*h:
            best_area = area
            best = (x,y,ww,hh)
    return best

def extract_line_series(crop, resample_points=300):
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    h, w = edges.shape
    pts = []
    for x in range(w):
        ys = np.where(edges[:, x] > 0)[0]
        if ys.size > 0:
            y = int(np.median(ys))
            pts.append((x, y))
    if len(pts) < 8:
        return None, edges
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    ys = pd.Series(ys).rolling(5, min_periods=1, center=True).mean().to_numpy()
    target_x = np.linspace(xs.min(), xs.max(), resample_points)
    res_y = np.interp(target_x, xs, ys)
    res_y = np.interp(res_y, (res_y.min(), res_y.max()), (1.0, 0.0))
    return res_y, edges

def detect_support_resistance(series):
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(series, distance=20)
        troughs, _ = find_peaks(-series, distance=20)
    except Exception:
        peaks = np.array([i for i in range(1,len(series)-1) if series[i]>series[i-1] and series[i]>series[i+1]])
        troughs = np.array([i for i in range(1,len(series)-1) if series[i]<series[i-1] and series[i]<series[i+1]])
    levels = []
    for p in peaks: levels.append(series[p])
    for t in troughs: levels.append(series[t])
    if not levels:
        qs = np.quantile(series, np.linspace(0.2,0.8,5))
        return list(qs)
    levels = sorted(levels)
    uniq = []
    for val in levels:
        if not uniq or abs(val-uniq[-1])>0.02:
            uniq.append(val)
    return uniq

def draw_levels_on_image(pil_img, crop_box, levels_rel, top_price, bottom_price):
    x,y,w,h = crop_box
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for i,lev in enumerate(levels_rel):
        py = int((1.0 - lev) * h) + y
        color = (0,200,0) if i==1 else (200,0,0) if i==0 else (255,165,0)
        draw.line([(x,py),(x+w,py)], fill=color, width=3)
        if top_price and bottom_price is not None:
            price_val = bottom_price + (1-lev)*(top_price-bottom_price)
            txt = f"{price_val:.6f}"
        else:
            txt = f"{lev:.3f}"
        draw.rectangle([(x+w-140, py-14),(x+w-2, py+14)], fill=(0,0,0))
        draw.text((x+w-136, py-12), txt, fill=(255,255,255), font=font)
    return pil_img

def recommend_levels(levels_rel, side, series, entry_price, invest_amount, custom_risk):
    sorted_levels = sorted(levels_rel)
    if side=="buy":
        support = sorted_levels[0] if sorted_levels else 0.4
        tp = sorted_levels[1] if len(sorted_levels)>1 else min(0.98, support+0.08)
        sl = max(0.0, support-0.01)
    elif side=="sell":
        resistance = sorted_levels[-1] if sorted_levels else 0.6
        tp = sorted_levels[-2] if len(sorted_levels)>1 else max(0.02, resistance-0.08)
        sl = min(1.0, resistance+0.01)
    else:
        vol = np.std(series)
        sl = max(0.02, 0.5 - vol*1.5)
        tp = min(0.98, 0.5 + vol*1.5)
    risk_pct = (custom_risk/100.0) if custom_risk and custom_risk>0 else 0.02
    tp_pct = 2.0 * risk_pct
    return {"sl_rel": sl, "tp_rel": tp, "risk_pct": risk_pct*100, "tp_pct": tp_pct*100}

# --------- Inputs (center screen for mobile) ---------
st.markdown("### Parámetros de operación (elige aquí)")
col1, col2 = st.columns([1,1])
with col1:
    asset = st.selectbox("Activo / Moneda", ["USD (spot)","XRP","BTC","ETH","Otro (precio por unidad)"])
    invest_amount = st.number_input("Cantidad a invertir (USD)", min_value=1.0, value=100.0, step=1.0, format="%.2f")
with col2:
    entry_price = st.number_input("Precio actual (opcional)", min_value=0.0, value=0.0, step=0.00000001, format="%.8f")
    custom_risk = st.number_input("Riesgo (%) (0 = automático)", min_value=0.0, value=0.0, step=0.1, format="%.2f")

uploaded = st.file_uploader("Sube la captura del gráfico (PNG/JPG)", type=["png","jpg","jpeg"])

if not uploaded:
    st.info("Sube la captura del gráfico para analizar. Ajusta parámetros arriba si quieres.")
else:
    img_arr, pil_img = load_image(uploaded)
    x,y,w,h = detect_plot_area(img_arr)
    crop = img_arr[y:y+h, x:x+w]
    st.subheader("Área detectada")
    st.image(crop, use_column_width=True)
    series, edges = extract_line_series(crop)
    st.subheader("Procesamiento (bordes detectados)")
    st.image(edges, use_column_width=True)
    if series is None:
        st.error("No se pudo extraer la serie; prueba otra imagen más clara.")
    else:
        diffs = np.diff(series)
        mean_diff = np.mean(diffs)
        if mean_diff > 0.001:
            side = "buy"; signal_text = "🟢 SEÑAL: COMPRAR"; signal_type = "buy"
        elif mean_diff < -0.001:
            side = "sell"; signal_text = "🔴 SEÑAL: VENDER"; signal_type = "sell"
        else:
            side = "neutral"; signal_text = "⚪ SEÑAL: MANTENER"; signal_type = "neutral"
        # show big signal prominently
        if signal_type=="buy":
            st.success(signal_text)
        elif signal_type=="sell":
            st.error(signal_text)
        else:
            st.warning(signal_text)
        # detect levels and recommend
        levels_rel = detect_support_resistance(series)
        rec = recommend_levels(levels_rel, side, series, entry_price, invest_amount, custom_risk)
        # draw levels on image (SL red, TP green)
        pil_copy = pil_img.copy()
        pil_with = draw_levels_on_image(pil_copy, (x,y,w,h), [rec['sl_rel'], rec['tp_rel']], entry_price if entry_price>0 else None, 0.0 if entry_price>0 else None)
        st.subheader("Imagen con SL/TP sugeridos")
        st.image(pil_with, use_column_width=True)
        st.markdown("### Recomendación numérica")
        if entry_price>0:
            sl_price = (entry_price * (1 - rec['sl_rel'])) if True else None
            tp_price = (entry_price * (1 + rec['tp_rel'])) if True else None
            st.write(f"Entry price: **{entry_price:.8f}**")
            st.write(f"Stop Loss sugerido (precio): **{sl_price:.8f}**")
            st.write(f"Take Profit sugerido (precio): **{tp_price:.8f}**")
        else:
            st.write(f"Stop Loss (relativo): **{rec['sl_rel']:.3f}**")
            st.write(f"Take Profit (relativo): **{rec['tp_rel']:.3f}**")
        st.write(f"Riesgo sugerido: **{rec['risk_pct']:.2f}%**")
        st.write(f"Cantidad que arriesgas aprox: **${invest_amount * (rec['risk_pct']/100.0):.2f}**")
        st.markdown("> ⚠️ Estimación visual. Valida con datos reales antes de operar.")


# ---------------------------
# NUEVA SECCIÓN: Brokers por país
# ---------------------------

import json
import streamlit as st

st.header("🌍 Encuentra tu Broker Ideal")
st.write("Selecciona tu país para ver los brokers disponibles, sus métodos de pago y plataformas.")

with open("brokers.json", "r", encoding="utf-8") as f:
    brokers = json.load(f)

pais = st.selectbox("Selecciona tu país:", [""] + list(brokers.keys()))

if pais:
    st.subheader(f"Brokers disponibles en {pais}")
    for b in brokers[pais]:
        st.markdown(f"""
        ### {b['nombre']}
        - 💵 **Depósito mínimo:** {b['deposito']}
        - 📊 **Plataformas:** {b['plataformas']}
        - 🪙 **Acepta PayPal:** {b['paypal']}
        - 📈 **Comisiones:** {b['comisiones']}
        - 🔗 [Ir al broker]({b['link']})
        ---
        """)
else:
    st.info("Selecciona un país para ver las opciones disponibles.")
