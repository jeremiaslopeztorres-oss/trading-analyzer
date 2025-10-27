# app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Analizador de Gráficos (imagen)", layout="wide")
st.title("📊 Analizador de Foto del Mercado")
st.write("Sube una captura de pantalla o foto de un gráfico. La app intenta extraer una serie de precios y devuelve: 🟢 COMPRAR / 🔴 VENDER / 🟡 MANTENER. Solo educativo.")

# ---------- Helpers ----------
def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)

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
        # prefer large rectangles but not the whole screen
        if area > best_area and 0.15*w*h < area < 0.95*w*h:
            best_area = area
            best = (x,y,ww,hh)
    return best

def extract_line_series(crop, resample_points=300):
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    # Try both threshold and Canny to detect lines/edges
    edges = cv2.Canny(gray, 60, 150)
    h, w = edges.shape
    pts = []
    for x in range(w):
        ys = np.where(edges[:, x] > 0)[0]
        if ys.size > 0:
            # choose median (robust to thick lines)
            y = int(np.median(ys))
            pts.append((x, y))
    if len(pts) < 8:
        return None, edges
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    # smooth y
    ys = pd.Series(ys).rolling(5, min_periods=1, center=True).mean().to_numpy()
    # resample to fixed length
    target_x = np.linspace(xs.min(), xs.max(), resample_points)
    res_y = np.interp(target_x, xs, ys)
    # invert y (pixel 0 is top) and normalize to 0..100 scale for processing
    res_y = np.interp(res_y, (res_y.min(), res_y.max()), (100, 0))
    return res_y, edges

def compute_signals(prices):
    s = pd.Series(prices)
    sma_short = s.rolling(5, min_periods=1).mean()
    sma_long = s.rolling(20, min_periods=1).mean()
    slope = np.polyfit(np.arange(min(10, len(prices))), prices[-min(10, len(prices)):], 1)[0]
    cur_short = sma_short.iloc[-1]
    cur_long = sma_long.iloc[-1]
    # rules
    if cur_short > cur_long and slope > 0:
        label = "🟢 COMPRAR (tendencia alcista)"
    elif cur_short < cur_long and slope < 0:
        label = "🔴 VENDER (tendencia bajista)"
    else:
        label = "🟡 MANTENER (mercado lateral)"
    meta = {
        "sma_short": float(cur_short),
        "sma_long": float(cur_long),
        "slope": float(slope)
    }
    return label, meta

def detect_pattern(prices):
    diff = np.diff(prices)
    pos = (diff > 0).sum()
    neg = (diff < 0).sum()
    if pos > 0.75 * len(diff):
        return "Tendencia alcista 📈"
    elif neg > 0.75 * len(diff):
        return "Tendencia bajista 📉"
    else:
        return "Rango lateral ↔️"

# ---------- UI ----------
uploaded = st.file_uploader("Sube la imagen del gráfico (PNG/JPG)", type=["png","jpg","jpeg"])
if not uploaded:
    st.info("Sube una captura de pantalla o foto con el gráfico. Preferible fondo claro y la menor cantidad de texto posible.")
else:
    img = load_image(uploaded)
    x,y,wc,hc = detect_plot_area(img)
    crop = img[y:y+hc, x:x+wc]
    st.subheader("Área detectada (puede recortarla si lo deseas)")
    st.image(crop, use_column_width=True)
    # extract series
    series, edges = extract_line_series(crop)
    st.subheader("Procesamiento")
    st.image(edges, caption="Bordes detectados (usado para extraer la línea)", use_column_width=True)
    if series is None:
        st.error("No se pudo extraer una serie clara. Prueba otra imagen más limpia o recorta manualmente antes de subir.")
    else:
        signal, meta = compute_signals(series)
        pattern = detect_pattern(series)
        # Results
        st.markdown("### Resultado del análisis")
        st.success(f"**Señal:** {signal}")
        st.info(f"**Patrón general:** {pattern}")
        st.write("**Detalles técnicos:**", meta)
        # plot recovered series and SMAs
        fig, ax = plt.subplots()
        ax.plot(series, label="Serie estimada", linewidth=2)
        sma5 = pd.Series(series).rolling(5, min_periods=1).mean()
        sma20 = pd.Series(series).rolling(20, min_periods=1).mean()
        ax.plot(sma5, label="SMA 5")
        ax.plot(sma20, label="SMA 20")
        ax.set_title("Serie de precios estimada (escala relativa)")
        ax.set_xlabel("Puntos re-muestreados")
        ax.set_ylabel("Precio (escala relativa)")
        ax.legend()
        st.pyplot(fig)
        st.markdown("> ⚠️ Esta señal es una estimación visual. No es asesoramiento financiero.")