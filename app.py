# app.py (v3) - Analizador avanzado: patrones, soporte/resistencia, SL/TP visual y cálculo según inversión
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Analizador Trading v3", layout="centered")
st.title("📊 Analizador de Gráficos (v3) — Patrones, SL/TP visuales")
st.write("Sube una captura del gráfico; la app detecta patrones, soportes/resistencias y propone SL y TP marcados en la imagen. Solo educativo.")

# ---------- Helpers ----------
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

def extract_line_series(crop, resample_points=400):
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
    # normalize invert to 0..1 relative scale
    res_y = np.interp(res_y, (res_y.min(), res_y.max()), (1.0, 0.0))
    return res_y, edges

def detect_support_resistance(series, bins=5):
    # find local peaks/troughs in series to propose levels
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(series, distance=20)
    troughs, _ = find_peaks(-series, distance=20)
    levels = []
    for p in peaks:
        levels.append(series[p])
    for t in troughs:
        levels.append(series[t])
    if not levels:
        # fallback to quantiles
        qs = np.quantile(series, np.linspace(0.1,0.9,bins))
        return list(qs)
    # cluster similar levels by rounding
    levels = np.array(levels)
    uniq = []
    for val in sorted(levels):
        if not uniq or abs(val-uniq[-1])>0.02:  # threshold in relative units
            uniq.append(val)
    return uniq

def relative_to_price(level_rel, top_price, bottom_price):
    # map relative (0..1) to price numeric (top->bottom)
    return bottom_price + (1-level_rel)*(top_price-bottom_price)

def map_pixels_to_price_on_crop(y_pixel, crop_h, top_price, bottom_price):
    # y_pixel in pixel coords (0..h), map to price
    rel = 1.0 - (y_pixel / float(crop_h))
    return relative_to_price(rel, top_price, bottom_price)

def draw_levels_on_image(pil_img, crop_box, levels_rel, top_price, bottom_price, label_prefix="L"):
    x,y,w,h = crop_box
    draw = ImageDraw.Draw(pil_img)
    font = None
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for i,lev in enumerate(levels_rel):
        # compute pixel y
        py = int((1.0 - lev) * h) + y
        color = (0,200,0) if i==0 else (200,0,0) if i==1 else (255,165,0)
        draw.line([(x,py),(x+w,py)], fill=color, width=2)
        price_val = relative_to_price(lev, top_price, bottom_price)
        txt = f"{label_prefix}{i+1}: {price_val:.6f}"
        draw.rectangle([(x+w-140, py-12),(x+w-2, py+12)], fill=(0,0,0,120))
        draw.text((x+w-136, py-10), txt, fill=(255,255,255), font=font)
    return pil_img

def recommend_levels_from_levels(levels_rel, side, series, entry_price, invest_amount, custom_risk):
    # side: "buy" or "sell" or "neutral"
    # naive logic: if buy -> SL just below nearest support, TP near next resistance
    sorted_levels = sorted(levels_rel)
    if side=="buy":
        support = sorted_levels[0] if sorted_levels else 0.05
        # choose next higher as TP
        tp = sorted_levels[1] if len(sorted_levels)>1 else min(0.98, support + 0.08)
        sl = max(0.0, support - 0.01)
    elif side=="sell":
        resistance = sorted_levels[-1] if sorted_levels else 0.95
        tp = sorted_levels[-2] if len(sorted_levels)>1 else max(0.02, resistance - 0.08)
        sl = min(1.0, resistance + 0.01)
    else:
        # neutral -> set SL/TP based on volatility
        vol = np.std(series)
        sl = 0.5 - vol*1.5
        tp = 0.5 + vol*1.5
    # risk percent
    risk_pct = (custom_risk/100.0) if custom_risk and custom_risk>0 else 0.02
    rr = 2.0
    tp_pct = rr * risk_pct
    return {"sl_rel": sl, "tp_rel": tp, "risk_pct": risk_pct*100, "tp_pct": tp_pct*100}

# ---------- UI ----------
st.sidebar.header("Parámetros de operación")
asset = st.sidebar.selectbox("Activo / Moneda", ["USD (spot)", "XRP", "BTC", "ETH", "Otro (precio por unidad)"])
invest_amount = st.sidebar.number_input("Cantidad a invertir (en USD)", min_value=1.0, value=100.0, step=1.0, format="%.2f")
entry_price = st.sidebar.number_input("Precio actual del activo (opcional)", min_value=0.0, value=0.0, step=0.00000001, format="%.8f")
custom_risk = st.sidebar.number_input("Riesgo (%) (opcional, 0 = automático)", min_value=0.0, value=0.0, step=0.1, format="%.2f")

uploaded = st.file_uploader("Sube la captura del gráfico (PNG/JPG)", type=["png","jpg","jpeg"])
if not uploaded:
    st.info("Sube una captura de tu gráfico. La app intentará detectar niveles y propondrá SL/TP visuales.")
else:
    img_arr, pil_img_orig = load_image(uploaded)
    x,y,wc,hc = detect_plot_area(img_arr)
    crop = img_arr[y:y+hc, x:x+wc]
    st.subheader("Área detectada")
    st.image(crop, use_column_width=True)
    series, edges = extract_line_series(crop)
    st.subheader("Procesamiento (bordes detectados)")
    st.image(edges, use_column_width=True)
    if series is None:
        st.error("No pude extraer una serie clara del gráfico. Prueba con una imagen más limpia o recorta antes de subir.")
    else:
        # detect support/resistance
        levels_rel = detect_support_resistance(series, bins=6)
        # decide side from trend
        diffs = np.diff(series)
        side = "neutral"
        if np.mean(diffs) > 0.001:
            side = "buy"
        elif np.mean(diffs) < -0.001:
            side = "sell"
        # detect simple patterns via peaks
        trend = "Indeterminado"
        try:
            from scipy.signal import find_peaks
        except Exception:
            pass
        # crude trend
        if np.mean(diffs) > 0:
            trend = "Alcista"
        elif np.mean(diffs) < 0:
            trend = "Bajista"
        else:
            trend = "Lateral"
        st.markdown("### Resultado del análisis")
        st.write(f"**Tendencia estimada:** {trend}")
        st.write(f"**Niveles detectados (relativos):** {['{:.3f}'.format(l) for l in levels_rel]}")
        # recommend SL/TP
        rec = recommend_levels_from_levels(levels_rel, side, series, entry_price, invest_amount, custom_risk)
        # map to prices if entry_price provided or show percent
        top_price = entry_price if entry_price>0 else 1.0
        bottom_price = 0.0 if entry_price>0 else 0.0
        # draw on original image
        pil_copy = pil_img_orig.copy()
        pil_with_levels = draw_levels_on_image(pil_copy, (x,y,wc,hc), [rec['sl_rel'], rec['tp_rel']], top_price, bottom_price, label_prefix="R")
        st.subheader("Imagen con SL (rojo) / TP (verde) sugeridos")
        st.image(pil_with_levels, use_column_width=True)
        # show numeric suggestions
        st.markdown("### Recomendación numérica (según tu inversión)")
        if entry_price>0:
            sl_price = relative_to_price(rec['sl_rel'], top_price, bottom_price)
            tp_price = relative_to_price(rec['tp_rel'], top_price, bottom_price)
            st.write(f"Entry price: **{entry_price:.8f}**")
            st.write(f"Stop Loss sugerido: **{sl_price:.8f}** ({rec['risk_pct']:.2f}% riesgo aprox.)")
            st.write(f"Take Profit sugerido: **{tp_price:.8f}** (TP {rec['tp_pct']:.2f}%)")
            risk_amount = invest_amount * (rec['risk_pct']/100.0)
            tp_amount = invest_amount * (rec['tp_pct']/100.0)
            st.write(f"Cantidad que arriesgas aproximada: **${risk_amount:.2f}**")
            st.write(f"Ganancia objetivo aproximada: **${tp_amount:.2f}**")
        else:
            st.write(f"Stop Loss (% relativo): **{rec['sl_rel']:.3f}**  Take Profit (% relativo): **{rec['tp_rel']:.3f}**")
            st.write(f"Riesgo sugerido: **{rec['risk_pct']:.2f}%**  Take Profit: **{rec['tp_pct']:.2f}%**")
            st.write(f"Cantidad que arriesgas aproximada: **${invest_amount * (rec['risk_pct']/100.0):.2f}**")
        st.markdown("> ⚠️ Esta recomendación es una estimación visual basada en la imagen. Valida siempre con datos reales y gestión de riesgo.")