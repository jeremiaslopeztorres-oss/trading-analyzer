# 📊 Analizador de Gráficos de Trading (imagen)

Esta aplicación permite subir una **foto o captura de pantalla** de un gráfico de mercado (por ejemplo, TradingView, MetaTrader, Binance, etc.)  
y analiza automáticamente la tendencia principal, devolviendo una **señal estimada**:  
🟢 **COMPRAR** | 🔴 **VENDER** | 🟡 **MANTENER**

---

## ¿Qué incluye este paquete?

- `app.py` — Código principal de la aplicación (Streamlit).
  - Extrae una serie aproximada de precios desde la imagen.
  - Calcula medias móviles (SMA) y pendiente para entregar una señal.
  - Muestra un gráfico de la serie estimada y las SMAs.
- `README.md` — Este archivo con instrucciones.

---

## Cómo usarlo (desde tu iPhone, paso a paso)

1. **Crea una cuenta en GitHub** si no la tienes: https://github.com
2. **Crea un repositorio nuevo** llamado `trading-analyzer`.
   - Desde Safari puedes abrir: `https://github.com/new`
3. **Sube el archivo ZIP** que vas a descargar (o sube `app.py` y `README.md` directamente).
   - En GitHub web (Safari) entra al repo → "Add file" → "Upload files" → selecciona el ZIP o los archivos → Commit changes.
4. **Ve a Streamlit Cloud**: https://share.streamlit.io y entra con tu cuenta de GitHub.
5. Pulsa **New app** → selecciona tu repositorio `trading-analyzer` → Branch `main` → File `app.py` → Deploy.
6. Streamlit instalará las dependencias automáticamente y te dará una URL pública.
7. Abre la URL en Safari y, si quieres, añade a pantalla de inicio (Compartir → Añadir a pantalla de inicio).

---

## Dependencias (Streamlit Cloud las instala automáticamente)

- streamlit
- numpy
- opencv-python-headless
- pillow
- matplotlib
- pandas

---

## Notas importantes

- La app trabaja con una **escala relativa** (0..100) cuando no hay números en los ejes; por eso los valores mostrados son relativos y sirven para detectar tendencias visuales.
- Funciona mejor con imágenes limpias, sin textos superpuestos ni muchas líneas/indicadores.
- **Solo educativo** — no es asesoramiento financiero.

---

## Autor

Creado por **Jeremías López** con ayuda de ChatGPT (GPT-5).
Puerto Rico 🇵🇷