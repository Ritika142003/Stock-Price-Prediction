import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from datetime import datetime


st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.title("📈 Stock Market Prediction Dashboard")
st.markdown("<h5 style='color:#888'>Accurate. Insightful. Attractive.</h5>", unsafe_allow_html=True)


st.sidebar.header("⚙️ Configuration Panel")

stocks = {
    "Nifty50": "^NSEI", "Tesla": "TSLA", "Meta": "META",
    "Bitcoin": "BTC-USD", "Google": "GOOGL",  # ✅ FIXED
    "Apple": "AAPL", "Amazon": "AMZN",
    "Microsoft": "MSFT", "Netflix": "NFLX"
}

stock_selection = st.sidebar.selectbox("Choose Stock", list(stocks.keys()))
stock_symbol = stocks[stock_selection]

start_date = st.sidebar.date_input("Start Date", datetime(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2024, 1, 1))

MODEL_PATH = "GOOGL_Stock_Prediction_Model.keras"

@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH)

model = load_model_cached()

@st.cache_data
def load_stock_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end, progress=False)

        if data.empty:
            return None

        # Fix multi-index issue
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data

    except:
        return None


data = load_stock_data(stock_symbol, start_date, end_date)

if data is None or data.empty:
    st.error("No data found.")
    st.stop()


close = pd.Series(data["Close"]).astype(float).dropna()
high = pd.Series(data["High"]).astype(float).dropna()
low = pd.Series(data["Low"]).astype(float).dropna()


if len(close) < 2:
    st.error("Not enough data to calculate metrics.")
    st.stop()

latest = float(close.iloc[-1])
prev = float(close.iloc[-2])

high_52w = float(high.tail(252).max())
low_52w = float(low.tail(252).min())
daily_change = 0 if prev == 0 else ((latest - prev) / prev) * 100

volatility = float(close.pct_change().dropna().std() * 100)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Latest Price", f"${latest:.2f}", f"{daily_change:+.2f}% Today")
col2.metric("52W High", f"${high_52w:.2f}")
col3.metric("52W Low", f"${low_52w:.2f}")
col4.metric("Volatility", f"{volatility:.2f}%")

data["MA50"] = close.rolling(50).mean()
data["MA100"] = close.rolling(100).mean()
data["MA200"] = close.rolling(200).mean()

ma50 = data["MA50"]
std50 = close.rolling(50).std()

data["BB_up"] = ma50 + (2 * std50)
data["BB_dn"] = ma50 - (2 * std50)

st.subheader("📊 Candlestick Chart")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="Price"
))

fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="MA50"))
fig.add_trace(go.Scatter(x=data.index, y=data["MA200"], name="MA200"))

fig.update_layout(template="plotly_dark", height=600)

st.plotly_chart(fig, use_container_width=True)

st.subheader("🔮 Prediction")

data_test = close[int(len(close) * 0.8):].dropna()

scaler = MinMaxScaler()

if len(data_test) > 100:

    scaled = scaler.fit_transform(data_test.values.reshape(-1, 1))

    x_test, y_test = [], []

    for i in range(100, len(scaled)):
        x_test.append(scaled[i-100:i])
        y_test.append(scaled[i, 0])

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    if len(x_test) > 0:
        pred = model.predict(x_test, verbose=0)

        # ✅ Correct scaling
        pred = scaler.inverse_transform(pred)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(y=y_test.flatten(), name="Actual"))
        fig2.add_trace(go.Scatter(y=pred.flatten(), name="Predicted"))

        fig2.update_layout(template="plotly_dark", height=400)

        st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("Not enough data for prediction")

st.subheader("📥 Download Data")

csv = data.to_csv().encode("utf-8")

st.download_button(
    "Download CSV",
    csv,
    f"{stock_symbol}_data.csv",
    "text/csv"
)