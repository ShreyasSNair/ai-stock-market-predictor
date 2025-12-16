import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("dark_background")
import os
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def apply_tradingview_theme():
    st.markdown(
        """
        <style>
        :root {
            --bg-main: #020617;
            --bg-panel: #020617;
            --bg-panel-soft: #020617;
            --accent: #10b981;
            --accent-soft: rgba(16, 185, 129, 0.16);
            --accent-warn: #f97316;
            --accent-sell: #ef4444;
            --text-main: #e5e7eb;
            --text-dim: #9ca3af;
            --border-subtle: rgba(148, 163, 184, 0.35);
            --radius-xl: 16px;
            --shadow-soft: 0 18px 45px rgba(15, 23, 42, 0.95);
        }

        /* App background */
        .stApp {
            background: radial-gradient(circle at 0% 0%, #1f2937 0, #020617 50%, #020617 100%);
            color: var(--text-main);
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
                         "Segoe UI", sans-serif;
        }

        /* Remove default top padding */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }

        /* Title styling */
        h1 {
            font-weight: 700 !important;
            letter-spacing: 0.04em;
        }
        h1 span.tv-badge {
            font-size: 0.8rem;
            padding: 0.12rem 0.6rem;
            margin-left: 0.6rem;
            border-radius: 999px;
            background: rgba(16, 185, 129, 0.12);
            color: #6ee7b7;
            border: 1px solid rgba(16,185,129,0.45);
            text-transform: uppercase;
        }

        /* Cards (we'll reuse for most containers) */
        .tv-card {
            background: radial-gradient(circle at 0 0, rgba(148, 163, 184, 0.35), transparent 45%),
                        linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(15, 23, 42, 0.98));
            border-radius: var(--radius-xl);
            border: 1px solid var(--border-subtle);
            box-shadow: var(--shadow-soft);
            padding: 1.1rem 1.25rem;
            margin-bottom: 1.2rem;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border-radius: 999px;
            padding: 0.35rem 0.9rem;
            font-size: 0.82rem;
            color: var(--text-dim);
            border: 1px solid transparent;
        }
        .stTabs [aria-selected="true"] {
            background: var(--accent-soft) !important;
            border-color: rgba(16, 185, 129, 0.7) !important;
            color: #a7f3d0 !important;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #10b981, #22c55e);
            color: #020617;
            border-radius: 999px;
            border: none;
            padding: 0.4rem 1.1rem;
            font-weight: 600;
            box-shadow: 0 12px 30px rgba(34, 197, 94, 0.25);
        }
        .stButton > button:hover {
            filter: brightness(1.05);
            transform: translateY(-1px);
        }

        /* Slider */
        div[data-baseweb="slider"] > div {
            background: rgba(15, 23, 42, 0.9);
        }
        div[data-baseweb="slider"] [data-testid="stTickBar"] {
            background: transparent;
        }

        /* Metrics / numbers */
        .metric-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: #e5e7eb;
        }
        .metric-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--text-dim);
        }

        /* Dataframes */
        .stDataFrame, .stTable {
            border-radius: var(--radius-xl);
            overflow: hidden;
            border: 1px solid rgba(55, 65, 81, 0.85);
        }
        .stDataFrame table {
            color: var(--text-main);
            background-color: #020617;
        }

        /* Chart padding */
        .stPlotlyChart, .st matplotlib-chart, .stPyplot {
            border-radius: var(--radius-xl) !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617, #020617);
            border-right: 1px solid rgba(31, 41, 55, 0.95);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.set_page_config(layout="wide")
apply_tradingview_theme()  

st.markdown(
    """
    <h1 style="font-weight:700; letter-spacing:0.5px;">
         STOCK MARKET PREDICTOR
        <span class="tv-badge">Pro LSTM</span>
    </h1>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# 📌 FILE FOR SELF-LEARNING
# ---------------------------
ERROR_MEMORY_FILE = "model_error_memory.json"

def load_error_memory():
    if os.path.exists(ERROR_MEMORY_FILE):
        try:
            with open(ERROR_MEMORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {"total_evals": 0, "avg_mae": None}
    else:
        return {"total_evals": 0, "avg_mae": None}

def update_error_memory(new_mae):
    mem = load_error_memory()
    n = mem.get("total_evals", 0)
    avg = mem.get("avg_mae", None)

    if avg is None:
        new_avg = new_mae
    else:
        new_avg = (avg * n + new_mae) / (n + 1)

    mem["total_evals"] = n + 1
    mem["avg_mae"] = float(new_avg)

    with open(ERROR_MEMORY_FILE, "w") as f:
        json.dump(mem, f)


# ---------------------------
# 📌 INDICATOR FUNCTIONS
# ---------------------------
def calc_bollinger(close, window=20):
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    return sma, upper, lower

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - macd_signal
    return macd, macd_signal, hist

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def prepare_lstm_data(scaled_series, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(scaled_series)):
        X.append(scaled_series[i-time_steps:i, 0])
        y.append(scaled_series[i, 0])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def is_bullish_engulfing(df, i):
    if i < 1: return False
    prev = df.iloc[i-1]
    curr = df.iloc[i]
    return (prev["Close"] < prev["Open"] and 
            curr["Close"] > curr["Open"] and 
            curr["Close"] > prev["Open"] and 
            curr["Open"] < prev["Close"])

def is_bearish_engulfing(df, i):
    if i < 1: return False
    prev = df.iloc[i-1]
    curr = df.iloc[i]
    return (prev["Close"] > prev["Open"] and 
            curr["Close"] < curr["Open"] and 
            curr["Open"] > prev["Close"] and 
            curr["Close"] < prev["Open"])

def is_hammer(df, i):
    curr = df.iloc[i]
    body = abs(curr["Close"] - curr["Open"])
    lower = curr["Open"] - curr["Low"] if curr["Open"] < curr["Close"] else curr["Close"] - curr["Low"]
    upper = curr["High"] - curr["Close"] if curr["Close"] > curr["Open"] else curr["High"] - curr["Open"]
    return lower > 2 * body and upper < body

def is_shooting_star(df, i):
    curr = df.iloc[i]
    body = abs(curr["Close"] - curr["Open"])
    upper = curr["High"] - curr["Open"] if curr["Open"] > curr["Close"] else curr["High"] - curr["Close"]
    lower = curr["Open"] - curr["Low"] if curr["Open"] < curr["Close"] else curr["Close"] - curr["Low"]
    return upper > 2 * body and lower < body

def is_morning_star(df, i):
    if i < 2: return False
    c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
    return (c1["Close"] < c1["Open"] and
            abs(c2["Close"] - c2["Open"]) < (c1["Open"] - c1["Close"]) * 0.3 and
            c3["Close"] > c3["Open"] and
            c3["Close"] > ((c1["Open"] + c1["Close"]) / 2))

def is_evening_star(df, i):
    if i < 2: return False
    c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
    return (c1["Close"] > c1["Open"] and
            abs(c2["Close"] - c2["Open"]) < (c1["Close"] - c1["Open"]) * 0.3 and
            c3["Close"] < c3["Open"] and
            c3["Close"] < ((c1["Open"] + c1["Close"]) / 2))

def generate_ai_reasoning(df):
    reasoning = []

    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    signal = df["Signal"].iloc[-1]
    sma20 = df["SMA20"].iloc[-1]
    sma50 = df["SMA50"].iloc[-1]
    adx = df["ADX"].iloc[-1]
    atr = df["ATR"].iloc[-1]

    # --- NEW: support & resistance ---
    support, resistance = calc_support_resistance(df["Close"])
    last_price = df["Close"].iloc[-1]

    # RSI
    if rsi < 30:
        reasoning.append("RSI is below 30 → Oversold (possible upward reversal).")
    elif rsi > 70:
        reasoning.append("RSI is above 70 → Overbought (possible downward reversal).")
    else:
        reasoning.append("RSI is in neutral zone → No strong momentum signal.")

    # MACD
    if macd > signal:
        reasoning.append("MACD is above Signal → Bullish momentum (buyers stronger).")
    else:
        reasoning.append("MACD is below Signal → Bearish momentum (sellers stronger).")

    # Moving averages
    if sma20 > sma50:
        reasoning.append("SMA20 is above SMA50 → Short-term uptrend vs long-term.")
    else:
        reasoning.append("SMA20 is below SMA50 → Short-term weakness vs long-term.")

    # Trend strength
    if adx > 25:
        reasoning.append("ADX > 25 → Trend is strong (moves are more reliable).")
    else:
        reasoning.append("ADX < 25 → Trend is weak / sideways (signals less reliable).")

    # Volatility
    if atr > df["ATR"].rolling(100).mean().iloc[-1]:
        reasoning.append("ATR is high → Market volatility is high (higher risk).")
    else:
        reasoning.append("ATR is normal → Volatility is within usual range.")

    # --- NEW: price vs support/resistance ---
    reasoning.append(
        f"Recent Support ≈ {support:.2f}, Resistance ≈ {resistance:.2f}, Current Price ≈ {last_price:.2f}."
    )

    if last_price <= support * 1.02:
        reasoning.append("Price is very close to SUPPORT → Good zone to accumulate / watch for bounce.")
    elif last_price >= resistance * 0.98:
        reasoning.append("Price is very close to RESISTANCE → Risk of pullback / profit-taking zone.")
    else:
        reasoning.append("Price is between support and resistance → Mid-range, no extreme level hit.")

    return reasoning

    if rsi < 30:
        reasoning.append("RSI is below 30 → Oversold (Possible Uptrend Reversal)")
    elif rsi > 70:
        reasoning.append("RSI is above 70 → Overbought (Possible Downtrend Reversal)")
    else:
        reasoning.append("RSI in neutral zone → No strong momentum")

    if macd > signal:
        reasoning.append("MACD crossover is bullish → Buying pressure increasing")
    else:
        reasoning.append("MACD crossover is bearish → Selling pressure increasing")

    if sma20 > sma50:
        reasoning.append("SMA20 above SMA50 → Uptrend confirmation")
    else:
        reasoning.append("SMA20 below SMA50 → Downtrend confirmation")

    if adx > 25:
        reasoning.append("ADX above 25 → Strong trend detected")
    else:
        reasoning.append("ADX below 25 → Weak or sideways trend")

    if atr > df['ATR'].rolling(100).mean().iloc[-1]:
        reasoning.append("High volatility → Risky market conditions")
    else:
        reasoning.append("Low volatility → Stable market")

    return reasoning


def final_ai_decision(df):
    reasons = []
    score_map = []

    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    signal = df["Signal"].iloc[-1]
    sma20 = df["SMA20"].iloc[-1]
    sma50 = df["SMA50"].iloc[-1]
    adx = df["ADX"].iloc[-1]

    support, resistance = calc_support_resistance(df["Close"])
    price = df["Close"].iloc[-1]

    buy_score = 0
    sell_score = 0

    # ---------- RSI ----------
    if rsi < 30:
        buy_score += 2
        reasons.append("RSI is oversold (<30)")
        score_map.append(("RSI Oversold", "BUY"))
    elif rsi > 70:
        sell_score += 2
        reasons.append("RSI is overbought (>70)")
        score_map.append(("RSI Overbought", "SELL"))

    # ---------- MACD ----------
    if macd > signal:
        buy_score += 1
        reasons.append("MACD bullish crossover")
        score_map.append(("MACD Bullish", "BUY"))
    else:
        sell_score += 1
        reasons.append("MACD bearish crossover")
        score_map.append(("MACD Bearish", "SELL"))

    # ---------- Moving Average ----------
    if sma20 > sma50:
        buy_score += 1
        reasons.append("Short-term trend above long-term (SMA20 > SMA50)")
        score_map.append(("Trend Up", "BUY"))
    else:
        sell_score += 1
        reasons.append("Short-term trend below long-term (SMA20 < SMA50)")
        score_map.append(("Trend Down", "SELL"))

    # ---------- ADX ----------
    if adx > 25:
        if buy_score > sell_score:
            buy_score += 1
            reasons.append("Strong trend confirmed (ADX > 25)")
        else:
            sell_score += 1
            reasons.append("Strong downtrend confirmed (ADX > 25)")

    # ---------- Support / Resistance ----------
    if price <= support * 1.02:
        buy_score += 1
        reasons.append("Price near strong support")
        score_map.append(("Near Support", "BUY"))
    elif price >= resistance * 0.98:
        sell_score += 1
        reasons.append("Price near strong resistance")
        score_map.append(("Near Resistance", "SELL"))

    # ---------- Final Decision ----------
    if buy_score > sell_score:
        decision = "BUY"
    elif sell_score > buy_score:
        decision = "SELL"
    else:
        decision = "HOLD"

    confidence = int(
        (max(buy_score, sell_score) / (buy_score + sell_score + 1e-6)) * 100
    )

    # ---------- Human Verdict ----------
    if decision == "BUY":
        verdict = "Bullish setup — buyers have control"
    elif decision == "SELL":
        verdict = "Bearish setup — sellers dominating"
    else:
        verdict = "No clear edge — wait & observe"

    return decision, confidence, reasons, verdict

def calculate_trendlines(df, lookback=100):
    data = df.tail(lookback).copy()

    data["Swing_High"] = data["High"][(data["High"] > data["High"].shift(1)) &
                                      (data["High"] > data["High"].shift(-1))]
    data["Swing_Low"] = data["Low"][(data["Low"] < data["Low"].shift(1)) &
                                    (data["Low"] < data["Low"].shift(-1))]

    highs = data.dropna(subset=["Swing_High"]).tail(5)
    lows = data.dropna(subset=["Swing_Low"]).tail(5)

    if len(highs) < 2 or len(lows) < 2:
        return None, None, None, "Not enough swing points detected"

    x_high = np.arange(len(highs))
    x_low = np.arange(len(lows))

    coef_high = np.polyfit(x_high, highs["Swing_High"].values, 1)
    coef_low = np.polyfit(x_low, lows["Swing_Low"].values, 1)

    trend_high = np.poly1d(coef_high)
    trend_low = np.poly1d(coef_low)

    if coef_low[0] > 0 and coef_high[0] > 0:
        trend_type = "UPTREND 📈"
    elif coef_low[0] < 0 and coef_high[0] < 0:
        trend_type = "DOWNTREND 📉"
    else:
        trend_type = "SIDEWAYS ➡"

    return highs, lows, (trend_high, trend_low), trend_type

# =======================
# ⭐ ADX Calculation
# =======================
def calc_adx(df, period=14):
    df = df.copy()
    
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
    
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)

    df["+DM"] = np.where((df["High"] - df["High"].shift(1)) > (df["Low"].shift(1) - df["Low"]), 
                         df["High"] - df["High"].shift(1), 0)

    df["-DM"] = np.where((df["Low"].shift(1) - df["Low"]) > (df["High"] - df["High"].shift(1)), 
                         df["Low"].shift(1) - df["Low"], 0)

    TRn = df["TR"].rolling(period).sum()
    plusDM = df["+DM"].rolling(period).sum()
    minusDM = df["-DM"].rolling(period).sum()

    plusDI = 100 * (plusDM / TRn)
    minusDI = 100 * (minusDM / TRn)

    DX = 100 * abs((plusDI - minusDI) / (plusDI + minusDI))
    ADX = DX.rolling(period).mean()

    return plusDI, minusDI, ADX


# =======================
# ⭐ ATR Calculation
# =======================
def calc_atr(df, period=14):
    df = df.copy()
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
    
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    ATR = df["TR"].rolling(period).mean()
    
    return ATR

# =======================
# ⭐ Support & Resistance (recent window)
# =======================
def calc_support_resistance(close, window=60):
    """
    Very simple support & resistance:
    - Look at last window closing prices
    - Support  = recent minimum
    - Resistance = recent maximum
    """
    recent = close.tail(window)
    support = recent.min()
    resistance = recent.max()
    return float(support), float(resistance)


# ---------------------------
# 📌 USER INPUTS
# ---------------------------
with st.container():
    st.markdown('<div class="tv-card">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2.3, 1.2, 0.9])

    with col1:
        stocks_input = st.text_input(
        "Enter NSE Stock Symbols (comma separated)",
        "RELIANCE.NS, TCS.NS, INFY.NS"
    )
    stock_list = [s.strip().upper() for s in stocks_input.split(",")]
    stock = stock_list[0]   # primary stock for deep analysis
    with col2:
        days = st.slider("Days to Predict", 1, 60, 30)
    with col3:
        st.write("")  # spacing
        run_btn = st.button("Run Analysis")

    st.markdown('</div>', unsafe_allow_html=True)

def portfolio_ranker(stock_list):
    results = []

    for symbol in stock_list:
        try:
            data = yf.download(symbol, period="3y", progress=False)
            if data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [c[0] for c in data.columns]

            df = data.copy()
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df = df.dropna(subset=["Close"]).reset_index()

            # Indicators
            df["RSI"] = calc_rsi(df["Close"])
            df["MACD"], df["Signal"], _ = calc_macd(df["Close"])
            df["SMA20"] = df["Close"].rolling(20).mean()
            df["SMA50"] = df["Close"].rolling(50).mean()
            _, _, df["ADX"] = calc_adx(df)
            df["ATR"] = calc_atr(df)

            decision, confidence, _, verdict = final_ai_decision(df)

            score = confidence
            if decision == "BUY":
                score += 20
            elif decision == "SELL":
                score -= 20

            results.append({
                "Stock": symbol,
                "Decision": decision,
                "Confidence (%)": confidence,
                "Score": score,
                "Verdict": verdict
            })

        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values("Score", ascending=False)

if run_btn:
    st.subheader("📊 Portfolio Opportunity Ranking")

    portfolio_df = portfolio_ranker(stock_list)

    if portfolio_df.empty:
        st.error("No valid stocks found in portfolio.")
        st.stop()

    st.dataframe(portfolio_df)
    st.success(f"🏆 Best Stock Today: {portfolio_df.iloc[0]['Stock']}")
    st.info(f"Downloading data for {stock} ...")

    # ---------------------------
    # 📌 DOWNLOAD & CLEAN DATA
    # ---------------------------
    data = yf.download(stock, period="5y")
    df = data.copy()

    if df.empty:
        st.error("❌ Invalid stock symbol or no data found.")
        st.stop()

    # Fix MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # Ensure Close is numeric
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"]).reset_index()  # now has 'Date' column

    st.success("✅ Data Loaded Successfully!")

    close_prices = df["Close"].values.reshape(-1, 1)

    # ---------------------------
    # 📌 SCALE DATA & PREPARE LSTM INPUT
    # ---------------------------
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    X, y = prepare_lstm_data(scaled_data, time_steps=60)

    if len(X) == 0:
        st.error("❌ Not enough data to train LSTM model.")
        st.stop()

    st.info("Training LSTM model... ⏳")

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=3, batch_size=32, verbose=0)

    st.success("✅ LSTM Model Trained!")

    # ---------------------------
    # 📌 FUTURE PREDICTION
    # ---------------------------
    last_60 = scaled_data[-60:]
    preds_scaled = []

    for _ in range(days):
        pred = model.predict(last_60.reshape(1, 60, 1), verbose=0)
        preds_scaled.append(pred[0][0])
        last_60 = np.append(last_60[1:], pred)

    future_prices = scaler.inverse_transform(
        np.array(preds_scaled).reshape(-1, 1)
    ).flatten()

    future_dates = pd.date_range(
        start=df["Date"].iloc[-1] + pd.Timedelta(days=1),
        periods=days
    )

    # ---------------------------
    # 📌 IN-SAMPLE PREDICTIONS (FOR ACCURACY)
    # ---------------------------
    # Use model to predict on all training sequences (X)
    y_pred_scaled = model.predict(X, verbose=0)
    y_true_scaled = y.reshape(-1, 1)

    y_true = scaler.inverse_transform(y_true_scaled).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

    # Align dates: first prediction corresponds to index 60 in original df
    pred_dates = df["Date"].iloc[60:]

    # Take last N points for accuracy display
    N = min(50, len(y_true))
    y_true_last = y_true[-N:]
    y_pred_last = y_pred[-N:]
    dates_last = pred_dates.iloc[-N:]

    errors = np.abs(y_true_last - y_pred_last)
    mae = float(np.mean(errors))

    # Update self-learning memory
    update_error_memory(mae)
    memory = load_error_memory()

    # ---------------------------
    # 🔧 PRE-COMPUTE ALL INDICATORS FOR ALL TABS + FINAL AI SYSTEM
    # ---------------------------

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["RSI"] = calc_rsi(df["Close"])
    df["BB_MID"], df["BB_UPPER"], df["BB_LOWER"] = calc_bollinger(df["Close"])

    # MACD
    df["MACD"], df["Signal"], df["Hist"] = calc_macd(df["Close"])

    # ADX
    plusDI, minusDI, ADX = calc_adx(df)
    df["PlusDI"] = plusDI
    df["MinusDI"] = minusDI
    df["ADX"] = ADX

    # ATR
    df["ATR"] = calc_atr(df)

    # ---------------------------
    # 📌 TABS
    # ---------------------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14 = st.tabs([
    "📈 Prediction",
    "📊 Indicators",
    "🟢 MACD",
    "🎯 Accuracy",
    "💡 Buy/Sell",
    "🤖 Final Decision",
    "📌 Support & Resistance",
    "📉 Trendlines",
    "🕯 Candle Patterns",
    "📊 Strategy Optimization",
    "🧠 AI Insights",
    "🔔 Live Alerts",
    "🎯 AI Classifier",
    "🧪 Backtesting"
    ])
    # ---------------------------
    # TAB 1 → PREDICTION
    # ---------------------------
    with tab1:
        st.subheader(f"{stock} — Next {days} Days Prediction")

        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(df["Date"], df["Close"], label="Past Prices", linewidth=1)
        ax1.plot(future_dates, future_prices, label="Future Prediction", color="orange")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price (INR)")
        ax1.legend()
        st.pyplot(fig1)

        predicted_table = pd.DataFrame({
            "Date": future_dates.strftime("%Y-%m-%d"),
            "Predicted Price (INR)": np.round(future_prices, 2)
        })
        st.write("### 📄 Predicted Prices Table")
        st.dataframe(predicted_table)

    # ---------------------------
    # TAB 2 → INDICATORS
    # ---------------------------
    with tab2:
        st.subheader("📊 Technical Indicators")

        close = df["Close"]

        # Moving averages
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        ema20 = close.ewm(span=20).mean()

        # RSI
        rsi = calc_rsi(close, period=14)

        # Bollinger Bands
        bb_mid, bb_upper, bb_lower = calc_bollinger(close, window=20)

        
        st.subheader("📈 ADX — Trend Strength Indicator")

        plusDI, minusDI, ADX = calc_adx(df)

        fig_adx, ax_adx = plt.subplots(figsize=(12, 3))
        ax_adx.plot(df["Date"], plusDI, label="+DI", color="green")
        ax_adx.plot(df["Date"], minusDI, label="-DI", color="red")
        ax_adx.plot(df["Date"], ADX, label="ADX", color="blue")
        ax_adx.axhline(25, color="yellow", linestyle="--", linewidth=1)
        ax_adx.set_title("ADX (Trend Strength)")
        ax_adx.legend()
        st.pyplot(fig_adx)

        # Interpretation Message
        if ADX.iloc[-1] > 25:
            st.success("Trend Strength: Strong Trend (ADX > 25)")
        else:
            st.info("Trend Strength: Weak / Sideways Trend (ADX < 25)")


    # =======================
    # 📌 ATR Plot
    # =======================
        st.subheader("📉 ATR — Volatility Indicator")

        ATR = calc_atr(df)

        fig_atr, ax_atr = plt.subplots(figsize=(12, 3))
        ax_atr.plot(df["Date"], ATR, color="orange", label="ATR")
        ax_atr.set_title("ATR (Volatility Index)")
        ax_atr.legend()
        st.pyplot(fig_atr)

    # Interpretation
        if ATR.iloc[-1] > ATR.mean():
            st.warning("Volatility: High → Market Risky Today")
        else:
            st.success("Volatility: Normal → Market Stable")

        # Bollinger + MAs
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(df["Date"], close, label="Close", linewidth=1)
        ax2.plot(df["Date"], sma20, label="SMA20", linewidth=1)
        ax2.plot(df["Date"], sma50, label="SMA50", linewidth=1)
        ax2.plot(df["Date"], ema20, label="EMA20", linewidth=1)
        ax2.plot(df["Date"], bb_upper, label="Upper Band", linestyle="--")
        ax2.plot(df["Date"], bb_mid, label="Middle Band", linestyle="--")
        ax2.plot(df["Date"], bb_lower, label="Lower Band", linestyle="--")
        ax2.legend()
        ax2.set_title(f"{stock} — Moving Averages + Bollinger Bands")
        st.pyplot(fig2)

        # RSI Plot
        fig3, ax3 = plt.subplots(figsize=(12, 3))
        ax3.plot(df["Date"], rsi, color="orange", linewidth=1)
        ax3.axhline(70, color="red", linestyle="--")
        ax3.axhline(30, color="green", linestyle="--")
        ax3.set_title("RSI (14)")
        st.pyplot(fig3)

    # ---------------------------
    # TAB 3 → MACD TRADING SIGNALS
    # ---------------------------
    with tab3:
        st.subheader("🟢 MACD Trading Signals")

        close = df["Close"]
        macd, macd_signal, hist = calc_macd(close)
        rsi_all = calc_rsi(close, period=14)

        # Conditions
        buy_cond = (macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1)) & (rsi_all < 60)
        sell_cond = (macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1)) & (rsi_all > 40)

        buys = df[buy_cond].copy()
        sells = df[sell_cond].copy()

        fig4, (ax_p, ax_m) = plt.subplots(
            2, 1, sharex=True, figsize=(12, 7),
            gridspec_kw={"height_ratios": [3, 2]}
        )

        # Price + signals
        ax_p.plot(df["Date"], close, label="Close Price", linewidth=1)
        ax_p.scatter(buys["Date"], buys["Close"], marker="^", color="green", s=80, label="BUY")
        ax_p.scatter(sells["Date"], sells["Close"], marker="v", color="red", s=80, label="SELL")
        ax_p.set_ylabel("Price (INR)")
        ax_p.legend()

        # MACD
        ax_m.plot(df["Date"], macd, label="MACD", linewidth=1)
        ax_m.plot(df["Date"], macd_signal, label="Signal", linewidth=1)
        ax_m.bar(df["Date"], hist, label="Histogram")
        ax_m.axhline(0, color="white", linewidth=0.5)
        ax_m.set_ylabel("MACD")
        ax_m.legend()

        fig4.suptitle(f"{stock} — MACD Trading Signals", y=0.95)
        st.pyplot(fig4)

        # Signals table
        signals = []
        for _, row in buys.iterrows():
            signals.append({
                "Date": row["Date"].date(),
                "Signal": "BUY",
                "Price (INR)": round(float(row["Close"]), 2),
                "Reason": "MACD crossed ABOVE Signal & RSI < 60"
            })
        for _, row in sells.iterrows():
            signals.append({
                "Date": row["Date"].date(),
                "Signal": "SELL",
                "Price (INR)": round(float(row["Close"]), 2),
                "Reason": "MACD crossed BELOW Signal & RSI > 40"
            })

        if not signals:
            st.info("No MACD-based buy/sell signals found for this period.")
        else:
            st.write("### 📄 Trade Signals Table")
            st.dataframe(pd.DataFrame(signals))

    # ---------------------------
    # TAB 4 → ACCURACY & SELF-LEARNING
    # ---------------------------
    with tab4:
        st.subheader("🎯 Model Accuracy (Past Data)")

        # Build accuracy DataFrame
        acc_df = pd.DataFrame({
            "Date": dates_last.values,
            "Actual Price (INR)": np.round(y_true_last, 2),
            "Predicted Price (INR)": np.round(y_pred_last, 2)
        })
        acc_df["Error (Abs)"] = np.round(np.abs(acc_df["Actual Price (INR)"] - acc_df["Predicted Price (INR)"]), 2)
        acc_df["Accuracy (%)"] = np.round(
            100 - (acc_df["Error (Abs)"] / acc_df["Actual Price (INR)"]) * 100,
            2
        )

        st.write("### 📄 Last Predictions vs Actual (Recent 50 Points)")
        st.dataframe(acc_df)

        st.write("### 📉 Error Over Time (Recent Points)")
        fig_err, ax_err = plt.subplots(figsize=(12, 3))
        ax_err.plot(acc_df["Date"], acc_df["Error (Abs)"], marker="o", linewidth=1)
        ax_err.set_ylabel("Absolute Error (INR)")
        st.pyplot(fig_err)

        st.write("### 📊 Current Session Accuracy")
        st.write(f"- Mean Absolute Error (MAE): {mae:.2f} INR")

        if memory.get("avg_mae") is not None:
            st.write("### 🧠 Long-Term Self-Learning Memory")
            st.write(f"- Total evaluations so far: {memory['total_evals']}")
            st.write(f"- Average MAE across runs: {memory['avg_mae']:.2f} INR")
        else:
            st.info("No past accuracy data stored yet. This is the first run.")
     # ---------------------------
    # TAB 5 → BUY/SELL RECOMMENDATION
    # ---------------------------     
    with tab5:
        st.subheader("💡 Buy/Sell Recommendations")

        # Your indicators
        rsi = calc_rsi(df["Close"])
        sma20 = df["Close"].rolling(20).mean()
        sma50 = df["Close"].rolling(50).mean()

        buy = []
        sell = []

    for i in range(len(df)):
        if rsi[i] < 30 and sma20[i] > sma50[i]:
                buy.append(df["Close"][i])
                sell.append(np.nan)
        elif rsi[i] > 70 and sma20[i] < sma50[i]:
                sell.append(df["Close"][i])
                buy.append(np.nan)
        else:
            buy.append(np.nan)
            sell.append(np.nan)

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df["Date"], df["Close"], label="Close")
    ax.plot(df["Date"], sma20, label="SMA20")
    ax.plot(df["Date"], sma50, label="SMA50")
    ax.scatter(df["Date"], buy, color="green", marker="^", s=80, label="BUY")
    ax.scatter(df["Date"], sell, color="red", marker="v", s=80, label="SELL")
    ax.legend()
    st.pyplot(fig)

    # Today's Recommendation
    if not np.isnan(buy[-1]):
        st.success("🟢 BUY — Strong upside signal detected.")
    elif not np.isnan(sell[-1]):
        st.error("🔴 SELL — Downside signal detected.")
    else:
        st.info("🟡 HOLD — No clear signal.")
      

    with tab6:
        st.header("🤖 FINAL AI Decision (BUY / SELL / HOLD)")

        decision, confidence, reasons,verdict = final_ai_decision(df)

        st.subheader(f"🔥 FINAL DECISION: {decision}")
        st.write(f"🎯 Confidence: {confidence}%")
        st.info(f"🧾 Verdict: {verdict}")
        st.write("### 🧠 Reasoning:")
        for r in reasons:
            st.write(f"- {r}")

    # ---------------------------
    # TAB 7 → SUPPORT & RESISTANCE
    # ---------------------------
    with tab7:
        st.subheader("📉 Support & Resistance Levels")

        support, resistance = calc_support_resistance(df["Close"])
        last_price = float(df["Close"].iloc[-1])

        col_sr1, col_sr2, col_sr3 = st.columns(3)
    with col_sr1:
        st.metric("Recent Support", f"{support:.2f} INR")
    with col_sr2:
        st.metric("Current Price", f"{last_price:.2f} INR")
    with col_sr3:
        st.metric("Recent Resistance", f"{resistance:.2f} INR")

        fig_sr, ax_sr = plt.subplots(figsize=(12, 5))
        ax_sr.plot(df["Date"], df["Close"], label="Close Price", linewidth=1)
        ax_sr.axhline(support, color="green", linestyle="--", linewidth=1.2,
                  label=f"Support ≈ {support:.2f}")
        ax_sr.axhline(resistance, color="red", linestyle="--", linewidth=1.2,
                  label=f"Resistance ≈ {resistance:.2f}")
        ax_sr.axhline(last_price, color="orange", linestyle=":", linewidth=1.2,
                  label=f"Current Price ≈ {last_price:.2f}")
        ax_sr.set_ylabel("Price (INR)")
        ax_sr.set_title(f"{stock} — Support & Resistance (last 60 days window)")
        ax_sr.legend()
        st.pyplot(fig_sr)

    # Simple text interpretation
        st.write("### 📝 Interpretation")
    if last_price <= support * 1.02:
        st.success("Price is very close to SUPPORT → Better zone to accumulate / watch for bounce.")
    elif last_price >= resistance * 0.98:
        st.error("Price is very close to RESISTANCE → Risk of reversal / profit-taking zone.")
    else:
        st.info("Price is between support and resistance → Neutral zone, no extreme level reached yet.")    


    with tab8:
        st.subheader("📐 Auto Trendline Detection")

        highs, lows, trends, trend_type = calculate_trendlines(df)

    if trends is None:
        st.warning("Not enough data to calculate trendlines.")
    else:
        trend_high, trend_low = trends

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["Date"].tail(200), df["Close"].tail(200), label="Close Price", linewidth=1)

        # Plot swing highs
        ax.scatter(highs["Date"], highs["Swing_High"], color="red", label="Swing Highs", s=60)

        # Plot swing lows
        ax.scatter(lows["Date"], lows["Swing_Low"], color="green", label="Swing Lows", s=60)

        # Draw trendlines
        xh = np.arange(len(highs))
        ax.plot(highs["Date"], trend_high(xh), color="red", linestyle="--", label="Resistance Trendline")

        xl = np.arange(len(lows))
        ax.plot(lows["Date"], trend_low(xl), color="green", linestyle="--", label="Support Trendline")

        ax.set_title(f"Trendlines ({trend_type})")
        ax.legend()
        st.pyplot(fig)

        st.write("### 📌 Trend Interpretation")
        if "UPTREND" in trend_type:
            st.success("Market is in UPTREND — Buy dips, strong momentum")
        elif "DOWNTREND" in trend_type:
            st.error("Market is in DOWNTREND — Avoid buying, trend is weak")
        else:
            st.info("Market is SIDEWAYS — No clear direction, wait for breakout")   

    with tab9:
        st.subheader("📉 Candlestick Pattern Recognition & Volume Spike Signals")

    df_plot = df.copy()

    # ================================
    # ⭐ CANDLESTICK PATTERN DETECTION
    # ================================

    def is_bullish_engulfing(d):
        # d has 2 rows: previous (0) and current (1)
        return (
            d["Close"].iloc[0] < d["Open"].iloc[0] and   # prev red
            d["Close"].iloc[1] > d["Open"].iloc[1] and   # current green
            d["Close"].iloc[1] > d["Open"].iloc[0] and
            d["Open"].iloc[1] < d["Close"].iloc[0]
        )

    def is_bearish_engulfing(d):
        return (
            d["Close"].iloc[0] > d["Open"].iloc[0] and   # prev green
            d["Close"].iloc[1] < d["Open"].iloc[1] and   # current red
            d["Open"].iloc[1] > d["Close"].iloc[0] and
            d["Close"].iloc[1] < d["Open"].iloc[0]
        )

    # Make lists EXACTLY same length as df_plot
    bullish = [np.nan] * len(df_plot)
    bearish = [np.nan] * len(df_plot)

    # Start from index 1 (we need i-1 and i)
    for i in range(1, len(df_plot)):
        window = df_plot.iloc[i-1:i+1]
        if len(window) < 2:
            continue

        if is_bullish_engulfing(window):
            bullish[i] = df_plot["Close"].iloc[i]
        elif is_bearish_engulfing(window):
            bearish[i] = df_plot["Close"].iloc[i]

    # ================================
    # ⭐ VOLUME SPIKE BUY/SELL SIGNALS
    # ================================

    df_plot["Volume_MA20"] = df_plot["Volume"].rolling(20).mean()
    df_plot["Volume_Spike"] = df_plot["Volume"] > (df_plot["Volume_MA20"] * 1.5)

    buy_vol = [np.nan] * len(df_plot)
    sell_vol = [np.nan] * len(df_plot)

    for i in range(1, len(df_plot)):
        if df_plot["Volume_Spike"].iloc[i] and df_plot["Close"].iloc[i] > df_plot["Close"].iloc[i - 1]:
            buy_vol[i] = df_plot["Close"].iloc[i]
        elif df_plot["Volume_Spike"].iloc[i] and df_plot["Close"].iloc[i] < df_plot["Close"].iloc[i - 1]:
            sell_vol[i] = df_plot["Close"].iloc[i]

    # ================================
    # ⭐ TRENDLINES (OPTIONAL)
    # ================================
    from scipy.signal import argrelextrema

    df_plot["Min"] = np.nan
    df_plot["Max"] = np.nan

    if len(df_plot) > 10:  # need enough data
        mins = argrelextrema(df_plot["Close"].values, np.less_equal, order=5)[0]
        maxs = argrelextrema(df_plot["Close"].values, np.greater_equal, order=5)[0]
        df_plot.loc[df_plot.index[mins], "Min"] = df_plot["Close"].iloc[mins]
        df_plot.loc[df_plot.index[maxs], "Max"] = df_plot["Close"].iloc[maxs]

    # ================================
    # ⭐ PLOT EVERYTHING
    # ================================

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(df_plot["Date"], df_plot["Close"], label="Close Price", color="blue", linewidth=1)

    # Candlestick patterns
    ax.scatter(df_plot["Date"], bullish, marker="^", color="green", s=80, label="Bullish Engulfing")
    ax.scatter(df_plot["Date"], bearish, marker="v", color="red", s=80, label="Bearish Engulfing")

    # Volume spike markers
    ax.scatter(df_plot["Date"], buy_vol, marker="^", color="lime", s=120, label="BUY (Volume Spike)")
    ax.scatter(df_plot["Date"], sell_vol, marker="v", color="orange", s=120, label="SELL (Volume Spike)")

    # Trendlines
    ax.plot(df_plot["Date"], df_plot["Min"], color="cyan", linewidth=1.2, label="Support Trendline")
    ax.plot(df_plot["Date"], df_plot["Max"], color="magenta", linewidth=1.2, label="Resistance Trendline")

    ax.set_title("📉 Candlestick Patterns + Volume Spikes + Trendlines")
    ax.legend()
    st.pyplot(fig)

    # Text Summary
    st.write("### 📝 Pattern Summary")
    st.write(f"• Bullish Engulfing Signals: {np.count_nonzero(~np.isnan(bullish))}")
    st.write(f"• Bearish Engulfing Signals: {np.count_nonzero(~np.isnan(bearish))}")
    st.write(f"• Volume Spike Buy Signals: {np.count_nonzero(~np.isnan(buy_vol))}")
    st.write(f"• Volume Spike Sell Signals: {np.count_nonzero(~np.isnan(sell_vol))}")

    with tab10:
        st.header("💹 Backtesting — Strategy Profit Simulation")

        st.info("Testing your trading strategy on historical data...")

    # Use Close, Buy Sell Signals
    close = df["Close"].values
    dates = df["Date"].values

    # Reuse Buy/Sell logic from your app
    rsi = calc_rsi(df["Close"])
    sma20 = df["Close"].rolling(20).mean()
    sma50 = df["Close"].rolling(50).mean()

    buy_signal = []
    sell_signal = []

    for i in range(len(df)):
        if rsi[i] < 30 and sma20[i] > sma50[i]:
            buy_signal.append(1)
            sell_signal.append(0)
        elif rsi[i] > 70 and sma20[i] < sma50[i]:
            buy_signal.append(0)
            sell_signal.append(1)
        else:
            buy_signal.append(0)
            sell_signal.append(0)

    # BACKTEST PARAMETERS
    balance = 100000     # Starting capital
    position = 0
    entry_price = 0
    trade_log = []
    capital_curve = []

    for i in range(len(df)):
        price = close[i]

        # BUY
        if buy_signal[i] == 1 and position == 0:
            position = balance / price
            entry_price = price
            trade_log.append({
                "Date": str(dates[i])[:10],
                "Type": "BUY",
                "Price": round(price, 2)
            })
            balance = 0

        # SELL
        elif sell_signal[i] == 1 and position > 0:
            balance = position * price
            profit = balance - (position * entry_price)
            trade_log.append({
                "Date": str(dates[i])[:10],
                "Type": "SELL",
                "Price": round(price, 2),
                "Profit": round(profit, 2)
            })
            position = 0

        # Track equity for graph
        if position > 0:
            capital_curve.append(position * price)
        else:
            capital_curve.append(balance)

    # Final value
    final_balance = balance if position == 0 else position * close[-1]

    total_profit = final_balance - 100000
    returns = total_profit / 100000 * 100

    # Win Rate
    wins = [t for t in trade_log if "Profit" in t and t["Profit"] > 0]
    losses = [t for t in trade_log if "Profit" in t and t["Profit"] <= 0]

    if len(wins) + len(losses) > 0:
        win_rate = len(wins) / (len(wins) + len(losses)) * 100
    else:
        win_rate = 0

    # Max Drawdown
    curve = np.array(capital_curve)
    roll_max = np.maximum.accumulate(curve)
    drawdown = (curve - roll_max) / roll_max
    max_drawdown = np.min(drawdown) * 100

    # Sharpe Ratio (daily)
    daily_returns = np.diff(curve) / curve[:-1]
    sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252) if np.std(daily_returns) != 0 else 0

    # Display Results
    st.subheader("📌 Backtest Summary")
    st.write(f"💰 Final Balance: ₹{final_balance:,.2f}")
    st.write(f"📈 Total Profit: ₹{total_profit:,.2f} ({returns:.2f}%)")
    st.write(f"🥇 Win Rate: {win_rate:.2f}%")
    st.write(f"📉 Max Drawdown: {max_drawdown:.2f}%")
    st.write(f"⚡ Sharpe Ratio: {sharpe:.2f}")

    # Graph
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], capital_curve, label="Equity Curve", color="cyan")
    ax.set_title("Equity Curve (Balance Over Time)")
    ax.set_ylabel("Portfolio Value (INR)")
    ax.legend()
    st.pyplot(fig)

    # Trade Log Table
    st.dataframe(pd.DataFrame(trade_log))

    with tab11:
        st.header("🧠 Strategy Optimization — Auto-Find Best Parameters")

        st.write("This tool tests multiple RSI + SMA combinations to find the most profitable strategy.")

        close = df["Close"]
        close_np = close.values   # IMPORTANT FIX

    # PARAMETER SEARCH SPACE
    rsi_periods = [7, 10, 14]
    sma_fast_list = [5, 10, 20]
    sma_slow_list = [20, 40, 50]

    results = []

    for rsi_p in rsi_periods:
        rsi_vals = calc_rsi(close, period=rsi_p).values  # convert to numpy

        for sma_f in sma_fast_list:
            for sma_s in sma_slow_list:
                if sma_f >= sma_s:
                    continue

                sma_fast = close.rolling(sma_f).mean().values
                sma_slow = close.rolling(sma_s).mean().values

                balance = 100000
                position = 0

                for i in range(len(close_np)):

                    # skip NaN values to avoid errors
                    if np.isnan(rsi_vals[i]) or np.isnan(sma_fast[i]) or np.isnan(sma_slow[i]):
                        continue

                    # BUY SIGNAL
                    if rsi_vals[i] < 30 and sma_fast[i] > sma_slow[i] and position == 0:
                        position = balance / close_np[i]
                        balance = 0

                    # SELL SIGNAL
                    elif rsi_vals[i] > 70 and sma_fast[i] < sma_slow[i] and position > 0:
                        balance = position * close_np[i]
                        position = 0

                # FINAL BALANCE (if still holding)
                final_balance = balance if position == 0 else position * close_np[-1]
                profit = final_balance - 100000

                results.append({
                    "RSI Period": rsi_p,
                    "SMA Fast": sma_f,
                    "SMA Slow": sma_s,
                    "Final Balance": round(final_balance, 2),
                    "Profit": round(profit, 2)
                })

    results_df = pd.DataFrame(results)

    # Best result
    best = results_df.loc[results_df["Profit"].idxmax()]

    st.subheader("🏆 Best Strategy Found")
    st.success(
        f"RSI={best['RSI Period']}, SMA Fast={best['SMA Fast']}, "
        f"SMA Slow={best['SMA Slow']} → Profit ₹{best['Profit']}"
    )

    st.subheader("📄 All Tested Strategies")
    st.dataframe(results_df)

    # Graph
    fig_opt, ax_opt = plt.subplots(figsize=(12, 5))
    ax_opt.bar(range(len(results_df)), results_df["Profit"], color="cyan")
    ax_opt.set_title("Profit for Each Strategy Combination")
    ax_opt.set_ylabel("Profit (INR)")
    st.pyplot(fig_opt)

    # -------------------------------------------------------------
# TAB 12 → LIVE ALERTS + RISK SIGNALS (DAY 22)
# -------------------------------------------------------------
    with tab12:
        st.header("🔔 Real-Time Alerts & Risk Signals")

    # Latest values
    curr_price = df["Close"].iloc[-1]
    prev_price = df["Close"].iloc[-2]
    curr_vol = df["Volume"].iloc[-1]
    avg_vol = df["Volume"].rolling(50).mean().iloc[-1]

    # Indicators
    rsi_val = calc_rsi(df["Close"]).iloc[-1]
    macd_val, macd_signal, _ = calc_macd(df["Close"])
    macd_val = macd_val.iloc[-1]
    macd_signal = macd_signal.iloc[-1]

    alerts = []

    # -------- PRICE SPIKE / CRASH --------
    price_change = ((curr_price - prev_price) / prev_price) * 100

    if price_change > 2:
        alerts.append(("🟢 Bullish Price Spike", 
                        f"Price jumped +{price_change:.2f}% today"))
    elif price_change < -2:
        alerts.append(("🔴 Bearish Price Drop", 
                        f"Price fell {price_change:.2f}% today"))

    # -------- VOLUME ALERT --------
    if curr_vol > avg_vol * 1.5:
        alerts.append(("⚡ Volume Breakout", 
                        "Volume exploded above 150% of normal → breakout coming"))

    # -------- RSI ALERT --------
    if rsi_val > 70:
        alerts.append(("🟥 Overbought Zone", 
                        f"RSI = {rsi_val:.1f} → Risk of reversal"))
    elif rsi_val < 30:
        alerts.append(("🟩 Oversold Zone", 
                        f"RSI = {rsi_val:.1f} → Possible rebound opportunity"))

    # -------- MACD ALERT --------
    if macd_val > macd_signal:
        alerts.append(("🟢 MACD Bullish Crossover", 
                        "MACD crossed ABOVE signal → Uptrend building"))
    elif macd_val < macd_signal:
        alerts.append(("🔻 MACD Bearish Crossover", 
                        "MACD crossed BELOW signal → Downtrend risk"))

    # ------------------------------------------------
    # DISPLAY ALERTS
    # ------------------------------------------------
    if len(alerts) == 0:
        st.info("😴 No major alerts today — market stable.")
    else:
        for title, msg in alerts:
            st.warning(f"{title}\n\n{msg}")

    # Table Summary
    alert_table = pd.DataFrame(alerts, columns=["Alert", "Details"])
    st.write("### 📄 Today’s Alert Summary")
    st.dataframe(alert_table)

    # -----------------------------------------------------
# TAB 13 → AI CLASSIFIER (BUY / SELL / HOLD PREDICTOR)
# -----------------------------------------------------
    with tab13:
        st.header("🎯 AI Classifier – Buy / Sell / Hold Prediction")

    # --------------- BUILD FEATURES ---------------
    df_feat = df.copy()

    df_feat["RSI"] = calc_rsi(df_feat["Close"])
    df_feat["MACD"], df_feat["Signal"], df_feat["Hist"] = calc_macd(df_feat["Close"])
    df_feat["SMA20"] = df_feat["Close"].rolling(20).mean()
    df_feat["SMA50"] = df_feat["Close"].rolling(50).mean()
    df_feat["ADX_plus"], df_feat["ADX_minus"], df_feat["ADX"] = calc_adx(df_feat)
    df_feat["ATR"] = calc_atr(df_feat)
    df_feat["Momentum"] = df_feat["Close"].pct_change()

    df_feat = df_feat.dropna().reset_index(drop=True)

    # --------------- LABEL CREATION ---------------
    # +1 = BUY, -1 = SELL, 0 = HOLD
    df_feat["Label"] = 0
    df_feat.loc[df_feat["Momentum"] > 0.01, "Label"] = 1
    df_feat.loc[df_feat["Momentum"] < -0.01, "Label"] = -1

    features = ["RSI", "MACD", "Signal", "SMA20", "SMA50", "ADX", "ATR", "Momentum"]

    X = df_feat[features]
    y = df_feat["Label"]

    # --------------- TRAIN MODEL ---------------
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )
    clf.fit(X, y)

    # --------------- MAKE PREDICTION ---------------
    latest = X.iloc[-1].values.reshape(1, -1)
    pred = clf.predict(latest)[0]
    proba = clf.predict_proba(latest)[0]

    if pred == 1:
        st.success("🟢 AI Classifier Signal: BUY")
    elif pred == -1:
        st.error("🔴 AI Classifier Signal: SELL")
    else:
        st.info("🟡 AI Classifier Signal: HOLD")

    # --------------- CONFIDENCE ---------------
    st.write("### 📊 Confidence Level")
    st.write(f"BUY: {proba[2]*100:.2f}%")
    st.write(f"HOLD: {proba[1]*100:.2f}%")
    st.write(f"SELL: {proba[0]*100:.2f}%")

    # --------------- FEATURE IMPORTANCE ---------------
    st.write("### 🧠 Feature Importance")
    importances = clf.feature_importances_

    fig_imp, ax_imp = plt.subplots(figsize=(10,4))
    ax_imp.bar(features, importances)
    ax_imp.set_xticklabels(features, rotation=45)
    st.pyplot(fig_imp)

    # Table output
    st.write("### 🔍 Full Feature Data")
    st.dataframe(df_feat.tail(10))

    # -----------------------------------------------------
# TAB 14 → BACKTESTING ENGINE (SIMULATE PROFITS)
# -----------------------------------------------------
    with tab14:
        st.header("🧪 Backtesting Simulator — Strategy Profit Analysis")

    df_bt = df.copy()

    # --- Indicators for Backtesting ---
    df_bt["RSI"] = calc_rsi(df_bt["Close"])
    df_bt["SMA20"] = df_bt["Close"].rolling(20).mean()
    df_bt["SMA50"] = df_bt["Close"].rolling(50).mean()
    df_bt["MACD"], df_bt["Signal"], df_bt["Hist"] = calc_macd(df_bt["Close"])

    df_bt = df_bt.dropna().reset_index(drop=True)

    # --- Trading Rules ---
    df_bt["Buy_Signal"] = (
        (df_bt["RSI"] < 30) &
        (df_bt["SMA20"] > df_bt["SMA50"]) &
        (df_bt["MACD"] > df_bt["Signal"])
    )

    df_bt["Sell_Signal"] = (
        (df_bt["RSI"] > 70) &
        (df_bt["SMA20"] < df_bt["SMA50"]) &
        (df_bt["MACD"] < df_bt["Signal"])
    )

    position = 0  
    buy_price = 0
    balance = 0
    trades = []

    # --- Backtesting Loop ---
    for i in range(len(df_bt)):
        if df_bt["Buy_Signal"][i] and position == 0:
            position = 1
            buy_price = df_bt["Close"][i]
            trades.append(("BUY", df_bt["Date"][i], buy_price))

        elif df_bt["Sell_Signal"][i] and position == 1:
            position = 0
            sell_price = df_bt["Close"][i]
            profit = sell_price - buy_price
            balance += profit
            trades.append(("SELL", df_bt["Date"][i], sell_price, profit))

    # --- Final closed trade summary ---
    st.subheader("📊 Backtest Summary")

    total_trades = len([t for t in trades if t[0] == "SELL"])
    win_trades = len([t for t in trades if len(t) == 4 and t[3] > 0])
    lose_trades = total_trades - win_trades
    win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0

    st.write(f"💰 Total Profit: {balance:.2f} INR")
    st.write(f"📈 Total Trades: {total_trades}")
    st.write(f"🏆 Winning Trades: {win_trades}")
    st.write(f"💔 Losing Trades: {lose_trades}")
    st.write(f"🎯 Win Rate: {win_rate:.2f}%")

    # --- Equity Curve (Balance Over Time) ---
    equity = []
    current_balance = 0

    for t in trades:
        if t[0] == "SELL":
            current_balance += t[3]
        equity.append(current_balance)

    if len(equity) > 0:
        fig_eq, ax_eq = plt.subplots(figsize=(12, 4))
        ax_eq.plot(equity, label="Equity Curve", linewidth=2)
        ax_eq.set_title("Equity Curve — Balance Growth Over Time")
        ax_eq.set_xlabel("Trade Number")
        ax_eq.set_ylabel("Profit (INR)")
        st.pyplot(fig_eq)

    # --- Trades Table ---
    st.write("### 📄 Detailed Trades")
    trade_rows = []

    for t in trades:
        if t[0] == "BUY":
            trade_rows.append({"Type": "BUY", "Date": t[1], "Price": t[2]})
        else:
            trade_rows.append({
                "Type": "SELL",
                "Date": t[1],
                "Price": t[2],
                "Profit": t[3]
            })

    st.dataframe(pd.DataFrame(trade_rows))
