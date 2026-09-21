import datetime
import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf

# મોબાઈલ સ્ક્રીન માટે વાઈડ લેઆઉટ
st.set_page_config(page_title="ETF Momentum Scanner", layout="wide")

st.markdown(
    """
    <h2 style='text-align: center; color: #1E88E5;'>📊 ETF Momentum & RS Scanner</h2>
    <p style='text-align: center; color: gray; font-size: 14px;'>ScreeningMantis Style Live Dashboard</p>
""",
    unsafe_allow_html=True,
)

ALL_15_ETFS = [
    {"ticker": "PHARMABEES", "symbol": "PHARMABEES.NS", "type": "Equity"},
    {"ticker": "HDFCSML250", "symbol": "HDFCSML250.NS", "type": "Equity"},
    {"ticker": "METALIETF", "symbol": "METALIETF.NS", "type": "Equity"},
    {"ticker": "GOLDBEES", "symbol": "GOLDBEES.NS", "type": "Commodity"},
    {"ticker": "MODEFENCE", "symbol": "MODEFENCE.NS", "type": "Equity"},
    {"ticker": "JUNIORBEES", "symbol": "JUNIORBEES.NS", "type": "Equity"},
    {"ticker": "MID150BEES", "symbol": "MID150BEES.NS", "type": "Equity"},
    {"ticker": "SILVERBEES", "symbol": "SILVERBEES.NS", "type": "Commodity"},
    {"ticker": "MOM30IETF", "symbol": "MOM30IETF.NS", "type": "Equity"},
    {"ticker": "COMMOIETF", "symbol": "COMMOIETF.NS", "type": "Equity"},
    {"ticker": "ITBEES", "symbol": "ITBEES.NS", "type": "Equity"},
    {"ticker": "VAL30IETF", "symbol": "VAL30IETF.NS", "type": "Equity"},
    {"ticker": "CPSEETF", "symbol": "CPSEETF.NS", "type": "Equity"},
    {"ticker": "FMCGIETF", "symbol": "FMCGIETF.NS", "type": "Equity"},
    {"ticker": "NV20IETF", "symbol": "NV20IETF.NS", "type": "Equity"},
]

BENCHMARK_SYMBOL = "^CRSLDX"  # Nifty 500

col_opt1, col_opt2 = st.columns(2)
with col_opt1:
  use_rs_filter = st.checkbox("Nifty 500 Mansfield RS ?", value=True)
with col_opt2:
  refresh = st.button("🔄 Data Refresh")


# TradingView Wilder's RMA RSI ફોર્મ્યુલા
def get_tv_wilder_rsi(series, period=14):
  delta = series.diff()
  gain = delta.clip(lower=0.0)
  loss = -delta.clip(upper=0.0)
  avg_gain = pd.Series(index=series.index, dtype=float)
  avg_loss = pd.Series(index=series.index, dtype=float)
  if len(series) <= period:
    return pd.Series(index=series.index, data=np.nan)
  avg_gain.iloc[period] = gain.iloc[1 : period + 1].mean()
  avg_loss.iloc[period] = loss.iloc[1 : period + 1].mean()
  for i in range(period + 1, len(series)):
    avg_gain.iloc[i] = (
        avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]
    ) / period
    avg_loss.iloc[i] = (
        avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]
    ) / period
  rs = avg_gain / avg_loss.replace(0, np.nan)
  return 100 - (100 / (1 + rs))


# ડેટા કેશિંગ: દર ૫ મિનિટે આપોઆપ નવો ડેટા રિફ્રેશ થશે
@st.cache_data(ttl=300)
def fetch_data():
  symbols = [x["symbol"] for x in ALL_15_ETFS] + [BENCHMARK_SYMBOL]
  raw = yf.download(
      symbols, period="5y", interval="1d", auto_adjust=False, progress=False
  )["Close"].ffill().dropna()
  weekly = raw.resample("W-FRI").last().ffill().dropna()
  return raw, weekly


with st.spinner("લાઈવ માર્કેટ ડેટા સ્કેન થઈ રહ્યો છે..."):
  raw_daily, df_weekly = fetch_data()
  bench_series = df_weekly[BENCHMARK_SYMBOL]

  data_rows = []
  for item in ALL_15_ETFS:
    sym = item["symbol"]
    if sym not in df_weekly.columns:
      continue
    series = df_weekly[sym]
    ema20 = float(series.ewm(span=20, adjust=False).mean().iloc[-1])
    rsi = float(get_tv_wilder_rsi(series, 14).iloc[-1])

    ratio = series / bench_series
    sma52 = ratio.rolling(52).mean()
    mrs = float((((ratio / sma52) - 1.0) * 100.0).iloc[-1])

    ltp = float(raw_daily[sym].iloc[-1])
    above_ema = ltp > ema20
    rsi_bull = rsi > 60.0
    mrs_bull = mrs > 0.0

    if item["type"] == "Commodity":
      rs_cond = True
      mrs_display = 0.0
    else:
      rs_cond = mrs_bull if use_rs_filter else True
      mrs_display = round(mrs, 2)

    qualified = above_ema and rsi_bull and rs_cond

    data_rows.append({
        "ETF": item["ticker"],
        "Type": item["type"],
        "LTP": round(ltp, 2),
        "EMA 20": round(ema20, 2),
        "RSI": round(rsi, 2),
        "Mansfield RS": mrs_display,
        "above_ema": above_ema,
        "qualified": qualified,
    })

  df = pd.DataFrame(data_rows)
  if use_rs_filter:
    df["score"] = df["RSI"] + (df["Mansfield RS"] * 2.0)
  else:
    df["score"] = df["RSI"]

  df = df.sort_values(
      by=["qualified", "score", "RSI"], ascending=[False, False, False]
  ).reset_index(drop=True)
  df["Rank"] = range(1, len(df) + 1)

  signals = []
  for _, r in df.iterrows():
    if r["qualified"] and r["Rank"] <= 5:
      signals.append("🟢 Entry")
    elif r["above_ema"]:
      signals.append("🟡 Hold")
    else:
      signals.append("🔴 Exit")
  df["Signal"] = signals

# ભારતીય સમય
ist_tz = pytz.timezone("Asia/Kolkata")
st.caption(
    f"🕒 Last Updated (IST):"
    f" {datetime.datetime.now(ist_tz).strftime('%d-%b-%Y %I:%M:%S %p')}"
)


def highlight_rsi(val):
  if val >= 65:
    return "background-color: #2E7D32; color: white;"
  elif val >= 60:
    return "background-color: #81C784; color: black;"
  elif val <= 40:
    return "background-color: #C62828; color: white;"
  elif val <= 50:
    return "background-color: #FFCDD2; color: black;"
  return ""


def highlight_mrs(val):
  if val > 5:
    return "background-color: #2E7D32; color: white;"
  elif val > 0:
    return "background-color: #81C784; color: black;"
  elif val < -5:
    return "background-color: #C62828; color: white;"
  elif val < 0:
    return "background-color: #FFCDD2; color: black;"
  return ""


view_df = df[
    ["Rank", "ETF", "Type", "LTP", "EMA 20", "RSI", "Mansfield RS", "Signal"]
]
styled_df = (
    view_df.style.map(highlight_rsi, subset=["RSI"])
    .map(highlight_mrs, subset=["Mansfield RS"])
    .format({
        "LTP": "₹{:.2f}",
        "EMA 20": "₹{:.2f}",
        "RSI": "{:.2f}",
        "Mansfield RS": "{:.2f}",
    })
)

st.dataframe(styled_df, use_container_width=True, hide_index=True)
