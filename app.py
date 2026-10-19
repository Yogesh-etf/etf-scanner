import datetime
import json
import urllib.request
import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf

# સાઇડબાર આપોઆપ ખુલેલું રહે તે માટે initial_sidebar_state="expanded" રાખ્યું છે
st.set_page_config(page_title="ETF Momentum Scanner", layout="wide", initial_sidebar_state="expanded")

# સાઇડબારનું ટોગલ બટન સચવાય અને પેજ કોમ્પેક્ટ ફિટ રહે તે માટે સુધારેલું CSS
st.markdown("""
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 0rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

DEFAULT_ETFS = [
    "PHARMABEES", "HDFCSML250", "METALIETF", "GOLDBEES",
    "MODEFENCE", "JUNIORBEES", "MIDCAPIETF", "SILVERBEES",
    "MOM30IETF", "COMMOIETF", "ITBEES", "VAL30IETF",
    "CPSEETF", "AUTOBEES", "MOREALTY"
]
BENCHMARK_SYMBOL = "^CRSLDX"  # Nifty 500

# સાઇડબાર સેટિંગ્સ અને ચેકબૉક્સ
st.sidebar.header("⚙️ Scanner Settings")
use_rs_filter = st.sidebar.checkbox("Include Nifty 500 Mansfield RS Filter", value=True)
use_mom_filter = st.sidebar.checkbox("Include 6M/12M Momentum Filter & Score", value=True)
refresh = st.sidebar.button("🔄 Refresh Live Data")

selected_tickers = list(DEFAULT_ETFS)

# હેડિંગ અને સમય
ist_tz = pytz.timezone("Asia/Kolkata")
now_ist = datetime.datetime.now(ist_tz)
day_str = now_ist.strftime("%A")

st.markdown(f"""
    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
        <h3 style='margin: 0; color: #1E88E5;'>📊 ETF Momentum & RS Scanner</h3>
        <span style='color: gray; font-size: 13px;'>{day_str} | {now_ist.strftime('%d-%b-%Y %I:%M %p')} IST</span>
    </div>
""", unsafe_allow_html=True)

def calculate_tradingview_rsi(series, period=14):
    series = series.dropna()
    if len(series) <= period + 1:
        return np.nan
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.iloc[1:period+1].mean()
    avg_loss = loss.iloc[1:period+1].mean()
    for i in range(period + 1, len(series)):
        avg_gain = (avg_gain * (period - 1) + gain.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss.iloc[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

@st.cache_data(ttl=300)
def fetch_etf_data(ticker_list):
    if not ticker_list:
        return None
    symbols = list(set([f"{t}.NS" for t in ticker_list] + [BENCHMARK_SYMBOL, "GOLDBEES.NS", "SILVERBEES.NS"]))
    df = yf.download(symbols, period="5y", interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        raw = df['Close'].copy() if 'Close' in df.columns.levels[0] else df.xs('Close', axis=1, level=0, drop_level=True).copy()
    else:
        raw = df['Close'].copy() if 'Close' in df else df.copy()
    return raw.ffill()

def fetch_realtime_quotes(ticker_list):
    quotes = {}
    for ticker in ticker_list:
        sym = f"{ticker}.NS"
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1m&range=1d"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                meta = data['chart']['result'][0]['meta']
                ltp = float(meta.get('regularMarketPrice', 0.0))
                prev_close = float(meta.get('chartPreviousClose', 0.0))
                change_pct = ((ltp - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
                quotes[ticker] = (ltp, change_pct)
        except Exception:
            try:
                t = yf.Ticker(sym)
                info = t.fast_info
                ltp = float(info['lastPrice'])
                prev_close = float(info['previousClose'])
                change_pct = ((ltp - prev_close) / prev
                
