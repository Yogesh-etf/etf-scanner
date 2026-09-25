import os
import datetime
import pytz
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="ETF Momentum Scanner", layout="wide")

st.markdown("""
    <h2 style='text-align: center; color: #1E88E5;'>📊 ETF Momentum & RS Scanner</h2>
    <p style='text-align: center; color: gray; font-size: 14px;'>Live Momentum & Relative Strength Dashboard</p>
""", unsafe_allow_html=True)

CONFIG_FILE = "etf_list.txt"
ADMIN_PIN = "120120"

DEFAULT_ETFS = [
    "PHARMABEES", "HDFCSML250", "METALIETF", "GOLDBEES",
    "MODEFENCE", "JUNIORBEES", "MID150BEES", "SILVERBEES",
    "MOM30IETF", "COMMOIETF", "ITBEES", "VAL30IETF",
    "CPSEETF", "FMCGIETF", "NV20IETF"
]
BENCHMARK_SYMBOL = "^CRSLDX"  # Nifty 500

def load_saved_etfs():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                lines = [line.strip().upper() for line in f.readlines() if line.strip()]
            if lines:
                return lines
        except Exception:
            pass
    return list(DEFAULT_ETFS)

def save_etfs(etf_list):
    try:
        with open(CONFIG_FILE, "w") as f:
            for item in etf_list:
                f.write(f"{item}\n")
    except Exception:
        pass

if "etf_pool" not in st.session_state:
    st.session_state.etf_pool = load_saved_etfs()

# સાઇડબાર
st.sidebar.header("⚙️ Scanner Settings")
use_rs_filter = st.sidebar.checkbox("Include Nifty 500 Mansfield RS Filter", value=True)
refresh = st.sidebar.button("🔄 Refresh Live Data")

st.sidebar.markdown("---")
st.sidebar.subheader("🔒 Manage ETF Basket")

admin_pass = st.sidebar.text_input("Enter Admin PIN to edit list:", type="password")
is_admin = (admin_pass == ADMIN_PIN)

if is_admin:
    st.sidebar.success("🔓 Admin Mode Active")
    new_etf_input = st.sidebar.text_input("➕ Add New ETF (e.g. AUTOIETF):").strip().upper()
    if new_etf_input:
        clean_sym = new_etf_input.replace(".NS", "").strip()
        if clean_sym and clean_sym not in st.session_state.etf_pool:
            st.session_state.etf_pool.append(clean_sym)
            save_etfs(st.session_state.etf_pool)
            st.sidebar.success(f"{clean_sym} added permanently!")

    selected_tickers = st.sidebar.multiselect(
        "📋 Active Basket (Click × to remove):",
        options=st.session_state.etf_pool,
        default=st.session_state.etf_pool
    )

    if set(selected_tickers) != set(st.session_state.etf_pool):
        st.session_state.etf_pool = list(selected_tickers)
        save_etfs(st.session_state.etf_pool)
else:
    if admin_pass:
        st.sidebar.error("Incorrect PIN")
    st.sidebar.info("👀 View-Only Mode: Locked by Admin. Enter PIN above to add/remove ETFs.")
    selected_tickers = list(st.session_state.etf_pool)

def get_tv_wilder_rsi(series, period=14):
    clean_s = series.dropna()
    if len(clean_s) <= period + 1:
        return pd.Series(index=series.index, data=np.nan)
    delta = clean_s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = pd.Series(index=clean_s.index, dtype=float)
    avg_loss = pd.Series(index=clean_s.index, dtype=float)
    avg_gain.iloc[period] = gain.iloc[1:period+1].mean()
    avg_loss.iloc[period] = loss.iloc[1:period+1].mean()
    for i in range(period + 1, len(clean_s)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# સાપ્તાહિક ડેટા (RSI, EMA 20, Mansfield RS માટે)
@st.cache_data(ttl=300)
def fetch_weekly_data(ticker_list):
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
    raw = raw.ffill()
    weekly = raw.resample('W-FRI').last().ffill()
    return weekly

# લાઈવ ભાવ અને ટકાવારી (કોઈ કેશ વગર, તાજો ડેટા)
def fetch_live_prices(ticker_list):
    symbols = [f"{t}.NS" for t in ticker_list]
    # તાજેતરના દિવસોનો ડેઇલી ડેટા મેળવવો
    df_recent = yf.download(symbols, period="5d", interval="1d", auto_adjust=False, progress=False)
    prices = {}
    if df_recent.empty:
        return prices

    if isinstance(df_recent.columns, pd.MultiIndex):
        close_df = df_recent['Close'].copy() if 'Close' in df_recent.columns.levels[0] else df_recent.xs('Close', axis=1, level=0, drop_level=True).copy()
    else:
        close_df = df_recent['Close'].copy() if 'Close' in df_recent else df_recent.copy()

    for ticker in ticker_list:
        sym = f"{ticker}.NS"
        if sym in close_df.columns:
            s = close_df[sym].dropna()
            if len(s) >= 2:
                ltp = float(s.iloc[-1])
                prev_close = float(s.iloc[-2])
                chg_pct = ((ltp - prev_close) / prev_close) * 100.0
                prices[ticker] = (ltp, chg_pct)
            elif len(s) == 1:
                ltp = float(s.iloc[-1])
                prices[ticker] = (ltp, 0.0)
    return prices

if not selected_tickers:
    st.warning("No ETFs available in the basket.")
else:
    with st.spinner("Fetching market data..."):
        df_weekly = fetch_weekly_data(selected_tickers)
        live_prices = fetch_live_prices(selected_tickers)

        if df_weekly is not None and not df_weekly.empty and BENCHMARK_SYMBOL in df_weekly.columns:
            bench_series = df_weekly[BENCHMARK_SYMBOL].dropna()

            has_gold = "GOLDBEES.NS" in df_weekly.columns and len(df_weekly["GOLDBEES.NS"].dropna()) >= 55
            has_silver = "SILVERBEES.NS" in df_weekly.columns and len(df_weekly["SILVERBEES.NS"].dropna()) >= 55
            silver_vs_gold_mrs = 0.0

            if has_gold and has_silver:
                g_series = df_weekly["GOLDBEES.NS"].dropna()
                s_series = df_weekly["SILVERBEES.NS"].dropna()
                combined_gs = pd.concat([s_series, g_series], axis=1, join='inner')
                ratio_series = combined_gs.iloc[:, 0] / combined_gs.iloc[:, 1]
                sma52_ratio = ratio_series.rolling(52).mean()
                mrs_gs_series = (((ratio_series / sma52_ratio) - 1.0) * 100.0).dropna()
                silver_vs_gold_mrs = float(mrs_gs_series.iloc[-1]) if not mrs_gs_series.empty else 0.0

            data_rows = []

            for ticker in selected_tickers:
                sym = f"{ticker}.NS"
                if sym not in df_weekly.columns or ticker not in live_prices:
                    continue

                series = df_weekly[sym].dropna()
                if len(series) < 55:
                    continue

                try:
                    ema20 = float(series.ewm(span=20, adjust=False).mean().iloc[-1])
                    rsi_series = get_tv_wilder_rsi(series, 14).dropna()
                    if rsi_series.empty:
                        continue
                    rsi = float(rsi_series.iloc[-1])

                    ltp, daily_change_pct = live_prices[ticker]

                    above_ema = ltp > ema20
                    rsi_bull = rsi > 60.0

                    is_gold = (ticker == "GOLDBEES")
                    is_silver = (ticker == "SILVERBEES")
                    is_commodity = is_gold or is_silver

                    if is_silver:
                        mrs_val = silver_vs_gold_mrs
                        if silver_vs_gold_mrs > 0:
                            mrs_display = f"Bullish (+{silver_vs_gold_mrs:.1f}%)"
                        else:
                            mrs_display = f"Bearish ({silver_vs_gold_mrs:.1f}%)"
                        rs_cond = True
                    elif is_gold:
                        mrs_val = -silver_vs_gold_mrs
                        if silver_vs_gold_mrs <= 0:
                            mrs_display = "Bullish (Favoured)"
                        else:
                            mrs_display = "Neutral (Defensive)"
                        rs_cond = True
                    else:
                        combined = pd.concat([series, bench_series], axis=1, join='inner')
                        ratio = combined.iloc[:, 0] / combined.iloc[:, 1]
                        sma52 = ratio.rolling(52).mean()
                        mrs_series = (((ratio / sma52) - 1.0) * 100.0).dropna()
                        mrs = float(mrs_series.iloc[-1]) if not mrs_series.empty else 0.0
                        mrs_display = f"{mrs:.2f}"
                        mrs_val = mrs
                        rs_cond = (mrs > 0.0) if use_rs_filter else True

                    qualified = above_ema and rsi_bull and rs_cond

                    data_rows.append({
                        'ETF': ticker,
                        'Type': "Commodity" if is_commodity else "Equity",
                        'LTP': round(ltp, 2),
                        'Change (%)': round(daily_change_pct, 2),
                        'EMA 20': round(ema20, 2),
                        'RSI': round(rsi, 2),
                        'Mansfield RS': mrs_display,
                        'mrs_val': mrs_val,
                        'above_ema': above_ema,
                        'qualified': qualified
                    })
                except Exception:
                    continue

            if data_rows:
                df = pd.DataFrame(data_rows)
                if use_rs_filter:
                    df['score'] = df['RSI'] + (df['mrs_val'] * 2.0)
                else:
                    df['score'] = df['RSI']

                df = df.sort_values(by=['qualified', 'score', 'RSI'], ascending=[False, False, False]).reset_index(drop=True)
                df['Rank'] = range(1, len(df) + 1)

                signals = []
                for _, r in df.iterrows():
                    if r['qualified'] and r['Rank'] <= 5:
                        signals.append("🟢 Entry")
                    elif r['above_ema']:
                        signals.append("🟡 Hold")
                    else:
                        signals.append("🔴 Exit")
                df['Signal'] = signals

                ist_tz = pytz.timezone("Asia/Kolkata")
                st.caption(f"🕒 Last Updated (IST): {datetime.datetime.now(ist_tz).strftime('%d-%b-%Y %I:%M:%S %p')}")

                def highlight_rsi(val):
                    if val >= 65: return 'background-color: #2E7D32; color: white;'
                    elif val >= 60: return 'background-color: #81C784; color: black;'
                    elif val <= 40: return 'background-color: #C62828; color: white;'
                    elif val <= 50: return 'background-color: #FFCDD2; color: black;'
                    return ''

                def highlight_change(val):
                    if val > 0: return 'color: #2E7D32; font-weight: bold;'
                    elif val < 0: return 'color: #C62828; font-weight: bold;'
                    return 'color: gray;'

                def highlight_mrs(val):
                    val_str = str(val)
                    if "Bullish" in val_str:
                        return 'background-color: #2E7D32; color: white;'
                    elif "Bearish" in val_str:
                        return 'background-color: #C62828; color: white;'
                    elif "Neutral" in val_str:
                        return 'background-color: #FFF9C4; color: black;'
                    try:
                        f_val = float(val)
                        if f_val > 5: return 'background-color: #2E7D32; color: white;'
                        elif f_val > 0: return 'background-color: #81C784; color: black;'
                        elif f_val < -5: return 'background-color: #C62828; color: white;'
                        elif f_val < 0: return 'background-color: #FFCDD2; color: black;'
                    except Exception:
                        pass
                    return ''

                view_df = df[['Rank', 'ETF', 'Type', 'LTP', 'Change (%)', 'EMA 20', 'RSI', 'Mansfield RS', 'Signal']]
                styled_df = view_df.style.map(highlight_rsi, subset=['RSI'])\
                                         .map(highlight_change, subset=['Change (%)'])\
                                         .map(highlight_mrs, subset=['Mansfield RS'])\
                                         .format({
                                             'LTP': '₹{:.2f}',
                                             'Change (%)': '{:+.2f}%',
                                             'EMA 20': '₹{:.2f}',
                                             'RSI': '{:.2f}'
                                         })

                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Market data is currently updating. Please refresh in a moment.")
        else:
            st.error("Failed to fetch data from Yahoo Finance. Please refresh.")
            
