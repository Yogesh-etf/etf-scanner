import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import pytz

st.set_page_config(page_title="ETF Momentum Scanner", layout="wide")

st.markdown("""
    <h2 style='text-align: center; color: #1E88E5;'>📊 ETF Momentum & RS Scanner</h2>
    <p style='text-align: center; color: gray; font-size: 14px;'>ScreeningMantis Style Live Dashboard</p>
""", unsafe_allow_html=True)

DEFAULT_ETFS = [
    "PHARMABEES", "HDFCSML250", "METALIETF", "GOLDBEES",
    "MODEFENCE", "JUNIORBEES", "MID150BEES", "SILVERBEES",
    "MOM30IETF", "COMMOIETF", "ITBEES", "VAL30IETF",
    "CPSEETF", "FMCGIETF", "NV20IETF"
]

BENCHMARK_SYMBOL = "^CRSLDX" # Nifty 500

st.sidebar.header("⚙️ ETF મેનેજર")

if "etf_pool" not in st.session_state:
    st.session_state.etf_pool = list(DEFAULT_ETFS)

new_etf_input = st.sidebar.text_input("➕ નવો ETF ઉમેરો (દા.ત. BANKBEES):").strip().upper()
if new_etf_input:
    clean_sym = new_etf_input.replace(".NS", "").strip()
    if clean_sym and clean_sym not in st.session_state.etf_pool:
        st.session_state.etf_pool.append(clean_sym)
        st.sidebar.success(f"{clean_sym} ઉમેરાઈ ગયો!")

selected_tickers = st.sidebar.multiselect(
    "📋 સક્રિય ETF (કાઢવા માટે × દબાવો):",
    options=st.session_state.etf_pool,
    default=st.session_state.etf_pool
)

use_rs_filter = st.sidebar.checkbox("Nifty 500 Mansfield RS ગણવું?", value=True)
refresh = st.sidebar.button("🔄 Data Refresh")

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

@st.cache_data(ttl=120)
def fetch_safe_data(ticker_list):
    if not ticker_list:
        return None, None
    symbols = list(set([f"{t}.NS" for t in ticker_list] + [BENCHMARK_SYMBOL, "GOLDBEES.NS", "SILVERBEES.NS"]))
    df = yf.download(symbols, period="5y", interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        return None, None
    if isinstance(df.columns, pd.MultiIndex):
        raw = df['Close'].copy() if 'Close' in df.columns.levels[0] else df.xs('Close', axis=1, level=0, drop_level=True).copy()
    else:
        raw = df['Close'].copy() if 'Close' in df else df.copy()
    raw = raw.ffill()
    weekly = raw.resample('W-FRI').last().ffill()
    return raw, weekly

if not selected_tickers:
    st.warning("⚠️ કૃપા કરીને સાઇડબારમાંથી ઓછામાં ઓછો એક ETF પસંદ કરો.")
else:
    with st.spinner("લાઈવ માર્કેટ ડેટા સ્કેન થઈ રહ્યો છે..."):
        raw_daily, df_weekly = fetch_safe_data(selected_tickers)

        if df_weekly is not None and not df_weekly.empty and BENCHMARK_SYMBOL in df_weekly.columns:
            bench_series = df_weekly[BENCHMARK_SYMBOL].dropna()

            # Gold અને Silver માટે Mansfield RS રેશિયો
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
                if sym not in df_weekly.columns:
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

                    daily_s = raw_daily[sym].dropna()
                    if daily_s.empty:
                        continue
                    ltp = float(daily_s.iloc[-1])

                    above_ema = ltp > ema20
                    rsi_bull = rsi > 60.0

                    is_gold = (ticker == "GOLDBEES")
                    is_silver = (ticker == "SILVERBEES")
                    is_commodity = is_gold or is_silver

                    # --- Mansfield RS લોજિક ---
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

                view_df = df[['Rank', 'ETF', 'Type', 'LTP', 'EMA 20', 'RSI', 'Mansfield RS', 'Signal']]
                styled_df = view_df.style.map(highlight_rsi, subset=['RSI'])\
                                         .map(highlight_mrs, subset=['Mansfield RS'])\
                                         .format({'LTP': '₹{:.2f}', 'EMA 20': '₹{:.2f}', 'RSI': '{:.2f}'})

                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.warning("હાલ માર્કેટ ડેટા અપડેટ થઈ રહ્યો છે, કૃપા કરીને 1 મિનિટ પછી ફરી રિફ્રેશ કરો.")
        else:
            st.error("ડેટા ફેચિંગમાં સમસ્યા છે, કૃપા કરીને થોડીવાર પછી ફરીથી પ્રયાસ કરો.")
