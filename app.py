import datetime
import json
import urllib.request
import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf

# પેજ સેટઅપ: સાઇડબાર આપોઆપ ઓપન રહેશે
st.set_page_config(page_title="ETF Momentum Scanner", layout="wide", initial_sidebar_state="expanded")

# સિંગલ-સ્ક્રીન ફિટિંગ માટે કોમ્પેક્ટ લેઆઉટ CSS
st.markdown("""
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 0rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ફાઇનલ ૧૫ ETF બાસ્કેટ
DEFAULT_ETFS = [
    "PHARMABEES", "HDFCSML250", "METALIETF", "GOLDBEES",
    "MODEFENCE", "JUNIORBEES", "MIDCAPIETF", "SILVERBEES",
    "MOM30IETF", "COMMOIETF", "ITBEES", "VAL30IETF",
    "CPSEETF", "AUTOBEES", "MOREALTY"
]
BENCHMARK_SYMBOL = "^CRSLDX"  # Nifty 500

# સાઇડબાર સેટિંગ્સ
st.sidebar.header("⚙️ Scanner Settings")
use_rs_filter = st.sidebar.checkbox("Include Nifty 500 Mansfield RS Filter", value=True)
use_mom_filter = st.sidebar.checkbox("Include 6M/12M Momentum Filter & Score", value=True)
refresh = st.sidebar.button("🔄 Refresh Live Data")

selected_tickers = list(DEFAULT_ETFS)

# ટાઇટલ અને લાઇવ સમય
ist_tz = pytz.timezone("Asia/Kolkata")
now_ist = datetime.datetime.now(ist_tz)
day_str = now_ist.strftime("%A")

st.markdown(f"""
    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
        <h3 style='margin: 0; color: #1E88E5;'>📊 ETF Momentum & RS Scanner</h3>
        <span style='color: gray; font-size: 13px;'>{day_str} | {now_ist.strftime('%d-%b-%Y %I:%M %p')} IST</span>
    </div>
""", unsafe_allow_html=True)

# TradingView-Exact Wilder's RSI Function
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

# ઐતિહાસિક ડેટા ફેચિંગ
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

# રિયલ-ટાઇમ માર્કેટ ડેટા (Yahoo Finance JSON API)
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
                change_pct = ((ltp - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
                quotes[ticker] = (ltp, change_pct)
            except Exception:
                quotes[ticker] = (None, 0.0)
    return quotes

if not selected_tickers:
    st.warning("No ETFs available in the basket.")
else:
    with st.spinner("Fetching live market data..."):
        df_daily = fetch_etf_data(selected_tickers)
        live_quotes = fetch_realtime_quotes(selected_tickers)

        if df_daily is not None and not df_daily.empty:
            bench_daily = df_daily[BENCHMARK_SYMBOL].dropna() if BENCHMARK_SYMBOL in df_daily.columns else None
            bench_weekly = bench_daily.resample('W-FRI').last().ffill() if bench_daily is not None else None

            # સિલ્વર vs ગોલ્ડ રેશિયો મોમેન્ટમ
            silver_vs_gold_mrs = 0.0
            if "GOLDBEES.NS" in df_daily.columns and "SILVERBEES.NS" in df_daily.columns:
                g_weekly = df_daily["GOLDBEES.NS"].dropna().resample('W-FRI').last().ffill()
                s_weekly = df_daily["SILVERBEES.NS"].dropna().resample('W-FRI').last().ffill()
                combined_gs = pd.concat([s_weekly, g_weekly], axis=1, join='inner')
                ratio_series = combined_gs.iloc[:, 0] / combined_gs.iloc[:, 1]
                sma52_ratio = ratio_series.rolling(52).mean()
                mrs_gs_series = (((ratio_series / sma52_ratio) - 1.0) * 100.0).dropna()
                silver_vs_gold_mrs = float(mrs_gs_series.iloc[-1]) if not mrs_gs_series.empty else 0.0

            data_rows = []

            for ticker in selected_tickers:
                sym = f"{ticker}.NS"
                if sym not in df_daily.columns:
                    continue

                d_series = df_daily[sym].dropna().copy()
                if len(d_series) < 252:
                    continue

                try:
                    ltp, daily_change_pct = live_quotes.get(ticker, (None, 0.0))
                    if ltp is None or ltp == 0.0:
                        ltp = float(d_series.iloc[-1])

                    # સાપ્તાહિક કેન્ડલમાં આજનો લાઈવ LTP ઉમેરવો
                    w_series = d_series.resample('W-FRI').last().ffill()
                    w_series_live = w_series.copy()
                    w_series_live.iloc[-1] = ltp

                    # Weekly 20 EMA & Wilder's RSI
                    ema20 = float(w_series_live.ewm(span=20, adjust=False).mean().iloc[-1])
                    rsi = calculate_tradingview_rsi(w_series_live, 14)
                    if np.isnan(rsi):
                        continue

                    # 52-Week High
                    high_52w = float(w_series_live.iloc[-52:].max())
                    if ltp > high_52w:
                        high_52w = ltp
                    dist_52w_pct = ((ltp - high_52w) / high_52w) * 100.0

                    # 6M/12M Composite Return & Annualized Volatility
                    ret_26w = ((ltp - float(w_series_live.iloc[-26])) / float(w_series_live.iloc[-26])) * 100.0 if len(w_series_live) >= 26 else 0.0
                    ret_52w = ((ltp - float(w_series_live.iloc[-52])) / float(w_series_live.iloc[-52])) * 100.0 if len(w_series_live) >= 52 else 0.0
                    composite_mom = (ret_26w + ret_52w) / 2.0

                    daily_ret = d_series.pct_change().dropna()[-252:]
                    volatility = float(daily_ret.std() * np.sqrt(252) * 100.0) if len(daily_ret) >= 50 else 0.0
                    raw_mom = (composite_mom / volatility) if volatility > 0 else -999.0

                    buffer_ema_pct = ((ltp - ema20) / ema20) * 100.0
                    above_ema = ltp > ema20
                    rsi_entry_band = (rsi >= 60.0) and (rsi <= 75.0)

                    is_gold = (ticker == "GOLDBEES")
                    is_silver = (ticker == "SILVERBEES")
                    is_commodity = is_gold or is_silver

                    if is_silver:
                        mrs_val = silver_vs_gold_mrs
                        mrs_display = f"Bullish (+{silver_vs_gold_mrs:.1f}%)" if silver_vs_gold_mrs > 0 else f"Bearish ({silver_vs_gold_mrs:.1f}%)"
                        rs_cond = True
                    elif is_gold:
                        mrs_val = -silver_vs_gold_mrs
                        mrs_display = "Bullish (Favoured)" if silver_vs_gold_mrs <= 0 else "Neutral (Defensive)"
                        rs_cond = True
                    else:
                        mrs = 0.0
                        mrs_display = "0.00"
                        if bench_weekly is not None:
                            combined = pd.concat([w_series_live, bench_weekly], axis=1, join='inner')
                            ratio = combined.iloc[:, 0] / combined.iloc[:, 1]
                            sma52 = ratio.rolling(52).mean()
                            mrs_series = (((ratio / sma52) - 1.0) * 100.0).dropna()
                            mrs = float(mrs_series.iloc[-1]) if not mrs_series.empty else 0.0
                            mrs_display = f"{mrs:.2f}"
                        mrs_val = mrs
                        rs_cond = (mrs > 0.0) if use_rs_filter else True

                    mom_cond = (composite_mom > 0.0) if use_mom_filter else True
                    qualified_entry = above_ema and rsi_entry_band and rs_cond and mom_cond

                    data_rows.append({
                        'ETF': ticker,
                        'Type': "Commodity" if is_commodity else "Equity",
                        'LTP': round(ltp, 2),
                        'Change (%)': round(daily_change_pct, 2),
                        '52W High (%)': round(dist_52w_pct, 2),
                        'EMA 20': round(ema20, 2),
                        'buffer_ema_pct': buffer_ema_pct,
                        'RSI': round(rsi, 2),
                        'Mansfield RS': mrs_display,
                        'mrs_val': mrs_val,
                        'raw_mom': raw_mom,
                        'above_ema': above_ema,
                        'qualified_entry': qualified_entry
                    })
                except Exception:
                    continue

            if data_rows:
                df = pd.DataFrame(data_rows)

                # ૧ થી ૧૦૦ પર્સેન્ટાઇલ મોમેન્ટમ સ્કોર
                df['Mom Score'] = (df['raw_mom'].rank(pct=True) * 100).round().astype(int)

                score_series = df['RSI'].copy()
                if use_rs_filter:
                    score_series += (df['mrs_val'] * 2.0)
                if use_mom_filter:
                    score_series += (df['Mom Score'] * 0.25)
                df['score'] = score_series

                df = df.sort_values(by=['qualified_entry', 'score', 'RSI'], ascending=[False, False, False]).reset_index(drop=True)
                df['Rank'] = range(1, len(df) + 1)

                actions = []
                for _, r in df.iterrows():
                    if not r['above_ema']:
                        actions.append("🔴 Exit Now")
                    elif r['qualified_entry'] and r['Rank'] <= 5:
                        actions.append("🟢 Fresh Buy")
                    elif r['buffer_ema_pct'] <= 2.0:
                        actions.append("⚠️ Near Exit")
                    else:
                        actions.append("🟡 Safe Hold")
                df['Friday Action'] = actions

                def highlight_rsi(val):
                    if 60 <= val <= 75: return 'background-color: #2E7D32; color: white;'
                    elif val > 75: return 'background-color: #FFA726; color: black;'
                    elif val <= 40: return 'background-color: #C62828; color: white;'
                    elif val <= 50: return 'background-color: #FFCDD2; color: black;'
                    return ''

                def highlight_change(val):
                    if val > 0: return 'color: #2E7D32; font-weight: bold;'
                    elif val < 0: return 'color: #C62828; font-weight: bold;'
                    return 'color: gray;'

                def highlight_52w(val):
                    if val >= -3.0: return 'color: #2E7D32; font-weight: bold;'
                    elif val <= -12.0: return 'color: #C62828;'
                    return ''

                def highlight_ema_row(row):
                    styles = [''] * len(row)
                    ema_idx = row.index.get_loc('EMA 20')
                    if row['LTP'] > row['EMA 20']:
                        styles[ema_idx] = 'background-color: #2E7D32; color: white; font-weight: bold;'
                    else:
                        styles[ema_idx] = 'background-color: #C62828; color: white; font-weight: bold;'
                    return styles

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

                def highlight_mom_score(val):
                    try:
                        score = int(val)
                        if score >= 80:
                            return 'background-color: #2E7D32; color: white; font-weight: bold;'
                        elif score >= 60:
                            return 'background-color: #81C784; color: black;'
                        elif score <= 35:
                            return 'background-color: #C62828; color: white; font-weight: bold;'
                        return ''
                    except Exception:
                        return ''

                # સાઇડબાર ચેકબૉક્સ મુજબ ડાયનેમિક કૉલમ્સ
                cols_to_show = ['Rank', 'ETF', 'Type', 'LTP', 'Change (%)', '52W High (%)', 'EMA 20', 'RSI']
                if use_rs_filter:
                    cols_to_show.append('Mansfield RS')
                if use_mom_filter:
                    cols_to_show.append('Mom Score')
                cols_to_show.append('Friday Action')

                view_df = df[cols_to_show]
                style_pipe = view_df.style.apply(highlight_ema_row, axis=1)\
                                          .map(highlight_rsi, subset=['RSI'])\
                                          .map(highlight_change, subset=['Change (%)'])\
                                          .map(highlight_52w, subset=['52W High (%)'])

                if use_rs_filter:
                    style_pipe = style_pipe.map(highlight_mrs, subset=['Mansfield RS'])
                if use_mom_filter:
                    style_pipe = style_pipe.map(highlight_mom_score, subset=['Mom Score'])

                format_dict = {
                    'LTP': '₹{:.2f}',
                    'Change (%)': '{:+.2f}%',
                    '52W High (%)': '{:+.2f}%',
                    'EMA 20': '₹{:.2f}',
                    'RSI': '{:.2f}'
                }
                if use_mom_filter:
                    format_dict['Mom Score'] = '{:d}'

                styled_df = style_pipe.format(format_dict)
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Market data is currently updating. Please refresh in a moment.")
        else:
            st.error("Failed to fetch data from Yahoo Finance. Please refresh.")
            
