# app.py (clean, focused dashboard for CAGR & Volatility)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

st.set_page_config(page_title="📊 Stock Return Analyzer", layout="wide")

# =======================
# Theme (finance-y)
# =======================
st.markdown(
    """
<style>
:root { --bg:#0d1117; --panel:#161b22; --panel2:#0c1526; --border:#30363d; --text:#e6edf3; --muted:#9db1d6;
        --green:#00cc66; --red:#ff4d4d; --gold:#d4af37; }
html, body, [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--text); font-size: 18px; }
section[data-testid="stSidebar"] { background: #0a1322; }
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] span { color: var(--text) !important; font-size: 16px; }
.section-title { font-size: 28px !important; font-weight: 700 !important; color: #00ff99 !important; margin-top: 20px !important; }
.grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }
.grid-4 { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
.tile { background: var(--panel); padding: 18px; border-radius: 12px; border: 1px solid var(--border); margin-bottom: 12px; }
.tile h3 { font-size: 18px; color: #f0f6fc; margin: 0 0 6px 0; opacity:.9; }
.tile .value { font-size: 34px; font-weight: 800; letter-spacing: .2px; margin-bottom: 2px; }
.tile .sub { font-size: 14px; color: var(--muted); }
.pill { display: inline-block; padding: 6px 10px; border-radius: 6px; font-size: 16px; margin: 2px; background:#0f1a2b; border:1px solid #193154; }
.pill.red { background-color: rgba(255,77,77,.12); color: #ff6b6b; border-color:#803b3b; }
.pill.green { background-color: rgba(0,204,102,.12); color: #00cc66; border-color:#1b6b44; }
.pill.gold { background-color: rgba(212,175,55,.12); color: #d4af37; border-color:#7a6022; }
.block { background: var(--panel2); border:1px solid var(--border); border-radius:16px; padding:14px; margin-top:14px; }
</style>
""",
    unsafe_allow_html=True,
)

TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_YEAR = 252
NORMAL_SIGMA_TO_PCTS = {1: (15.865, 84.135), 2: (2.275, 97.725), 3: (0.135, 99.865)}

def get_price_series(ticker: str, start_date, end_date) -> pd.Series:
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        sub = None
        for lvl0 in ("Adj Close", "Close"):
            if lvl0 in df.columns.levels[0]:
                sub = df[lvl0]
                break
        if sub is None or sub.empty:
            return pd.Series(dtype=float)
        s = sub[ticker] if ticker in sub.columns else sub.iloc[:, 0]
    else:
        col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
        if col is None:
            return pd.Series(dtype=float)
        s = df[col]
    s = pd.to_numeric(s.squeeze(), errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    return s.astype(float)

def compute_returns(price: pd.Series, use_log: bool) -> pd.Series:
    if price.empty or len(price) < 3:
        return pd.Series(dtype=float)
    rets = np.log(price / price.shift(1)) if use_log else price.pct_change()
    return rets.dropna().astype(float)

def safe_cagr(price: pd.Series) -> float:
    if len(price) < 2:
        return np.nan
    years = (price.index[-1] - price.index[0]).days / 365.25
    if years <= 0:
        return np.nan
    start, end = float(price.iloc[0]), float(price.iloc[-1])
    if start <= 0:
        return np.nan
    return float((end / start) ** (1 / years) - 1)

def pick_horizons(n_trading_days: int):
    if n_trading_days < 15:
        return [("Day", 1)]
    elif n_trading_days < 40:
        return [("Day", 1), ("Month (~21d)", TRADING_DAYS_PER_MONTH)]
    else:
        return [("Day", 1), ("Month (~21d)", TRADING_DAYS_PER_MONTH), ("Year (252d)", TRADING_DAYS_PER_YEAR)]

def compounded_window_returns(price_series: pd.Series, window_days: int, use_log: bool) -> pd.Series:
    if window_days <= 0 or len(price_series) <= window_days:
        return pd.Series(dtype=float)
    if use_log:
        lr = np.log(price_series / price_series.shift(1)).dropna()
        roll_sum = lr.rolling(window_days).sum().dropna()
        return (np.exp(roll_sum) - 1).astype(float)
    else:
        sr = price_series.pct_change().dropna()
        roll_prod = (1 + sr).rolling(window_days).apply(lambda x: np.prod(x), raw=True).dropna()
        return (roll_prod - 1).astype(float)

def empirical_band(price_series: pd.Series, window_days: int, N: int, use_log: bool):
    pct_lo, pct_hi = NORMAL_SIGMA_TO_PCTS.get(N, (15.865, 84.135))
    rets = compounded_window_returns(price_series, window_days, use_log)
    if len(rets) < 30:
        return None
    lo = float(np.nanpercentile(rets, pct_lo))
    hi = float(np.nanpercentile(rets, pct_hi))
    return lo, hi

def logspace_band_from_daily(mu_d_log: float, sd_d_log: float, periods: int, N: int):
    mu_h = mu_d_log * periods
    sd_h = sd_d_log * np.sqrt(periods)
    low = np.expm1(mu_h - N * sd_h)
    high = np.expm1(mu_h + N * sd_h)
    return float(low), float(high)

def year_band_from_cagr_log(cagr: float, sd_d_log: float, N: int):
    mu_year = np.log1p(cagr)
    sd_year = sd_d_log * np.sqrt(TRADING_DAYS_PER_YEAR)
    low = np.expm1(mu_year - N * sd_year)
    high = np.expm1(mu_year + N * sd_year)
    return float(low), float(high)

def fmt_pct(x: float) -> str:
    return "—" if x is None or np.isnan(x) else f"{x:.2%}"

def render_metric(title: str, value: str, subtitle: str = "") -> str:
    return f"""
    <div class="tile">
      <h3>{title}</h3>
      <div class="value">{value}</div>
      <div class="sub">{subtitle}</div>
    </div>
    """

def render_tile(title: str, low_r, high_r, last_price, invest_amt) -> str:
    low_r = float(low_r)
    high_r = float(high_r)
    last_price = float(last_price)
    invest_amt = float(invest_amt)
    low_price = last_price * (1 + low_r)
    high_price = last_price * (1 + high_r)
    invest_line = ""
    if invest_amt > 0:
        lower_dollar = invest_amt * (1 + low_r)
        upper_dollar = invest_amt * (1 + high_r)
        invest_line = f'<div class="pill gold">💵 ${invest_amt:,.0f} → ${lower_dollar:,.0f} → ${upper_dollar:,.0f}</div>'
    return f"""
    <div class="tile">
      <h3>{title}</h3>
      <div class="pill red">📉 Min return: {low_r:.2%}</div>
      <div class="pill green">📈 Max return: {high_r:.2%}</div>
      <div class="pill">💲 Price range: ${low_price:,.2f} → ${high_price:,.2f}</div>
      {invest_line}
    </div>
    """

st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", "AAPL").strip().upper()
start_date = st.sidebar.date_input("Start Date", value=date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())
use_log = st.sidebar.checkbox("Use Log Returns (affects charts/empirical bands)", value=True)
N = st.sidebar.selectbox("Std Dev Multiplier (N)", [1, 2, 3], index=1)
invest_amt = st.sidebar.number_input("Investment Amount ($)", value=0, step=100)

try:
    px = get_price_series(ticker, start_date, end_date)
    if px.empty or len(px) < 2:
        st.error("Not enough data for the selected period.")
    else:
        last_price = float(px.iloc[-1])
        rets_d = compute_returns(px, use_log=use_log)
        if rets_d.empty:
            st.error("Not enough data to compute returns.")
        else:
            mean_d = float(rets_d.mean())
            std_d = float(rets_d.std())
            rets_log = np.log(px / px.shift(1)).dropna()
            mu_d_log = float(rets_log.mean())
            sd_d_log = float(rets_log.std())
            ann_vol_log = sd_d_log * np.sqrt(TRADING_DAYS_PER_YEAR)
            # MA20 for yearly range
            rets_log_ma20 = rets_log.rolling(20).mean().dropna()
            mu_d_log_ma20 = float(rets_log_ma20.mean()) if len(rets_log_ma20) else np.nan
            sd_d_log_ma20 = float(rets_log_ma20.std()) if len(rets_log_ma20) else np.nan
            avg_ret_ma20_annual = np.expm1(mu_d_log_ma20 * TRADING_DAYS_PER_YEAR) if not np.isnan(mu_d_log_ma20) else np.nan
            growth_cagr = safe_cagr(px)

            st.markdown('<div class="section-title">📌 Key Dashboard</div>', unsafe_allow_html=True)
            st.markdown('<div class="grid-4">', unsafe_allow_html=True)
            st.markdown(render_metric("Ticker", value=ticker, subtitle=f"{start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}"), unsafe_allow_html=True)
            st.markdown(render_metric("CAGR (start→end)", value=fmt_pct(growth_cagr), subtitle="Compound annual growth over your selected dates"), unsafe_allow_html=True)
            st.markdown(render_metric("Annualized Volatility", value=fmt_pct(ann_vol_log), subtitle="From daily log returns × √252"), unsafe_allow_html=True)
            st.markdown(render_metric("Last Price", value=f"${last_price:,.2f}", subtitle="Auto-adjusted close"), unsafe_allow_html=True)
            st.markdown(render_metric("Average Return (MA20, annualized)", value=fmt_pct(avg_ret_ma20_annual), subtitle="From 20 day moving avg of daily log returns"), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            n_trading_days = len(rets_d)
            horizons = pick_horizons(n_trading_days)
            st.markdown(f'<div class="section-title">🎯 Expected Total Return Ranges (±{int(N)}σ)</div>', unsafe_allow_html=True)
            st.markdown('<div class="grid-3">', unsafe_allow_html=True)
            for label, periods in horizons:
                if periods == TRADING_DAYS_PER_YEAR and not np.isnan(mu_d_log_ma20) and not np.isnan(sd_d_log_ma20):
                    low_r, high_r = logspace_band_from_daily(mu_d_log_ma20, sd_d_log_ma20, periods, int(N))
                else:
                    band = empirical_band(px, periods, int(N), use_log)
                    if band is None:
                        if periods == TRADING_DAYS_PER_YEAR and not np.isnan(growth_cagr):
                            low_r, high_r = year_band_from_cagr_log(growth_cagr, sd_d_log, int(N))
                        else:
                            low_r, high_r = logspace_band_from_daily(mu_d_log, sd_d_log, periods, int(N))
                    else:
                        low_r, high_r = band
                st.markdown(render_tile(label, low_r, high_r, last_price, invest_amt), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">📊 Price History</div>', unsafe_allow_html=True)
            st.markdown('<div class="block">', unsafe_allow_html=True)
            st.line_chart(px.rename("Price"))
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">📉 Distribution of Daily Returns</div>', unsafe_allow_html=True)
            st.caption("Histogram of daily returns with a normal curve overlay (based on your log/simple choice above).")
            mu, sigma = mean_d, std_d
            if sigma == 0 or np.isnan(sigma):
                st.info("Not enough variability to plot a distribution.")
            else:
                x = np.linspace(rets_d.min() * 1.2, rets_d.max() * 1.2, 400)
                norm_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                fig, ax = plt.subplots()
                ax.hist(rets_d.values, bins=60, density=True, alpha=0.65)
                ax.plot(x, norm_pdf)
                ax.set_xlabel("Daily Return" + (" (log)" if use_log else " (simple)"))
                ax.set_ylabel("Density")
                st.pyplot(fig, clear_figure=True)
except Exception as e:
    st.error(f"Error loading data: {e}")


