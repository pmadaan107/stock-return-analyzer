import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =======================
# Finance Dashboard Theme
# =======================
st.set_page_config(page_title="📊 Stock Return Analyzer", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #0d1117;
        color: #e6edf3;
    }
    .section-title {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #00ff99 !important;
        margin-top: 20px !important;
    }
    .tile {
        background-color: #161b22;
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #30363d;
        margin-bottom: 12px;
    }
    .tile h3 {
        font-size: 20px;
        color: #f0f6fc;
        margin-bottom: 10px;
    }
    .pill {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 16px;
        margin: 2px;
    }
    .pill.red { background-color: #ff4d4d; color: white; }
    .pill.green { background-color: #00cc66; color: white; }
    .pill.gold { background-color: #d4af37; color: black; }
    .grid-3 {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# =======================
# Constants
# =======================
TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_YEAR = 252
NORMAL_SIGMA_TO_PCTS = {
    1: (15.865, 84.135),
    2: (2.275, 97.725),
    3: (0.135, 99.865),
}

# =======================
# Helper Functions
# =======================
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
        return np.exp(roll_sum) - 1
    else:
        sr = price_series.pct_change().dropna()
        roll_prod = (1 + sr).rolling(window_days).apply(lambda x: np.prod(x), raw=True).dropna()
        return roll_prod - 1

def empirical_band(price_series: pd.Series, window_days: int, N: int, use_log: bool):
    pct_lo, pct_hi = NORMAL_SIGMA_TO_PCTS.get(N, (15.865, 84.135))
    rets = compounded_window_returns(price_series, window_days, use_log)
    if len(rets) < 30:
        return None
    lo = float(np.nanpercentile(rets, pct_lo))
    hi = float(np.nanpercentile(rets, pct_hi))
    lo = max(lo, -1.0)
    return lo, hi

def range_from_daily(mean_d, std_d, periods: int, N: int, use_log: bool, cagr_for_year: float | None = None):
    if cagr_for_year is not None and periods == TRADING_DAYS_PER_YEAR and not np.isnan(cagr_for_year):
        sigma_year = std_d * np.sqrt(TRADING_DAYS_PER_YEAR)
        low = cagr_for_year - N * sigma_year
        high = cagr_for_year + N * sigma_year
        return float(max(low, -1.0)), float(high)

    if use_log:
        mu_h = mean_d * periods
        sigma_h = std_d * np.sqrt(periods)
        low = np.exp(mu_h - N * sigma_h) - 1
        high = np.exp(mu_h + N * sigma_h) - 1
    else:
        mean_h = (1 + mean_d) ** periods - 1
        sigma_h = std_d * np.sqrt(periods)
        low = max(mean_h - N * sigma_h, -1.0)
        high = mean_h + N * sigma_h
    return float(low), float(high)

def render_tile(title, low_r, high_r, last_price, invest_amt):
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

# =======================
# Sidebar Inputs
# =======================
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
use_log = st.sidebar.checkbox("Use Log Returns", value=False)
N = st.sidebar.selectbox("Std Dev Multiplier (N)", [1, 2, 3], index=1)
invest_amt = st.sidebar.number_input("Investment Amount ($)", value=0, step=100)

# =======================
# Data Download & Metrics
# =======================
try:
    data = yf.download(ticker, start=start_date, end=end_date)
    px = data['Adj Close'].dropna()

    if len(px) < 2:
        st.error("Not enough data for the selected period.")
    else:
        rets_d = px.pct_change().dropna() if not use_log else np.log(px / px.shift(1)).dropna()
        mean_d = rets_d.mean()
        std_d = rets_d.std()

        # CAGR
        total_period_years = (px.index[-1] - px.index[0]).days / 365.25
        growth_cagr = (px.iloc[-1] / px.iloc[0]) ** (1 / total_period_years) - 1

        # Annualized volatility
        ann_vol = std_d * np.sqrt(TRADING_DAYS_PER_YEAR)

        last_price = px.iloc[-1]

        # =======================
        # Key Metrics
        # =======================
        st.markdown(f'<div class="section-title">📌 Key Metrics</div>', unsafe_allow_html=True)
        st.markdown('<div class="grid-3">', unsafe_allow_html=True)
        st.markdown(render_tile("CAGR", growth_cagr, growth_cagr, last_price, 0), unsafe_allow_html=True)
        st.markdown(render_tile("Annual Volatility", ann_vol, ann_vol, last_price, 0), unsafe_allow_html=True)
        st.markdown(render_tile("Last Price", 0, 0, last_price, 0), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # =======================
        # Ranges Section
        # =======================
        n_trading_days = len(rets_d)
        horizons = pick_horizons(n_trading_days)
        st.markdown(f'<div class="section-title">🎯 Expected Total Return Ranges (±{N}σ)</div>', unsafe_allow_html=True)
        st.markdown('<div class="grid-3">', unsafe_allow_html=True)
        for label, periods in horizons:
            band = empirical_band(px, periods, N, use_log)
            if band is None:
                cagr_center = growth_cagr if periods == TRADING_DAYS_PER_YEAR else None
                low_r, high_r = range_from_daily(mean_d, std_d, periods, N, use_log, cagr_for_year=cagr_center)
            else:
                low_r, high_r = band
            st.markdown(render_tile(label, low_r, high_r, last_price, invest_amt), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # =======================
        # Distribution Curve
        # =======================
        st.markdown(f'<div class="section-title">📊 Distribution of Daily Returns</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(rets_d, bins=50, density=True, alpha=0.6, color='#00cc66')
        mu, sigma = mean_d, std_d
        x = np.linspace(rets_d.min(), rets_d.max(), 100)
        from scipy.stats import norm
        ax.plot(x, norm.pdf(x, mu, sigma), 'r', linewidth=2)
        ax.set_title(f"Distribution of {ticker} Daily Returns", fontsize=14)
        ax.set_xlabel("Return")
        ax.set_ylabel("Density")
        st.pyplot(fig)

except Exception as e:
    st.error(f"Error loading data: {e}")

    

       
