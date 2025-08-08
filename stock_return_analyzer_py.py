# app.py
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date

st.set_page_config(page_title="Return Range Explorer", layout="wide")

# =========================
# Constants (finance math)
# =========================
PPY = 252              # trading days per year
TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_YEAR = 252

# =========================
# Core functions
# =========================
def get_price_series(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if data.empty:
        return pd.Series(dtype=float)
    price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
    s = pd.to_numeric(data[price_col].squeeze(), errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    return s

def compute_returns(price: pd.Series, use_log: bool):
    if price.empty or len(price) < 3:
        return pd.Series(dtype=float)
    rets = np.log(price / price.shift(1)) if use_log else price.pct_change()
    return rets.dropna()

def annualize_from_daily(mean_d, std_d, use_log: bool):
    ann_return = np.exp(mean_d * PPY) - 1 if use_log else (1 + mean_d) ** PPY - 1
    ann_vol = std_d * np.sqrt(PPY)
    return float(ann_return), float(ann_vol)

def cagr(price: pd.Series):
    if len(price) < 2:
        return np.nan
    start, end = float(price.iloc[0]), float(price.iloc[-1])
    years = (price.index[-1] - price.index[0]).days / 365.25
    if years <= 0 or start <= 0:
        return np.nan
    return (end / start) ** (1 / years) - 1

# ----- Adaptive horizons helpers -----
def pick_horizons(n_trading_days: int):
    """
    Decide which horizons to show based on selected window size.
    Returns a list of (label, periods) in trading days.
    """
    if n_trading_days < 15:
        return [("Day", 1)]
    elif n_trading_days < 40:
        return [("Day", 1), ("Month (~21d)", TRADING_DAYS_PER_MONTH)]
    else:
        return [
            ("Day", 1),
            ("Month (~21d)", TRADING_DAYS_PER_MONTH),
            ("Year (252d)", TRADING_DAYS_PER_YEAR),
        ]

def range_from_daily(mean_d, std_d, periods: int, N: int, use_log: bool, cagr_for_year: float | None = None):
    """
    Min/max total return range for a given horizon.
    - For the Year (252d) horizon, if cagr_for_year is provided, use CAGR as the center.
    - Otherwise scale from daily mean/std.
    """
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

# =========================
# Sidebar Controls
# =========================
st.sidebar.markdown("## ⚙️ Controls")
ticker = st.sidebar.text_input("Ticker", value="AAPL").strip().upper()
start_date = st.sidebar.date_input("Start Date", value=date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())
use_log = st.sidebar.toggle("Use log returns (recommended)", value=True)
N = st.sidebar.selectbox("Confidence (±σ)", [1, 2, 3], index=1)
invest_amt = st.sidebar.number_input("Investment Amount ($)", min_value=0.0, value=10000.0, step=100.0)
go = st.sidebar.button("🚀 Run Analysis", type="primary")

# =========================
# Finance Theme CSS (bigger text)
# =========================
st.markdown("""
<style>
:root {
  --bg:#0b1220; --panel:#0f1a2b; --panel2:#0c1526; --text:#e6eefc; --muted:#9db1d6;
  --green:#2ecc71; --red:#ff5c5c; --gold:#f5c15c; --blue:#3fa9ff;
}
html, body, [data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg, #09111f, #0b1220 20%, #09111f 100%);
  color: var(--text); font-size: 18px;
}
section[data-testid="stSidebar"] { background: #0a1322; }
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] span { color: var(--text) !important; font-size: 16px; }

.hero {
  background: linear-gradient(140deg, #0d1628, #0a1221);
  border: 1px solid #16233a; border-radius: 18px;
  padding: 28px 30px; margin-bottom: 18px;
  box-shadow: 0 12px 28px rgba(0,0,0,0.35);
}
.hero h1 { margin:0 0 4px 0; font-size: 34px; letter-spacing: .3px; }
.hero p { color: var(--muted); margin: 2px 0 0 0; font-size: 18px; }

.grid-4 { display:grid; grid-template-columns:repeat(4,minmax(180px,1fr)); gap:16px; margin:8px 0 18px; }
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border:1px solid #15243d; border-radius:16px; padding:18px;
  box-shadow: 0 10px 26px rgba(0,0,0,0.35);
  transition: transform .12s ease, border-color .12s ease, box-shadow .12s ease;
}
.card:hover { transform: translateY(-2px); border-color:#25507f; box-shadow: 0 14px 32px rgba(0,0,0,0.45); }
.card .label { font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
.card .value { font-size: 28px; margin-top: 6px; font-weight: 600; }
.card .value.green { color: var(--green); }
.card .value.red { color: var(--red); }
.card .value.gold { color: var(--gold); }

.section-title { margin:16px 0 8px; color: var(--text); font-weight:700; font-size: 22px; }
.grid-3 { display:grid; grid-template-columns:repeat(3,minmax(260px,1fr)); gap:16px; }
.tile {
  background: var(--panel); border:1px solid #17273f; border-radius:16px; padding:16px;
  box-shadow: 0 8px 22px rgba(0,0,0,0.35);
}
.tile h3 { margin-top:0; font-size: 20px; font-weight: 700; }
.badge { display:inline-block; font-size:12px; padding:4px 10px; border-radius:999px; background:#102038; color: var(--muted); margin-left:8px; }
.pill { display:inline-block; border-radius:10px; padding:7px 12px; font-size:15px; margin:6px 6px 0 0; background:#0f1a2b; border:1px solid #193154; }
.pill.green { color: var(--green); border-color: #1b6b44; background: rgba(46,204,113,0.08); }
.pill.red   { color: var(--red); border-color: #803b3b; background: rgba(255,92,92,0.08); }
.pill.gold  { color: var(--gold); border-color: #7a6022; background: rgba(245,193,92,0.08); }

.block { background: var(--panel2); border:1px solid #162742; border-radius:16px; padding:14px; margin-top:14px; box-shadow: 0 8px 22px rgba(0,0,0,0.35); }

/* Built-in metric text sizes */
[data-testid="stMetricValue"]{ font-size: 28px !important; }
[data-testid="stMetricLabel"]{ font-size: 16px !important; color: var(--muted) !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown(f"""
<div class="hero">
  <h1>📈 Return Range Explorer — Finance Theme</h1>
  <p>Auto-adaptive day / month / year ranges (±σ), a return distribution view, and clean price history — for {ticker or "your ticker"}.</p>
</div>
""", unsafe_allow_html=True)

# =========================
# Analysis
# =========================
if not go:
    st.info("Set your options in the left sidebar and click **Run Analysis**. Try AAPL / MSFT / AMZN.")
else:
    px = get_price_series(ticker, start_date, end_date)
    if px.empty:
        st.error("No data found. Try another ticker or a wider date range.")
        st.stop()

    last_price = float(px.iloc[-1])
    rets_d = compute_returns(px, use_log=use_log)
    if rets_d.empty:
        st.error("Not enough data to compute returns.")
        st.stop()

    mean_d, std_d = float(rets_d.mean()), float(rets_d.std())
    ann_return, ann_vol = annualize_from_daily(mean_d, std_d, use_log=use_log)
    growth_cagr = cagr(px)

    # =========================
    # Stat cards row
    # =========================
    def colored_value(val):
        if pd.isna(val): return '<span class="value">—</span>'
        return f'<span class="value {"green" if val >= 0 else "red"}">{val:.2%}</span>'

    st.markdown('<div class="grid-4">', unsafe_allow_html=True)
    st.markdown(f"""
      <div class="card"><div class="label">Last Price</div><div class="value gold">${last_price:,.2f}</div></div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
      <div class="card"><div class="label">CAGR (start → end)</div>{colored_value(0 if np.isnan(growth_cagr) else growth_cagr)}</div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
      <div class="card"><div class="label">Annualized Return</div>{colored_value(ann_return)}</div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
      <div class="card"><div class="label">Annualized Volatility</div><span class="value">{ann_vol:.2%}</span></div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # Adaptive Ranges Section
    # =========================
    n_trading_days = len(rets_d)  # number of daily return observations
    horizons = pick_horizons(n_trading_days)

    st.markdown(f'<div class="section-title">🎯 Expected Total Return Ranges (±{N}σ)</div>', unsafe_allow_html=True)

    def render_tile(title, low_r, high_r):
        low_price = last_price * (1 + low_r)
        high_price = last_price * (1 + high_r)
        invest_line = ""
        if invest_amt > 0:
            lower_dollar = invest_amt * (1 + low_r)
            upper_dollar = invest_amt * (1 + high_r)
            invest_line = f'<div class="pill gold">💵 ${invest_amt:,.0f} → ${lower_dollar:,.0f} → ${upper_dollar:,.0f}</div>'
        return f"""
        <div class="tile">
          <h3>{title} <span class="badge">{'Log' if use_log else 'Simple'} returns</span></h3>
          <div class="pill red">📉 Min return: {low_r:.2%}</div>
          <div class="pill green">📈 Max return: {high_r:.2%}</div>
          <div class="pill">💲 Price range: ${low_price:,.2f} → ${high_price:,.2f}</div>
          {invest_line}
        </div>
        """

    st.markdown('<div class="grid-3">', unsafe_allow_html=True)
    for label, periods in horizons:
        cagr_center = growth_cagr if periods == TRADING_DAYS_PER_YEAR else None
        low_r, high_r = range_from_daily(mean_d, std_d, periods, N, use_log, cagr_for_year=cagr_center)
        st.markdown(render_tile(label, low_r, high_r), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # Price history
    # =========================
    st.markdown('<div class="section-title">📊 Price History</div>', unsafe_allow_html=True)
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.line_chart(px.rename("Price"))
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # Return Distribution (histogram + normal overlay)
    # =========================
    st.markdown('<div class="section-title">📉 Return Distribution (Daily)</div>', unsafe_allow_html=True)
    st.caption("Histogram of daily returns with a normal curve overlay. Toggle log/simple returns in the sidebar.")
    st.markdown('<div class="block">', unsafe_allow_html=True)

    mu, sigma = rets_d.mean(), rets_d.std()
    if sigma == 0 or np.isnan(sigma):
        st.write("Not enough variability to plot a distribution.")
    else:
        x = np.linspace(rets_d.min()*1.2, rets_d.max()*1.2, 500)
        norm_pdf = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mu)/sigma)**2)

        fig_hist, axh = plt.subplots()
        axh.hist(rets_d.values, bins=60, density=True, alpha=0.65)
        axh.plot(x, norm_pdf)
        axh.set_xlabel("Daily Return" + (" (log)" if use_log else " (simple)"))
        axh.set_ylabel("Density")
        st.pyplot(fig_hist, clear_figure=True)

    # =========================
    # Notes
    # =========================
    st.markdown('<div class="section-title">ℹ️ Notes</div>', unsafe_allow_html=True)
    st.markdown("""
- Ranges use historical daily mean and volatility scaled to each horizon.  
- **Year** range uses **CAGR** as its center (more realistic for realized growth).  
- **Log returns** are generally more consistent for compounding.  
- ±σ ranges assume roughly normal returns; markets can exceed these bounds.  
""")



    

       
