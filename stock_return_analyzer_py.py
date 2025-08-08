# app.py
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date

st.set_page_config(page_title="Return Range Explorer", layout="wide")

# =========================
# Constants
# =========================
PPY = 252              # trading days per year
DAYS_PER_MONTH = 21    # avg trading days per month

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
    if use_log:
        ann_return = np.exp(mean_d * PPY) - 1
    else:
        ann_return = (1 + mean_d) ** PPY - 1
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

def horizon_range_from_daily(mean_d, std_d, periods: int, N: int, use_log: bool):
    if use_log:
        mu_h = mean_d * periods
        sigma_h = std_d * np.sqrt(periods)
        low = np.exp(mu_h - N * sigma_h) - 1
        high = np.exp(mu_h + N * sigma_h) - 1
    else:
        mean_h = (1 + mean_d) ** periods - 1
        std_h = std_d * np.sqrt(periods)
        low = max(mean_h - N * std_h, -1.0)
        high = mean_h + N * std_h
    return float(low), float(high)

def mc_price_cone(S0: float, mu_ann: float, sigma_ann: float, days: int = 252, paths: int = 400, seed: int = 7):
    if S0 <= 0 or np.isnan(mu_ann) or np.isnan(sigma_ann) or sigma_ann < 0:
        return None
    rng = np.random.default_rng(seed)
    dt = 1 / PPY
    out = np.empty((days + 1, paths))
    out[0, :] = S0
    drift = (mu_ann - 0.5 * sigma_ann**2) * dt
    vol_dt = sigma_ann * np.sqrt(dt)
    for t in range(1, days + 1):
        z = rng.standard_normal(paths)
        out[t, :] = out[t - 1, :] * np.exp(drift + vol_dt * z)
    return out

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
mc_paths = st.sidebar.slider("Monte Carlo paths", 100, 1000, 500, step=50)
go = st.sidebar.button("🚀 Run Analysis", type="primary")

# =========================
# Global CSS (HTML styling)
# =========================
st.markdown("""
<style>
/* Overall page */
.main {
  padding-top: 0 !important;
}
:root {
  --bg:#0f1220;
  --card:#151a2e;
  --card2:#11172a;
  --text:#e7ecff;
  --muted:#92a0c7;
  --accent:#6aa0ff;
  --success:#38d996;
  --warn:#f7b955;
  --danger:#ff6a6a;
}

/* App header */
.hero {
  background: radial-gradient(1000px 400px at 10% -10%, rgba(106,160,255,0.25), transparent 60%),
              radial-gradient(800px 300px at 110% 10%, rgba(56,217,150,0.20), transparent 60%),
              linear-gradient(160deg, #0F1320, #0a0d18);
  border-radius: 18px;
  padding: 26px 28px;
  margin-bottom: 18px;
  color: var(--text);
  border: 1px solid rgba(255,255,255,0.06);
  box-shadow: 0 10px 30px rgba(0,0,0,0.35), inset 0 0 60px rgba(255,255,255,0.03);
}
.hero h1 {
  margin: 0;
  font-size: 30px;
  letter-spacing: 0.3px;
}
.hero p {
  color: var(--muted);
  margin: 6px 0 0 0;
}

/* Stat cards */
.grid-4 {
  display: grid;
  grid-template-columns: repeat(4, minmax(160px, 1fr));
  gap: 14px;
  margin: 8px 0 18px;
}
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 16px;
  color: var(--text);
  box-shadow: 0 12px 22px rgba(0,0,0,0.25);
  position: relative;
  overflow: hidden;
  transition: transform .15s ease, border-color .15s ease;
}
.card:hover {
  transform: translateY(-2px);
  border-color: rgba(106,160,255,0.35);
}
.card .label {
  font-size: 12px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 1px;
}
.card .value {
  font-size: 24px;
  margin-top: 4px;
}

/* Range tiles */
.grid-3 {
  display: grid;
  grid-template-columns: repeat(3, minmax(240px, 1fr));
  gap: 14px;
}
.tile {
  background: var(--card);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 16px;
  color: var(--text);
  box-shadow: 0 8px 20px rgba(0,0,0,0.25);
}
.tile h3 {
  margin-top: 0;
  font-size: 18px;
}
.badge {
  display: inline-block;
  font-size: 11px;
  padding: 3px 8px;
  border-radius: 999px;
  background: rgba(255,255,255,0.07);
  color: var(--muted);
  margin-left: 6px;
}
.pill {
  display: inline-block;
  border-radius: 10px;
  padding: 6px 10px;
  font-size: 13px;
  margin: 4px 4px 0 0;
  background: rgba(255,255,255,0.06);
  color: var(--text);
}
.pill.green { background: rgba(56,217,150,0.18); }
.pill.red   { background: rgba(255,106,106,0.18); }

/* Section header */
.section-title {
  margin: 16px 0 8px;
  color: var(--text);
  font-weight: 600;
  font-size: 20px;
}

/* Dark background for charts */
.block {
  background: var(--card2);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 12px;
  margin-top: 14px;
  box-shadow: 0 8px 20px rgba(0,0,0,0.25);
}

/* Make Streamlit body darker */
section[data-testid="stSidebar"] div[role="radiogroup"] label { color: var(--text) }
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg);
  color: var(--text);
}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
  color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown(f"""
<div class="hero">
  <h1>📊 Return Range Explorer</h1>
  <p>Clean day / month / year ranges (±σ) and a simple 1-year Monte Carlo price cone — for {ticker or "your ticker"}.</p>
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

    # Ranges
    r_day_lo, r_day_hi = horizon_range_from_daily(mean_d, std_d, 1, N, use_log)
    r_mon_lo, r_mon_hi = horizon_range_from_daily(mean_d, std_d, DAYS_PER_MONTH, N, use_log)
    r_yr_lo,  r_yr_hi  = horizon_range_from_daily(mean_d, std_d, PPY, N, use_log)

    # =========================
    # Stat cards
    # =========================
    st.markdown('<div class="grid-4">', unsafe_allow_html=True)
    st.markdown(f"""
      <div class="card">
        <div class="label">Last Price</div>
        <div class="value">${last_price:,.2f}</div>
      </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
      <div class="card">
        <div class="label">CAGR (start → end)</div>
        <div class="value">{(0 if np.isnan(growth_cagr) else growth_cagr):.2%}</div>
      </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
      <div class="card">
        <div class="label">Annualized Return</div>
        <div class="value">{ann_return:.2%}</div>
      </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
      <div class="card">
        <div class="label">Annualized Volatility</div>
        <div class="value">{ann_vol:.2%}</div>
      </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # Ranges Section
    # =========================
    st.markdown(f'<div class="section-title">🎯 Expected Total Return Ranges (±{N}σ)</div>', unsafe_allow_html=True)

    def tile_html(title, low_r, high_r):
        low_price = last_price * (1 + low_r)
        high_price = last_price * (1 + high_r)
        invest_low = invest_amt * (1 + low_r) if invest_amt > 0 else None
        invest_high = invest_amt * (1 + high_r) if invest_amt > 0 else None
        invest_line = ""
        if invest_amt > 0:
            invest_line = f'<div class="pill">💵 ${invest_amt:,.0f} → ${invest_low:,.0f} → ${invest_high:,.0f}</div>'
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
    st.markdown(tile_html("Day", r_day_lo, r_day_hi), unsafe_allow_html=True)
    st.markdown(tile_html("Month (~21 trading days)", r_mon_lo, r_mon_hi), unsafe_allow_html=True)
    st.markdown(tile_html("Year (252 trading days)", r_yr_lo, r_yr_hi), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # Price history (dark card)
    # =========================
    st.markdown('<div class="section-title">📈 Price History</div>', unsafe_allow_html=True)
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.line_chart(px.rename("Price"))
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # Monte Carlo Cone
    # =========================
    st.markdown('<div class="section-title">🌀 Monte Carlo Price Cone (1 Year)</div>', unsafe_allow_html=True)
    st.caption("We simulate many price paths using your annualized return & volatility, then show the 25–75% and 5–95% bands (easy to read).")
    paths = mc_price_cone(last_price, ann_return, ann_vol, days=PPY, paths=mc_paths)

    if paths is None:
        st.warning("Could not simulate paths — check the inputs.")
    else:
        pct = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
        t = np.arange(paths.shape[0])

        fig, ax = plt.subplots()
        ax.fill_between(t, pct[0], pct[4], alpha=0.2, label="5–95% band")
        ax.fill_between(t, pct[1], pct[3], alpha=0.3, label="25–75% band")
        ax.plot(t, pct[2], label="Median (50th pct)")
        ax.set_xlabel("Trading Days Ahead")
        ax.set_ylabel("Simulated Price")
        ax.legend(loc="best")
        st.pyplot(fig, clear_figure=True)

        end5, end25, end50, end75, end95 = pct[0, -1], pct[1, -1], pct[2, -1], pct[3, -1], pct[4, -1]
        # Nice HTML summary row
        st.markdown("""
        <div class="grid-3" style="margin-top:10px;">
          <div class="tile">
            <h3>Median Outcome</h3>
            <div class="pill">Price ≈ ${:,.2f}</div>
          </div>
          <div class="tile">
            <h3>Typical Range (25–75%)</h3>
            <div class="pill">Price ≈ ${:,.2f} → ${:,.2f}</div>
          </div>
          <div class="tile">
            <h3>Wide Range (5–95%)</h3>
            <div class="pill">Price ≈ ${:,.2f} → ${:,.2f}</div>
          </div>
        </div>
        """.format(end50, end25, end75, end5, end95), unsafe_allow_html=True)

    # Notes
    st.markdown('<div class="section-title">ℹ️ Notes</div>', unsafe_allow_html=True)
    st.markdown("""
- Ranges use historical daily mean and volatility and scale them to each horizon.  
- **Log returns** are generally better for compounding/aggregation.  
- ±σ ranges rely on a normal-return assumption — real markets can exceed these bounds.  
- The Monte Carlo cone is **illustrative**, not predictive.
""")




    

       
