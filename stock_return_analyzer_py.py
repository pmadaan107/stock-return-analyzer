# app.py
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date

st.set_page_config(page_title="📊 Return Range Explorer", layout="wide")

# -----------------------------
# CONSTANTS
# -----------------------------
# Periods per year (trading calendar approximation)
PPY = 252
DAYS_PER_MONTH = 21  # average trading days/month

# -----------------------------
# CORE FUNCTIONS
# -----------------------------
def get_price_series(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if data.empty:
        return pd.Series(dtype=float)
    price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
    s = pd.to_numeric(data[price_col].squeeze(), errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    return s

def compute_returns(price: pd.Series, use_log: bool):
    """Return a Series of daily returns (simple or log)."""
    if price.empty or len(price) < 3:
        return pd.Series(dtype=float)
    return (np.log(price / price.shift(1)) if use_log else price.pct_change()).dropna()

def annualize_from_daily(mean_d, std_d, use_log: bool):
    """Annualized mean return & volatility from daily stats."""
    if use_log:
        ann_return = np.exp(mean_d * PPY) - 1
    else:
        ann_return = (1 + mean_d) ** PPY - 1  # approximation in simple space
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
    """
    Expected total return range over 'periods' trading days using mean ± N*std.
    - If use_log: aggregate in log space (exact for normal-lognormal model).
      low/high = exp((mu_d * p) ± N * (sigma_d * sqrt(p))) - 1
    - If simple: use compounding for mean and sqrt-time for vol (approx).
      range ≈ ( (1+mean_d)^p - 1 ) ± N * (std_d * sqrt(p))
      Lower is floored at -100%.
    """
    if use_log:
        mu_h = mean_d * periods
        sigma_h = std_d * np.sqrt(periods)
        low = np.exp(mu_h - N * sigma_h) - 1
        high = np.exp(mu_h + N * sigma_h) - 1
    else:
        mean_h = (1 + mean_d) ** periods - 1
        std_h = std_d * np.sqrt(periods)
        low = mean_h - N * std_h
        high = mean_h + N * std_h
        low = max(low, -1.0)  # cap at -100%
    return float(low), float(high)

def mc_price_cone(S0: float, mu_ann: float, sigma_ann: float, days: int = 252, paths: int = 300, seed: int = 7):
    """Simple GBM Monte Carlo to draw a 1-year price cone."""
    if S0 <= 0 or np.isnan(mu_ann) or np.isnan(sigma_ann) or sigma_ann < 0:
        return None
    rng = np.random.default_rng(seed)
    dt = 1 / PPY
    mu = mu_ann
    sigma = sigma_ann
    out = np.empty((days + 1, paths))
    out[0, :] = S0
    for t in range(1, days + 1):
        z = rng.standard_normal(paths)
        out[t, :] = out[t - 1, :] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return out

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Controls")
ticker = st.sidebar.text_input("Ticker", value="AAPL").strip().upper()
start_date = st.sidebar.date_input("Start Date", value=date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())
use_log = st.sidebar.toggle("Use log returns (recommended)", value=True)
N = st.sidebar.selectbox("Confidence (±σ)", [1, 2, 3], index=1)
invest_amt = st.sidebar.number_input("Investment Amount ($)", min_value=0.0, value=10000.0, step=100.0)
mc_paths = st.sidebar.slider("Monte Carlo paths", 100, 1000, 400, step=50)
analyze = st.sidebar.button("🚀 Analyze", type="primary")

# -----------------------------
# MAIN
# -----------------------------
st.title("📊 Return Range Explorer")
st.caption("Clean ranges for day / month / year + a simple Monte Carlo cone. No fuss.")

if analyze:
    # -- Data
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

    # -- Layout
    tab1, tab2 = st.tabs(["🎯 Expected Ranges", "🌀 Monte Carlo (1 year)"])

    # =============================
    # TAB 1: EXPECTED RANGES
    # =============================
    with tab1:
        st.subheader(f"Results for {ticker}")
        colA, colB, colC, colD = st.columns(4)
        with colA:
            st.metric("Last Price", f"${last_price:,.2f}")
        with colB:
            st.metric("CAGR (start→end)", f"{0 if np.isnan(growth_cagr) else growth_cagr:.2%}")
        with colC:
            st.metric("Annualized Return (from daily avg)", f"{ann_return:.2%}")
        with colD:
            st.metric("Annualized Volatility (Std Dev)", f"{ann_vol:.2%}")

        st.markdown("---")
        st.markdown(f"### 📆 Min / Max **Total Return** you can expect (±{N}σ)")

        # Compute ranges for Day / Month / Year
        ranges = {
            "Day": horizon_range_from_daily(mean_d, std_d, 1, N, use_log),
            "Month (~21d)": horizon_range_from_daily(mean_d, std_d, DAYS_PER_MONTH, N, use_log),
            "Year (252d)": horizon_range_from_daily(mean_d, std_d, PPY, N, use_log),
        }

        col1, col2, col3 = st.columns(3)
        for col, (label, (low_r, high_r)) in zip([col1, col2, col3], ranges.items()):
            with col:
                st.markdown(f"#### {label}")
                st.write(f"**Return Range:** {low_r:.2%} → {high_r:.2%}")
                low_price = last_price * (1 + low_r)
                high_price = last_price * (1 + high_r)
                st.write(f"**Price Range:** ${low_price:,.2f} → ${high_price:,.2f}")
                if invest_amt > 0:
                    lower_dollar = invest_amt * (1 + low_r)
                    upper_dollar = invest_amt * (1 + high_r)
                    st.write(f"**${invest_amt:,.0f} → ${lower_dollar:,.0f} → ${upper_dollar:,.0f}**")

        st.markdown("##### Notes")
        st.write(
            "- Ranges use historical daily mean and volatility, scaled to each horizon.\n"
            "- Log-returns give the most consistent scaling; simple returns use an approximation.\n"
            "- ±σ ranges assume roughly normal returns, so real markets can exceed these bounds."
        )

        st.markdown("### 📈 Price History")
        st.line_chart(px.rename("Price"))

    # =============================
    # TAB 2: MONTE CARLO
    # =============================
    with tab2:
        st.subheader("Easy 1-Year Price Cone")
        st.caption("We simulate many price paths using your annualized return & volatility. The shaded areas show typical outcomes.")

        paths = mc_price_cone(last_price, ann_return, ann_vol, days=PPY, paths=mc_paths)
        if paths is None:
            st.warning("Could not simulate paths—check your inputs.")
        else:
            percentiles = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
            t = np.arange(paths.shape[0])

            fig, ax = plt.subplots()
            ax.fill_between(t, percentiles[0], percentiles[4], alpha=0.2, label="5–95% band")
            ax.fill_between(t, percentiles[1], percentiles[3], alpha=0.3, label="25–75% band")
            ax.plot(t, percentiles[2], label="Median (50th pct)")
            ax.set_xlabel("Trading Days Ahead")
            ax.set_ylabel("Simulated Price")
            ax.legend(loc="best")
            st.pyplot(fig, clear_figure=True)

            end_pcts = percentiles[:, -1]
            colL, colR = st.columns(2)
            with colL:
                st.metric("Median Price (≈ 50%)", f"${end_pcts[2]:,.2f}")
                st.metric("Typical Range (25–75%)", f"${end_pcts[1]:,.2f} → ${end_pcts[3]:,.2f}")
            with colR:
                st.metric("Wide Range (5–95%)", f"${end_pcts[0]:,.2f} → ${end_pcts[4]:,.2f}")
                st.caption("These are simulated outcomes, not guarantees.")

else:
    st.info("Set your options in the sidebar and click **Analyze**. Try `AAPL`, `MSFT`, or `AMZN`.")



    

       
