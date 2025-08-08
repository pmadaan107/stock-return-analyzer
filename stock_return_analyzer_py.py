# app.py
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import io
from datetime import date
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 Stock Return & Risk Analyzer", layout="wide")

# -----------------------------
# CONSTANTS
# -----------------------------
FREQ_MAP = {
    "Daily":   {"ppy": 252, "resample": None},
    "Weekly":  {"ppy": 52,  "resample": "W-FRI"},
    "Monthly": {"ppy": 12,  "resample": "M"},
}

# -----------------------------
# FUNCTIONS
# -----------------------------
def compute_returns(price_series: pd.Series, freq: str, use_log: bool):
    """Resample prices, then compute simple or log returns."""
    if FREQ_MAP[freq]["resample"]:
        px = price_series.resample(FREQ_MAP[freq]["resample"]).last().dropna()
    else:
        px = price_series.dropna()

    if len(px) < 3:
        return px, pd.Series(dtype=float)

    rets = np.log(px / px.shift(1)).dropna() if use_log else px.pct_change().dropna()
    return px, rets

def annualize_mean_return(avg_period_return: float, ppy: int, use_log: bool) -> float:
    """Annualized mean return from period mean."""
    return float(np.exp(avg_period_return * ppy) - 1) if use_log else float((1 + avg_period_return) ** ppy - 1)

def annualize_vol(period_std: float, ppy: int) -> float:
    """sigma_annual = sigma_period * sqrt(ppy)"""
    return float(period_std * np.sqrt(ppy))

def compute_cagr(px) -> float:
    """CAGR based on start and end prices."""
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    px = px.dropna()
    if len(px) < 2:
        return np.nan
    idx = pd.to_datetime(px.index)
    n_years = (idx[-1] - idx[0]).days / 365.25
    if n_years <= 0:
        return np.nan
    start_price = float(px.iloc[0])
    end_price = float(px.iloc[-1])
    if start_price <= 0:
        return np.nan
    return (end_price / start_price) ** (1 / n_years) - 1

def sharpe_ratio(ann_return: float, ann_vol: float, rf: float) -> float:
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return (ann_return - rf) / ann_vol

def bollinger_bands(price: pd.Series, window: int = 20, k: float = 2.0):
    """Classic Bollinger Bands on price: SMA ± k * rolling_std(price)."""
    sma = price.rolling(window=window).mean()
    stdp = price.rolling(window=window).std()
    upper = sma + k * stdp
    lower = sma - k * stdp
    return sma, upper, lower

def rolling_annual_vol(returns: pd.Series, window: int, ppy: int):
    """Rolling annualized volatility from period returns."""
    return returns.rolling(window=window).std() * np.sqrt(ppy)

def simulate_gbm_paths(S0: float, mu: float, sigma: float, days: int = 252, n_paths: int = 200, seed: int = 42):
    """Monte Carlo GBM paths for price cone visualization."""
    rng = np.random.default_rng(seed)
    dt = 1/252
    # preallocate
    paths = np.empty((days+1, n_paths))
    paths[0, :] = S0
    # simulate
    for t in range(1, days+1):
        z = rng.standard_normal(n_paths)
        paths[t, :] = paths[t-1, :] * np.exp((mu - 0.5 * sigma**2)*dt + sigma * np.sqrt(dt) * z)
    return paths

def download_df_button(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv().encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

# -----------------------------
# SIDEBAR (Controls)
# -----------------------------
st.sidebar.header("⚙️ Controls")
tickers_raw = st.sidebar.text_input("Tickers (comma-separated)", value="AAPL, MSFT").upper()
freq = st.sidebar.selectbox("Return Frequency", list(FREQ_MAP.keys()), index=0)
start_date = st.sidebar.date_input("Start Date", value=date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())
use_log = st.sidebar.toggle("Use Log Returns (recommended)", value=True)
sigma_choice = st.sidebar.selectbox("Confidence Level (σ)", [1, 2, 3], index=1)
invest_amt = st.sidebar.number_input("Investment Amount ($)", min_value=0.0, value=10000.0, step=100.0)
rf_input = st.sidebar.number_input("Risk-free rate (annual, %)", value=2.0, step=0.25) / 100.0
bb_window = st.sidebar.number_input("Bollinger window (days)", min_value=10, value=20, step=1)
bb_k = st.sidebar.number_input("Bollinger k (std devs)", min_value=1.0, value=2.0, step=0.5)
roll_vol_window = st.sidebar.number_input("Rolling Vol window (periods)", min_value=10, value=30, step=5)
n_paths = st.sidebar.number_input("Monte Carlo paths", min_value=50, value=200, step=50)
analyze = st.sidebar.button("🔍 Analyze", type="primary")

# -----------------------------
# MAIN
# -----------------------------
st.title("📊 Stock Return & Risk Analyzer (Pro)")

if analyze:
    tickers = [t.strip() for t in tickers_raw.split(",") if t.strip()]
    if not tickers:
        st.error("Please enter at least one ticker.")
        st.stop()

    # Download all tickers in one call where possible
    try:
        data_all = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
    except Exception as e:
        st.error(f"Download error: {e}")
        st.stop()

    if data_all.empty:
        st.error("No data found for these settings.")
        st.stop()

    # Normalize to DataFrame with columns per ticker for Close/Adj Close
    # yfinance shape differs for single vs multiple tickers. Handle both.
    if isinstance(data_all.columns, pd.MultiIndex):
        close_df = data_all["Adj Close"] if "Adj Close" in data_all.columns.levels[0] else data_all["Close"]
    else:
        # single ticker returns a Series per column; make DataFrame with one column name
        col = "Adj Close" if "Adj Close" in data_all.columns else "Close"
        close_df = data_all[[col]].rename(columns={col: tickers[0]})

    close_df = close_df.dropna(how="all")
    close_df.index = pd.to_datetime(close_df.index)

    # ---------- MULTI-TICKER METRICS TABLE ----------
    rows = []
    details = {}
    ppy = FREQ_MAP[freq]["ppy"]

    for t in tickers:
        if t not in close_df.columns:
            continue
        px_raw = pd.to_numeric(close_df[t], errors="coerce").dropna()
        if px_raw.empty:
            continue

        px, rets = compute_returns(px_raw, freq=freq, use_log=use_log)
        if rets.empty:
            continue

        avg_period_ret = float(rets.mean())
        period_std = float(rets.std())
        ann_return = annualize_mean_return(avg_period_ret, ppy, use_log)
        ann_vol = annualize_vol(period_std, ppy)
        cagr = compute_cagr(px)
        sr = sharpe_ratio(ann_return, ann_vol, rf_input)

        rows.append({
            "Ticker": t,
            f"{freq} Mean Ret": avg_period_ret,
            f"{freq} Std Dev": period_std,
            "Ann Return (mean)": ann_return,
            "Ann Vol": ann_vol,
            "CAGR": cagr,
            "Sharpe": sr
        })

        details[t] = {"px": px, "rets": rets, "ann_return": ann_return, "ann_vol": ann_vol, "cagr": cagr}

    if not rows:
        st.error("No valid series to compute metrics.")
        st.stop()

    metrics_df = pd.DataFrame(rows).set_index("Ticker")
    fmt_df = metrics_df.copy()
    for c in fmt_df.columns:
        if "Std Dev" in c or "Ret" in c or "Ann" in c or "CAGR" in c or "Vol" in c:
            fmt_df[c] = fmt_df[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
        if c == "Sharpe":
            fmt_df[c] = fmt_df[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    st.subheader("📋 Summary (Performance & Risk)")
    st.dataframe(fmt_df, use_container_width=True)

    download_df_button(metrics_df, "metrics.csv", "⬇️ Download metrics CSV")

    # ---------- DETAILED SECTION FOR FIRST TICKER ----------
    focus = tickers[0]
    st.markdown(f"---\n## 🔎 Detailed View: **{focus}**")

    px = details[focus]["px"]
    rets = details[focus]["rets"]
    ann_return = details[focus]["ann_return"]
    ann_vol = details[focus]["ann_vol"]
    cagr = details[focus]["cagr"]

    N = sigma_choice
    low_ann = ann_return - N * ann_vol
    high_ann = ann_return + N * ann_vol
    last_price = float(px.iloc[-1])
    low_price = last_price * (1 + low_ann)
    high_price = last_price * (1 + high_ann)

    colP, colR = st.columns([2, 1])

    with colP:
        st.markdown("### 📈 Price with Bollinger Bands")
        sma, upper, lower = bollinger_bands(px, window=int(bb_window), k=float(bb_k))

        fig1, ax1 = plt.subplots()
        ax1.plot(px.index, px.values, label="Price")
        ax1.plot(sma.index, sma.values, label=f"SMA {bb_window}")
        ax1.plot(upper.index, upper.values, label=f"Upper ({bb_k}σ)")
        ax1.plot(lower.index, lower.values, label=f"Lower ({bb_k}σ)")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        ax1.legend()
        st.pyplot(fig1, clear_figure=True)

    with colR:
        st.markdown("### 🎯 Expected Annual Range")
        st.metric("Last Price", f"${last_price:,.2f}")
        st.metric("Annualized Return (mean)", f"{ann_return:.2%}")
        st.metric("Annualized Volatility", f"{ann_vol:.2%}")
        st.write(f"**{N}σ Return Range:** {low_ann:.2%} → {high_ann:.2%}")
        st.write(f"**Implied 1y Price:** ${low_price:,.2f} → ${high_price:,.2f}")
        st.write(f"**CAGR (start→end):** {0 if np.isnan(cagr) else cagr:.2%}")

    # Rolling annualized volatility
    st.markdown("### 📉 Rolling Annualized Volatility")
    roll_vol = rolling_annual_vol(rets, int(roll_vol_window), ppy)
    fig2, ax2 = plt.subplots()
    ax2.plot(roll_vol.index, roll_vol.values)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Annualized Volatility")
    st.pyplot(fig2, clear_figure=True)

    # Histogram of returns with normal overlay
    st.markdown(f"### 🧰 Distribution of {freq} Returns")
    mu = rets.mean()
    sigma = rets.std()
    fig3, ax3 = plt.subplots()
    ax3.hist(rets.values, bins=50, density=True, alpha=0.6)
    # Normal overlay
    x = np.linspace(rets.min()*1.2, rets.max()*1.2, 400)
    norm_pdf = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mu)/sigma)**2)
    ax3.plot(x, norm_pdf)
    ax3.set_xlabel("Return")
    ax3.set_ylabel("Density")
    st.pyplot(fig3, clear_figure=True)

    # Monte Carlo price cone
    st.markdown("### 🌀 Monte Carlo Price Cone (1 Year)")
    paths = simulate_gbm_paths(S0=last_price, mu=ann_return, sigma=ann_vol, days=252, n_paths=int(n_paths))
    pc = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)  # 5-95 fan
    t = np.arange(paths.shape[0])

    fig4, ax4 = plt.subplots()
    ax4.fill_between(t, pc[0], pc[4], alpha=0.2, label="5–95%")
    ax4.fill_between(t, pc[1], pc[3], alpha=0.3, label="25–75%")
    ax4.plot(t, pc[2], label="Median")
    ax4.set_xlabel("Trading Days Ahead")
    ax4.set_ylabel("Simulated Price")
    ax4.legend()
    st.pyplot(fig4, clear_figure=True)

    # Downloads: detailed series
    ret_df = pd.DataFrame({
        "price": px,
        f"{freq}_return": rets
    })
    download_df_button(ret_df, f"{focus}_price_returns.csv", f"⬇️ Download {focus} price & returns")

else:
    st.info("Set your options in the sidebar and click **Analyze**. Try multiple tickers (e.g., `AAPL, MSFT, AMZN`).")


    

       
