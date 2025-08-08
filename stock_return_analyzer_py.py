import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import date

st.set_page_config(page_title="Return & Volatility Analyzer", layout="centered")

# ---------- Helpers ----------
FREQ_MAP = {
    "Daily": {"ppy": 252, "resample": None},         # Use raw daily
    "Weekly": {"ppy": 52, "resample": "W-FRI"},      # Friday close
    "Monthly": {"ppy": 12, "resample": "M"},         # Month-end close
}

def compute_returns(price_series: pd.Series, freq: str, use_log: bool):
    """Resample price if needed, then compute % returns (simple or log)."""
    if FREQ_MAP[freq]["resample"]:
        px = price_series.resample(FREQ_MAP[freq]["resample"]).last().dropna()
    else:
        px = price_series.dropna()

    if len(px) < 3:
        return px, pd.Series(dtype=float)  # not enough data

    if use_log:
        rets = np.log(px / px.shift(1)).dropna()
    else:
        rets = px.pct_change().dropna()
    return px, rets

def annualize_mean_return(avg_period_return: float, ppy: int, use_log: bool):
    if use_log:
        # log returns compound via exponent
        return float(np.exp(avg_period_return * ppy) - 1)
    else:
        return float((1 + avg_period_return) ** ppy - 1)

def annualize_vol(period_std: float, ppy: int):
    return float(period_std * np.sqrt(ppy))

def compute_cagr(px: pd.Series):
    if len(px) < 2:
        return np.nan
    start_price = px.iloc[0]
    end_price = px.iloc[-1]
    n_years = (px.index[-1] - px.index[0]).days / 365.25
    if n_years <= 0 or start_price <= 0:
        return np.nan
    return float((end_price / start_price) ** (1 / n_years) - 1)

# ---------- UI ----------
st.title("📈 Return & Volatility Analyzer")

colA, colB = st.columns(2)
with colA:
    ticker = st.text_input("Ticker", value="AAPL").strip()
with colB:
    freq = st.selectbox("Return Frequency", list(FREQ_MAP.keys()), index=0)

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=date(2020, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=date.today())

col3, col4, col5 = st.columns(3)
with col3:
    use_log = st.toggle("Use log returns (recommended for volatility)", value=True)
with col4:
    sigma_choice = st.selectbox("Confidence (σ)", [1, 2, 3], index=1)
with col5:
    invest_amt = st.number_input("Investment Amount ($)", min_value=0.0, value=10000.0, step=100.0)

if st.button("Analyze", type="primary"):
    if not ticker:
        st.error("Please enter a ticker.")
        st.stop()

    with st.spinner("Downloading and crunching numbers…"):
        try:
            # yfinance changed defaults; be explicit
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        except Exception as e:
            st.error(f"Download error: {e}")
            st.stop()

        if data.empty or "Adj Close" not in data.columns and "Close" not in data.columns:
            st.error("No price data found for that period/ticker.")
            st.stop()

        # Prefer Adj Close, else Close
        price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
        px_raw = data[price_col].copy()
        px_raw.index = pd.to_datetime(px_raw.index)

        # Returns & resampling
        px, rets = compute_returns(px_raw, freq=freq, use_log=use_log)
        if rets.empty:
            st.error("Not enough data to compute returns. Try a longer date range.")
            st.stop()

        ppy = FREQ_MAP[freq]["ppy"]

        # Stats
        avg_period_ret = float(rets.mean())
        period_std = float(rets.std())
        ann_return_from_avg = annualize_mean_return(avg_period_ret, ppy, use_log=use_log)
        ann_vol = annualize_vol(period_std, ppy)
        cagr = compute_cagr(px)

        # Return bands (mean ± Nσ on annual scale)
        N = sigma_choice
        low_ann = ann_return_from_avg - N * ann_vol
        high_ann = ann_return_from_avg + N * ann_vol

        # Convert return bands to price bands (based on last price)
        last_price = float(px.iloc[-1])
        low_price = last_price * (1 + low_ann)
        high_price = last_price * (1 + high_ann)

        # Dollar outcomes for user-entered investment
        lower_dollar = invest_amt * (1 + low_ann)
        upper_dollar = invest_amt * (1 + high_ann)

    # ---------- Output ----------
    st.subheader(f"Results for {ticker.upper()}")
    st.markdown(
        f"""
**Window:** {px.index[0].date()} → {px.index[-1].date()}  
**Frequency:** {freq} | **Returns:** {"Log" if use_log else "Simple"}
"""
    )

    colL, colR = st.columns(2)
    with colL:
        st.metric("Last Price", f"${last_price:,.2f}")
        st.metric(f"{freq} Mean Return", f"{avg_period_ret:.4%}")
        st.metric(f"{freq} Std Dev", f"{period_std:.4%}")
    with colR:
        st.metric("Annualized Return (from mean)", f"{ann_return_from_avg:.2%}")
        st.metric("Annualized Volatility (Std Dev)", f"{ann_vol:.2%}")
        st.metric("CAGR (start→end)", f"{(cagr if not np.isnan(cagr) else 0):.2%}")

    st.markdown(
        f"""
### 🎯 {N}σ Annual Return Range
- **Expected annual return range:** **{low_ann:.2%}** to **{high_ann:.2%}**  
- **Implied price range (1y from now, based on last price ${last_price:,.2f}):**  
  **${low_price:,.2f}** to **${high_price:,.2f}**

### 💰 Portfolio Lens (Investment = ${invest_amt:,.0f})
- Ending value (1y) with {N}σ range: **${lower_dollar:,.2f}** to **${upper_dollar:,.2f}**
"""
    )

    with st.expander("Notes & Tips"):
        st.write(
            "- **Annualized Return (from mean)** uses your selected frequency and compounds via simple or log math accordingly.\n"
            "- **CAGR** is the realized growth from first to last price; it’s the clean metric for *historical performance*.\n"
            "- **Annualized Volatility** scales the period std dev by √periods/year.\n"
            "- The ±Nσ range assumes approximate normality; real returns can have skew & fat tails."
        )

    # Optional mini chart (price)
    st.line_chart(px.rename("Price"))


       
