import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Stock Return Analyzer", layout="centered")

st.title("📈 Stock Return Analyzer")

ticker = st.text_input("Enter Stock Symbol (e.g., AAPL):", value="AAPL")
start_date = st.date_input("Start Date:", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date:", value=pd.to_datetime("2024-12-31"))

if st.button("Analyze"):
    try:
        # Download data
        data = yf.download(ticker, start=start_date, end=end_date)
        data = data[['Close']].rename(columns={'Close': 'price'})

        # Calculate returns
        data['daily_return'] = data['price'].pct_change()
        data.dropna(inplace=True)

        # Metrics
        trading_days = len(data)
       # Metrics
        avg_daily_return = float(data['daily_return'].mean())
        std_dev = float(data['daily_return'].std())
        
        start_price = float(data['price'].iloc[0])
        end_price = float(data['price'].iloc[-1])
        num_years = (end_date - start_date).days / 365.25
        CAGR = (end_price / start_price) ** (1 / num_years) - 1
        total_return = float(data['cumulative_return'].iloc[-1] - 1)


        # Show metrics
        st.subheader(f"{ticker} Performance Metrics")
        st.write(f"**CAGR:** {CAGR:.2%}")
        st.write(f"**Average Daily Return:** {avg_daily_return:.4%}")
        st.write(f"**Standard Deviation:** {std_dev:.4%}")
        st.write(f"**Total Return:** {total_return * 100:.2f}%")

   
    

    except Exception as e:
        st.error(f"Error: {e}")
       
