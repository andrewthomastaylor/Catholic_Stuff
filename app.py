import streamlit as st
import pandas as pd
import numpy as np
from black_scholes_strategy.model import BlackScholesModel
from black_scholes_strategy.strategy import MispricingStrategy

st.set_page_config(page_title="Black-Scholes Option Calculator", layout="wide")

st.title("📊 Black-Scholes Option Pricing & Strategy")

# Sidebar for inputs
st.sidebar.header("Input Parameters")

S = st.sidebar.number_input("Stock Price (S)", value=100.0, min_value=0.01)
K = st.sidebar.number_input("Strike Price (K)", value=105.0, min_value=0.01)
T = st.sidebar.number_input("Time to Expiry (T in years)", value=0.5, min_value=0.0, step=0.01)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, step=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.20, min_value=0.01, step=0.01)

st.sidebar.markdown("---")
market_price = st.sidebar.number_input("Market Price (for Signal)", value=4.50, min_value=0.0)
threshold = st.sidebar.slider("Signal Threshold (%)", 0, 20, 5) / 100.0

# Model initialization
bs_model = BlackScholesModel(S, K, T, r, sigma)
strategy = MispricingStrategy(threshold=threshold)

# Calculations
call_price = bs_model.calc_price('call')
put_price = bs_model.calc_price('put')

call_delta = bs_model.calc_delta('call')
put_delta = bs_model.calc_delta('put')
gamma = bs_model.calc_gamma()
vega = bs_model.calc_vega()
call_theta = bs_model.calc_theta('call')
put_theta = bs_model.calc_theta('put')
call_rho = bs_model.calc_rho('call')
put_rho = bs_model.calc_rho('put')

call_signal = strategy.generate_signal(call_price, market_price)

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Theoretical Prices")
    price_data = {
        "Option Type": ["Call", "Put"],
        "Theoretical Price": [f"{call_price:.4f}", f"{put_price:.4f}"]
    }
    st.table(pd.DataFrame(price_data))

with col2:
    st.subheader("Trading Signal (Call)")
    st.metric("Signal", call_signal)
    if call_price > 0:
        diff_pct = (call_price - market_price) / call_price * 100
        st.write(f"Difference: {diff_pct:.2f}%")

st.markdown("---")
st.subheader("Greeks")

greeks_data = {
    "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
    "Call Value": [f"{call_delta:.4f}", f"{gamma:.4f}", f"{vega:.4f}", f"{call_theta:.4f}", f"{call_rho:.4f}"],
    "Put Value": [f"{put_delta:.4f}", f"{gamma:.4f}", f"{vega:.4f}", f"{put_theta:.4f}", f"{put_rho:.4f}"]
}
st.table(pd.DataFrame(greeks_data))

st.info("Note: Vega is per 1% change in volatility. Theta is per day. Rho is per 1% change in interest rate.")

st.markdown("---")
st.subheader("Visualizations")

# Generate data for charts
s_range = np.linspace(S * 0.5, S * 1.5, 50)
viz_data = []

for s_val in s_range:
    temp_model = BlackScholesModel(s_val, K, T, r, sigma)
    viz_data.append({
        "Stock Price": s_val,
        "Call Price": temp_model.calc_price('call'),
        "Put Price": temp_model.calc_price('put'),
        "Call Delta": temp_model.calc_delta('call'),
        "Put Delta": temp_model.calc_delta('put')
    })

df_viz = pd.DataFrame(viz_data)

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    st.write("Option Price vs Stock Price")
    st.line_chart(df_viz.set_index("Stock Price")[["Call Price", "Put Price"]])

with viz_col2:
    st.write("Delta vs Stock Price")
    st.line_chart(df_viz.set_index("Stock Price")[["Call Delta", "Put Delta"]])
