import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="SMIF Stock & ETF Analysis", layout="wide")

st.title("SMIF Stock & ETF Analysis Website")
st.write("""
This interactive website is designed for SMIF and Advanced Financial Modeling students.
Students can compare individual stocks and ETFs using Python, AI-assisted coding, and financial modeling tools.
""")
st.info("Educational demo only. This app does not provide investment, tax, legal, or financial planning advice.")

# Sidebar inputs
st.sidebar.header("User Inputs")
tickers_input = st.sidebar.text_input(
    "Enter stock or ETF tickers separated by commas",
    "AAPL, MSFT, NVDA, SPY, QQQ, TLT, GLD, SGOV"
)
benchmark = st.sidebar.text_input("Benchmark ticker", "SPY").strip().upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
risk_free_rate = st.sidebar.number_input(
    "Annual Risk-Free Rate",
    min_value=0.0,
    max_value=0.20,
    value=0.04,
    step=0.005,
    format="%.3f"
)

st.sidebar.markdown("---")
st.sidebar.write("Example stocks: AAPL, MSFT, NVDA, AMZN, META, JPM, LLY")
st.sidebar.write("Example ETFs: SPY, QQQ, VOO, VTI, TLT, GLD, SGOV")

# Clean ticker inputs
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
all_tickers = tickers.copy()
if benchmark not in all_tickers:
    all_tickers.append(benchmark)

@st.cache_data
def load_price_data(ticker_list, start, end):
    data = yf.download(ticker_list, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = ticker_list
    return prices.dropna(how="all")

prices = load_price_data(all_tickers, start_date, end_date)

if prices.empty:
    st.error("No price data found. Please check tickers and dates.")
    st.stop()

available_tickers = [t for t in tickers if t in prices.columns]
if not available_tickers:
    st.error("None of the selected stock or ETF tickers were found.")
    st.stop()
if benchmark not in prices.columns:
    st.error("Benchmark ticker was not found. Please check the benchmark ticker.")
    st.stop()

returns = prices.pct_change().dropna()
asset_returns = returns[available_tickers]
benchmark_returns = returns[benchmark]

# Helper functions
def calculate_metrics(returns_df, benchmark_series, rf_rate):
    trading_days = 252
    rows = []
    for ticker in returns_df.columns:
        r = returns_df[ticker].dropna()
        aligned = pd.concat([r, benchmark_series], axis=1).dropna()
        aligned.columns = ["Asset", "Benchmark"]

        cumulative_return = (1 + r).prod() - 1
        years = len(r) / trading_days
        cagr = (1 + cumulative_return) ** (1 / years) - 1 if years > 0 else np.nan
        volatility = r.std() * np.sqrt(trading_days)
        sharpe = (cagr - rf_rate) / volatility if volatility != 0 else np.nan

        if len(aligned) > 2:
            covariance = np.cov(aligned["Asset"], aligned["Benchmark"])[0][1]
            benchmark_variance = np.var(aligned["Benchmark"])
            beta = covariance / benchmark_variance if benchmark_variance != 0 else np.nan
        else:
            beta = np.nan

        cumulative_curve = (1 + r).cumprod()
        drawdown = cumulative_curve / cumulative_curve.cummax() - 1
        max_drawdown = drawdown.min()

        annual = r.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        rows.append({
            "Ticker": ticker,
            "Cumulative Return": cumulative_return,
            "CAGR": cagr,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe,
            "Beta vs Benchmark": beta,
            "Max Drawdown": max_drawdown,
            "Best Year": annual.max(),
            "Worst Year": annual.min()
        })
    return pd.DataFrame(rows)

def calculate_drawdowns(returns_df):
    result = pd.DataFrame(index=returns_df.index)
    for ticker in returns_df.columns:
        cumulative = (1 + returns_df[ticker]).cumprod()
        result[ticker] = cumulative / cumulative.cummax() - 1
    return result

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Price & Returns",
    "Performance Summary",
    "Annual Returns",
    "Risk & Drawdown",
    "Teaching Notes"
])

with tab1:
    st.subheader("Price Data Preview")
    st.dataframe(prices[available_tickers].tail(), use_container_width=True)

    st.subheader("Normalized Price Chart")
    normalized = prices[available_tickers] / prices[available_tickers].iloc[0]
    fig = px.line(normalized, x=normalized.index, y=normalized.columns, title="Normalized Price Performance")
    fig.update_layout(xaxis_title="Date", yaxis_title="Normalized Price")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cumulative Growth of $1")
    cumulative = (1 + asset_returns).cumprod()
    fig = px.line(cumulative, x=cumulative.index, y=cumulative.columns, title="Cumulative Return Comparison")
    fig.update_layout(xaxis_title="Date", yaxis_title="Growth of $1")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Performance Summary")
    metrics = calculate_metrics(asset_returns, benchmark_returns, risk_free_rate)
    st.dataframe(metrics.style.format({
        "Cumulative Return": "{:.2%}",
        "CAGR": "{:.2%}",
        "Volatility": "{:.2%}",
        "Sharpe Ratio": "{:.2f}",
        "Beta vs Benchmark": "{:.2f}",
        "Max Drawdown": "{:.2%}",
        "Best Year": "{:.2%}",
        "Worst Year": "{:.2%}"
    }), use_container_width=True)

    st.subheader("Risk-Return Scatter Plot")
    fig = px.scatter(metrics, x="Volatility", y="CAGR", text="Ticker", size="Sharpe Ratio", title="Risk vs Return")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Beta vs Sharpe Ratio")
    fig = px.scatter(metrics, x="Beta vs Benchmark", y="Sharpe Ratio", text="Ticker", title=f"Beta vs Sharpe Ratio, Benchmark = {benchmark}")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Annual Returns")
    annual_returns = asset_returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    annual_returns.index = annual_returns.index.year
    st.dataframe(annual_returns.style.format("{:.2%}"), use_container_width=True)

    fig = px.bar(annual_returns, x=annual_returns.index, y=annual_returns.columns, barmode="group", title="Annual Return Comparison")
    fig.update_layout(xaxis_title="Year", yaxis_title="Annual Return")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Drawdown Analysis")
    drawdowns = calculate_drawdowns(asset_returns)
    fig = px.line(drawdowns, x=drawdowns.index, y=drawdowns.columns, title="Drawdown Comparison")
    fig.update_layout(xaxis_title="Date", yaxis_title="Drawdown")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Matrix")
    corr = asset_returns.corr()
    fig = px.imshow(corr, text_auto=True, title="Return Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rolling 12-Month Volatility")
    rolling_vol = asset_returns.rolling(window=252).std() * np.sqrt(252)
    fig = px.line(rolling_vol, x=rolling_vol.index, y=rolling_vol.columns, title="Rolling 12-Month Volatility")
    fig.update_layout(xaxis_title="Date", yaxis_title="Annualized Volatility")
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Teaching Notes for SMIF Students")
    st.markdown("""
    ## How to Use This Website

    This website helps SMIF and Advanced Financial Modeling students compare individual stocks and ETFs in a structured way.

    Students should not simply choose the asset with the highest return. They should compare return, risk, beta, drawdown, correlation, and the role of each asset in the portfolio.

    ## Key Questions

    1. Which asset had the highest cumulative return?
    2. Which asset had the highest Sharpe ratio?
    3. Which asset had the largest maximum drawdown?
    4. Which asset has the highest beta relative to the benchmark?
    5. Are these assets highly correlated with one another?
    6. Does this asset improve portfolio diversification?
    7. Is this asset suitable as a core holding, satellite holding, hedge, or watchlist position?

    ## Suggested Student Output

    For each stock or ETF, students should summarize:

    - Investment thesis
    - Return performance
    - Risk profile
    - Beta and market sensitivity
    - Drawdown risk
    - Correlation with existing holdings
    - Final recommendation: Buy, Add, Hold, Trim, Sell, or Watchlist

    ## Important Reminder

    This app is a learning tool. It is not a substitute for fundamental analysis, valuation, macro analysis, or professional judgment.
    """)
