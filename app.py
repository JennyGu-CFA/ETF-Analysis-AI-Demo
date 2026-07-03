import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="AI-Powered ETF Analysis",
    layout="wide"
)

st.title("AI-Powered ETF Analysis Website")
st.write(
    "This demo was created for an Advanced Financial Modeling class. "
    "It uses Python, AI-assisted coding, and prompt engineering to compare ETFs "
    "by cumulative returns, annual returns, Sharpe ratios, beta, volatility, and drawdowns."
)

st.sidebar.header("User Inputs")

tickers_input = st.sidebar.text_input(
    "Enter ETF tickers separated by commas",
    "SPY, QQQ, TLT, GLD, SGOV"
)

benchmark_input = st.sidebar.text_input(
    "Benchmark ETF",
    "SPY"
)

start_date = st.sidebar.date_input(
    "Start Date",
    pd.to_datetime("2018-01-01")
)

end_date = st.sidebar.date_input(
    "End Date",
    pd.to_datetime("today")
)

risk_free_rate = st.sidebar.number_input(
    "Annual Risk-Free Rate",
    min_value=0.0,
    max_value=0.20,
    value=0.04,
    step=0.005,
    format="%.3f"
)

tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
benchmark = benchmark_input.strip().upper()

if benchmark not in tickers:
    all_tickers = tickers + [benchmark]
else:
    all_tickers = tickers

@st.cache_data
def load_price_data(ticker_list, start, end):
    data = yf.download(
        ticker_list,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = ticker_list

    prices = prices.dropna(how="all")
    return prices

prices = load_price_data(all_tickers, start_date, end_date)

if prices.empty:
    st.error("No price data found. Please check your tickers and date range.")
    st.stop()

available_tickers = [ticker for ticker in tickers if ticker in prices.columns]
if benchmark not in prices.columns:
    st.error("Benchmark data was not found. Please check the benchmark ticker.")
    st.stop()

returns = prices.pct_change().dropna()
etf_returns = returns[available_tickers]
benchmark_returns = returns[benchmark]

st.subheader("Price Data Preview")
st.dataframe(prices.tail(), use_container_width=True)

st.subheader("Cumulative Growth of $1")

cumulative_returns = (1 + etf_returns).cumprod()

fig_cum = px.line(
    cumulative_returns,
    x=cumulative_returns.index,
    y=cumulative_returns.columns,
    title="Cumulative ETF Returns"
)
fig_cum.update_layout(
    xaxis_title="Date",
    yaxis_title="Growth of $1"
)
st.plotly_chart(fig_cum, use_container_width=True)

def calculate_metrics(returns_df, benchmark_series, rf_rate):
    trading_days = 252
    metrics = []

    for ticker in returns_df.columns:
        r = returns_df[ticker].dropna()

        aligned = pd.concat([r, benchmark_series], axis=1).dropna()
        aligned.columns = ["ETF", "Benchmark"]

        cumulative_return = (1 + r).prod() - 1
        years = len(r) / trading_days

        if years > 0:
            cagr = (1 + cumulative_return) ** (1 / years) - 1
        else:
            cagr = np.nan

        volatility = r.std() * np.sqrt(trading_days)

        if volatility != 0:
            sharpe_ratio = (cagr - rf_rate) / volatility
        else:
            sharpe_ratio = np.nan

        if len(aligned) > 2:
            covariance = np.cov(aligned["ETF"], aligned["Benchmark"])[0][1]
            benchmark_variance = np.var(aligned["Benchmark"])
            beta = covariance / benchmark_variance if benchmark_variance != 0 else np.nan
        else:
            beta = np.nan

        cumulative_curve = (1 + r).cumprod()
        running_max = cumulative_curve.cummax()
        drawdown = cumulative_curve / running_max - 1
        max_drawdown = drawdown.min()

        annual_returns = r.resample("YE").apply(lambda x: (1 + x).prod() - 1)

        best_year = annual_returns.max()
        worst_year = annual_returns.min()

        metrics.append({
            "Ticker": ticker,
            "Cumulative Return": cumulative_return,
            "CAGR": cagr,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Beta vs Benchmark": beta,
            "Max Drawdown": max_drawdown,
            "Best Year": best_year,
            "Worst Year": worst_year
        })

    return pd.DataFrame(metrics)

metrics_df = calculate_metrics(etf_returns, benchmark_returns, risk_free_rate)

st.subheader("Performance Summary")

st.dataframe(
    metrics_df.style.format({
        "Cumulative Return": "{:.2%}",
        "CAGR": "{:.2%}",
        "Volatility": "{:.2%}",
        "Sharpe Ratio": "{:.2f}",
        "Beta vs Benchmark": "{:.2f}",
        "Max Drawdown": "{:.2%}",
        "Best Year": "{:.2%}",
        "Worst Year": "{:.2%}"
    }),
    use_container_width=True
)

st.subheader("Annual Returns")

annual_returns = etf_returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
annual_returns.index = annual_returns.index.year

st.dataframe(
    annual_returns.style.format("{:.2%}"),
    use_container_width=True
)

fig_annual = px.bar(
    annual_returns,
    x=annual_returns.index,
    y=annual_returns.columns,
    barmode="group",
    title="Annual ETF Returns"
)
fig_annual.update_layout(
    xaxis_title="Year",
    yaxis_title="Annual Return"
)
st.plotly_chart(fig_annual, use_container_width=True)

st.subheader("Drawdown Analysis")

drawdowns = pd.DataFrame(index=etf_returns.index)

for ticker in etf_returns.columns:
    cumulative_curve = (1 + etf_returns[ticker]).cumprod()
    running_max = cumulative_curve.cummax()
    drawdowns[ticker] = cumulative_curve / running_max - 1

fig_drawdown = px.line(
    drawdowns,
    x=drawdowns.index,
    y=drawdowns.columns,
    title="ETF Drawdowns"
)
fig_drawdown.update_layout(
    xaxis_title="Date",
    yaxis_title="Drawdown"
)
st.plotly_chart(fig_drawdown, use_container_width=True)

st.subheader("AI and Financial Modeling Teaching Reflection")

st.write(
    """
    This project demonstrates how AI and prompt engineering can support advanced financial modeling.
    Students can use AI to structure Python code, debug errors, explain financial metrics, and improve
    dashboard design. However, students must verify all calculations, understand the financial logic,
    and apply professional judgment when interpreting investment analytics.
    """
)

st.info(
    "Educational demo only. This app does not provide investment, tax, legal, or financial planning advice."
)
