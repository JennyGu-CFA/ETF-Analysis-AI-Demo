import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="SMIF Stock & ETF Analysis Website",
    layout="wide"
)

st.title("SMIF Stock & ETF Analysis Website")

st.write(
    """
    This interactive website is designed for SMIF and Advanced Financial Modeling students.
    Students can compare individual stocks and ETFs using Python, AI-assisted coding,
    and financial modeling tools.

    The website allows users to compare cumulative returns, annual returns, CAGR,
    volatility, Sharpe ratios, beta, maximum drawdowns, correlations, and rolling volatility.
    """
)

st.info(
    "Educational demo only. This app does not provide investment, tax, legal, or financial planning advice."
)

# --------------------------------------------------
# Sidebar inputs
# --------------------------------------------------

st.sidebar.header("User Inputs")

tickers_input = st.sidebar.text_input(
    "Enter stock or ETF tickers separated by commas",
    "AAPL, MSFT, NVDA, SPY, QQQ, TLT, GLD, SGOV"
)

benchmark_input = st.sidebar.text_input(
    "Benchmark ticker",
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

st.sidebar.markdown("---")
st.sidebar.write("Example stocks: AAPL, MSFT, NVDA, AMZN, META, JPM, LLY")
st.sidebar.write("Example ETFs: SPY, QQQ, VOO, VTI, TLT, GLD, SGOV")

# --------------------------------------------------
# Clean ticker inputs
# --------------------------------------------------

tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
benchmark = benchmark_input.strip().upper()

if not tickers:
    st.error("Please enter at least one stock or ETF ticker.")
    st.stop()

if benchmark and benchmark not in tickers:
    all_tickers = tickers + [benchmark]
else:
    all_tickers = tickers

# Remove duplicates while preserving order
all_tickers = list(dict.fromkeys(all_tickers))
tickers = list(dict.fromkeys(tickers))

# --------------------------------------------------
# Load data
# --------------------------------------------------

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
        if "Close" not in data.columns.get_level_values(0):
            return pd.DataFrame()
        prices = data["Close"].copy()
    else:
        if "Close" not in data.columns:
            return pd.DataFrame()
        prices = data[["Close"]].copy()
        if len(ticker_list) == 1:
            prices.columns = ticker_list

    prices = prices.dropna(how="all")
    return prices

prices = load_price_data(all_tickers, start_date, end_date)

if prices.empty:
    st.error("No price data found. Please check your tickers and date range.")
    st.stop()

available_tickers = [ticker for ticker in tickers if ticker in prices.columns and prices[ticker].notna().sum() > 1]

if len(available_tickers) == 0:
    st.error("None of the selected stock or ETF tickers were found. Please check ticker symbols.")
    st.stop()

if benchmark not in prices.columns or prices[benchmark].notna().sum() <= 1:
    st.error("Benchmark data was not found. Please check the benchmark ticker.")
    st.stop()

# Use only selected tickers with valid data
prices_selected = prices[available_tickers].dropna(how="all")
returns = prices.pct_change(fill_method=None).dropna(how="all")
asset_returns = returns[available_tickers].dropna(how="all")
benchmark_returns = returns[benchmark].dropna()

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def calculate_metrics(returns_df, benchmark_series, rf_rate):
    trading_days = 252
    metrics = []

    for ticker in returns_df.columns:
        r = returns_df[ticker].dropna()

        if len(r) < 2:
            continue

        aligned = pd.concat([r, benchmark_series], axis=1).dropna()
        aligned.columns = ["Asset", "Benchmark"]

        cumulative_return = (1 + r).prod() - 1
        years = len(r) / trading_days
        cagr = (1 + cumulative_return) ** (1 / years) - 1 if years > 0 else np.nan
        volatility = r.std() * np.sqrt(trading_days)
        sharpe_ratio = (cagr - rf_rate) / volatility if volatility and volatility != 0 else np.nan

        if len(aligned) > 2:
            covariance = np.cov(aligned["Asset"], aligned["Benchmark"])[0][1]
            benchmark_variance = np.var(aligned["Benchmark"])
            beta = covariance / benchmark_variance if benchmark_variance != 0 else np.nan
        else:
            beta = np.nan

        cumulative_curve = (1 + r).cumprod()
        running_max = cumulative_curve.cummax()
        drawdown = cumulative_curve / running_max - 1
        max_drawdown = drawdown.min()

        annual_returns = r.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        best_year = annual_returns.max() if not annual_returns.empty else np.nan
        worst_year = annual_returns.min() if not annual_returns.empty else np.nan

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


def calculate_drawdowns(returns_df):
    drawdowns = pd.DataFrame(index=returns_df.index)

    for ticker in returns_df.columns:
        r = returns_df[ticker].dropna()
        cumulative_curve = (1 + r).cumprod()
        running_max = cumulative_curve.cummax()
        drawdowns[ticker] = cumulative_curve / running_max - 1

    return drawdowns


def safe_metric_table(df):
    if df.empty:
        return df
    return df.replace([np.inf, -np.inf], np.nan)

# --------------------------------------------------
# Tabs
# --------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Price & Returns",
        "Performance Summary",
        "Annual Returns",
        "Risk & Drawdown",
        "Teaching Notes"
    ]
)

# --------------------------------------------------
# Tab 1
# --------------------------------------------------

with tab1:
    st.subheader("Price Data Preview")
    st.dataframe(prices_selected.tail(), use_container_width=True)

    st.subheader("Normalized Price Chart")
    normalized_prices = prices_selected / prices_selected.ffill().bfill().iloc[0]

    fig_price = px.line(
        normalized_prices,
        title="Normalized Price Performance"
    )
    fig_price.update_layout(
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        legend_title_text="Ticker"
    )
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("Cumulative Growth of $1")
    cumulative_returns = (1 + asset_returns.fillna(0)).cumprod()

    fig_cum = px.line(
        cumulative_returns,
        title="Cumulative Return Comparison"
    )
    fig_cum.update_layout(
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        legend_title_text="Ticker"
    )
    st.plotly_chart(fig_cum, use_container_width=True)

# --------------------------------------------------
# Tab 2
# --------------------------------------------------

with tab2:
    st.subheader("Performance Summary")

    metrics_df = calculate_metrics(asset_returns, benchmark_returns, risk_free_rate)
    metrics_df = safe_metric_table(metrics_df)

    if metrics_df.empty:
        st.warning("Not enough return data to calculate performance metrics.")
    else:
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

        st.subheader("Risk-Return Scatter Plot")

        scatter_df = metrics_df.dropna(subset=["Volatility", "CAGR"]).copy()

        if scatter_df.empty:
            st.warning("Not enough data to build the risk-return scatter plot.")
        else:
            fig_scatter = px.scatter(
                scatter_df,
                x="Volatility",
                y="CAGR",
                text="Ticker",
                title="Risk vs Return: CAGR vs Volatility"
            )
            fig_scatter.update_traces(textposition="top center", marker=dict(size=12))
            fig_scatter.update_layout(
                xaxis_title="Annualized Volatility",
                yaxis_title="CAGR"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Beta vs Sharpe Ratio")

        beta_df = metrics_df.dropna(subset=["Beta vs Benchmark", "Sharpe Ratio"]).copy()

        if beta_df.empty:
            st.warning("Not enough data to build the beta vs Sharpe ratio chart.")
        else:
            fig_beta = px.scatter(
                beta_df,
                x="Beta vs Benchmark",
                y="Sharpe Ratio",
                text="Ticker",
                title=f"Beta vs Sharpe Ratio, Benchmark = {benchmark}"
            )
            fig_beta.update_traces(textposition="top center", marker=dict(size=12))
            st.plotly_chart(fig_beta, use_container_width=True)

# --------------------------------------------------
# Tab 3
# --------------------------------------------------

with tab3:
    st.subheader("Annual Returns")

    annual_returns = asset_returns.resample("YE").apply(lambda x: (1 + x.dropna()).prod() - 1)
    annual_returns.index = annual_returns.index.year

    st.dataframe(
        annual_returns.style.format("{:.2%}"),
        use_container_width=True
    )

    fig_annual = px.bar(
        annual_returns,
        barmode="group",
        title="Annual Return Comparison"
    )
    fig_annual.update_layout(
        xaxis_title="Year",
        yaxis_title="Annual Return",
        legend_title_text="Ticker"
    )
    st.plotly_chart(fig_annual, use_container_width=True)

# --------------------------------------------------
# Tab 4
# --------------------------------------------------

with tab4:
    st.subheader("Drawdown Analysis")

    drawdowns = calculate_drawdowns(asset_returns)

    fig_drawdown = px.line(
        drawdowns,
        title="Drawdown Comparison"
    )
    fig_drawdown.update_layout(
        xaxis_title="Date",
        yaxis_title="Drawdown",
        legend_title_text="Ticker"
    )
    st.plotly_chart(fig_drawdown, use_container_width=True)

    st.subheader("Correlation Matrix")

    corr_matrix = asset_returns.corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        title="Return Correlation Matrix",
        zmin=-1,
        zmax=1
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Rolling 12-Month Volatility")

    rolling_vol = asset_returns.rolling(window=252).std() * np.sqrt(252)

    fig_rolling_vol = px.line(
        rolling_vol,
        title="Rolling 12-Month Volatility"
    )
    fig_rolling_vol.update_layout(
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        legend_title_text="Ticker"
    )
    st.plotly_chart(fig_rolling_vol, use_container_width=True)

# --------------------------------------------------
# Tab 5
# --------------------------------------------------

with tab5:
    st.subheader("Teaching Notes for SMIF Students")

    st.markdown(
        """
        ## How to Use This Website

        This website is designed to help SMIF and Advanced Financial Modeling students compare
        individual stocks and ETFs in a structured way.

        Students should not simply choose the asset with the highest return. They should compare
        return, risk, beta, drawdown, correlation, and the role of each asset in the portfolio.

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

        This app is a learning tool. It is not a substitute for fundamental analysis,
        valuation, macro analysis, or professional judgment.
        """
    )
