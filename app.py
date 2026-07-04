import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO

st.set_page_config(
    page_title="SMIF Stock & ETF Analysis Website",
    layout="wide"
)

st.title("SMIF Stock & ETF Analysis Website")

st.write(
    """
    This interactive website is designed for SMIF and Advanced Financial Modeling students. 
    Students can compare individual stocks and ETFs using Python, AI-assisted coding, and financial modeling tools.

    The website allows users to compare cumulative returns, annual returns, CAGR, volatility, Sharpe ratios, beta, 
    maximum drawdowns, correlations, rolling volatility, and professional performance evaluation metrics.
    """
)

st.info(
    "Educational demo only. This app does not provide investment, tax, legal, or financial planning advice."
)

# -------------------------------------------------------------------
# Default SMIF tickers
# -------------------------------------------------------------------

DEFAULT_SMIF_TICKERS = {
    "All SMIF Sample Holdings": [
        "AAPL", "MSFT", "NVDA", "AVGO", "META", "GOOG", "AMZN", "ASML",
        "LLY", "JNJ", "UNH", "ABT", "PFE", "BMY", "BIIB", "REGN",
        "JPM", "BLK", "SCHW", "MA", "V", "ICE",
        "XOM", "CVX", "SHEL", "OXY", "URA",
        "LMT", "EMR", "FDX", "UNP", "DE",
        "WMT", "COST", "GLD", "SGOV", "TLT", "SPY", "QQQ", "VOO"
    ],
    "Technology / Communication": ["AAPL", "MSFT", "NVDA", "AVGO", "META", "GOOG", "AMZN", "ASML", "QQQ", "SPY"],
    "Healthcare": ["LLY", "JNJ", "UNH", "ABT", "PFE", "BMY", "BIIB", "REGN", "XLV", "SPY"],
    "Financials": ["JPM", "BLK", "SCHW", "MA", "V", "ICE", "XLF", "SPY"],
    "Energy / Materials": ["XOM", "CVX", "SHEL", "OXY", "URA", "XLE", "SPY"],
    "Industrials": ["LMT", "EMR", "FDX", "UNP", "DE", "XLI", "SPY"],
    "Consumer": ["AMZN", "WMT", "COST", "XLY", "XLP", "SPY"],
    "Defensive / Hedge / Cash": ["GLD", "SGOV", "TLT", "SPY", "QQQ"]
}

# -------------------------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------------------------

st.sidebar.header("User Inputs")

preset = st.sidebar.selectbox(
    "Choose a SMIF preset list",
    list(DEFAULT_SMIF_TICKERS.keys()),
    index=1
)

default_ticker_text = ", ".join(DEFAULT_SMIF_TICKERS[preset])

tickers_input = st.sidebar.text_area(
    "Enter stock or ETF tickers separated by commas",
    default_ticker_text,
    height=110
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

# -------------------------------------------------------------------
# Data Loading
# -------------------------------------------------------------------

tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
# Remove duplicates while preserving order
tickers = list(dict.fromkeys(tickers))
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
        progress=False,
        group_by="column"
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"]
        else:
            return pd.DataFrame()
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
missing_tickers = [ticker for ticker in tickers if ticker not in prices.columns]

if len(available_tickers) == 0:
    st.error("None of the selected stock or ETF tickers were found.")
    st.stop()

if benchmark not in prices.columns:
    st.error("Benchmark data was not found. Please check the benchmark ticker.")
    st.stop()

if missing_tickers:
    st.warning(f"Some tickers were not found or had no data: {', '.join(missing_tickers)}")

returns = prices.pct_change().dropna(how="all")
asset_returns = returns[available_tickers].dropna(how="all")
benchmark_returns = returns[benchmark].dropna()

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------

def annualized_return(r, trading_days=252):
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    cumulative_return = (1 + r).prod() - 1
    years = len(r) / trading_days
    return (1 + cumulative_return) ** (1 / years) - 1 if years > 0 else np.nan


def max_drawdown(r):
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    cumulative_curve = (1 + r).cumprod()
    running_max = cumulative_curve.cummax()
    drawdown = cumulative_curve / running_max - 1
    return drawdown.min()


def downside_deviation(r, mar=0.0, trading_days=252):
    downside = r[r < mar]
    if len(downside) == 0:
        return np.nan
    return downside.std() * np.sqrt(trading_days)


def calculate_metrics(returns_df, benchmark_series, rf_rate):
    trading_days = 252
    metrics = []

    for ticker in returns_df.columns:
        r = returns_df[ticker].dropna()
        aligned = pd.concat([r, benchmark_series], axis=1).dropna()
        aligned.columns = ["Asset", "Benchmark"]

        cumulative_return = (1 + r).prod() - 1
        cagr = annualized_return(r, trading_days)
        volatility = r.std() * np.sqrt(trading_days)
        sharpe_ratio = (cagr - rf_rate) / volatility if volatility and volatility != 0 else np.nan

        dd = downside_deviation(r, 0.0, trading_days)
        sortino_ratio = (cagr - rf_rate) / dd if dd and dd != 0 else np.nan

        if len(aligned) > 2:
            covariance = np.cov(aligned["Asset"], aligned["Benchmark"])[0][1]
            benchmark_variance = np.var(aligned["Benchmark"])
            beta = covariance / benchmark_variance if benchmark_variance != 0 else np.nan
            benchmark_cagr = annualized_return(aligned["Benchmark"], trading_days)
            alpha = cagr - (rf_rate + beta * (benchmark_cagr - rf_rate)) if pd.notna(beta) else np.nan
            active_return = aligned["Asset"] - aligned["Benchmark"]
            tracking_error = active_return.std() * np.sqrt(trading_days)
            information_ratio = (cagr - benchmark_cagr) / tracking_error if tracking_error and tracking_error != 0 else np.nan
            treynor_ratio = (cagr - rf_rate) / beta if beta and beta != 0 else np.nan

            up_market = aligned[aligned["Benchmark"] > 0]
            down_market = aligned[aligned["Benchmark"] < 0]

            up_capture = (
                up_market["Asset"].mean() / up_market["Benchmark"].mean()
                if len(up_market) > 0 and up_market["Benchmark"].mean() != 0 else np.nan
            )
            down_capture = (
                down_market["Asset"].mean() / down_market["Benchmark"].mean()
                if len(down_market) > 0 and down_market["Benchmark"].mean() != 0 else np.nan
            )
            batting_average = (aligned["Asset"] > aligned["Benchmark"]).mean()
        else:
            beta = alpha = tracking_error = information_ratio = treynor_ratio = np.nan
            up_capture = down_capture = batting_average = np.nan

        mdd = max_drawdown(r)
        calmar_ratio = cagr / abs(mdd) if mdd and mdd != 0 else np.nan

        annual_returns = r.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        best_year = annual_returns.max()
        worst_year = annual_returns.min()

        metrics.append({
            "Ticker": ticker,
            "Cumulative Return": cumulative_return,
            "CAGR": cagr,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Beta vs Benchmark": beta,
            "Jensen Alpha": alpha,
            "Treynor Ratio": treynor_ratio,
            "Tracking Error": tracking_error,
            "Information Ratio": information_ratio,
            "Max Drawdown": mdd,
            "Calmar Ratio": calmar_ratio,
            "Upside Capture": up_capture,
            "Downside Capture": down_capture,
            "Batting Average": batting_average,
            "Best Year": best_year,
            "Worst Year": worst_year
        })

    return pd.DataFrame(metrics)


def calculate_drawdowns(returns_df):
    drawdowns = pd.DataFrame(index=returns_df.index)
    for ticker in returns_df.columns:
        cumulative_curve = (1 + returns_df[ticker].dropna()).cumprod()
        running_max = cumulative_curve.cummax()
        drawdowns[ticker] = cumulative_curve / running_max - 1
    return drawdowns


def format_metrics_table(df):
    fmt = {
        "Cumulative Return": "{:.2%}",
        "CAGR": "{:.2%}",
        "Volatility": "{:.2%}",
        "Sharpe Ratio": "{:.2f}",
        "Sortino Ratio": "{:.2f}",
        "Beta vs Benchmark": "{:.2f}",
        "Jensen Alpha": "{:.2%}",
        "Treynor Ratio": "{:.2f}",
        "Tracking Error": "{:.2%}",
        "Information Ratio": "{:.2f}",
        "Max Drawdown": "{:.2%}",
        "Calmar Ratio": "{:.2f}",
        "Upside Capture": "{:.2f}",
        "Downside Capture": "{:.2f}",
        "Batting Average": "{:.2%}",
        "Best Year": "{:.2%}",
        "Worst Year": "{:.2%}"
    }
    return df.style.format(fmt)

metrics_df = calculate_metrics(asset_returns, benchmark_returns, risk_free_rate)

# Safe scatter size: Plotly size cannot be negative or NaN
metrics_df["Bubble Size"] = metrics_df["Sharpe Ratio"].fillna(0).clip(lower=0) + 0.10

# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Price & Returns",
        "Performance Summary",
        "Annual Returns",
        "Risk & Drawdown",
        "Manager Evaluation",
        "Teaching Notes"
    ]
)

with tab1:
    st.subheader("Price Data Preview")
    st.dataframe(prices[available_tickers].tail(), use_container_width=True)

    st.subheader("Normalized Price Chart")
    normalized_prices = prices[available_tickers] / prices[available_tickers].iloc[0]
    fig_price = px.line(normalized_prices, title="Normalized Price Performance")
    fig_price.update_layout(xaxis_title="Date", yaxis_title="Normalized Price")
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("Cumulative Growth of $1")
    cumulative_returns = (1 + asset_returns).cumprod()
    fig_cum = px.line(cumulative_returns, title="Cumulative Return Comparison")
    fig_cum.update_layout(xaxis_title="Date", yaxis_title="Growth of $1")
    st.plotly_chart(fig_cum, use_container_width=True)

with tab2:
    st.subheader("Performance Summary")
    st.dataframe(format_metrics_table(metrics_df.drop(columns=["Bubble Size"])), use_container_width=True)

    csv = metrics_df.drop(columns=["Bubble Size"]).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download performance metrics as CSV",
        data=csv,
        file_name="smif_performance_metrics.csv",
        mime="text/csv"
    )

    st.subheader("Risk-Return Scatter Plot")
    fig_scatter = px.scatter(
        metrics_df,
        x="Volatility",
        y="CAGR",
        text="Ticker",
        size="Bubble Size",
        title="Risk vs Return: CAGR vs Volatility"
    )
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(xaxis_title="Annualized Volatility", yaxis_title="CAGR")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Beta vs Sharpe Ratio")
    fig_beta = px.scatter(
        metrics_df,
        x="Beta vs Benchmark",
        y="Sharpe Ratio",
        text="Ticker",
        title=f"Beta vs Sharpe Ratio, Benchmark = {benchmark}"
    )
    fig_beta.update_traces(textposition="top center")
    st.plotly_chart(fig_beta, use_container_width=True)

with tab3:
    st.subheader("Annual Returns")
    annual_returns = asset_returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    annual_returns.index = annual_returns.index.year
    st.dataframe(annual_returns.style.format("{:.2%}"), use_container_width=True)

    fig_annual = px.bar(
        annual_returns,
        barmode="group",
        title="Annual Return Comparison"
    )
    fig_annual.update_layout(xaxis_title="Year", yaxis_title="Annual Return")
    st.plotly_chart(fig_annual, use_container_width=True)

with tab4:
    st.subheader("Drawdown Analysis")
    drawdowns = calculate_drawdowns(asset_returns)
    fig_drawdown = px.line(drawdowns, title="Drawdown Comparison")
    fig_drawdown.update_layout(xaxis_title="Date", yaxis_title="Drawdown")
    st.plotly_chart(fig_drawdown, use_container_width=True)

    st.subheader("Correlation Matrix")
    corr_matrix = asset_returns.corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, title="Return Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Rolling 12-Month Volatility")
    rolling_vol = asset_returns.rolling(window=252).std() * np.sqrt(252)
    fig_rolling_vol = px.line(rolling_vol, title="Rolling 12-Month Volatility")
    fig_rolling_vol.update_layout(xaxis_title="Date", yaxis_title="Annualized Volatility")
    st.plotly_chart(fig_rolling_vol, use_container_width=True)

with tab5:
    st.subheader("Professional Manager Performance Evaluation")

    st.markdown(
        f"""
        This section evaluates each stock or ETF relative to the selected benchmark: **{benchmark}**.

        Key CFA-style metrics include:
        - **Jensen Alpha:** Excess return after adjusting for beta and the risk-free rate.
        - **Information Ratio:** Active return relative to tracking error.
        - **Tracking Error:** Volatility of active returns versus the benchmark.
        - **Treynor Ratio:** Excess return per unit of beta risk.
        - **Upside Capture:** How much the asset participates when the benchmark is up.
        - **Downside Capture:** How much the asset participates when the benchmark is down.
        - **Batting Average:** Frequency of outperforming the benchmark.
        """
    )

    eval_cols = [
        "Ticker", "Jensen Alpha", "Information Ratio", "Tracking Error", "Treynor Ratio",
        "Upside Capture", "Downside Capture", "Batting Average", "Sharpe Ratio", "Max Drawdown"
    ]
    st.dataframe(format_metrics_table(metrics_df[eval_cols]), use_container_width=True)

    st.subheader("Alpha vs Information Ratio")
    fig_alpha_ir = px.scatter(
        metrics_df,
        x="Information Ratio",
        y="Jensen Alpha",
        text="Ticker",
        title="Skill Diagnostic: Jensen Alpha vs Information Ratio"
    )
    fig_alpha_ir.update_traces(textposition="top center")
    st.plotly_chart(fig_alpha_ir, use_container_width=True)

    st.subheader("Skill vs Luck Teaching Framework")
    st.markdown(
        """
        Strong performance may come from market beta, factor exposure, a few lucky winners, or persistent skill.
        Students should not conclude that outperformance automatically means skill.

        A more professional evaluation asks:
        1. Is alpha positive after adjusting for benchmark beta?
        2. Is the information ratio meaningful and not driven by one short period?
        3. Is tracking error reasonable relative to the strategy?
        4. Does the asset or portfolio outperform consistently, or only in one regime?
        5. Is upside capture high while downside capture is controlled?
        6. Is performance explained by exposure to AI, mega-cap growth, rates, or another factor?
        7. Does performance align with the stated investment process?
        """
    )

with tab6:
    st.subheader("Teaching Notes for SMIF Students")
    st.markdown(
        """
        ## How to Use This Website

        This website is designed to help SMIF and Advanced Financial Modeling students compare
        individual stocks and ETFs in a structured way.

        Students should not simply choose the asset with the highest return. They should compare
        return, risk, beta, drawdown, correlation, benchmark-relative performance, and portfolio role.

        ## Suggested Student Output

        For each stock or ETF, students should summarize:

        - Investment thesis
        - Return performance
        - Risk profile
        - Beta and market sensitivity
        - Drawdown risk
        - Correlation with existing holdings
        - Jensen alpha and information ratio
        - Upside and downside capture
        - Final recommendation: Buy, Add, Hold, Trim, Sell, or Watchlist

        ## Next Version

        The next upgrade can add portfolio optimization, including equal-weight portfolio,
        current SMIF portfolio, minimum variance portfolio, maximum Sharpe portfolio, and efficient frontier.
        """
    )
