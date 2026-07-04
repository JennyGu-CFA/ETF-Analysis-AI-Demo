import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

st.set_page_config(page_title="SMIF Stock & ETF Analysis Website", layout="wide")

# -----------------------------
# Preset ticker lists
# -----------------------------
SMIF_PRESETS = {
    "Technology / Communication": "AAPL, MSFT, NVDA, AVGO, META, GOOG, AMZN, ASML, QQQ, SPY",
    "Healthcare": "LLY, JNJ, PFE, BMY, UNH, ABT, REGN, BIIB, XLV, SPY",
    "Financials": "JPM, BLK, SCHW, MA, V, XLF, SPY",
    "Energy / Industrials": "XOM, CVX, SHEL, OXY, LMT, EMR, FDX, UNP, XLE, XLI, SPY",
    "Consumer": "AMZN, WMT, COST, XLY, XLP, SPY",
    "Defensive / Hedge": "SGOV, TLT, GLD, SPY, QQQ",
    "All Sample SMIF Holdings": "AAPL, MSFT, NVDA, AVGO, META, GOOG, AMZN, ASML, LLY, JNJ, PFE, BMY, UNH, ABT, REGN, BIIB, JPM, BLK, SCHW, MA, V, XOM, CVX, SHEL, OXY, LMT, EMR, FDX, UNP, WMT, COST, GLD, SGOV, TLT, SPY, QQQ",
}

st.title("SMIF Stock & ETF Analysis Website")

st.write(
    """
    This interactive website is designed for SMIF and Advanced Financial Modeling students. Students can compare individual stocks and ETFs using Python, AI-assisted coding, and financial modeling tools.

    The website allows users to compare cumulative returns, annual returns, CAGR, volatility, Sharpe ratios, beta, maximum drawdowns, correlations, rolling volatility, professional manager evaluation metrics, and portfolio optimization results.
    """
)

st.info("Educational demo only. This app does not provide investment, tax, legal, or financial planning advice.")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("User Inputs")

preset_name = st.sidebar.selectbox("Choose a SMIF preset list", list(SMIF_PRESETS.keys()))

tickers_input = st.sidebar.text_area(
    "Enter stock or ETF tickers separated by commas",
    SMIF_PRESETS[preset_name],
    height=100,
)

benchmark_input = st.sidebar.text_input("Benchmark ticker", "SPY")

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

risk_free_rate = st.sidebar.number_input(
    "Annual Risk-Free Rate",
    min_value=0.0,
    max_value=0.20,
    value=0.04,
    step=0.005,
    format="%.3f",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Optimization Settings")
max_weight = st.sidebar.slider("Maximum weight per asset", 0.05, 1.00, 0.20, 0.05)
num_simulations = st.sidebar.slider("Random portfolios for efficient frontier", 500, 8000, 2500, 500)

st.sidebar.markdown("---")
st.sidebar.write("Example stocks: AAPL, MSFT, NVDA, AMZN, META, JPM, LLY")
st.sidebar.write("Example ETFs: SPY, QQQ, VOO, VTI, TLT, GLD, SGOV")

# -----------------------------
# Data Loading
# -----------------------------
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
benchmark = benchmark_input.strip().upper()

# Remove duplicates while keeping order
tickers = list(dict.fromkeys(tickers))

all_tickers = tickers.copy()
if benchmark and benchmark not in all_tickers:
    all_tickers.append(benchmark)

@st.cache_data(show_spinner=False)
def load_price_data(ticker_list, start, end):
    data = yf.download(ticker_list, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = ticker_list
    prices = prices.dropna(how="all")
    return prices

with st.spinner("Downloading price data..."):
    prices = load_price_data(all_tickers, start_date, end_date)

if prices.empty:
    st.error("No price data found. Please check your tickers and date range.")
    st.stop()

available_tickers = [ticker for ticker in tickers if ticker in prices.columns]
if len(available_tickers) == 0:
    st.error("None of the selected stock or ETF tickers were found.")
    st.stop()
if benchmark not in prices.columns:
    st.error("Benchmark data was not found. Please check the benchmark ticker.")
    st.stop()

# Drop assets with too little data; align returns for optimization/evaluation
returns = prices.pct_change().dropna(how="all")
asset_returns = returns[available_tickers].dropna(how="all")
benchmark_returns = returns[benchmark].dropna()

# -----------------------------
# Helper Functions
# -----------------------------
def calculate_metrics(returns_df, benchmark_series, rf_rate):
    trading_days = 252
    metrics = []
    for ticker in returns_df.columns:
        r = returns_df[ticker].dropna()
        aligned = pd.concat([r, benchmark_series], axis=1).dropna()
        aligned.columns = ["Asset", "Benchmark"]
        if len(r) < 2:
            continue
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
            "Worst Year": worst_year,
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

def calculate_manager_metrics(returns_df, benchmark_series, rf_rate):
    trading_days = 252
    rows = []
    daily_rf = rf_rate / trading_days
    for ticker in returns_df.columns:
        aligned = pd.concat([returns_df[ticker], benchmark_series], axis=1).dropna()
        aligned.columns = ["Asset", "Benchmark"]
        if len(aligned) < 30:
            continue
        asset = aligned["Asset"]
        bench = aligned["Benchmark"]
        excess_asset = asset - daily_rf
        excess_bench = bench - daily_rf
        beta = np.cov(asset, bench)[0][1] / np.var(bench) if np.var(bench) != 0 else np.nan
        alpha_daily = (asset.mean() - daily_rf) - beta * (bench.mean() - daily_rf) if not np.isnan(beta) else np.nan
        jensen_alpha = alpha_daily * trading_days if not np.isnan(alpha_daily) else np.nan
        active_return = asset - bench
        tracking_error = active_return.std() * np.sqrt(trading_days)
        information_ratio = (active_return.mean() * trading_days) / tracking_error if tracking_error != 0 else np.nan
        asset_cagr = (1 + asset).prod() ** (trading_days / len(asset)) - 1
        vol = asset.std() * np.sqrt(trading_days)
        sharpe = (asset_cagr - rf_rate) / vol if vol != 0 else np.nan
        treynor = (asset_cagr - rf_rate) / beta if beta and beta != 0 else np.nan
        downside = asset[asset < 0]
        downside_dev = downside.std() * np.sqrt(trading_days) if len(downside) > 1 else np.nan
        sortino = (asset_cagr - rf_rate) / downside_dev if downside_dev and downside_dev != 0 else np.nan
        up_periods = bench > 0
        down_periods = bench < 0
        upside_capture = asset[up_periods].mean() / bench[up_periods].mean() if up_periods.sum() > 0 and bench[up_periods].mean() != 0 else np.nan
        downside_capture = asset[down_periods].mean() / bench[down_periods].mean() if down_periods.sum() > 0 and bench[down_periods].mean() != 0 else np.nan
        batting_average = (active_return > 0).mean()
        cumulative_curve = (1 + asset).cumprod()
        max_drawdown = (cumulative_curve / cumulative_curve.cummax() - 1).min()
        rows.append({
            "Ticker": ticker,
            "Jensen Alpha": jensen_alpha,
            "Information Ratio": information_ratio,
            "Tracking Error": tracking_error,
            "Beta": beta,
            "Treynor Ratio": treynor,
            "Sortino Ratio": sortino,
            "Upside Capture": upside_capture,
            "Downside Capture": downside_capture,
            "Batting Average": batting_average,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_drawdown,
        })
    return pd.DataFrame(rows)

def portfolio_stats(weights, mean_returns, cov_matrix, rf_rate):
    ret = float(np.dot(weights, mean_returns))
    vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    sharpe = (ret - rf_rate) / vol if vol != 0 else np.nan
    return ret, vol, sharpe

def optimize_portfolio(mean_returns, cov_matrix, rf_rate, objective="max_sharpe", max_w=0.20):
    n = len(mean_returns)
    bounds = tuple((0, max_w) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
    x0 = np.array([1 / n] * n)
    if n * max_w < 1:
        return None
    def neg_sharpe(w):
        ret, vol, sharpe = portfolio_stats(w, mean_returns, cov_matrix, rf_rate)
        return -sharpe if not np.isnan(sharpe) else 1e6
    def min_vol(w):
        return portfolio_stats(w, mean_returns, cov_matrix, rf_rate)[1]
    objective_func = neg_sharpe if objective == "max_sharpe" else min_vol
    result = minimize(objective_func, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x if result.success else None

def random_portfolios(mean_returns, cov_matrix, rf_rate, num=2500, max_w=0.20):
    n = len(mean_returns)
    rows = []
    if n * max_w < 1:
        return pd.DataFrame()
    attempts = 0
    while len(rows) < num and attempts < num * 20:
        attempts += 1
        w = np.random.dirichlet(np.ones(n), size=1)[0]
        if np.max(w) > max_w:
            continue
        ret, vol, sharpe = portfolio_stats(w, mean_returns, cov_matrix, rf_rate)
        rows.append({"Return": ret, "Volatility": vol, "Sharpe": sharpe, "Weights": w})
    return pd.DataFrame(rows)

metrics_df = calculate_metrics(asset_returns, benchmark_returns, risk_free_rate)
manager_df = calculate_manager_metrics(asset_returns, benchmark_returns, risk_free_rate)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Price & Returns",
    "Performance Summary",
    "Annual Returns",
    "Risk & Drawdown",
    "Manager Evaluation",
    "Portfolio Optimization",
    "Teaching Notes",
])

with tab1:
    st.subheader("Price Data Preview")
    st.dataframe(prices[available_tickers].tail(), use_container_width=True)
    st.subheader("Normalized Price Chart")
    normalized_prices = prices[available_tickers].dropna(how="all") / prices[available_tickers].dropna(how="all").iloc[0]
    fig_price = px.line(normalized_prices, title="Normalized Price Performance")
    fig_price.update_layout(xaxis_title="Date", yaxis_title="Normalized Price")
    st.plotly_chart(fig_price, use_container_width=True)
    st.subheader("Cumulative Growth of $1")
    cumulative_returns = (1 + asset_returns.fillna(0)).cumprod()
    fig_cum = px.line(cumulative_returns, title="Cumulative Return Comparison")
    fig_cum.update_layout(xaxis_title="Date", yaxis_title="Growth of $1")
    st.plotly_chart(fig_cum, use_container_width=True)

with tab2:
    st.subheader("Performance Summary")
    st.dataframe(metrics_df.style.format({
        "Cumulative Return": "{:.2%}", "CAGR": "{:.2%}", "Volatility": "{:.2%}",
        "Sharpe Ratio": "{:.2f}", "Beta vs Benchmark": "{:.2f}", "Max Drawdown": "{:.2%}",
        "Best Year": "{:.2%}", "Worst Year": "{:.2%}",
    }), use_container_width=True)
    st.download_button("Download performance metrics as CSV", metrics_df.to_csv(index=False), "performance_metrics.csv", "text/csv")
    st.subheader("Risk-Return Scatter Plot")
    scatter_df = metrics_df.copy()
    scatter_df["Bubble Size"] = scatter_df["Sharpe Ratio"].fillna(0).clip(lower=0) + 0.1
    fig_scatter = px.scatter(scatter_df, x="Volatility", y="CAGR", text="Ticker", size="Bubble Size", title="Risk vs Return: CAGR vs Volatility")
    fig_scatter.update_traces(textposition="top center")
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.subheader("Beta vs Sharpe Ratio")
    fig_beta = px.scatter(scatter_df, x="Beta vs Benchmark", y="Sharpe Ratio", text="Ticker", title=f"Beta vs Sharpe Ratio, Benchmark = {benchmark}")
    fig_beta.update_traces(textposition="top center")
    st.plotly_chart(fig_beta, use_container_width=True)

with tab3:
    st.subheader("Annual Returns")
    annual_returns = asset_returns.resample("YE").apply(lambda x: (1 + x.dropna()).prod() - 1)
    annual_returns.index = annual_returns.index.year
    st.dataframe(annual_returns.style.format("{:.2%}"), use_container_width=True)
    fig_annual = px.bar(annual_returns, x=annual_returns.index, y=annual_returns.columns, barmode="group", title="Annual Return Comparison")
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
    st.write(f"This section evaluates each stock or ETF relative to the selected benchmark: **{benchmark}**.")
    st.markdown("""
    Key CFA-style metrics include:
    - **Jensen Alpha:** Excess return after adjusting for beta and the risk-free rate.
    - **Information Ratio:** Active return relative to tracking error.
    - **Tracking Error:** Volatility of active returns versus the benchmark.
    - **Treynor Ratio:** Excess return per unit of beta risk.
    - **Sortino Ratio:** Excess return per unit of downside risk.
    - **Upside/Downside Capture:** Participation in benchmark up and down periods.
    - **Batting Average:** Frequency of outperforming the benchmark.
    """)
    st.dataframe(manager_df.style.format({
        "Jensen Alpha": "{:.2%}", "Information Ratio": "{:.2f}", "Tracking Error": "{:.2%}",
        "Beta": "{:.2f}", "Treynor Ratio": "{:.2f}", "Sortino Ratio": "{:.2f}",
        "Upside Capture": "{:.2f}", "Downside Capture": "{:.2f}", "Batting Average": "{:.2%}",
        "Sharpe Ratio": "{:.2f}", "Max Drawdown": "{:.2%}",
    }), use_container_width=True)
    st.download_button("Download manager evaluation as CSV", manager_df.to_csv(index=False), "manager_evaluation.csv", "text/csv")
    if not manager_df.empty:
        st.subheader("Information Ratio vs Jensen Alpha")
        fig_ir_alpha = px.scatter(manager_df, x="Information Ratio", y="Jensen Alpha", text="Ticker", title="Skill Screen: Information Ratio vs Jensen Alpha")
        fig_ir_alpha.update_traces(textposition="top center")
        st.plotly_chart(fig_ir_alpha, use_container_width=True)

with tab6:
    st.subheader("Portfolio Optimization")
    st.write("This section creates simple long-only optimized portfolios using historical returns. It is for educational purposes and should not be treated as an investment recommendation.")
    opt_returns = asset_returns.dropna(axis=1, how="any")
    if opt_returns.shape[1] < 2:
        st.warning("Optimization needs at least two assets with complete return histories. Try a shorter date range or fewer tickers.")
    elif opt_returns.shape[1] * max_weight < 1:
        st.warning("The maximum weight constraint is too tight for the number of available assets. Increase max weight or add more assets.")
    else:
        mean_returns = opt_returns.mean() * 252
        cov_matrix = opt_returns.cov() * 252
        asset_names = list(opt_returns.columns)
        n = len(asset_names)
        equal_w = np.array([1 / n] * n)
        max_sharpe_w = optimize_portfolio(mean_returns.values, cov_matrix.values, risk_free_rate, "max_sharpe", max_weight)
        min_var_w = optimize_portfolio(mean_returns.values, cov_matrix.values, risk_free_rate, "min_var", max_weight)
        portfolios = []
        for name, weights in [("Equal Weight", equal_w), ("Maximum Sharpe", max_sharpe_w), ("Minimum Variance", min_var_w)]:
            if weights is not None:
                ret, vol, sharpe = portfolio_stats(weights, mean_returns.values, cov_matrix.values, risk_free_rate)
                portfolios.append({"Portfolio": name, "Expected Return": ret, "Volatility": vol, "Sharpe Ratio": sharpe})
        portfolio_summary = pd.DataFrame(portfolios)
        st.subheader("Optimized Portfolio Summary")
        st.dataframe(portfolio_summary.style.format({"Expected Return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe Ratio": "{:.2f}"}), use_container_width=True)
        st.subheader("Optimized Weights")
        weight_rows = []
        for name, weights in [("Equal Weight", equal_w), ("Maximum Sharpe", max_sharpe_w), ("Minimum Variance", min_var_w)]:
            if weights is not None:
                row = {"Portfolio": name}
                row.update({asset_names[i]: weights[i] for i in range(n)})
                weight_rows.append(row)
        weights_df = pd.DataFrame(weight_rows)
        st.dataframe(weights_df.style.format({col: "{:.2%}" for col in weights_df.columns if col != "Portfolio"}), use_container_width=True)
        st.download_button("Download optimized weights as CSV", weights_df.to_csv(index=False), "optimized_weights.csv", "text/csv")
        st.subheader("Efficient Frontier Simulation")
        frontier_df = random_portfolios(mean_returns.values, cov_matrix.values, risk_free_rate, num_simulations, max_weight)
        if frontier_df.empty:
            st.warning("Unable to generate frontier with the current constraints.")
        else:
            fig_frontier = px.scatter(frontier_df, x="Volatility", y="Return", color="Sharpe", title="Efficient Frontier: Random Long-Only Portfolios")
            for _, row in portfolio_summary.iterrows():
                fig_frontier.add_trace(go.Scatter(x=[row["Volatility"]], y=[row["Expected Return"]], mode="markers+text", text=[row["Portfolio"]], textposition="top center", marker=dict(size=12, symbol="star"), name=row["Portfolio"]))
            fig_frontier.update_layout(xaxis_title="Annualized Volatility", yaxis_title="Expected Annual Return")
            st.plotly_chart(fig_frontier, use_container_width=True)
        st.markdown("""
        **Teaching interpretation:** Optimization is only as good as its inputs. Students should compare optimized weights with their investment thesis, valuation work, liquidity constraints, and risk limits. A model may over-allocate to assets that performed well historically, so professional judgment is required.
        """)

with tab7:
    st.subheader("Teaching Notes for SMIF Students")
    st.markdown("""
    ## How to Use This Website
    This website is designed to help SMIF and Advanced Financial Modeling students compare individual stocks and ETFs in a structured way.

    Students should not simply choose the asset with the highest return. They should compare return, risk, beta, drawdown, correlation, manager evaluation metrics, and the role of each asset in the portfolio.

    ## Key Questions
    1. Which asset had the highest cumulative return?
    2. Which asset had the highest Sharpe ratio?
    3. Which asset had the largest maximum drawdown?
    4. Which asset has the highest beta relative to the benchmark?
    5. Are these assets highly correlated with one another?
    6. Does this asset improve portfolio diversification?
    7. Is performance due to alpha, beta exposure, factor exposure, or one lucky period?
    8. How do optimized weights compare with your qualitative investment thesis?

    ## Suggested Student Output
    For each stock or ETF, students should summarize:
    - Investment thesis
    - Return performance
    - Risk profile
    - Beta and market sensitivity
    - Drawdown risk
    - Correlation with existing holdings
    - Jensen alpha, information ratio, and tracking error
    - Optimization role: core holding, satellite holding, hedge, or excluded asset
    - Final recommendation: Buy, Add, Hold, Trim, Sell, or Watchlist

    ## Important Reminder
    This app is a learning tool. It is not a substitute for fundamental analysis, valuation, macro analysis, or professional judgment.
    """)
