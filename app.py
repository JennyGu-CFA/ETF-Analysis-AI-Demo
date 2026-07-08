from macro_panel import render_macro_market_regime_tab

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize

try:
    import feedparser
except Exception:
    feedparser = None

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="SMIF Stock & ETF Analysis Website", layout="wide")

# -----------------------------
# Preset ticker groups
# -----------------------------
PRESETS = {
    "Technology / Communication": "AAPL, MSFT, NVDA, AVGO, META, GOOG, AMZN, ASML, QQQ, SPY",
    "Healthcare": "LLY, JNJ, PFE, BMY, UNH, ABT, REGN, BIIB, XLV, SPY",
    "Financials": "JPM, BLK, SCHW, MA, V, XLF, SPY",
    "Energy / Industrials": "XOM, CVX, SHEL, OXY, LMT, EMR, FDX, UNP, XLE, XLI, SPY",
    "Consumer": "AMZN, WMT, COST, XLY, XLP, SPY",
    "Defensive / Hedge": "SPY, QQQ, TLT, GLD, SGOV, VOO, VTI",
    "Custom": "AAPL, MSFT, NVDA, SPY, QQQ, TLT, GLD, SGOV",
}

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("User Inputs")

preset = st.sidebar.selectbox("Choose a SMIF preset list", list(PRESETS.keys()), index=0)

tickers_input = st.sidebar.text_area(
    "Enter stock or ETF tickers separated by commas",
    PRESETS[preset],
    height=95,
)

benchmark_input = st.sidebar.text_input("Benchmark ticker", "SPY")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
risk_free_rate = st.sidebar.number_input(
    "Annual Risk-Free Rate",
    min_value=0.0,
    max_value=0.25,
    value=0.04,
    step=0.005,
    format="%.3f",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Optimization Settings")
max_weight = st.sidebar.slider("Maximum weight per asset", 0.05, 1.0, 0.20, 0.01)
num_random_portfolios = st.sidebar.slider("Random portfolios for efficient frontier", 500, 10000, 2500, 500)

st.sidebar.subheader("Custom Optimization")
custom_objective = st.sidebar.selectbox(
    "Custom objective",
    [
        "Maximize Sharpe Ratio",
        "Minimize Volatility",
        "Target Return: Minimize Volatility",
        "Target Risk: Maximize Return",
        "Balanced Return-Risk Score",
    ],
)
target_return = st.sidebar.number_input("Target annual return for custom optimization", 0.0, 1.0, 0.10, 0.01)
target_volatility = st.sidebar.number_input("Target annual volatility for custom optimization", 0.01, 1.0, 0.20, 0.01)
risk_penalty = st.sidebar.slider("Risk penalty for balanced score", 0.0, 10.0, 3.0, 0.25)

st.sidebar.markdown("---")
st.sidebar.write("Example stocks: AAPL, MSFT, NVDA, AMZN, META, JPM, LLY")
st.sidebar.write("Example ETFs: SPY, QQQ, VOO, VTI, TLT, GLD, SGOV")

# -----------------------------
# Data helpers
# -----------------------------
def clean_tickers(text):
    tickers = []
    for item in text.replace("\n", ",").split(","):
        ticker = item.strip().upper()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers

tickers = clean_tickers(tickers_input)
benchmark = benchmark_input.strip().upper()
all_tickers = list(dict.fromkeys(tickers + ([benchmark] if benchmark and benchmark not in tickers else [])))

@st.cache_data(ttl=900)
def load_price_data(ticker_list, start, end):
    if not ticker_list:
        return pd.DataFrame()
    data = yf.download(ticker_list, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = ticker_list[:1]
    prices = prices.dropna(how="all")
    return prices

prices = load_price_data(all_tickers, start_date, end_date)

# Only keep tickers that have some valid data
available_tickers = []
if not prices.empty:
    available_tickers = [t for t in tickers if t in prices.columns and prices[t].dropna().shape[0] > 5]

# Prepare returns only if data is available
if not prices.empty and available_tickers and benchmark in prices.columns:
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    asset_returns = returns[available_tickers].dropna(how="all")
    benchmark_returns = returns[benchmark].dropna()
else:
    returns = pd.DataFrame()
    asset_returns = pd.DataFrame()
    benchmark_returns = pd.Series(dtype=float)

# -----------------------------
# Finance helper functions
# -----------------------------
def annualized_return(r):
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    years = len(r) / 252
    total = (1 + r).prod()
    if years <= 0 or total <= 0:
        return np.nan
    return total ** (1 / years) - 1

def annualized_volatility(r):
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return r.std() * np.sqrt(252)

def max_drawdown(r):
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    curve = (1 + r).cumprod()
    running_max = curve.cummax()
    dd = curve / running_max - 1
    return dd.min()

def downside_deviation(r, mar=0.0):
    r = r.dropna()
    downside = r[r < mar / 252]
    if len(downside) == 0:
        return np.nan
    return downside.std() * np.sqrt(252)

def calculate_beta(asset_r, bench_r):
    aligned = pd.concat([asset_r, bench_r], axis=1).dropna()
    if len(aligned) < 5:
        return np.nan
    aligned.columns = ["asset", "benchmark"]
    bench_var = np.var(aligned["benchmark"])
    if bench_var == 0:
        return np.nan
    return np.cov(aligned["asset"], aligned["benchmark"])[0][1] / bench_var

def calculate_metrics(returns_df, benchmark_series, rf_rate):
    columns = [
        "Ticker", "Cumulative Return", "CAGR", "Volatility", "Sharpe Ratio", "Sortino Ratio",
        "Beta vs Benchmark", "Max Drawdown", "Best Year", "Worst Year"
    ]
    rows = []
    if returns_df.empty:
        return pd.DataFrame(columns=columns)
    for ticker in returns_df.columns:
        r = returns_df[ticker].dropna()
        if len(r) < 5:
            continue
        cumulative_return = (1 + r).prod() - 1
        cagr = annualized_return(r)
        vol = annualized_volatility(r)
        sharpe = (cagr - rf_rate) / vol if pd.notna(vol) and vol != 0 else np.nan
        dd = downside_deviation(r)
        sortino = (cagr - rf_rate) / dd if pd.notna(dd) and dd != 0 else np.nan
        beta = calculate_beta(r, benchmark_series)
        mdd = max_drawdown(r)
        annual = r.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        rows.append({
            "Ticker": ticker,
            "Cumulative Return": cumulative_return,
            "CAGR": cagr,
            "Volatility": vol,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Beta vs Benchmark": beta,
            "Max Drawdown": mdd,
            "Best Year": annual.max() if len(annual) else np.nan,
            "Worst Year": annual.min() if len(annual) else np.nan,
        })
    return pd.DataFrame(rows, columns=columns)

def calculate_drawdowns(returns_df):
    out = pd.DataFrame(index=returns_df.index)
    for ticker in returns_df.columns:
        r = returns_df[ticker].dropna()
        curve = (1 + r).cumprod()
        out[ticker] = curve / curve.cummax() - 1
    return out

def calculate_manager_metrics(returns_df, benchmark_series, rf_rate):
    columns = [
        "Ticker", "Jensen Alpha", "Information Ratio", "Tracking Error", "Treynor Ratio",
        "Upside Capture", "Downside Capture", "Batting Average", "Sharpe Ratio", "Max Drawdown"
    ]
    rows = []
    if returns_df.empty or benchmark_series.empty:
        return pd.DataFrame(columns=columns)
    bench_ann = annualized_return(benchmark_series)
    for ticker in returns_df.columns:
        r = returns_df[ticker].dropna()
        aligned = pd.concat([r, benchmark_series], axis=1).dropna()
        if len(aligned) < 20:
            continue
        aligned.columns = ["asset", "benchmark"]
        asset_ann = annualized_return(aligned["asset"])
        asset_vol = annualized_volatility(aligned["asset"])
        beta = calculate_beta(aligned["asset"], aligned["benchmark"])
        sharpe = (asset_ann - rf_rate) / asset_vol if pd.notna(asset_vol) and asset_vol != 0 else np.nan
        jensen_alpha = asset_ann - (rf_rate + beta * (bench_ann - rf_rate)) if pd.notna(beta) and pd.notna(bench_ann) else np.nan
        active = aligned["asset"] - aligned["benchmark"]
        tracking_error = active.std() * np.sqrt(252)
        active_ann = annualized_return(aligned["asset"]) - annualized_return(aligned["benchmark"])
        information_ratio = active_ann / tracking_error if tracking_error != 0 else np.nan
        treynor = (asset_ann - rf_rate) / beta if pd.notna(beta) and beta != 0 else np.nan
        up = aligned[aligned["benchmark"] > 0]
        down = aligned[aligned["benchmark"] < 0]
        upside_capture = up["asset"].mean() / up["benchmark"].mean() if len(up) > 0 and up["benchmark"].mean() != 0 else np.nan
        downside_capture = down["asset"].mean() / down["benchmark"].mean() if len(down) > 0 and down["benchmark"].mean() != 0 else np.nan
        batting = (active > 0).mean() if len(active) > 0 else np.nan
        rows.append({
            "Ticker": ticker,
            "Jensen Alpha": jensen_alpha,
            "Information Ratio": information_ratio,
            "Tracking Error": tracking_error,
            "Treynor Ratio": treynor,
            "Upside Capture": upside_capture,
            "Downside Capture": downside_capture,
            "Batting Average": batting,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_drawdown(aligned["asset"]),
        })
    return pd.DataFrame(rows, columns=columns)

def portfolio_stats(weights, mu, cov, rf_rate):
    ret = float(np.dot(weights, mu))
    vol = float(np.sqrt(np.dot(weights.T, np.dot(cov, weights))))
    sharpe = (ret - rf_rate) / vol if vol != 0 else np.nan
    return ret, vol, sharpe

def optimize_portfolio(mu, cov, rf_rate, objective="max_sharpe", max_w=0.2, target_ret=0.10, target_vol=0.20, risk_pen=3.0):
    n = len(mu)
    if n == 0:
        return None
    bounds = [(0, max_w)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    x0 = np.repeat(1 / n, n)

    if objective == "max_sharpe":
        def obj(w):
            _, _, s = portfolio_stats(w, mu, cov, rf_rate)
            return -s if pd.notna(s) else 1e6
    elif objective == "min_vol":
        def obj(w):
            return portfolio_stats(w, mu, cov, rf_rate)[1]
    elif objective == "target_return":
        constraints.append({"type": "ineq", "fun": lambda w: np.dot(w, mu) - target_ret})
        def obj(w):
            return portfolio_stats(w, mu, cov, rf_rate)[1]
    elif objective == "target_risk":
        constraints.append({"type": "ineq", "fun": lambda w: target_vol - portfolio_stats(w, mu, cov, rf_rate)[1]})
        def obj(w):
            return -np.dot(w, mu)
    else:
        def obj(w):
            ret, vol, _ = portfolio_stats(w, mu, cov, rf_rate)
            return -(ret - risk_pen * vol)

    result = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        return None
    return result.x

def random_portfolios(mu, cov, rf_rate, n_ports=2500, max_w=0.2):
    n = len(mu)
    rows = []
    if n == 0:
        return pd.DataFrame()
    for _ in range(n_ports):
        # rejection sampling to respect max weight; fallback if not possible
        w = np.random.dirichlet(np.ones(n))
        attempts = 0
        while np.max(w) > max_w and attempts < 100:
            w = np.random.dirichlet(np.ones(n))
            attempts += 1
        if np.max(w) > max_w:
            continue
        ret, vol, sharpe = portfolio_stats(w, mu, cov, rf_rate)
        rows.append({"Return": ret, "Volatility": vol, "Sharpe Ratio": sharpe})
    return pd.DataFrame(rows)

@st.cache_data(ttl=300)
def load_rss_news(url, source):
    if feedparser is None:
        return pd.DataFrame(columns=["Source", "Published", "Headline", "Link"])
    feed = feedparser.parse(url)
    rows = []
    for entry in feed.entries:
        rows.append({
            "Source": source,
            "Published": getattr(entry, "published", ""),
            "Headline": getattr(entry, "title", ""),
            "Link": getattr(entry, "link", ""),
        })
    return pd.DataFrame(rows)

# -----------------------------
# Header
# -----------------------------
st.title("SMIF Stock & ETF Analysis Website")
st.write(
    "This interactive website is designed for SMIF and Advanced Financial Modeling students. "
    "Students can compare individual stocks and ETFs using Python, AI-assisted coding, and financial modeling tools."
)
st.write(
    "The website allows users to compare cumulative returns, annual returns, CAGR, volatility, Sharpe ratios, beta, "
    "maximum drawdowns, correlations, rolling volatility, professional manager evaluation metrics, "
    "mean-variance optimization, custom optimization results, and market headline updates."
)
st.info("Educational demo only. This app does not provide investment, tax, legal, or financial planning advice.")

# Data warning
if prices.empty:
    st.error("No price data found. Please check your tickers and date range.")
elif not available_tickers:
    st.error("No selected ticker has enough valid price data. Please check ticker symbols and date range.")
elif benchmark not in prices.columns:
    st.error("Benchmark data was not found. Please check the benchmark ticker.")

# -----------------------------
# Tabs
# -----------------------------
 tab_news, tab_price, tab_summary, tab_annual, tab_risk, tab_manager, tab_opt, tab_macro, tab_notes = st.tabs([
    "Market News",
    "Price & Returns",
    "Performance Summary",
    "Annual Returns",
    "Risk & Drawdown",
    "Manager Evaluation",
    "Portfolio Optimization",
    "Macro & Market Regime",
    "Teaching Notes",
])

# -----------------------------
# Market News tab
# -----------------------------
with tab_news:
    st.subheader("Market News Briefing")
    st.write("This tab shows public RSS headlines and links. Students with WSJ access can click WSJ links and read articles using their own subscription.")
    col1, col2, col3 = st.columns(3)
    with col1:
        keyword = st.text_input("Keyword filter", "")
    with col2:
        max_rows = st.slider("Max headlines", 10, 100, 30, 5)
    with col3:
        refresh_minutes = st.slider("Auto-refresh minutes", 5, 60, 15, 5)
    if st_autorefresh is not None:
        st_autorefresh(interval=refresh_minutes * 60 * 1000, key="news_refresh")

    feeds = [
        ("https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "WSJ Markets"),
        ("https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml", "WSJ Business"),
        ("https://www.cnbc.com/id/100003114/device/rss/rss.html", "CNBC Top News"),
        ("https://www.cnbc.com/id/15839135/device/rss/rss.html", "CNBC Markets"),
    ]
    news_frames = [load_rss_news(url, src) for url, src in feeds]
    news_df = pd.concat(news_frames, ignore_index=True) if news_frames else pd.DataFrame()
    if keyword.strip() and not news_df.empty:
        mask = news_df["Headline"].str.contains(keyword.strip(), case=False, na=False)
        news_df = news_df[mask]
    if news_df.empty:
        st.warning("No headlines loaded. RSS feeds may be temporarily unavailable.")
    else:
        st.dataframe(news_df.head(max_rows), use_container_width=True)
        for _, row in news_df.head(max_rows).iterrows():
            st.markdown(f"**{row['Source']}** — [{row['Headline']}]({row['Link']})")
            if row.get("Published"):
                st.caption(str(row["Published"]))

# Stop non-news tabs if data invalid
valid_data = (not prices.empty) and bool(available_tickers) and (benchmark in prices.columns) and (not asset_returns.empty)

# -----------------------------
# Price tab
# -----------------------------
with tab_price:
    if not valid_data:
        st.warning("Price analysis is unavailable until valid ticker data is loaded.")
    else:
        st.subheader("Price Data Preview")
        st.dataframe(prices[available_tickers].tail(), use_container_width=True)
        st.subheader("Normalized Price Chart")
        normalized_prices = prices[available_tickers].dropna(how="all") / prices[available_tickers].dropna(how="all").iloc[0]
        st.plotly_chart(px.line(normalized_prices, title="Normalized Price Performance"), use_container_width=True)
        st.subheader("Cumulative Growth of $1")
        cumulative_returns = (1 + asset_returns).cumprod()
        st.plotly_chart(px.line(cumulative_returns, title="Cumulative Return Comparison"), use_container_width=True)

# -----------------------------
# Summary tab
# -----------------------------
with tab_summary:
    if not valid_data:
        st.warning("Performance summary is unavailable until valid ticker data is loaded.")
    else:
        st.subheader("Performance Summary")
        metrics_df = calculate_metrics(asset_returns, benchmark_returns, risk_free_rate)
        st.dataframe(metrics_df.style.format({
            "Cumulative Return": "{:.2%}", "CAGR": "{:.2%}", "Volatility": "{:.2%}",
            "Sharpe Ratio": "{:.2f}", "Sortino Ratio": "{:.2f}", "Beta vs Benchmark": "{:.2f}",
            "Max Drawdown": "{:.2%}", "Best Year": "{:.2%}", "Worst Year": "{:.2%}"
        }), use_container_width=True)
        st.download_button("Download performance metrics as CSV", metrics_df.to_csv(index=False), "performance_metrics.csv", "text/csv")
        if not metrics_df.empty:
            scatter_df = metrics_df.copy()
            scatter_df["Bubble Size"] = scatter_df["Sharpe Ratio"].fillna(0).clip(lower=0) + 0.1
            st.subheader("Risk-Return Scatter Plot")
            fig = px.scatter(scatter_df, x="Volatility", y="CAGR", text="Ticker", size="Bubble Size", title="Risk vs Return: CAGR vs Volatility")
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Beta vs Sharpe Ratio")
            fig2 = px.scatter(scatter_df, x="Beta vs Benchmark", y="Sharpe Ratio", text="Ticker", title=f"Beta vs Sharpe Ratio, Benchmark = {benchmark}")
            fig2.update_traces(textposition="top center")
            st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Annual tab
# -----------------------------
with tab_annual:
    if not valid_data:
        st.warning("Annual returns are unavailable until valid ticker data is loaded.")
    else:
        st.subheader("Annual Returns")
        annual_returns = asset_returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        annual_returns.index = annual_returns.index.year
        st.dataframe(annual_returns.style.format("{:.2%}"), use_container_width=True)
        st.plotly_chart(px.bar(annual_returns, barmode="group", title="Annual Return Comparison"), use_container_width=True)

# -----------------------------
# Risk tab
# -----------------------------
with tab_risk:
    if not valid_data:
        st.warning("Risk analysis is unavailable until valid ticker data is loaded.")
    else:
        st.subheader("Drawdown Analysis")
        drawdowns = calculate_drawdowns(asset_returns)
        st.plotly_chart(px.line(drawdowns, title="Drawdown Comparison"), use_container_width=True)
        st.subheader("Correlation Matrix")
        corr_matrix = asset_returns.corr()
        st.plotly_chart(px.imshow(corr_matrix, text_auto=True, title="Return Correlation Matrix"), use_container_width=True)
        st.subheader("Rolling 12-Month Volatility")
        rolling_vol = asset_returns.rolling(window=252).std() * np.sqrt(252)
        st.plotly_chart(px.line(rolling_vol, title="Rolling 12-Month Volatility"), use_container_width=True)

# -----------------------------
# Manager evaluation tab
# -----------------------------
with tab_manager:
    if not valid_data:
        st.warning("Manager evaluation is unavailable until valid ticker data is loaded.")
    else:
        st.subheader("Professional Manager Performance Evaluation")
        st.write(f"This section evaluates each stock or ETF relative to the selected benchmark: **{benchmark}**.")
        st.markdown(
            """
            Key CFA-style metrics include:
            - **Jensen Alpha:** Excess return after adjusting for beta and the risk-free rate.
            - **Information Ratio:** Active return relative to tracking error.
            - **Tracking Error:** Volatility of active returns versus the benchmark.
            - **Treynor Ratio:** Excess return per unit of beta risk.
            - **Upside Capture:** Participation when the benchmark is up.
            - **Downside Capture:** Participation when the benchmark is down.
            - **Batting Average:** Frequency of outperforming the benchmark.
            """
        )
        manager_df = calculate_manager_metrics(asset_returns, benchmark_returns, risk_free_rate)
        st.dataframe(manager_df.style.format({
            "Jensen Alpha": "{:.2%}", "Information Ratio": "{:.2f}", "Tracking Error": "{:.2%}",
            "Treynor Ratio": "{:.2f}", "Upside Capture": "{:.2f}", "Downside Capture": "{:.2f}",
            "Batting Average": "{:.2%}", "Sharpe Ratio": "{:.2f}", "Max Drawdown": "{:.2%}"
        }), use_container_width=True)
        st.download_button("Download manager evaluation as CSV", manager_df.to_csv(index=False), "manager_evaluation.csv", "text/csv")

# -----------------------------
# Optimization tab
# -----------------------------
with tab_opt:
    if not valid_data or len(asset_returns.columns) < 2:
        st.warning("Portfolio optimization requires at least two valid tickers with return data.")
    else:
        st.subheader("Portfolio Optimization")
        opt_returns = asset_returns.dropna(axis=1, how="any")
        if opt_returns.shape[1] < 2:
            st.warning("Not enough complete return series for optimization. Try a shorter date range or fewer tickers.")
        else:
            mu = opt_returns.mean() * 252
            cov = opt_returns.cov() * 252
            names = opt_returns.columns.tolist()
            n = len(names)
            equal_w = np.repeat(1 / n, n)
            eq_ret, eq_vol, eq_sharpe = portfolio_stats(equal_w, mu, cov, risk_free_rate)
            min_w = optimize_portfolio(mu.values, cov.values, risk_free_rate, "min_vol", max_weight)
            maxs_w = optimize_portfolio(mu.values, cov.values, risk_free_rate, "max_sharpe", max_weight)
            obj_map = {
                "Maximize Sharpe Ratio": "max_sharpe",
                "Minimize Volatility": "min_vol",
                "Target Return: Minimize Volatility": "target_return",
                "Target Risk: Maximize Return": "target_risk",
                "Balanced Return-Risk Score": "balanced",
            }
            custom_w = optimize_portfolio(mu.values, cov.values, risk_free_rate, obj_map[custom_objective], max_weight, target_return, target_volatility, risk_penalty)
            rows = []
            for label, w in [("Equal Weight", equal_w), ("Minimum Variance", min_w), ("Maximum Sharpe", maxs_w), ("Custom Optimization", custom_w)]:
                if w is not None:
                    ret, vol, sh = portfolio_stats(w, mu.values, cov.values, risk_free_rate)
                    rows.append({"Portfolio": label, "Expected Return": ret, "Volatility": vol, "Sharpe Ratio": sh})
            port_summary = pd.DataFrame(rows)
            st.subheader("Optimized Portfolio Summary")
            st.dataframe(port_summary.style.format({"Expected Return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe Ratio": "{:.2f}"}), use_container_width=True)
            weight_rows = []
            for label, w in [("Equal Weight", equal_w), ("Minimum Variance", min_w), ("Maximum Sharpe", maxs_w), ("Custom Optimization", custom_w)]:
                if w is not None:
                    for name, weight in zip(names, w):
                        weight_rows.append({"Portfolio": label, "Ticker": name, "Weight": weight})
            weights_df = pd.DataFrame(weight_rows)
            st.subheader("Optimized Weights")
            st.dataframe(weights_df.pivot(index="Ticker", columns="Portfolio", values="Weight").fillna(0).style.format("{:.2%}"), use_container_width=True)
            st.download_button("Download optimized weights as CSV", weights_df.to_csv(index=False), "optimized_weights.csv", "text/csv")
            st.subheader("Simulated Efficient Frontier")
            sim_df = random_portfolios(mu.values, cov.values, risk_free_rate, num_random_portfolios, max_weight)
            if not sim_df.empty:
                st.plotly_chart(px.scatter(sim_df, x="Volatility", y="Return", color="Sharpe Ratio", title="Efficient Frontier Simulation"), use_container_width=True)
            else:
                st.warning("Efficient frontier simulation could not generate portfolios under the current constraints. Increase maximum weight per asset.")


# -----------------------------
# Macro & Market Regime tab
# -----------------------------
with tab_macro:
    render_macro_market_regime_tab()
# -----------------------------
# Teaching notes tab
# -----------------------------
with tab_notes:
    st.subheader("Teaching Notes for SMIF Students")
    st.markdown(
        """
        ## How to Use This Website

        This website is designed to help SMIF and Advanced Financial Modeling students compare individual stocks and ETFs in a structured way.

        Students should not simply choose the asset with the highest return. They should compare return, risk, beta, drawdown, correlation, and the role of each asset in the portfolio.

        ## Suggested Student Output

        For each stock or ETF, students should summarize:

        - Investment thesis
        - Return performance
        - Risk profile
        - Beta and market sensitivity
        - Drawdown risk
        - Correlation with existing holdings
        - Manager evaluation metrics: alpha, information ratio, tracking error, capture ratios
        - Portfolio optimization role
        - Final recommendation: Buy, Add, Hold, Trim, Sell, or Watchlist

        ## Optimization Reminder

        Optimization is a decision-support tool, not an automatic investment decision. Students must explain whether the optimized weights make economic sense and whether they are consistent with the fund's investment process.
        """
    )
