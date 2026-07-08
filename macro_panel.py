"""Macro & Market Regime panel for ETF-Analysis-AI-Demo.

Drop this file next to app.py, then import and call render_macro_market_regime_tab()
inside a new Streamlit tab.

Educational / research demo only. Not investment advice.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf


MAG7 = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]

SEMIS_HBM = [
    "SMH",
    "SOXX",
    "NVDA",
    "AVGO",
    "TSM",
    "ASML",
    "AMD",
    "MU",
    "005930.KS",
    "000660.KS",
]

CROSS_ASSET = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
    "EFA": "Developed ex-US",
    "EEM": "Emerging Markets",
    "VXUS": "International ex-US",
    "GLD": "Gold ETF",
    "IAU": "Gold ETF low-fee proxy",
    "TLT": "Long-duration Treasury",
    "SHY": "Short Treasury",
    "SMH": "Semiconductor ETF",
    "SOXX": "Semiconductor ETF",
}

RATE_PROXIES = {
    "^TNX": "10Y Treasury Yield Proxy",
    "^IRX": "13W T-Bill Yield Proxy",
    "DX-Y.NYB": "US Dollar Index Proxy",
    "CL=F": "Crude Oil Futures Proxy",
    "GC=F": "Gold Futures Proxy",
}


@dataclass
class RegimeResult:
    label: str
    score: float
    drivers: List[str]
    warnings: List[str]


@st.cache_data(ttl=900, show_spinner=False)
def _download_prices(tickers: Tuple[str, ...], period: str = "2y") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    data = yf.download(list(tickers), period=period, auto_adjust=True, progress=False)

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        prices = data.get("Close", pd.DataFrame()).copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = [tickers[0]]

    prices = prices.dropna(how="all")
    return prices


@st.cache_data(ttl=3600, show_spinner=False)
def _download_info(tickers: Tuple[str, ...]) -> pd.DataFrame:
    rows = []

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).get_info()
        except Exception:
            info = {}

        rows.append(
            {
                "Ticker": ticker,
                "Market Cap": info.get("marketCap"),
                "Trailing PE": info.get("trailingPE"),
                "Forward PE": info.get("forwardPE"),
                "EV / EBITDA": info.get("enterpriseToEbitda"),
                "Revenue Growth": info.get("revenueGrowth"),
                "Gross Margin": info.get("grossMargins"),
                "Operating Margin": info.get("operatingMargins"),
                "Operating Cash Flow": info.get("operatingCashflow"),
                "Free Cash Flow": info.get("freeCashflow"),
                "Operating Income": info.get("operatingIncome"),
            }
        )

    return pd.DataFrame(rows)


def _normalized(prices: pd.DataFrame) -> pd.DataFrame:
    clean = prices.dropna(how="all")

    if clean.empty:
        return clean

    first = clean.apply(lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan)
    return clean / first * 100


def _total_return(prices: pd.DataFrame, days: int = 252) -> pd.Series:
    if prices.empty:
        return pd.Series(dtype=float)

    recent = prices.tail(days + 1)
    return recent.ffill().iloc[-1] / recent.ffill().iloc[0] - 1


def _volatility(prices: pd.DataFrame, days: int = 252) -> pd.Series:
    returns = prices.pct_change(fill_method=None).tail(days)
    return returns.std() * np.sqrt(252)


def _safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _label_from_score(score: float) -> str:
    if score >= 70:
        return "Risk-on / Momentum-led"
    if score >= 55:
        return "Constructive but selective"
    if score >= 40:
        return "Mixed / transition regime"
    if score >= 25:
        return "Defensive / liquidity-sensitive"
    return "Risk-off / stress regime"


def calculate_market_regime_score(
    cross_prices: pd.DataFrame,
    mag7_info: pd.DataFrame,
    fear_greed: Optional[float] = None,
    buffett_indicator: Optional[float] = None,
    shiller_cape: Optional[float] = None,
) -> RegimeResult:
    """Simple transparent scoring model for teaching/demo purposes."""

    score = 50.0
    drivers: List[str] = []
    warnings: List[str] = []

    returns_1y = _total_return(cross_prices, 252)

    spy = _safe_float(returns_1y.get("SPY"))
    qqq = _safe_float(returns_1y.get("QQQ"))
    iwm = _safe_float(returns_1y.get("IWM"))
    eem = _safe_float(returns_1y.get("EEM"))
    efa = _safe_float(returns_1y.get("EFA"))
    gld = _safe_float(returns_1y.get("GLD"))
    smh = _safe_float(returns_1y.get("SMH"))
    tlt = _safe_float(returns_1y.get("TLT"))

    if spy is not None and spy > 0.12:
        score += 8
        drivers.append("SPY has positive 1-year momentum.")
    elif spy is not None and spy < -0.05:
        score -= 10
        warnings.append("SPY 1-year return is negative, suggesting broad market pressure.")

    if qqq is not None and spy is not None and qqq - spy > 0.08:
        score += 4
        drivers.append("QQQ is outperforming SPY, indicating growth/technology leadership.")
        warnings.append("Leadership may be concentrated in growth and mega-cap technology.")

    if iwm is not None and spy is not None and iwm - spy > 0.03:
        score += 6
        drivers.append("Small caps are participating, which improves market breadth.")
    elif iwm is not None and spy is not None and iwm - spy < -0.10:
        score -= 6
        warnings.append("Small caps are lagging SPY, suggesting weak breadth.")

    if eem is not None and efa is not None and spy is not None:
        intl_avg = np.nanmean([eem, efa])
        if intl_avg - spy > 0.03:
            score += 3
            drivers.append("International and emerging markets are participating.")
        elif spy - intl_avg > 0.10:
            warnings.append("US outperformance remains large versus international markets.")

    if gld is not None and spy is not None and gld - spy > 0.05:
        score -= 4
        warnings.append(
            "Gold is outperforming equities, which can signal defensive or inflation hedging demand."
        )
    elif gld is not None and gld > 0.10:
        drivers.append(
            "Gold has positive momentum; monitor inflation, real rates, and safe-haven demand."
        )

    if smh is not None and spy is not None and smh - spy > 0.15:
        score += 5
        drivers.append("Semiconductors are strongly outperforming the market.")
        warnings.append("AI/semiconductor concentration risk may be elevated.")

    if tlt is not None and spy is not None and tlt < -0.05 and spy > 0:
        warnings.append(
            "Long-duration bonds are weak while equities are up; rate sensitivity remains important."
        )

    if fear_greed is not None:
        if fear_greed >= 75:
            score += 2
            warnings.append(
                "Fear & Greed is in extreme greed territory; sentiment risk is elevated."
            )
        elif fear_greed <= 25:
            score -= 8
            warnings.append("Fear & Greed is in fear/extreme fear territory.")
        else:
            drivers.append("Fear & Greed is not at an extreme.")

    if buffett_indicator is not None:
        if buffett_indicator >= 1.8:
            score -= 8
            warnings.append(
                "Buffett Indicator is elevated versus GDP, suggesting valuation stress."
            )
        elif buffett_indicator <= 1.1:
            score += 5
            drivers.append("Buffett Indicator is relatively moderate.")

    if shiller_cape is not None:
        if shiller_cape >= 35:
            score -= 6
            warnings.append("Shiller CAPE is elevated; long-term valuation risk is high.")
        elif shiller_cape <= 20:
            score += 4
            drivers.append("Shiller CAPE is relatively moderate.")

    if not mag7_info.empty and "Market Cap" in mag7_info.columns:
        total_cap = mag7_info["Market Cap"].dropna().sum()
        largest = mag7_info["Market Cap"].dropna().max()

        if total_cap > 0 and largest / total_cap > 0.25:
            warnings.append(
                "MAG7 market cap concentration is high inside the mega-cap basket."
            )

        median_fpe = (
            mag7_info["Forward PE"].dropna().median()
            if "Forward PE" in mag7_info
            else np.nan
        )

        if pd.notna(median_fpe) and median_fpe > 35:
            score -= 4
            warnings.append("MAG7 median forward PE is elevated.")

    score = float(np.clip(score, 0, 100))
    return RegimeResult(_label_from_score(score), score, drivers, warnings)


def calculate_mag7_concentration(
    mag7_info: pd.DataFrame, mag7_prices: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    info = mag7_info.copy()

    if info.empty:
        return pd.DataFrame(), pd.DataFrame()

    total_cap = info["Market Cap"].dropna().sum() if "Market Cap" in info else np.nan

    if total_cap and total_cap > 0:
        info["Weight within MAG7"] = info["Market Cap"] / total_cap
    else:
        info["Weight within MAG7"] = np.nan

    for col in ["Free Cash Flow", "Operating Income"]:
        if col in info and "Market Cap" in info:
            info[f"{col} Yield"] = info[col] / info["Market Cap"]

    prices = mag7_prices[MAG7].dropna(how="all") if not mag7_prices.empty else pd.DataFrame()

    if prices.empty:
        return info, pd.DataFrame()

    norm = _normalized(prices)
    equal_weight = norm.mean(axis=1).rename("MAG7 Equal Weight")

    market_weights = (
        info.set_index("Ticker")["Weight within MAG7"]
        .reindex(prices.columns)
        .fillna(0)
    )

    if market_weights.sum() > 0:
        market_weights = market_weights / market_weights.sum()
        cap_weight = (norm * market_weights).sum(axis=1).rename("MAG7 Cap-Weight Proxy")
        panel = pd.concat([equal_weight, cap_weight], axis=1)
    else:
        panel = equal_weight.to_frame()

    return info, panel


def build_event_calendar(custom_events: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Calendar placeholders. Replace with official CSV when available."""

    today = pd.Timestamp.today().normalize()

    events = [
        {
            "date": pd.Timestamp(year=2026, month=11, day=3),
            "event": "US midterm election",
            "category": "Political / Fiscal Policy",
            "risk_level": "High",
            "why_it_matters": (
                "Midterm elections can affect fiscal policy, regulation, taxation, "
                "healthcare, energy, and defense expectations."
            ),
            "source_type": "static calendar placeholder",
        },
        {
            "date": today + pd.Timedelta(days=7),
            "event": "CPI release window",
            "category": "Inflation",
            "risk_level": "High",
            "why_it_matters": (
                "Inflation surprises can shift Fed-rate expectations and valuation multiples."
            ),
            "source_type": "placeholder; replace with official BLS calendar date",
        },
        {
            "date": today + pd.Timedelta(days=14),
            "event": "Retail sales release window",
            "category": "Consumption",
            "risk_level": "Medium",
            "why_it_matters": (
                "Consumption data affects views on economic resilience, margins, and earnings risk."
            ),
            "source_type": "placeholder; replace with official Census calendar date",
        },
        {
            "date": today + pd.Timedelta(days=21),
            "event": "PCE inflation release window",
            "category": "Inflation / Fed",
            "risk_level": "High",
            "why_it_matters": (
                "PCE is a key inflation measure watched by the Federal Reserve."
            ),
            "source_type": "placeholder; replace with official BEA calendar date",
        },
        {
            "date": today + pd.Timedelta(days=30),
            "event": "FOMC decision window",
            "category": "Interest Rates",
            "risk_level": "High",
            "why_it_matters": (
                "Fed communication can reprice growth stocks, long-duration assets, and financials."
            ),
            "source_type": "placeholder; replace with official Fed calendar date",
        },
        {
            "date": today + pd.Timedelta(days=45),
            "event": "Earnings season window",
            "category": "Earnings",
            "risk_level": "Medium",
            "why_it_matters": (
                "Earnings season tests whether valuation and AI-growth expectations "
                "are supported by fundamentals."
            ),
            "source_type": "placeholder",
        },
    ]

    base = pd.DataFrame(events)

    if custom_events is not None and not custom_events.empty:
        custom = custom_events.copy()

        if "date" in custom.columns:
            custom["date"] = pd.to_datetime(custom["date"], errors="coerce")

            for col in [
                "event",
                "category",
                "risk_level",
                "why_it_matters",
                "source_type",
            ]:
                if col not in custom.columns:
                    custom[col] = ""

            base = pd.concat([base, custom[base.columns]], ignore_index=True)

    base = base.dropna(subset=["date"]).sort_values("date")
    base["days_until"] = (base["date"] - today).dt.days

    return base[base["days_until"] >= 0]


def _format_large_number(x):
    if pd.isna(x):
        return ""

    x = float(x)

    if abs(x) >= 1e12:
        return f"{x / 1e12:.2f}T"
    if abs(x) >= 1e9:
        return f"{x / 1e9:.2f}B"
    if abs(x) >= 1e6:
        return f"{x / 1e6:.2f}M"

    return f"{x:,.0f}"


def render_macro_market_regime_tab():
    st.subheader("Macro & Market Regime Dashboard")
    st.caption(
        "Top-down market context for ETF, stock, MAG7, semiconductor, gold, "
        "and international-market analysis. Educational demo only."
    )

    with st.expander("Inputs: valuation and sentiment indexes", expanded=True):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            fear_greed = st.number_input(
                "Fear & Greed Index",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=1.0,
            )

        with c2:
            buffett_indicator = st.number_input(
                "Buffett Indicator proxy: market cap / GDP",
                min_value=0.0,
                max_value=5.0,
                value=1.9,
                step=0.05,
            )

        with c3:
            shiller_cape = st.number_input(
                "Shiller CAPE / Shiller Index",
                min_value=0.0,
                max_value=80.0,
                value=34.0,
                step=0.5,
            )

        with c4:
            lookback = st.selectbox(
                "Market data lookback",
                ["6mo", "1y", "2y", "5y"],
                index=2,
            )

    custom_calendar_file = st.file_uploader(
        "Optional: upload official market calendar CSV",
        type=["csv"],
        help="Columns: date,event,category,risk_level,why_it_matters,source_type",
    )

    custom_events = None

    if custom_calendar_file is not None:
        custom_events = pd.read_csv(custom_calendar_file)

    cross_tickers = tuple(dict.fromkeys(list(CROSS_ASSET.keys()) + list(RATE_PROXIES.keys())))

    with st.spinner("Loading macro, ETF, MAG7, semiconductor, and gold data..."):
        cross_prices = _download_prices(cross_tickers, period=lookback)
        mag7_prices = _download_prices(tuple(MAG7), period=lookback)
        semi_prices = _download_prices(tuple(SEMIS_HBM), period=lookback)
        mag7_info = _download_info(tuple(MAG7))
        semi_info = _download_info(tuple(["NVDA", "AVGO", "TSM", "ASML", "AMD", "MU"]))

    regime = calculate_market_regime_score(
        cross_prices=cross_prices,
        mag7_info=mag7_info,
        fear_greed=fear_greed,
        buffett_indicator=buffett_indicator,
        shiller_cape=shiller_cape,
    )

    st.markdown("### 1. Market Regime Score")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Regime Score", f"{regime.score:.0f}/100")
    k2.metric("Regime Label", regime.label)
    k3.metric("Fear & Greed", f"{fear_greed:.0f}")
    k4.metric("Buffett / CAPE", f"{buffett_indicator:.2f} / {shiller_cape:.1f}")

    d1, d2 = st.columns(2)

    with d1:
        st.markdown("**Positive drivers**")

        if regime.drivers:
            for item in regime.drivers:
                st.write("- " + item)
        else:
            st.write("- No strong positive drivers detected.")

    with d2:
        st.markdown("**Risks / warnings**")

        if regime.warnings:
            for item in regime.warnings:
                st.write("- " + item)
        else:
            st.write("- No major warnings detected by the simple scoring model.")

    st.markdown(
        "### 2. Cross-Asset, International Market, Emerging Market, Gold, and Semiconductor Comparison"
    )

    if not cross_prices.empty:
        display_cols = [c for c in CROSS_ASSET.keys() if c in cross_prices.columns]
        norm = _normalized(cross_prices[display_cols])

        st.plotly_chart(
            px.line(norm, title="Normalized Cross-Asset Performance"),
            use_container_width=True,
        )

        perf = pd.DataFrame(
            {
                "Ticker": display_cols,
                "Description": [CROSS_ASSET.get(t, t) for t in display_cols],
                "1Y Return": _total_return(cross_prices[display_cols], 252)
                .reindex(display_cols)
                .values,
                "1Y Volatility": _volatility(cross_prices[display_cols], 252)
                .reindex(display_cols)
                .values,
            }
        )

        st.dataframe(
            perf.style.format(
                {
                    "1Y Return": "{:.2%}",
                    "1Y Volatility": "{:.2%}",
                }
            ),
            use_container_width=True,
        )
    else:
        st.warning(
            "Cross-asset price data did not load. Try again later or check internet/data-source access."
        )

    st.markdown("### 3. MAG7 Concentration Dashboard")

    mag7_table, mag7_panel = calculate_mag7_concentration(mag7_info, mag7_prices)

    if not mag7_panel.empty:
        comparison = mag7_panel.copy()

        if not cross_prices.empty:
            for t in ["SPY", "QQQ"]:
                if t in cross_prices.columns:
                    comparison[t] = _normalized(cross_prices[[t]])[t]

        st.plotly_chart(
            px.line(
                comparison,
                title="MAG7 Equal Weight / Cap-Weight Proxy vs SPY and QQQ",
            ),
            use_container_width=True,
        )

    if not mag7_table.empty:
        cols = [
            "Ticker",
            "Market Cap",
            "Weight within MAG7",
            "Trailing PE",
            "Forward PE",
            "EV / EBITDA",
            "Free Cash Flow",
            "Free Cash Flow Yield",
            "Operating Income",
            "Operating Income Yield",
            "Revenue Growth",
            "Gross Margin",
            "Operating Margin",
        ]

        cols = [c for c in cols if c in mag7_table.columns]

        st.dataframe(
            mag7_table[cols].style.format(
                {
                    "Market Cap": _format_large_number,
                    "Weight within MAG7": "{:.2%}",
                    "Trailing PE": "{:.2f}",
                    "Forward PE": "{:.2f}",
                    "EV / EBITDA": "{:.2f}",
                    "Free Cash Flow": _format_large_number,
                    "Free Cash Flow Yield": "{:.2%}",
                    "Operating Income": _format_large_number,
                    "Operating Income Yield": "{:.2%}",
                    "Revenue Growth": "{:.2%}",
                    "Gross Margin": "{:.2%}",
                    "Operating Margin": "{:.2%}",
                }
            ),
            use_container_width=True,
        )

    st.markdown("### 4. Semiconductor / SMH / HBM Supply Chain")

    if not semi_prices.empty:
        semi_cols = [c for c in SEMIS_HBM if c in semi_prices.columns]

        st.plotly_chart(
            px.line(
                _normalized(semi_prices[semi_cols]),
                title="Semiconductor and HBM Supply Chain Performance",
            ),
            use_container_width=True,
        )

        st.caption(
            "HBM chain examples: MU, Samsung Electronics, SK Hynix. ETF proxies: SMH, SOXX."
        )

    if not semi_info.empty:
        st.dataframe(semi_info, use_container_width=True)

    st.markdown("### 5. Election / Fed / Inflation / Consumption Risk Calendar")

    calendar = build_event_calendar(custom_events)
    st.dataframe(calendar, use_container_width=True)

    high_risk = calendar[calendar["risk_level"].str.lower().eq("high")].head(5)

    if not high_risk.empty:
        st.markdown("**Nearest high-risk events to monitor**")

        for _, row in high_risk.iterrows():
            st.write(f"- {row['date'].date()}: {row['event']} — {row['why_it_matters']}")

    st.markdown("### 6. Congressional Disclosure / Pelosi Holdings Monitor")
    st.caption(
        "Use this as a public-disclosure monitor only. Congressional trades are disclosed "
        "with lag and should not be treated as causal trading signals."
    )

    holdings_file = st.file_uploader(
        "Optional: upload congressional / Pelosi holdings CSV",
        type=["csv"],
        help=(
            "Suggested columns: politician,ticker,transaction_type,"
            "transaction_date,disclosure_date,amount_low,amount_high"
        ),
        key="congress_upload",
    )

    if holdings_file is not None:
        holdings = pd.read_csv(holdings_file)
        st.dataframe(holdings, use_container_width=True)

        if "ticker" in holdings.columns:
            st.write("Most frequent disclosed tickers:")

            st.dataframe(
                holdings["ticker"]
                .value_counts()
                .head(20)
                .rename_axis("Ticker")
                .reset_index(name="Count"),
                use_container_width=True,
            )
    else:
        st.info(
            "Upload a CSV export from your preferred public disclosure source to enable this monitor."
        )

    st.markdown("### 7. AI-ready interpretation prompt")

    prompt = f"""
You are an equity research copilot. Interpret the following market-regime snapshot for educational research use only.

Market regime label: {regime.label}
Regime score: {regime.score:.0f}/100
Fear & Greed Index: {fear_greed:.0f}
Buffett Indicator proxy: {buffett_indicator:.2f}
Shiller CAPE: {shiller_cape:.1f}
Positive drivers: {regime.drivers}
Warnings: {regime.warnings}

Discuss:
1. Whether the environment is risk-on, risk-off, or mixed.
2. Whether valuation stress is elevated.
3. Whether MAG7 and semiconductor concentration is a concern.
4. Whether gold, international markets, and emerging markets confirm or contradict the equity signal.
5. Which macro events deserve attention: Fed, CPI/PCE, consumption, earnings season, and midterm election risk.
6. What a student analyst should verify before making a Buy/Add/Hold/Trim/Sell recommendation.
""".strip()

    st.text_area(
        "Copy this prompt into Claude / ChatGPT for AI macro interpretation",
        prompt,
        height=300,
    )
