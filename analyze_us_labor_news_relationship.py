from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
LAYOFFS_PATH = ROOT / "layoffs_events.csv"
US_LABOR_PATH = ROOT / "us_labor_indicators.csv"
NEWS_PATH = ROOT / "news_sentiment.csv"
OUT_DIR = ROOT / "outputs"


def fmt_count(value: float) -> str:
    return f"{value:,.0f}"


def normalize_country(value: object) -> str:
    text = str(value).strip()
    if text in {"US", "USA", "United States"}:
        return "United States"
    return text


def monthly_corr(df: pd.DataFrame, metric: str) -> dict[str, float]:
    base = df[["layoff_events", metric]].dropna()
    if len(base) < 3:
        return {"same_month": float("nan"), "metric_leads_1m": float("nan"), "layoffs_lead_1m": float("nan")}
    return {
        "same_month": base["layoff_events"].corr(base[metric]),
        "metric_leads_1m": df["layoff_events"].corr(df[metric].shift(1)),
        "layoffs_lead_1m": df["layoff_events"].corr(df[metric].shift(-1)),
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    layoffs = pd.read_csv(LAYOFFS_PATH)
    us_labor = pd.read_csv(US_LABOR_PATH)
    news = pd.read_csv(NEWS_PATH)

    layoffs["date"] = pd.to_datetime(layoffs["date"], errors="coerce")
    layoffs["month"] = layoffs["date"].dt.to_period("M").dt.to_timestamp()
    layoffs["country_clean"] = layoffs["country"].map(normalize_country)
    layoffs["layoff_count_num"] = pd.to_numeric(layoffs["layoff_count"], errors="coerce")

    us_layoffs = layoffs[layoffs["country_clean"] == "United States"].copy()
    monthly_layoffs = (
        us_layoffs.dropna(subset=["month"])
        .groupby("month", as_index=False)
        .agg(
            layoff_events=("company", "size"),
            known_layoff_count=("layoff_count_num", "sum"),
            known_count_events=("layoff_count_num", "count"),
            companies=("company", "nunique"),
        )
    )

    us_labor["date"] = pd.to_datetime(us_labor["date"], errors="coerce")
    numeric_cols = [
        "unemployment_rate",
        "jolts_job_openings_k",
        "initial_jobless_claims_k",
        "information_sector_emp_k",
        "computer_math_emp_k",
        "financial_emp_k",
        "openings_per_unemployed",
        "tech_emp_yoy_pct",
        "claims_4w_avg",
    ]
    for col in numeric_cols:
        us_labor[col] = pd.to_numeric(us_labor[col], errors="coerce")

    monthly_labor = (
        us_labor.set_index("date")[numeric_cols]
        .resample("MS")
        .mean()
        .reset_index()
        .rename(columns={"date": "month"})
    )

    news["date"] = pd.to_datetime(news["date"], errors="coerce")
    news["month"] = news["date"].dt.to_period("M").dt.to_timestamp()
    news["sentiment"] = pd.to_numeric(news["sentiment"], errors="coerce")
    monthly_news = (
        news.dropna(subset=["month"])
        .groupby("month", as_index=False)
        .agg(
            news_count=("title", "size"),
            avg_sentiment=("sentiment", "mean"),
            negative_news=("sentiment_cat", lambda s: (s == "negative").sum()),
            positive_news=("sentiment_cat", lambda s: (s == "positive").sum()),
            layoff_news=("is_layoff_news", lambda s: (s.astype(str).str.lower() == "true").sum()),
            hiring_news=("is_hiring_news", lambda s: (s.astype(str).str.lower() == "true").sum()),
        )
    )
    monthly_news["negative_share_pct"] = monthly_news["negative_news"] / monthly_news["news_count"] * 100

    calendar = pd.DataFrame(
        {
            "month": pd.date_range(
                min(monthly_layoffs["month"].min(), monthly_labor["month"].min()),
                max(monthly_layoffs["month"].max(), monthly_labor["month"].max()),
                freq="MS",
            )
        }
    )
    merged = (
        calendar.merge(monthly_layoffs, on="month", how="left")
        .merge(monthly_labor, on="month", how="left")
        .merge(monthly_news, on="month", how="left")
    )
    fill_zero_cols = [
        "layoff_events",
        "known_layoff_count",
        "known_count_events",
        "companies",
        "news_count",
        "negative_news",
        "positive_news",
        "layoff_news",
        "hiring_news",
    ]
    merged[fill_zero_cols] = merged[fill_zero_cols].fillna(0)
    merged["negative_share_pct"] = merged["negative_share_pct"].fillna(0)

    merged["layoff_events_3m_avg"] = merged["layoff_events"].rolling(3, min_periods=1).mean()
    merged["known_layoff_count_3m_avg"] = merged["known_layoff_count"].rolling(3, min_periods=1).mean()
    merged["claims_4w_avg_k"] = merged["claims_4w_avg"] / 1000
    merged["initial_jobless_claims_k_scaled"] = merged["initial_jobless_claims_k"] / 1000

    merged.to_csv(OUT_DIR / "us_labor_news_monthly_merged.csv", index=False)

    corr_metrics = [
        "initial_jobless_claims_k",
        "claims_4w_avg",
        "jolts_job_openings_k",
        "openings_per_unemployed",
        "information_sector_emp_k",
        "computer_math_emp_k",
        "tech_emp_yoy_pct",
        "avg_sentiment",
        "negative_share_pct",
        "news_count",
    ]
    corr_rows = []
    for metric in corr_metrics:
        values = monthly_corr(merged, metric)
        for relation, corr in values.items():
            corr_rows.append(
                {
                    "metric": metric,
                    "relation": relation,
                    "correlation_with_layoff_events": corr,
                }
            )
    corr_df = pd.DataFrame(corr_rows).round(3)
    corr_df.to_csv(OUT_DIR / "us_layoff_labor_news_correlations.csv", index=False)

    same_month_corr = (
        merged[
            [
                "layoff_events",
                "known_layoff_count",
                "initial_jobless_claims_k",
                "claims_4w_avg",
                "jolts_job_openings_k",
                "openings_per_unemployed",
                "information_sector_emp_k",
                "computer_math_emp_k",
                "tech_emp_yoy_pct",
                "avg_sentiment",
                "negative_share_pct",
                "news_count",
            ]
        ]
        .corr()
        .round(2)
    )

    sns.set_theme(style="whitegrid")
    blue = "#3366A8"
    orange = "#D95F43"
    green = "#4B8F6A"
    purple = "#8A5FBF"

    fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True, constrained_layout=True)
    fig.suptitle("US Layoffs vs Labor Market Indicators", fontsize=18, fontweight="bold")

    ax = axes[0]
    ax.bar(merged["month"], merged["layoff_events"], width=24, color=blue, alpha=0.65, label="US layoff events")
    ax.plot(merged["month"], merged["layoff_events_3m_avg"], color="#1F3F6F", linewidth=2.5, label="3-month avg")
    ax.set_title("US Layoff Event Volume", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("Events")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.plot(merged["month"], merged["claims_4w_avg_k"], color=orange, linewidth=2.2, label="Jobless claims 4w avg, thousands")
    ax2 = ax.twinx()
    ax2.plot(merged["month"], merged["unemployment_rate"], color=purple, linewidth=2.0, label="Unemployment rate")
    ax.set_title("Jobless Claims and Unemployment", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("Claims, thousands")
    ax2.set_ylabel("Unemployment rate (%)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax = axes[2]
    ax.plot(merged["month"], merged["jolts_job_openings_k"], color=green, linewidth=2.2, label="JOLTS openings, thousands")
    ax2 = ax.twinx()
    ax2.plot(merged["month"], merged["openings_per_unemployed"], color=purple, linewidth=2.0, label="Openings per unemployed")
    ax.set_title("Labor Demand: JOLTS Openings", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("Openings, thousands")
    ax2.set_ylabel("Openings per unemployed")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax = axes[3]
    ax.plot(merged["month"], merged["information_sector_emp_k"], color=blue, linewidth=2.0, label="Information sector employment")
    ax.plot(merged["month"], merged["computer_math_emp_k"], color=green, linewidth=2.0, label="Computer/math employment")
    ax2 = ax.twinx()
    ax2.plot(merged["month"], merged["tech_emp_yoy_pct"], color=orange, linewidth=2.0, label="Tech employment YoY %")
    ax.set_title("Tech-Adjacent Employment Indicators", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("Employment, thousands")
    ax2.set_ylabel("YoY growth (%)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.set_xlabel("Month")

    fig.text(
        0.01,
        0.01,
        "Note: US layoffs are monthly counts from layoffs_events.csv. Labor indicators are monthly averages from us_labor_indicators.csv.",
        fontsize=10,
        color="#555555",
    )
    fig.savefig(OUT_DIR / "us_layoffs_labor_market_relationship.png", dpi=180)

    fig2, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True, constrained_layout=True)
    fig2.suptitle("News Sentiment vs US Layoff Events", fontsize=18, fontweight="bold")

    ax = axes[0]
    ax.bar(merged["month"], merged["layoff_events"], width=24, color=blue, alpha=0.65, label="US layoff events")
    ax.plot(merged["month"], merged["layoff_events_3m_avg"], color="#1F3F6F", linewidth=2.5, label="3-month avg")
    ax.set_title("US Layoff Events", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("Events")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.bar(merged["month"], merged["negative_news"], width=24, color=orange, alpha=0.65, label="Negative news count")
    ax.plot(merged["month"], merged["negative_share_pct"], color=purple, linewidth=2.2, label="Negative share %")
    ax.set_title("Negative News Volume and Share", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count / share")
    ax.legend(loc="upper left")

    ax = axes[2]
    ax.plot(merged["month"], merged["avg_sentiment"], color=green, linewidth=2.2, marker="o", markersize=3, label="Average sentiment")
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_title("Average News Sentiment", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sentiment score")
    ax.set_xlabel("Month")
    ax.legend(loc="upper left")

    fig2.text(
        0.01,
        0.01,
        "Note: News archive starts later than labor and layoff data, so early months have no sentiment observations.",
        fontsize=10,
        color="#555555",
    )
    fig2.savefig(OUT_DIR / "news_sentiment_vs_us_layoffs.png", dpi=180)

    fig3, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    ax = axes[0]
    sns.heatmap(
        same_month_corr,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Same-Month Correlation Matrix", fontsize=13, fontweight="bold")

    ax = axes[1]
    pivot_corr = corr_df.pivot(index="metric", columns="relation", values="correlation_with_layoff_events")
    sns.heatmap(
        pivot_corr,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Lagged Correlations with Layoff Events", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig3.savefig(OUT_DIR / "us_layoff_labor_news_correlations.png", dpi=180)

    print("Monthly merged range")
    print(f"{merged['month'].min().date()} to {merged['month'].max().date()}")
    print("\nSelected lagged correlations")
    selected = corr_df[corr_df["metric"].isin(
        [
            "initial_jobless_claims_k",
            "claims_4w_avg",
            "jolts_job_openings_k",
            "tech_emp_yoy_pct",
            "avg_sentiment",
            "negative_share_pct",
        ]
    )]
    print(selected.to_string(index=False))
    print(f"\nSaved: {OUT_DIR / 'us_layoffs_labor_market_relationship.png'}")
    print(f"Saved: {OUT_DIR / 'news_sentiment_vs_us_layoffs.png'}")
    print(f"Saved: {OUT_DIR / 'us_layoff_labor_news_correlations.png'}")
    print(f"Saved: {OUT_DIR / 'us_labor_news_monthly_merged.csv'}")
    print(f"Saved: {OUT_DIR / 'us_layoff_labor_news_correlations.csv'}")


if __name__ == "__main__":
    main()
