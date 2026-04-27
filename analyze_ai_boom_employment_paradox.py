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


def is_ai_company(value: object) -> bool:
    return str(value).strip().lower() == "true"


def normalize_country(value: object) -> str:
    text = str(value).strip()
    if text in {"US", "USA", "United States"}:
        return "United States"
    return text


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    layoffs = pd.read_csv(LAYOFFS_PATH)
    labor = pd.read_csv(US_LABOR_PATH)
    news = pd.read_csv(NEWS_PATH)

    layoffs["date"] = pd.to_datetime(layoffs["date"], errors="coerce")
    layoffs["year"] = layoffs["date"].dt.year
    layoffs["month"] = layoffs["date"].dt.to_period("M").dt.to_timestamp()
    layoffs["country_clean"] = layoffs["country"].map(normalize_country)
    layoffs["layoff_count_num"] = pd.to_numeric(layoffs["layoff_count"], errors="coerce")
    layoffs["ai_company"] = layoffs["is_ai_company"].map(is_ai_company)
    layoffs["company_type"] = layoffs["ai_company"].map(
        {True: "AI company", False: "Non-AI company"}
    )

    yearly = (
        layoffs.dropna(subset=["year"])
        .groupby(["year", "company_type"], as_index=False)
        .agg(
            events=("company", "size"),
            known_layoff_count=("layoff_count_num", "sum"),
            known_count_events=("layoff_count_num", "count"),
            companies=("company", "nunique"),
            avg_layoff_count=("layoff_count_num", "mean"),
            median_layoff_count=("layoff_count_num", "median"),
        )
    )
    yearly["year"] = yearly["year"].astype(int)
    totals = yearly.groupby("year", as_index=False).agg(
        total_events=("events", "sum"),
        total_known_layoff_count=("known_layoff_count", "sum"),
    )
    yearly = yearly.merge(totals, on="year", how="left")
    yearly["event_share_pct"] = yearly["events"] / yearly["total_events"] * 100
    yearly["headcount_share_pct"] = (
        yearly["known_layoff_count"] / yearly["total_known_layoff_count"] * 100
    )

    ai_yearly = yearly[yearly["company_type"] == "AI company"].copy()

    labor["date"] = pd.to_datetime(labor["date"], errors="coerce")
    labor_cols = [
        "unemployment_rate",
        "jolts_job_openings_k",
        "information_sector_emp_k",
        "computer_math_emp_k",
        "tech_emp_yoy_pct",
        "claims_4w_avg",
    ]
    for col in labor_cols:
        labor[col] = pd.to_numeric(labor[col], errors="coerce")
    monthly_labor = (
        labor.set_index("date")[labor_cols]
        .resample("MS")
        .mean()
        .reset_index()
        .rename(columns={"date": "month"})
    )

    us_layoffs = layoffs[layoffs["country_clean"] == "United States"].copy()
    monthly_us_layoffs = (
        us_layoffs.groupby(["month", "company_type"], as_index=False)
        .agg(
            events=("company", "size"),
            known_layoff_count=("layoff_count_num", "sum"),
        )
        .dropna(subset=["month"])
    )
    monthly_us_total = (
        us_layoffs.groupby("month", as_index=False)
        .agg(us_layoff_events=("company", "size"), us_known_layoff_count=("layoff_count_num", "sum"))
        .dropna(subset=["month"])
    )
    monthly_us_ai = (
        monthly_us_layoffs[monthly_us_layoffs["company_type"] == "AI company"]
        .rename(columns={"events": "us_ai_layoff_events", "known_layoff_count": "us_ai_known_layoff_count"})
        [["month", "us_ai_layoff_events", "us_ai_known_layoff_count"]]
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
            layoff_news=("is_layoff_news", lambda s: (s.astype(str).str.lower() == "true").sum()),
            hiring_news=("is_hiring_news", lambda s: (s.astype(str).str.lower() == "true").sum()),
        )
    )
    monthly_news["negative_share_pct"] = monthly_news["negative_news"] / monthly_news["news_count"] * 100

    calendar = pd.DataFrame(
        {
            "month": pd.date_range(
                min(layoffs["month"].min(), monthly_labor["month"].min()),
                max(layoffs["month"].max(), monthly_labor["month"].max()),
                freq="MS",
            )
        }
    )
    monthly = (
        calendar.merge(monthly_us_total, on="month", how="left")
        .merge(monthly_us_ai, on="month", how="left")
        .merge(monthly_labor, on="month", how="left")
        .merge(monthly_news, on="month", how="left")
    )
    zero_cols = [
        "us_layoff_events",
        "us_known_layoff_count",
        "us_ai_layoff_events",
        "us_ai_known_layoff_count",
        "news_count",
        "negative_news",
        "layoff_news",
        "hiring_news",
    ]
    monthly[zero_cols] = monthly[zero_cols].fillna(0)
    monthly["negative_share_pct"] = monthly["negative_share_pct"].fillna(0)
    monthly["us_ai_event_share_pct"] = (
        monthly["us_ai_layoff_events"] / monthly["us_layoff_events"].replace(0, pd.NA) * 100
    ).fillna(0)
    monthly["us_layoff_events_3m_avg"] = monthly["us_layoff_events"].rolling(3, min_periods=1).mean()
    monthly["us_ai_event_share_3m_avg"] = monthly["us_ai_event_share_pct"].rolling(3, min_periods=1).mean()

    yearly.to_csv(OUT_DIR / "ai_boom_yearly_ai_vs_non_ai.csv", index=False)
    monthly.to_csv(OUT_DIR / "ai_boom_monthly_us_context.csv", index=False)

    sns.set_theme(style="whitegrid")
    blue = "#3366A8"
    orange = "#D95F43"
    green = "#4B8F6A"
    purple = "#8A5FBF"

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    fig.suptitle(
        "Did the AI Boom Create Jobs, or Change the Face of Layoffs?",
        fontsize=18,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.bar(ai_yearly["year"].astype(str), ai_yearly["event_share_pct"], color=orange)
    ax.set_title("AI Companies' Share of Layoff Events", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("Share of all layoff events (%)")
    for _, row in ai_yearly.iterrows():
        ax.text(
            str(row["year"]),
            row["event_share_pct"],
            f"{row['event_share_pct']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax = axes[0, 1]
    ax.bar(ai_yearly["year"].astype(str), ai_yearly["headcount_share_pct"], color=purple)
    ax.set_title("AI Companies' Share of Known Layoff Headcount", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("Share of known laid-off employees (%)")
    for _, row in ai_yearly.iterrows():
        ax.text(
            str(row["year"]),
            row["headcount_share_pct"],
            f"{row['headcount_share_pct']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax = axes[1, 0]
    ax.plot(monthly["month"], monthly["us_layoff_events_3m_avg"], color=blue, linewidth=2.3, label="US layoff events, 3m avg")
    ax2 = ax.twinx()
    ax2.plot(monthly["month"], monthly["tech_emp_yoy_pct"], color=orange, linewidth=2.1, label="Tech employment YoY %")
    ax.axvspan(pd.Timestamp("2022-11-01"), pd.Timestamp("2024-12-01"), color="#DDDDDD", alpha=0.3)
    ax.set_title("US Layoff Pressure vs Tech Employment Growth", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("Layoff events")
    ax2.set_ylabel("Tech employment YoY (%)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax = axes[1, 1]
    ax.plot(monthly["month"], monthly["us_ai_event_share_3m_avg"], color=purple, linewidth=2.2, label="AI share of US layoff events, 3m avg")
    ax2 = ax.twinx()
    ax2.plot(monthly["month"], monthly["avg_sentiment"], color=green, linewidth=1.9, marker="o", markersize=3, label="Avg news sentiment")
    ax2.axhline(0, color="#555555", linewidth=1)
    ax.set_title("AI Layoff Share vs News Mood", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("AI share of US layoff events (%)")
    ax2.set_ylabel("Average sentiment")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    fig.text(
        0.01,
        0.01,
        "Note: The dataset contains layoffs, labor indicators, and news sentiment, not direct AI hiring postings. "
        "So this analysis tests whether AI-era layoff pressure rose while tech employment growth weakened.",
        fontsize=10,
        color="#555555",
    )
    fig.savefig(OUT_DIR / "ai_boom_employment_paradox_dashboard.png", dpi=180)

    fig2, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    scatter = monthly.dropna(subset=["tech_emp_yoy_pct"]).copy()
    scatter = scatter[scatter["us_layoff_events"] > 0]
    scatter["us_ai_event_share_pct"] = pd.to_numeric(
        scatter["us_ai_event_share_pct"], errors="coerce"
    ).fillna(0)
    bubble_size = scatter["us_ai_event_share_pct"].astype(float) * 12 + 60
    sc = ax.scatter(
        scatter["tech_emp_yoy_pct"],
        scatter["us_layoff_events"],
        s=bubble_size,
        c=scatter["us_ai_event_share_pct"],
        cmap="plasma",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set_title("Layoff Intensity Rises as Tech Employment Growth Weakens", fontsize=15, fontweight="bold")
    ax.set_xlabel("Tech employment YoY growth (%)")
    ax.set_ylabel("US layoff events per month")
    cbar = fig2.colorbar(sc, ax=ax)
    cbar.set_label("AI share of US layoff events (%)")
    fig2.text(
        0.01,
        0.01,
        "Bubble size also reflects AI share of US layoff events in the month.",
        fontsize=10,
        color="#555555",
    )
    fig2.savefig(OUT_DIR / "ai_boom_layoffs_vs_tech_growth.png", dpi=180)

    corr = monthly[
        [
            "us_layoff_events",
            "us_ai_layoff_events",
            "us_ai_event_share_pct",
            "tech_emp_yoy_pct",
            "jolts_job_openings_k",
            "information_sector_emp_k",
            "computer_math_emp_k",
            "avg_sentiment",
            "negative_share_pct",
        ]
    ].corr().round(3)
    corr.to_csv(OUT_DIR / "ai_boom_signal_correlations.csv")

    report_path = OUT_DIR / "ai_boom_employment_paradox_report.md"
    latest_month = monthly["month"].max().date()
    ai_overall = yearly[yearly["company_type"] == "AI company"]
    non_ai_overall = yearly[yearly["company_type"] == "Non-AI company"]
    ai_events = int(ai_overall["events"].sum())
    non_ai_events = int(non_ai_overall["events"].sum())
    ai_headcount = int(ai_overall["known_layoff_count"].sum())
    non_ai_headcount = int(non_ai_overall["known_layoff_count"].sum())
    ai_event_share = ai_events / (ai_events + non_ai_events) * 100
    ai_headcount_share = ai_headcount / (ai_headcount + non_ai_headcount) * 100
    peak_ai_event_share = ai_yearly.loc[ai_yearly["event_share_pct"].idxmax()]
    peak_ai_headcount_share = ai_yearly.loc[ai_yearly["headcount_share_pct"].idxmax()]
    tech_corr = monthly["us_layoff_events"].corr(monthly["tech_emp_yoy_pct"])

    report = f"""# AI Boom Employment Paradox

## Question

Did the AI boom create jobs, or did it change the face of layoffs?

## Dataset Boundary

This dataset does not contain direct AI hiring postings or job creation counts. It contains layoff events, US labor indicators, global labor indicators, and news sentiment. So the cleanest answer is about layoff pressure and tech labor-market cooling, not net AI job creation.

## Main Findings

- AI companies account for {ai_events:,} layoff events versus {non_ai_events:,} non-AI company events.
- AI companies are {ai_event_share:.1f}% of layoff events but {ai_headcount_share:.1f}% of known layoff headcount.
- AI company event share peaks in {int(peak_ai_event_share['year'])} at {peak_ai_event_share['event_share_pct']:.1f}%.
- AI company known-headcount share peaks in {int(peak_ai_headcount_share['year'])} at {peak_ai_headcount_share['headcount_share_pct']:.1f}%.
- US layoff events and tech employment YoY growth have a same-month correlation of {tech_corr:.3f}.

## Interpretation

The data does not prove that AI reduced total employment. But it does show that the AI era coincides with a shift in layoff composition: AI companies became a larger share of layoff events after 2024, and their share of known layoff headcount is much larger than their event share.

The strongest signal is not broad unemployment. It is tech-specific cooling. As US layoff pressure rises, tech employment YoY weakens. That fits a story where AI investment and automation hype did not simply create a smooth hiring boom. It also came with restructuring, concentration, and larger layoff events.

## Best One-Line Answer

The AI boom did not show up here as a simple jobs boom; in this dataset, it looks more like a labor-market reshuffle where AI companies became a more visible and sometimes larger-scale part of the layoff story.

Generated through {latest_month}.
"""
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"Saved: {OUT_DIR / 'ai_boom_employment_paradox_dashboard.png'}")
    print(f"Saved: {OUT_DIR / 'ai_boom_layoffs_vs_tech_growth.png'}")
    print(f"Saved: {OUT_DIR / 'ai_boom_yearly_ai_vs_non_ai.csv'}")
    print(f"Saved: {OUT_DIR / 'ai_boom_monthly_us_context.csv'}")
    print(f"Saved: {OUT_DIR / 'ai_boom_signal_correlations.csv'}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
