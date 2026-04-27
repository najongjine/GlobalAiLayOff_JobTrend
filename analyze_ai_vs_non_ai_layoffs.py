from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "layoffs_events.csv"
OUT_DIR = ROOT / "outputs"


def fmt_count(value: float) -> str:
    return f"{value:,.0f}"


def normalize_ai_flag(value: object) -> str:
    return "AI company" if str(value).strip().lower() == "true" else "Non-AI company"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    layoffs = pd.read_csv(DATA_PATH)
    layoffs["date"] = pd.to_datetime(layoffs["date"], errors="coerce")
    layoffs["year"] = layoffs["date"].dt.year
    layoffs["layoff_count_num"] = pd.to_numeric(layoffs["layoff_count"], errors="coerce")
    layoffs["company_type"] = layoffs["is_ai_company"].map(normalize_ai_flag)

    clean = layoffs.dropna(subset=["year"]).copy()
    clean["year"] = clean["year"].astype(int)

    yearly = (
        clean.groupby(["year", "company_type"], as_index=False)
        .agg(
            events=("company", "size"),
            known_count_events=("layoff_count_num", "count"),
            layoff_count=("layoff_count_num", "sum"),
            avg_layoff_count=("layoff_count_num", "mean"),
            median_layoff_count=("layoff_count_num", "median"),
            companies=("company", "nunique"),
        )
        .sort_values(["year", "company_type"])
    )

    totals = yearly.groupby("year", as_index=False).agg(
        total_events=("events", "sum"),
        total_layoff_count=("layoff_count", "sum"),
    )
    yearly = yearly.merge(totals, on="year", how="left")
    yearly["event_share_pct"] = yearly["events"] / yearly["total_events"] * 100
    yearly["headcount_share_pct"] = (
        yearly["layoff_count"] / yearly["total_layoff_count"] * 100
    )
    yearly["known_count_coverage_pct"] = (
        yearly["known_count_events"] / yearly["events"] * 100
    )
    yearly = yearly.round(
        {
            "event_share_pct": 1,
            "headcount_share_pct": 1,
            "known_count_coverage_pct": 1,
            "avg_layoff_count": 1,
            "median_layoff_count": 1,
        }
    )
    yearly.to_csv(OUT_DIR / "ai_vs_non_ai_yearly_summary.csv", index=False)

    overall = (
        clean.groupby("company_type", as_index=False)
        .agg(
            events=("company", "size"),
            known_count_events=("layoff_count_num", "count"),
            layoff_count=("layoff_count_num", "sum"),
            avg_layoff_count=("layoff_count_num", "mean"),
            median_layoff_count=("layoff_count_num", "median"),
            companies=("company", "nunique"),
        )
        .sort_values("events", ascending=False)
    )
    overall["event_share_pct"] = overall["events"] / overall["events"].sum() * 100
    overall["headcount_share_pct"] = (
        overall["layoff_count"] / overall["layoff_count"].sum() * 100
    )
    overall = overall.round(
        {
            "avg_layoff_count": 1,
            "median_layoff_count": 1,
            "event_share_pct": 1,
            "headcount_share_pct": 1,
        }
    )
    overall.to_csv(OUT_DIR / "ai_vs_non_ai_overall_summary.csv", index=False)

    sns.set_theme(style="whitegrid")
    colors = {"AI company": "#D95F43", "Non-AI company": "#3366A8"}
    order = ["Non-AI company", "AI company"]

    pivot_events = (
        yearly.pivot(index="year", columns="company_type", values="events")
        .reindex(columns=order)
        .fillna(0)
    )
    pivot_headcount = (
        yearly.pivot(index="year", columns="company_type", values="layoff_count")
        .reindex(columns=order)
        .fillna(0)
    )
    ai_share = yearly[yearly["company_type"] == "AI company"].set_index("year")

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), constrained_layout=True)
    fig.suptitle("AI vs Non-AI Company Layoffs", fontsize=18, fontweight="bold")

    ax = axes[0]
    bottom = pd.Series(0, index=pivot_events.index)
    for label in order:
        bars = ax.bar(
            pivot_events.index.astype(str),
            pivot_events[label],
            bottom=bottom,
            label=label,
            color=colors[label],
        )
        bottom += pivot_events[label]
    ax.set_title("Yearly Layoff Events", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("Event count")
    ax.legend(loc="upper left")
    for year, total in bottom.items():
        ax.text(str(year), total, fmt_count(total), ha="center", va="bottom", fontsize=10)

    ax = axes[1]
    ax.plot(
        ai_share.index.astype(str),
        ai_share["event_share_pct"],
        marker="o",
        linewidth=2.5,
        color=colors["AI company"],
        label="AI share of events",
    )
    ax.plot(
        ai_share.index.astype(str),
        ai_share["headcount_share_pct"],
        marker="o",
        linewidth=2.5,
        color="#8A5FBF",
        label="AI share of known headcount",
    )
    ax.set_title("AI Company Share by Year", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("Share (%)")
    ax.set_ylim(0, max(35, ai_share[["event_share_pct", "headcount_share_pct"]].max().max() + 5))
    ax.legend(loc="upper left")
    for year, row in ai_share.iterrows():
        ax.text(
            str(year),
            row["event_share_pct"] + 1,
            f"{row['event_share_pct']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color=colors["AI company"],
        )

    ax = axes[2]
    bottom = pd.Series(0, index=pivot_headcount.index)
    for label in order:
        ax.bar(
            pivot_headcount.index.astype(str),
            pivot_headcount[label],
            bottom=bottom,
            label=label,
            color=colors[label],
        )
        bottom += pivot_headcount[label]
    ax.set_title(
        "Known Layoff Headcount by Year", loc="left", fontsize=13, fontweight="bold"
    )
    ax.set_ylabel("Known laid-off employees")
    ax.legend(loc="upper left")
    for year, total in bottom.items():
        ax.text(str(year), total, fmt_count(total), ha="center", va="bottom", fontsize=10)

    fig.text(
        0.01,
        0.01,
        "Note: 2026 is partial-year data through 2026-04-16. "
        "Headcount totals include only rows with reported layoff_count.",
        fontsize=10,
        color="#555555",
    )
    fig.savefig(OUT_DIR / "ai_vs_non_ai_layoff_trends.png", dpi=180)

    fig2, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    ax = axes[0]
    sns.barplot(
        data=overall,
        x="company_type",
        y="events",
        hue="company_type",
        palette=colors,
        legend=False,
        ax=ax,
    )
    ax.set_title("Total Events", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Event count")
    for patch in ax.patches:
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height(),
            fmt_count(patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax = axes[1]
    sns.barplot(
        data=overall,
        x="company_type",
        y="layoff_count",
        hue="company_type",
        palette=colors,
        legend=False,
        ax=ax,
    )
    ax.set_title("Known Headcount", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Known laid-off employees")
    for patch in ax.patches:
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height(),
            fmt_count(patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=11,
        )

    fig2.suptitle("Overall AI vs Non-AI Layoff Comparison", fontsize=16, fontweight="bold")
    fig2.savefig(OUT_DIR / "ai_vs_non_ai_overall.png", dpi=180)

    print("Overall")
    print(overall.to_string(index=False))
    print("\nYearly")
    print(yearly.to_string(index=False))
    print(f"\nSaved: {OUT_DIR / 'ai_vs_non_ai_layoff_trends.png'}")
    print(f"Saved: {OUT_DIR / 'ai_vs_non_ai_overall.png'}")
    print(f"Saved: {OUT_DIR / 'ai_vs_non_ai_yearly_summary.csv'}")
    print(f"Saved: {OUT_DIR / 'ai_vs_non_ai_overall_summary.csv'}")


if __name__ == "__main__":
    main()
