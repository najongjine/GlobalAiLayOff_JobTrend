from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "layoffs_events.csv"
OUT_DIR = ROOT / "outputs"


def fmt_count(value: float) -> str:
    return f"{value:,.0f}"


def normalize_industry(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "Unknown"
    if text.startswith("Transportat"):
        return "Transportation"
    return text


def normalize_ai_flag(value: object) -> bool:
    return str(value).strip().lower() == "true"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    layoffs = pd.read_csv(DATA_PATH)
    layoffs["date"] = pd.to_datetime(layoffs["date"], errors="coerce")
    layoffs["year"] = layoffs["date"].dt.year
    layoffs["layoff_count_num"] = pd.to_numeric(layoffs["layoff_count"], errors="coerce")
    layoffs["industry_clean"] = layoffs["industry"].map(normalize_industry)
    layoffs["is_ai_bool"] = layoffs["is_ai_company"].map(normalize_ai_flag)

    industry = (
        layoffs.groupby("industry_clean", as_index=False)
        .agg(
            events=("company", "size"),
            known_count_events=("layoff_count_num", "count"),
            layoff_count=("layoff_count_num", "sum"),
            avg_layoff_count=("layoff_count_num", "mean"),
            median_layoff_count=("layoff_count_num", "median"),
            companies=("company", "nunique"),
            ai_events=("is_ai_bool", "sum"),
        )
        .sort_values("events", ascending=False)
    )
    industry["event_share_pct"] = industry["events"] / industry["events"].sum() * 100
    industry["headcount_share_pct"] = (
        industry["layoff_count"] / industry["layoff_count"].sum() * 100
    )
    industry["known_count_coverage_pct"] = (
        industry["known_count_events"] / industry["events"] * 100
    )
    industry["ai_event_share_pct"] = industry["ai_events"] / industry["events"] * 100
    industry = industry.round(
        {
            "avg_layoff_count": 1,
            "median_layoff_count": 1,
            "event_share_pct": 1,
            "headcount_share_pct": 1,
            "known_count_coverage_pct": 1,
            "ai_event_share_pct": 1,
        }
    )
    industry.to_csv(OUT_DIR / "industry_layoff_concentration_summary.csv", index=False)

    top_events = industry.head(12).copy()
    top_headcount = industry.sort_values("layoff_count", ascending=False).head(12).copy()

    yearly_top_names = top_events["industry_clean"].head(8).tolist()
    yearly_top = layoffs[layoffs["industry_clean"].isin(yearly_top_names)].copy()
    yearly_counts = (
        yearly_top.groupby(["year", "industry_clean"], as_index=False)
        .size()
        .rename(columns={"size": "events"})
        .dropna(subset=["year"])
    )
    yearly_counts["year"] = yearly_counts["year"].astype(int)
    pivot_yearly = (
        yearly_counts.pivot(index="year", columns="industry_clean", values="events")
        .fillna(0)
        .sort_index()
    )

    sns.set_theme(style="whitegrid")
    event_color = "#3366A8"
    headcount_color = "#D95F43"
    ai_color = "#4B8F6A"

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    fig.suptitle("Industry Concentration of Layoffs", fontsize=18, fontweight="bold")

    ax = axes[0, 0]
    plot_data = top_events.sort_values("events", ascending=True)
    ax.barh(plot_data["industry_clean"], plot_data["events"], color=event_color)
    ax.set_title("Top Industries by Event Count", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("Layoff events")
    ax.set_ylabel("")
    for i, row in enumerate(plot_data.itertuples(index=False)):
        ax.text(row.events, i, f" {fmt_count(row.events)}", va="center", fontsize=10)

    ax = axes[0, 1]
    plot_data = top_headcount.sort_values("layoff_count", ascending=True)
    ax.barh(plot_data["industry_clean"], plot_data["layoff_count"], color=headcount_color)
    ax.set_title(
        "Top Industries by Known Headcount", loc="left", fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Known laid-off employees")
    ax.set_ylabel("")
    for i, row in enumerate(plot_data.itertuples(index=False)):
        ax.text(row.layoff_count, i, f" {fmt_count(row.layoff_count)}", va="center", fontsize=10)

    ax = axes[1, 0]
    plot_data = top_events.sort_values("ai_event_share_pct", ascending=True)
    ax.barh(plot_data["industry_clean"], plot_data["ai_event_share_pct"], color=ai_color)
    ax.set_title(
        "AI Company Share Within Top Industries",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("AI company events (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, max(35, plot_data["ai_event_share_pct"].max() + 5))
    for i, row in enumerate(plot_data.itertuples(index=False)):
        ax.text(
            row.ai_event_share_pct,
            i,
            f" {row.ai_event_share_pct:.1f}%",
            va="center",
            fontsize=10,
        )

    ax = axes[1, 1]
    sns.heatmap(
        pivot_yearly.T,
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".0f",
        cbar_kws={"label": "Event count"},
        ax=ax,
    )
    ax.set_title(
        "Yearly Pattern for Top Event Industries",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("")

    fig.text(
        0.01,
        0.01,
        "Note: Headcount totals include only rows with reported layoff_count. "
        "Transportation labels were normalized from truncated source values.",
        fontsize=10,
        color="#555555",
    )
    fig.savefig(OUT_DIR / "industry_layoff_concentration.png", dpi=180)

    fig2, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    scatter_data = industry[industry["events"] >= 10].copy()
    sizes = (scatter_data["companies"] / scatter_data["companies"].max() * 900) + 80
    ax.scatter(
        scatter_data["events"],
        scatter_data["layoff_count"],
        s=sizes,
        c=scatter_data["ai_event_share_pct"],
        cmap="viridis",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.8,
    )
    for _, row in scatter_data.sort_values("events", ascending=False).head(12).iterrows():
        ax.text(
            row["events"],
            row["layoff_count"],
            f" {row['industry_clean']}",
            fontsize=9,
            va="center",
        )
    ax.set_title(
        "Industry Footprint: Frequency vs Known Headcount",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Layoff events")
    ax.set_ylabel("Known laid-off employees")
    cbar = fig2.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("AI company share of events (%)")
    fig2.text(
        0.01,
        0.01,
        "Bubble size reflects number of unique companies. Industries with at least 10 events are shown.",
        fontsize=10,
        color="#555555",
    )
    fig2.savefig(OUT_DIR / "industry_layoff_footprint.png", dpi=180)

    print("Top industries by events")
    print(
        industry[
            [
                "industry_clean",
                "events",
                "event_share_pct",
                "layoff_count",
                "headcount_share_pct",
                "companies",
                "ai_event_share_pct",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )
    print(f"\nSaved: {OUT_DIR / 'industry_layoff_concentration.png'}")
    print(f"Saved: {OUT_DIR / 'industry_layoff_footprint.png'}")
    print(f"Saved: {OUT_DIR / 'industry_layoff_concentration_summary.csv'}")


if __name__ == "__main__":
    main()
