from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "layoffs_events.csv"
OUT_DIR = ROOT / "outputs"


def fmt_count(value: float) -> str:
    return f"{value:,.0f}"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    layoffs = pd.read_csv(DATA_PATH)
    layoffs["date"] = pd.to_datetime(layoffs["date"], errors="coerce")
    layoffs["year"] = layoffs["date"].dt.year
    layoffs["layoff_count_num"] = pd.to_numeric(layoffs["layoff_count"], errors="coerce")

    yearly = (
        layoffs.dropna(subset=["year"])
        .groupby("year", as_index=False)
        .agg(
            events=("company", "size"),
            known_count_events=("layoff_count_num", "count"),
            layoff_count=("layoff_count_num", "sum"),
            avg_layoff_count=("layoff_count_num", "mean"),
            median_layoff_count=("layoff_count_num", "median"),
            companies=("company", "nunique"),
        )
        .sort_values("year")
    )
    yearly["year"] = yearly["year"].astype(int)
    yearly["known_count_coverage_pct"] = (
        yearly["known_count_events"] / yearly["events"] * 100
    ).round(1)

    yearly.to_csv(OUT_DIR / "yearly_layoff_trend_summary.csv", index=False)

    sns.set_theme(style="whitegrid")
    palette_events = "#3366A8"
    palette_people = "#D95F43"
    palette_coverage = "#4B8F6A"

    fig, axes = plt.subplots(3, 1, figsize=(12, 13), constrained_layout=True)
    fig.suptitle(
        "Yearly Layoff Trend: Events and Known Headcount",
        fontsize=18,
        fontweight="bold",
    )

    ax = axes[0]
    bars = ax.bar(yearly["year"].astype(str), yearly["events"], color=palette_events)
    ax.set_title("Layoff Events by Year", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("Event count")
    ax.set_xlabel("")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt_count(bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.annotate(
        "Peak event count",
        xy=(3, yearly.loc[yearly["year"] == 2023, "events"].iloc[0]),
        xytext=(2.25, yearly["events"].max() * 1.12),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        fontsize=10,
    )

    ax = axes[1]
    bars = ax.bar(
        yearly["year"].astype(str), yearly["layoff_count"], color=palette_people
    )
    ax.set_title(
        "Known Layoff Headcount by Year", loc="left", fontsize=13, fontweight="bold"
    )
    ax.set_ylabel("Known laid-off employees")
    ax.set_xlabel("")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt_count(bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.annotate(
        "Largest known headcount impact",
        xy=(3, yearly.loc[yearly["year"] == 2023, "layoff_count"].iloc[0]),
        xytext=(1.85, yearly["layoff_count"].max() * 1.12),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        fontsize=10,
    )

    ax = axes[2]
    ax.plot(
        yearly["year"].astype(str),
        yearly["known_count_coverage_pct"],
        marker="o",
        linewidth=2.5,
        color=palette_coverage,
    )
    ax.set_title(
        "How Complete Are Headcount Totals?",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("Events with known layoff count (%)")
    ax.set_xlabel("Year")
    ax.set_ylim(0, 100)
    for _, row in yearly.iterrows():
        ax.text(
            str(int(row["year"])),
            row["known_count_coverage_pct"] + 2,
            f"{row['known_count_coverage_pct']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.text(
        0.01,
        0.01,
        "Note: 2026 is partial-year data through 2026-04-16. "
        "Headcount totals use only events with a reported layoff_count.",
        fontsize=10,
        color="#555555",
    )
    fig.savefig(OUT_DIR / "yearly_layoff_trend.png", dpi=180)

    fig2, ax1 = plt.subplots(figsize=(12, 6), constrained_layout=True)
    x = range(len(yearly))
    ax1.bar(
        [i - 0.2 for i in x],
        yearly["events"],
        width=0.4,
        label="Events",
        color=palette_events,
    )
    ax1.set_ylabel("Event count", color=palette_events)
    ax1.tick_params(axis="y", labelcolor=palette_events)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(yearly["year"].astype(str))

    ax2 = ax1.twinx()
    ax2.bar(
        [i + 0.2 for i in x],
        yearly["layoff_count"],
        width=0.4,
        label="Known headcount",
        color=palette_people,
    )
    ax2.set_ylabel("Known laid-off employees", color=palette_people)
    ax2.tick_params(axis="y", labelcolor=palette_people)

    ax1.set_title(
        "Yearly Layoffs: Event Volume vs Known Headcount",
        fontsize=15,
        fontweight="bold",
    )
    fig2.text(
        0.01,
        0.01,
        "2026 is partial-year data through 2026-04-16.",
        fontsize=10,
        color="#555555",
    )
    fig2.savefig(OUT_DIR / "yearly_layoff_events_vs_headcount.png", dpi=180)

    print(yearly.to_string(index=False))
    print(f"\nSaved: {OUT_DIR / 'yearly_layoff_trend.png'}")
    print(f"Saved: {OUT_DIR / 'yearly_layoff_events_vs_headcount.png'}")
    print(f"Saved: {OUT_DIR / 'yearly_layoff_trend_summary.csv'}")


if __name__ == "__main__":
    main()
