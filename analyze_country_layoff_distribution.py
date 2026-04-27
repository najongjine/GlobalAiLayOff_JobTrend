from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
LAYOFFS_PATH = ROOT / "layoffs_events.csv"
GLOBAL_LABOR_PATH = ROOT / "global_labor_indicators.csv"
OUT_DIR = ROOT / "outputs"


def fmt_count(value: float) -> str:
    return f"{value:,.0f}"


def normalize_country(value: object) -> str:
    text = str(value).strip()
    if text.startswith("United King"):
        return "United Kingdom"
    aliases = {
        "United States": "United States",
        "USA": "United States",
        "US": "United States",
        "United Kingdom": "United Kingdom",
        "UK": "United Kingdom",
    }
    return aliases.get(text, text if text else "Unknown")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    layoffs = pd.read_csv(LAYOFFS_PATH)
    labor = pd.read_csv(GLOBAL_LABOR_PATH)

    layoffs["date"] = pd.to_datetime(layoffs["date"], errors="coerce")
    layoffs["year"] = layoffs["date"].dt.year
    layoffs["country_clean"] = layoffs["country"].map(normalize_country)
    layoffs["layoff_count_num"] = pd.to_numeric(layoffs["layoff_count"], errors="coerce")

    country = (
        layoffs.groupby("country_clean", as_index=False)
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
    country["event_share_pct"] = country["events"] / country["events"].sum() * 100
    country["headcount_share_pct"] = country["layoff_count"] / country["layoff_count"].sum() * 100
    country["known_count_coverage_pct"] = (
        country["known_count_events"] / country["events"] * 100
    )
    country = country.round(
        {
            "avg_layoff_count": 1,
            "median_layoff_count": 1,
            "event_share_pct": 1,
            "headcount_share_pct": 1,
            "known_count_coverage_pct": 1,
        }
    )
    country.to_csv(OUT_DIR / "country_layoff_distribution_summary.csv", index=False)

    top_countries = country.head(12).copy()

    yearly_country = (
        layoffs[layoffs["country_clean"].isin(top_countries["country_clean"])]
        .dropna(subset=["year"])
        .groupby(["year", "country_clean"], as_index=False)
        .agg(events=("company", "size"), layoff_count=("layoff_count_num", "sum"))
    )
    yearly_country["year"] = yearly_country["year"].astype(int)
    yearly_country.to_csv(OUT_DIR / "country_yearly_layoff_summary.csv", index=False)

    event_heatmap = (
        yearly_country.pivot(index="country_clean", columns="year", values="events")
        .fillna(0)
        .loc[top_countries["country_clean"]]
    )

    labor["country_name_clean"] = labor["country_name"].map(normalize_country)
    labor["year"] = pd.to_numeric(labor["year"], errors="coerce").astype("Int64")
    labor_cols = [
        "country_name_clean",
        "year",
        "unemployment_rate_pct",
        "youth_unemployment_pct",
        "employment_to_pop_pct",
    ]
    labor_small = labor[labor_cols].dropna(subset=["year"]).copy()
    labor_small["year"] = labor_small["year"].astype(int)

    joined = yearly_country.merge(
        labor_small,
        left_on=["country_clean", "year"],
        right_on=["country_name_clean", "year"],
        how="inner",
    )
    joined.to_csv(OUT_DIR / "country_layoffs_with_labor_indicators.csv", index=False)

    labor_change = (
        labor_small.sort_values(["country_name_clean", "year"])
        .groupby("country_name_clean", as_index=False)
        .agg(
            first_year=("year", "first"),
            last_year=("year", "last"),
            unemployment_first=("unemployment_rate_pct", "first"),
            unemployment_last=("unemployment_rate_pct", "last"),
            youth_unemployment_first=("youth_unemployment_pct", "first"),
            youth_unemployment_last=("youth_unemployment_pct", "last"),
        )
    )
    labor_change["unemployment_change_pp"] = (
        labor_change["unemployment_last"] - labor_change["unemployment_first"]
    )
    labor_change["youth_unemployment_change_pp"] = (
        labor_change["youth_unemployment_last"]
        - labor_change["youth_unemployment_first"]
    )

    country_labor = country.merge(
        labor_change,
        left_on="country_clean",
        right_on="country_name_clean",
        how="inner",
    )
    country_labor.to_csv(OUT_DIR / "country_layoffs_labor_change_summary.csv", index=False)

    sns.set_theme(style="whitegrid")
    event_color = "#3366A8"
    headcount_color = "#D95F43"

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    fig.suptitle("Country Distribution of Layoffs", fontsize=18, fontweight="bold")

    ax = axes[0, 0]
    plot_data = top_countries.sort_values("events", ascending=True)
    ax.barh(plot_data["country_clean"], plot_data["events"], color=event_color)
    ax.set_title("Top Countries by Event Count", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("Layoff events")
    ax.set_ylabel("")
    for i, row in enumerate(plot_data.itertuples(index=False)):
        ax.text(row.events, i, f" {fmt_count(row.events)}", va="center", fontsize=10)

    ax = axes[0, 1]
    plot_data = top_countries.sort_values("layoff_count", ascending=True)
    ax.barh(plot_data["country_clean"], plot_data["layoff_count"], color=headcount_color)
    ax.set_title(
        "Top Countries by Known Headcount", loc="left", fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Known laid-off employees")
    ax.set_ylabel("")
    for i, row in enumerate(plot_data.itertuples(index=False)):
        ax.text(row.layoff_count, i, f" {fmt_count(row.layoff_count)}", va="center", fontsize=10)

    ax = axes[1, 0]
    sns.heatmap(
        event_heatmap,
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".0f",
        cbar_kws={"label": "Event count"},
        ax=ax,
    )
    ax.set_title("Yearly Event Pattern for Top Countries", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("")

    ax = axes[1, 1]
    scatter_data = country_labor[country_labor["events"] >= 5].copy()
    sizes = (scatter_data["companies"] / scatter_data["companies"].max() * 900) + 80
    sc = ax.scatter(
        scatter_data["events"],
        scatter_data["unemployment_change_pp"],
        s=sizes,
        c=scatter_data["unemployment_last"],
        cmap="magma",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.8,
    )
    for _, row in scatter_data.sort_values("events", ascending=False).head(12).iterrows():
        ax.text(
            row["events"],
            row["unemployment_change_pp"],
            f" {row['country_clean']}",
            fontsize=9,
            va="center",
        )
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_title(
        "Layoff Events vs Unemployment Change",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Total layoff events")
    ax.set_ylabel("Unemployment rate change, first to last labor year (pp)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Latest unemployment rate (%)")

    fig.text(
        0.01,
        0.01,
        "Note: Headcount totals include only rows with reported layoff_count. "
        "Global labor comparison uses countries present in global_labor_indicators.csv.",
        fontsize=10,
        color="#555555",
    )
    fig.savefig(OUT_DIR / "country_layoff_distribution.png", dpi=180)

    fig2, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    us_share = country.head(10).copy()
    ax = axes[0]
    ax.pie(
        us_share["events"],
        labels=us_share["country_clean"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 9},
    )
    ax.set_title("Event Share Among Top 10 Countries", fontsize=13, fontweight="bold")

    ax = axes[1]
    line_data = joined[
        joined["country_clean"].isin(country_labor.sort_values("events", ascending=False).head(6)["country_clean"])
    ].copy()
    sns.lineplot(
        data=line_data,
        x="year",
        y="unemployment_rate_pct",
        hue="country_clean",
        marker="o",
        linewidth=2,
        ax=ax,
    )
    ax.set_title("Unemployment Rate Trend for High-Layoff Countries", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Unemployment rate (%)")
    ax.legend(title="")

    fig2.savefig(OUT_DIR / "country_layoff_labor_context.png", dpi=180)

    print("Top countries by events")
    print(
        country[
            [
                "country_clean",
                "events",
                "event_share_pct",
                "layoff_count",
                "headcount_share_pct",
                "companies",
                "known_count_coverage_pct",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )
    print("\nLabor-linked countries")
    print(
        country_labor[
            [
                "country_clean",
                "events",
                "layoff_count",
                "first_year",
                "last_year",
                "unemployment_first",
                "unemployment_last",
                "unemployment_change_pp",
            ]
        ]
        .sort_values("events", ascending=False)
        .head(15)
        .to_string(index=False)
    )
    print(f"\nSaved: {OUT_DIR / 'country_layoff_distribution.png'}")
    print(f"Saved: {OUT_DIR / 'country_layoff_labor_context.png'}")
    print(f"Saved: {OUT_DIR / 'country_layoff_distribution_summary.csv'}")
    print(f"Saved: {OUT_DIR / 'country_yearly_layoff_summary.csv'}")
    print(f"Saved: {OUT_DIR / 'country_layoffs_with_labor_indicators.csv'}")
    print(f"Saved: {OUT_DIR / 'country_layoffs_labor_change_summary.csv'}")


if __name__ == "__main__":
    main()
