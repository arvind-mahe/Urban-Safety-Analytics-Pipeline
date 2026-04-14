import os
import polars as pl

INPUT_PATH = "data/processed/chicago_crimes_cleaned_polars.csv"
OUTPUT_PATH = "data/processed/chicago_crimes_features.csv"
CURATED_DIR = "data/curated"


def main():
    os.makedirs(CURATED_DIR, exist_ok=True)

    print("Reading processed Polars dataset...")
    df = pl.read_csv(INPUT_PATH, try_parse_dates=True)

    print(f"Input shape: {df.shape}")

    # Create weekend flag
    df = df.with_columns([
        pl.when(pl.col("crime_dayofweek").is_in([5, 6]))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("is_weekend")
    ])

    # High-risk time flag
    df = df.with_columns([
        pl.when(
            (pl.col("crime_hour") >= 20) | (pl.col("crime_hour") <= 4)
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("high_risk_time")
    ])

    # Simple target for later modeling:
    # mark severe crime incidents as 1, others as 0
    df = df.with_columns([
        pl.when(pl.col("severity_score") >= 2)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("high_severity_target")
    ])

    # Create district-level frequency features
    district_counts = (
        df.group_by("district")
        .agg(pl.len().alias("district_crime_count"))
    )

    df = df.join(district_counts, on="district", how="left")

    # Create community-area frequency features
    community_counts = (
        df.filter(pl.col("community_area").is_not_null())
        .group_by("community_area")
        .agg(pl.len().alias("community_crime_count"))
    )

    df = df.join(community_counts, on="community_area", how="left")

    # Fill nulls for areas that may not have community_area
    df = df.with_columns([
        pl.col("community_crime_count").fill_null(0),
        pl.col("district_crime_count").fill_null(0)
    ])

    # Area risk score
    df = df.with_columns([
        (
            pl.col("severity_score") * 0.5 +
            pl.col("high_risk_time") * 1.0 +
            (pl.col("district_crime_count") / 1000) +
            (pl.col("community_crime_count") / 1000)
        ).round(3).alias("risk_score")
    ])

    print(f"Output shape after feature engineering: {df.shape}")

    df.write_csv(OUTPUT_PATH)

    # Curated summary tables for later dashboarding
    risk_summary_by_district = (
        df.group_by("district")
        .agg([
            pl.len().alias("crime_count"),
            pl.col("risk_score").mean().round(3).alias("avg_risk_score"),
            pl.col("high_severity_target").mean().round(3).alias("high_severity_rate")
        ])
        .sort("avg_risk_score", descending=True)
    )

    risk_summary_by_hour = (
        df.group_by("crime_hour")
        .agg([
            pl.len().alias("crime_count"),
            pl.col("risk_score").mean().round(3).alias("avg_risk_score")
        ])
        .sort("crime_hour")
    )

    risk_summary_by_type = (
        df.group_by("primary_type")
        .agg([
            pl.len().alias("crime_count"),
            pl.col("risk_score").mean().round(3).alias("avg_risk_score")
        ])
        .sort("crime_count", descending=True)
    )

    risk_summary_by_district.write_csv(
        os.path.join(CURATED_DIR, "risk_summary_by_district.csv")
    )
    risk_summary_by_hour.write_csv(
        os.path.join(CURATED_DIR, "risk_summary_by_hour.csv")
    )
    risk_summary_by_type.write_csv(
        os.path.join(CURATED_DIR, "risk_summary_by_type.csv")
    )

    print("Feature engineering complete.")
    print(f"Feature dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()