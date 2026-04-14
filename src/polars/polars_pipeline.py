import os
import polars as pl

RAW_PATH = "data/raw/chicago_crimes_2023_2026.csv"
PROCESSED_DIR = "data/processed"
CURATED_DIR = "data/curated"


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(CURATED_DIR, exist_ok=True)

    print("Reading raw data with Polars...")
    df = pl.read_csv(RAW_PATH, try_parse_dates=True)

    print(f"Initial shape: {df.shape}")

    # Parse date safely
    if "date" in df.columns:
        if df.schema["date"]==pl.Datetime:
            df=df.with_columns(
                pl.col("date").alias("date_ts")
            )
        else:
            df=df.with_columns(
                pl.col("date").str.strptime(pl.Datetime, strict=False).alias("date_ts"))
    else:
         raise ValueError("Column 'date' not found in the dataset.")
            

    # Keep only rows with essential fields
    df = df.filter(
        pl.col("date_ts").is_not_null() &
        pl.col("primary_type").is_not_null() &
        pl.col("latitude").is_not_null() &
        pl.col("longitude").is_not_null()
    )

    # Feature engineering
    df = df.with_columns([
        pl.col("date_ts").dt.year().alias("crime_year"),
        pl.col("date_ts").dt.month().alias("crime_month"),
        pl.col("date_ts").dt.day().alias("crime_day"),
        pl.col("date_ts").dt.hour().alias("crime_hour"),
        pl.col("date_ts").dt.weekday().alias("crime_dayofweek"),
    ])

    # Time bucket
    df = df.with_columns([
        pl.when((pl.col("crime_hour") >= 0) & (pl.col("crime_hour") < 6))
        .then(pl.lit("Late Night"))
        .when((pl.col("crime_hour") >= 6) & (pl.col("crime_hour") < 12))
        .then(pl.lit("Morning"))
        .when((pl.col("crime_hour") >= 12) & (pl.col("crime_hour") < 18))
        .then(pl.lit("Afternoon"))
        .otherwise(pl.lit("Evening"))
        .alias("time_bucket")
    ])

    # Convert arrest/domestic to numeric flags
    df = df.with_columns([
        pl.when(
            (pl.col("arrest") == True) |
            (pl.col("arrest") == "true") |
            (pl.col("arrest") == "True")
        ).then(pl.lit(1)).otherwise(pl.lit(0)).alias("arrest_flag"),

        pl.when(
            (pl.col("domestic") == True) |
            (pl.col("domestic") == "true") |
            (pl.col("domestic") == "True")
        ).then(pl.lit(1)).otherwise(pl.lit(0)).alias("domestic_flag")
    ])

    # Severity proxy
    df = df.with_columns([
        pl.when(pl.col("primary_type").is_in(["HOMICIDE", "CRIMINAL SEXUAL ASSAULT", "ROBBERY"]))
        .then(pl.lit(3))
        .when(pl.col("primary_type").is_in(["AGGRAVATED ASSAULT", "BURGLARY", "MOTOR VEHICLE THEFT"]))
        .then(pl.lit(2))
        .otherwise(pl.lit(1))
        .alias("severity_score")
    ])

    print(f"Shape after cleaning: {df.shape}")

    # Save processed file
    processed_path = os.path.join(PROCESSED_DIR, "chicago_crimes_cleaned_polars.csv")
    df.write_csv(processed_path)

    print("Creating curated Polars tables...")

    crime_by_type = (
        df.group_by("primary_type")
        .agg(pl.len().alias("crime_count"))
        .sort("crime_count", descending=True)
    )

    crime_by_hour = (
        df.group_by("crime_hour")
        .agg(pl.len().alias("crime_count"))
        .sort("crime_hour")
    )

    crime_by_district = (
        df.group_by("district")
        .agg([
            pl.len().alias("crime_count"),
            pl.col("severity_score").mean().round(2).alias("avg_severity"),
            pl.col("arrest_flag").mean().round(3).alias("arrest_rate")
        ])
        .sort("crime_count", descending=True)
    )

    crime_by_time_bucket = (
        df.group_by("time_bucket")
        .agg(pl.len().alias("crime_count"))
        .sort("crime_count", descending=True)
    )

    crime_hotspots = (
        df.filter(pl.col("community_area").is_not_null())
        .group_by("community_area")
        .agg([
            pl.len().alias("crime_count"),
            pl.col("severity_score").mean().round(2).alias("avg_severity"),
            pl.col("latitude").mean().round(6).alias("center_latitude"),
            pl.col("longitude").mean().round(6).alias("center_longitude")
        ])
        .sort("crime_count", descending=True)
    )

    # Save curated outputs
    crime_by_type.write_csv(os.path.join(CURATED_DIR, "crime_by_type_polars.csv"))
    crime_by_hour.write_csv(os.path.join(CURATED_DIR, "crime_by_hour_polars.csv"))
    crime_by_district.write_csv(os.path.join(CURATED_DIR, "crime_by_district_polars.csv"))
    crime_by_time_bucket.write_csv(os.path.join(CURATED_DIR, "crime_by_time_bucket_polars.csv"))
    crime_hotspots.write_csv(os.path.join(CURATED_DIR, "crime_hotspots_polars.csv"))

    print("Polars pipeline complete.")
    print(f"Processed file saved to: {processed_path}")
    print(f"Curated CSV files saved to: {CURATED_DIR}")


if __name__ == "__main__":
    main()