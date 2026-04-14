import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    to_timestamp,
    year,
    month,
    dayofmonth,
    hour,
    dayofweek,
    when,
    count,
    avg,
    round
)

RAW_PATH = "data/raw/chicago_crimes_2023_2026.csv"
PROCESSED_DIR = "data/processed"
CURATED_DIR = "data/curated"


def create_spark_session():
    return (
        SparkSession.builder
        .appName("UrbanSafetyAnalytics")
        .getOrCreate()
    )


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(CURATED_DIR, exist_ok=True)

    spark = create_spark_session()

    print("Reading raw data...")
    df = spark.read.csv(RAW_PATH, header=True, inferSchema=True)

    print("Initial shape estimate:")
    print("Rows:", df.count())
    print("Columns:", len(df.columns))

    print("Cleaning and transforming data...")

    # Convert date column to timestamp
    df = df.withColumn("date_ts", to_timestamp(col("date"), "yyyy-MM-dd'T'HH:mm:ss.SSS"))

    # If the above format fails for some rows, Spark will leave nulls.
    # We keep only rows with essential values.
    df = df.filter(
        col("date_ts").isNotNull() &
        col("primary_type").isNotNull() &
        col("latitude").isNotNull() &
        col("longitude").isNotNull()
    )

    # Create time-based features
    df = (
        df.withColumn("crime_year", year(col("date_ts")))
          .withColumn("crime_month", month(col("date_ts")))
          .withColumn("crime_day", dayofmonth(col("date_ts")))
          .withColumn("crime_hour", hour(col("date_ts")))
          .withColumn("crime_dayofweek", dayofweek(col("date_ts")))
    )

    # Create risk period bucket
    df = df.withColumn(
        "time_bucket",
        when((col("crime_hour") >= 0) & (col("crime_hour") < 6), "Late Night")
        .when((col("crime_hour") >= 6) & (col("crime_hour") < 12), "Morning")
        .when((col("crime_hour") >= 12) & (col("crime_hour") < 18), "Afternoon")
        .otherwise("Evening")
    )

    # Convert arrest/domestic text values to numeric flags where possible
    df = (
        df.withColumn(
            "arrest_flag",
            when(col("arrest") == True, 1)
            .when(col("arrest") == "true", 1)
            .when(col("arrest") == "True", 1)
            .otherwise(0)
        )
        .withColumn(
            "domestic_flag",
            when(col("domestic") == True, 1)
            .when(col("domestic") == "true", 1)
            .when(col("domestic") == "True", 1)
            .otherwise(0)
        )
    )

    # Severity proxy
    df = df.withColumn(
        "severity_score",
        when(col("primary_type").isin("HOMICIDE", "CRIMINAL SEXUAL ASSAULT", "ROBBERY"), 3)
        .when(col("primary_type").isin("AGGRAVATED ASSAULT", "BURGLARY", "MOTOR VEHICLE THEFT"), 2)
        .otherwise(1)
    )

    print("Saving cleaned Spark dataset...")
    processed_output = os.path.join(PROCESSED_DIR, "chicago_crimes_cleaned_spark.parquet")
    df.toPandas().to_csv(
    os.path.join(PROCESSED_DIR, "chicago_crimes_cleaned_spark.csv"),
    index=False
    )

    print("Creating curated tables...")

    # 1. Crime by type
    crime_by_type = (
        df.groupBy("primary_type")
          .agg(count("*").alias("crime_count"))
          .orderBy(col("crime_count").desc())
    )

    # 2. Crime by hour
    crime_by_hour = (
        df.groupBy("crime_hour")
          .agg(count("*").alias("crime_count"))
          .orderBy(col("crime_hour").asc())
    )

    # 3. Crime by district
    crime_by_district = (
        df.groupBy("district")
          .agg(
              count("*").alias("crime_count"),
              round(avg("severity_score"), 2).alias("avg_severity"),
              round(avg("arrest_flag"), 3).alias("arrest_rate")
          )
          .orderBy(col("crime_count").desc())
    )

    # 4. Crime by time bucket
    crime_by_time_bucket = (
        df.groupBy("time_bucket")
          .agg(count("*").alias("crime_count"))
          .orderBy(col("crime_count").desc())
    )

    # 5. Hotspot table by community area
    crime_hotspots = (
        df.groupBy("community_area")
          .agg(
              count("*").alias("crime_count"),
              round(avg("severity_score"), 2).alias("avg_severity"),
              round(avg("latitude"), 6).alias("center_latitude"),
              round(avg("longitude"), 6).alias("center_longitude")
          )
          .filter(col("community_area").isNotNull())
          .orderBy(col("crime_count").desc())
    )

    # Save curated tables as CSV
    crime_by_type.toPandas().to_csv(
        os.path.join(CURATED_DIR, "crime_by_type_spark.csv"),
        index=False
    )

    crime_by_hour.toPandas().to_csv(
        os.path.join(CURATED_DIR, "crime_by_hour_spark.csv"),
        index=False
    )

    crime_by_district.toPandas().to_csv(
        os.path.join(CURATED_DIR, "crime_by_district_spark.csv"),
        index=False
    )

    crime_by_time_bucket.toPandas().to_csv(
        os.path.join(CURATED_DIR, "crime_by_time_bucket_spark.csv"),
        index=False
    )

    crime_hotspots.toPandas().to_csv(
        os.path.join(CURATED_DIR, "crime_hotspots_spark.csv"),
        index=False
    )

    print("Spark pipeline complete.")
    print(f"Processed parquet saved to: {processed_output}")
    print(f"Curated CSV files saved to: {CURATED_DIR}")

    spark.stop()


if __name__ == "__main__":
    main()