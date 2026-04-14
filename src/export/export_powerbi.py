import os
import pandas as pd

FEATURES_PATH = "data/processed/chicago_crimes_features.csv"
PREDICTIONS_PATH = "outputs/predictions.csv"
CURATED_DIR = "data/curated"
OUTPUT_DIR = "outputs"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Reading feature dataset...")
    features_df = pd.read_csv(FEATURES_PATH)

    print("Reading predictions...")
    predictions_df = pd.read_csv(PREDICTIONS_PATH)

    # Main detailed table for dashboarding
    powerbi_main = features_df.copy()

    # Keep a compact predictions table
    powerbi_predictions = predictions_df.copy()

    # District summary
    district_summary = (
        features_df.groupby("district", dropna=False)
        .agg(
            crime_count=("district", "size"),
            avg_risk_score=("risk_score", "mean"),
            high_severity_rate=("high_severity_target", "mean"),
            avg_arrest_rate=("arrest_flag", "mean"),
            avg_domestic_rate=("domestic_flag", "mean")
        )
        .reset_index()
    )

    district_summary["avg_risk_score"] = district_summary["avg_risk_score"].round(3)
    district_summary["high_severity_rate"] = district_summary["high_severity_rate"].round(3)
    district_summary["avg_arrest_rate"] = district_summary["avg_arrest_rate"].round(3)
    district_summary["avg_domestic_rate"] = district_summary["avg_domestic_rate"].round(3)

    # Hour summary
    hour_summary = (
        features_df.groupby("crime_hour", dropna=False)
        .agg(
            crime_count=("crime_hour", "size"),
            avg_risk_score=("risk_score", "mean"),
            high_severity_rate=("high_severity_target", "mean")
        )
        .reset_index()
        .sort_values("crime_hour")
    )

    hour_summary["avg_risk_score"] = hour_summary["avg_risk_score"].round(3)
    hour_summary["high_severity_rate"] = hour_summary["high_severity_rate"].round(3)

    # Crime type summary
    type_summary = (
        features_df.groupby("primary_type", dropna=False)
        .agg(
            crime_count=("primary_type", "size"),
            avg_risk_score=("risk_score", "mean"),
            high_severity_rate=("high_severity_target", "mean")
        )
        .reset_index()
        .sort_values("crime_count", ascending=False)
    )

    type_summary["avg_risk_score"] = type_summary["avg_risk_score"].round(3)
    type_summary["high_severity_rate"] = type_summary["high_severity_rate"].round(3)

    # Save outputs
    main_output = os.path.join(OUTPUT_DIR, "powerbi_main_table.csv")
    pred_output = os.path.join(OUTPUT_DIR, "powerbi_predictions_table.csv")
    district_output = os.path.join(OUTPUT_DIR, "powerbi_district_summary.csv")
    hour_output = os.path.join(OUTPUT_DIR, "powerbi_hour_summary.csv")
    type_output = os.path.join(OUTPUT_DIR, "powerbi_type_summary.csv")

    powerbi_main.to_csv(main_output, index=False)
    powerbi_predictions.to_csv(pred_output, index=False)
    district_summary.to_csv(district_output, index=False)
    hour_summary.to_csv(hour_output, index=False)
    type_summary.to_csv(type_output, index=False)

    print("Power BI export complete.")
    print(f"Saved: {main_output}")
    print(f"Saved: {pred_output}")
    print(f"Saved: {district_output}")
    print(f"Saved: {hour_output}")
    print(f"Saved: {type_output}")


if __name__ == "__main__":
    main()