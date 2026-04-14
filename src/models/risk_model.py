import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

INPUT_PATH = "data/processed/chicago_crimes_features.csv"
OUTPUT_DIR = "outputs"
REPORTS_DIR = "outputs/reports"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Reading feature dataset...")
    df = pd.read_csv(INPUT_PATH)

    print("Initial shape:", df.shape)

    # Keep only columns we need for modeling
    feature_cols = [
        "primary_type",
        "district",
        "community_area",
        "crime_hour",
        "crime_month",
        "crime_dayofweek",
        "time_bucket",
        "arrest_flag",
        "domestic_flag",
        "severity_score",
        "is_weekend",
        "high_risk_time",
        "district_crime_count",
        "community_crime_count",
        "risk_score"
    ]

    target_col = "high_severity_target"

    model_df = df[feature_cols + [target_col]].copy()

    # Drop rows with missing target
    model_df = model_df.dropna(subset=[target_col])

    X = model_df[feature_cols]
    y = model_df[target_col].astype(int)

    categorical_features = [
        "primary_type",
        "district",
        "community_area",
        "time_bucket"
    ]

    numeric_features = [
    "crime_hour",
    "crime_month",
    "crime_dayofweek",
    "arrest_flag",
    "domestic_flag",
    "is_weekend",
    "high_risk_time",
    "district_crime_count",
    "community_crime_count"
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Generating predictions...")
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", round(acc, 4))
    print("F1 Score:", round(f1, 4))
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Save predictions
    predictions_df = X_test.copy()
    predictions_df["actual_target"] = y_test.values
    predictions_df["predicted_target"] = y_pred

    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        predictions_df["predicted_probability"] = y_prob

    predictions_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)

    # Save metrics report
    report_path = os.path.join(REPORTS_DIR, "model_metrics.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    # Feature importance summary
    try:
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = pipeline.named_steps["model"].feature_importances_

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        importance_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
        importance_df.to_csv(importance_path, index=False)

        print(f"Feature importance saved to: {importance_path}")
    except Exception as e:
        print("Could not save feature importance:", e)

    print(f"Predictions saved to: {predictions_path}")
    print(f"Metrics report saved to: {report_path}")


if __name__ == "__main__":
    main()