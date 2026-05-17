from __future__ import annotations

import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import MODELS_DIR, PROCESSED_DATA_DIR
from evaluate import evaluate_classification

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


RICH_TARGET_COLUMN = "high_justice_risk"
RICH_MODEL_FILE_NAME = "province_year_modeling_2011_2021.csv"
MASTER_FILE_NAME = "province_year_master_2011_2021.csv"
RANDOM_STATE = 42


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def build_model_candidates() -> tuple[dict[str, object], list[dict[str, str]]]:
    candidates: dict[str, object] = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    skipped_models: list[dict[str, str]] = []

    if XGBClassifier is None:
        skipped_models.append(
            {
                "model_name": "xgboost",
                "reason": "xgboost is not installed. Run `pip install -r requirements.txt` to enable it.",
            }
        )
    else:
        candidates["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        )

    return candidates, skipped_models


def build_scoring() -> dict[str, object]:
    return {
        "accuracy": make_scorer(accuracy_score),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }


def get_cv_splits(y: pd.Series, groups: pd.Series | None = None, max_splits: int = 5) -> int:
    min_class_count = int(y.value_counts().min())
    if groups is not None:
        return max(2, min(max_splits, min_class_count, groups.nunique()))
    return max(2, min(max_splits, min_class_count))


def get_group_labels(modeling_df: pd.DataFrame) -> pd.Series:
    if "province_key" in modeling_df.columns:
        return modeling_df["province_key"].astype(str)
    return modeling_df["province"].astype(str)


def save_benchmark_results(results: list[dict]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = MODELS_DIR / "benchmark_results.json"
    summary_path = MODELS_DIR / "benchmark_summary.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, default=float)

    summary_rows: list[dict[str, object]] = []
    for variant in results:
        for model_result in variant["model_results"]:
            test_metrics = model_result["test_metrics"]
            cv_metrics = model_result["cross_validation"]
            summary_rows.append(
                {
                    "variant": variant["title"],
                    "model_name": model_result["model_name"],
                    "rows_used": variant["rows_used"],
                    "feature_count": len(variant["features_used"]),
                    "cv_folds": variant["cv_folds"],
                    "evaluation_strategy": variant["evaluation_strategy"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                    "test_precision": test_metrics["precision"],
                    "test_recall": test_metrics["recall"],
                    "test_f1_score": test_metrics["f1_score"],
                    "test_roc_auc": test_metrics.get("roc_auc"),
                    "cv_accuracy_mean": cv_metrics["accuracy"]["mean"],
                    "cv_balanced_accuracy_mean": cv_metrics["balanced_accuracy"]["mean"],
                    "cv_precision_mean": cv_metrics["precision"]["mean"],
                    "cv_recall_mean": cv_metrics["recall"]["mean"],
                    "cv_f1_mean": cv_metrics["f1"]["mean"],
                    "cv_roc_auc_mean": cv_metrics["roc_auc"]["mean"],
                }
            )

    pd.DataFrame(summary_rows).sort_values(
        by=["variant", "test_f1_score", "cv_f1_mean"],
        ascending=[True, False, False],
    ).to_csv(summary_path, index=False)


def save_model_predictions(prediction_rows: list[dict[str, object]]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    prediction_path = MODELS_DIR / "model_predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False)


def print_variant_summary(variant_result: dict) -> None:
    print()
    print(f"=== {variant_result['title']} ===")
    print(f"Rows used: {variant_result['rows_used']}")
    print(f"Features used: {', '.join(variant_result['features_used'])}")
    print(f"Cross-validation folds: {variant_result['cv_folds']}")
    print(f"Evaluation strategy: {variant_result['evaluation_strategy']}")

    for skipped in variant_result["skipped_models"]:
        print(f"- Skipped {skipped['model_name']}: {skipped['reason']}")

    for model_result in variant_result["model_results"]:
        test_metrics = model_result["test_metrics"]
        cv_metrics = model_result["cross_validation"]
        print()
        print(f"Model: {model_result['model_name']}")
        print(
            "Test metrics: "
            f"accuracy={test_metrics['accuracy']:.3f}, "
            f"balanced_accuracy={test_metrics['balanced_accuracy']:.3f}, "
            f"precision={test_metrics['precision']:.3f}, "
            f"recall={test_metrics['recall']:.3f}, "
            f"f1={test_metrics['f1_score']:.3f}, "
            f"roc_auc={test_metrics.get('roc_auc', float('nan')):.3f}"
        )
        print(
            "CV mean metrics: "
            f"accuracy={cv_metrics['accuracy']['mean']:.3f}, "
            f"balanced_accuracy={cv_metrics['balanced_accuracy']['mean']:.3f}, "
            f"precision={cv_metrics['precision']['mean']:.3f}, "
            f"recall={cv_metrics['recall']['mean']:.3f}, "
            f"f1={cv_metrics['f1']['mean']:.3f}, "
            f"roc_auc={cv_metrics['roc_auc']['mean']:.3f}"
        )


def load_csv_dataset(file_name: str) -> pd.DataFrame:
    file_path = PROCESSED_DATA_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. "
            "Run `python src/prepare_raw_data.py` and `python src/merge_master_data.py` first."
        )
    return pd.read_csv(file_path)


def train_variant(
    title: str,
    df: pd.DataFrame,
    target_column: str,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict:
    available_numeric = [column for column in numeric_features if column in df.columns]
    available_categorical = [column for column in categorical_features if column in df.columns]

    feature_columns = available_numeric + available_categorical
    if not feature_columns:
        raise ValueError(f"No usable feature columns were found for variant: {title}")

    modeling_df = df.dropna(subset=[target_column]).copy()
    x = modeling_df[feature_columns].copy()
    y = modeling_df[target_column].astype(int)

    if y.nunique() < 2:
        raise ValueError(f"Variant '{title}' needs at least two target classes.")

    groups = get_group_labels(modeling_df)
    cv_splits = get_cv_splits(y, groups)
    cv_strategy = StratifiedGroupKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    holdout_splits = get_cv_splits(y, groups, max_splits=4)
    holdout_strategy = StratifiedGroupKFold(
        n_splits=holdout_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    train_index, test_index = next(holdout_strategy.split(x, y, groups=groups))
    x_train = x.iloc[train_index]
    x_test = x.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]
    scoring = build_scoring()
    model_candidates, skipped_models = build_model_candidates()

    model_results: list[dict] = []
    prediction_rows: list[dict[str, object]] = []
    for model_name, classifier in model_candidates.items():
        model = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(available_numeric, available_categorical)),
                ("classifier", classifier),
            ]
        )

        cv_scores = cross_validate(
            model,
            x,
            y,
            cv=cv_strategy,
            groups=groups,
            scoring=scoring,
            n_jobs=1,
        )

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
        test_metrics = evaluate_classification(y_test, predictions, probabilities)

        oof_predictions = cross_val_predict(
            model,
            x,
            y,
            cv=cv_strategy,
            groups=groups,
            method="predict",
            n_jobs=1,
        )
        if hasattr(classifier, "predict_proba"):
            oof_probabilities = cross_val_predict(
                model,
                x,
                y,
                cv=cv_strategy,
                groups=groups,
                method="predict_proba",
                n_jobs=1,
            )[:, 1]
        else:
            oof_probabilities = [None] * len(x)

        for row_index, (_, row) in enumerate(modeling_df.iterrows()):
            prediction_rows.append(
                {
                    "variant": title,
                    "model_name": model_name,
                    "province": row.get("province"),
                    "year": row.get("year"),
                    "target_column": target_column,
                    "actual_target": int(y.iloc[row_index]),
                    "predicted_class": int(oof_predictions[row_index]),
                    "predicted_probability": (
                        float(oof_probabilities[row_index])
                        if oof_probabilities[row_index] is not None
                        else None
                    ),
                    "prediction_source": "out_of_fold",
                }
            )

        cross_validation = {
            metric_name.replace("test_", ""): {
                "mean": float(cv_scores[metric_name].mean()),
                "std": float(cv_scores[metric_name].std()),
            }
            for metric_name in cv_scores
            if metric_name.startswith("test_")
        }

        model_results.append(
            {
                "model_name": model_name,
                "test_metrics": test_metrics,
                "cross_validation": cross_validation,
            }
        )

    return {
        "title": title,
        "rows_used": len(modeling_df),
        "features_used": feature_columns,
        "cv_folds": cv_splits,
        "evaluation_strategy": "stratified_group_by_province",
        "model_results": model_results,
        "prediction_rows": prediction_rows,
        "skipped_models": skipped_models,
    }


def build_wide_coverage_dataset() -> pd.DataFrame:
    df = load_csv_dataset(MASTER_FILE_NAME)
    df = df.dropna(subset=["investigation_files_opened", "active_insured_total"]).copy()

    # Year-relative target is fairer because justice file volume trends upward over time.
    yearly_median = df.groupby("year")["investigation_files_opened"].transform("median")
    df["high_justice_flow"] = (df["investigation_files_opened"] >= yearly_median).astype(int)
    return df


def train_baseline_models() -> list[dict]:
    rich_df = load_csv_dataset(RICH_MODEL_FILE_NAME)
    benchmark_results: list[dict] = []
    all_prediction_rows: list[dict[str, object]] = []

    rich_numeric_features = [
        "population",
        "in_migration",
        "out_migration",
        "net_migration",
        "active_insured_total",
        "active_insured_share_of_population",
        "general_secondary_gross_enrollment_rate",
        "vocational_secondary_gross_enrollment_rate",
        "illiterate_rate",
        "upper_secondary_rate",
        "university_rate",
        "postgraduate_rate",
        "higher_education_share",
    ]
    rich_categorical_features = ["geographical_region", "statistical_region"]

    rich_result = train_variant(
        title="Rich Feature Rate Model",
        df=rich_df,
        target_column=RICH_TARGET_COLUMN,
        numeric_features=rich_numeric_features,
        categorical_features=rich_categorical_features,
    )
    benchmark_results.append(rich_result)
    all_prediction_rows.extend(rich_result.pop("prediction_rows"))

    wide_df = build_wide_coverage_dataset()
    wide_numeric_features = [
        "year",
        "active_insured_total",
        "general_secondary_gross_enrollment_rate",
        "vocational_secondary_gross_enrollment_rate",
    ]
    wide_categorical_features = ["geographical_region", "statistical_region"]

    wide_result = train_variant(
        title="Wide Coverage Flow Model",
        df=wide_df,
        target_column="high_justice_flow",
        numeric_features=wide_numeric_features,
        categorical_features=wide_categorical_features,
    )
    benchmark_results.append(wide_result)
    all_prediction_rows.extend(wide_result.pop("prediction_rows"))

    save_benchmark_results(benchmark_results)
    save_model_predictions(all_prediction_rows)

    for variant_result in benchmark_results:
        print_variant_summary(variant_result)

    print()
    print(f"Saved benchmark outputs to {MODELS_DIR / 'benchmark_results.json'}")
    print(f"Saved benchmark summary to {MODELS_DIR / 'benchmark_summary.csv'}")
    print(f"Saved model predictions to {MODELS_DIR / 'model_predictions.csv'}")

    return benchmark_results


if __name__ == "__main__":
    train_baseline_models()
