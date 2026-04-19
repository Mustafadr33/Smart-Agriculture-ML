from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"
RANDOM_STATE = 42
VAL_SIZE = 0.15
TEST_SIZE = 0.15


def get_dataset_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "datasets" / "crop_recommendation" / "crop_recommendation_dataset.csv"


def get_model_output_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "models" / "random_forest_crop_model.joblib"


def load_and_validate_data(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)

    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Colonnes manquantes dans le dataset: {sorted(missing_columns)}")

    print("=== Vérification des données ===")
    print(f"Shape: {df.shape}")

    print("=== Statistiques descriptives ===")
    print(df.describe())

    print("Valeurs manquantes par colonne:")
    print(df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum())

    return df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()


def plot_data_overview(df: pd.DataFrame) -> None:
    corr = df[FEATURE_COLUMNS].corr(numeric_only=True)

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matrice de corrélation des features")
    plt.tight_layout()
    plt.show()


def prepare_data(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    print(f"Variables numériques: {num_cols}")

    temp_size = VAL_SIZE + TEST_SIZE
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=temp_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=TEST_SIZE / temp_size,
        random_state=RANDOM_STATE,
        stratify=y_tmp,
    )

    print(
        f"Répartition des jeux: train={len(X_train)} | validation={len(X_val)} | test={len(X_test)}"
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
        ],
        remainder="drop",
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, y, preprocessor


def train_random_forest(X_train, X_val, X_test, y_train, y_val, y_test, y, preprocessor):
    labels = sorted(y.unique())

    baseline_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    baseline_pipeline.fit(X_train, y_train)
    y_val_pred_baseline = baseline_pipeline.predict(X_val)

    baseline_val_acc = accuracy_score(y_val, y_val_pred_baseline)
    baseline_val_f1 = f1_score(y_val, y_val_pred_baseline, average="weighted")

    print("\n=== Random Forest baseline ===")
    print(f"Validation accuracy: {baseline_val_acc:.4f}")
    print(f"Validation F1 pondéré: {baseline_val_f1:.4f}")

    param_grid = {
        "model__n_estimators": [100, 300],
        "model__max_depth": [None, 20, 40],
        "model__min_samples_split": [2, 5],
    }

    optimized_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )

    grid_search = GridSearchCV(
        optimized_pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    y_val_pred_optimized = best_pipeline.predict(X_val)

    optimized_val_acc = accuracy_score(y_val, y_val_pred_optimized)
    optimized_val_f1 = f1_score(y_val, y_val_pred_optimized, average="weighted")

    print("\n=== Random Forest optimisé ===")
    print(f"Meilleurs paramètres: {grid_search.best_params_}")
    print(f"Validation accuracy: {optimized_val_acc:.4f}")
    print(f"Validation F1 pondéré: {optimized_val_f1:.4f}")

    final_pipeline = baseline_pipeline
    final_model_name = "Random Forest"
    final_val_acc = baseline_val_acc
    final_val_f1 = baseline_val_f1
    best_params = {
        "model__n_estimators": 300,
        "model__max_depth": None,
        "model__min_samples_split": 2,
    }

    if optimized_val_f1 > baseline_val_f1:
        final_pipeline = best_pipeline
        final_model_name = "Random Forest optimisé"
        final_val_acc = optimized_val_acc
        final_val_f1 = optimized_val_f1
        best_params = grid_search.best_params_

    y_test_pred = final_pipeline.predict(X_test)

    metrics = {
        "model": final_model_name,
        "validation_accuracy": final_val_acc,
        "validation_f1_score_weighted": final_val_f1,
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_f1_score_weighted": f1_score(y_test, y_test_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_test_pred, labels=labels),
        "pipeline": final_pipeline,
        "best_params": best_params,
    }

    print("\n=== Modèle final retenu ===")
    print(f"Modèle: {metrics['model']}")
    print(f"Validation accuracy: {metrics['validation_accuracy']:.4f}")
    print(f"Validation F1 pondéré: {metrics['validation_f1_score_weighted']:.4f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test F1 pondéré: {metrics['test_f1_score_weighted']:.4f}")

    return metrics, labels, y_test_pred


def plot_confusion_matrix(cm, labels) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="magma")
    plt.title("Confusion Matrix — Random Forest")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.tight_layout()
    plt.show()


def compare_predictions_vs_reality(y_test, y_pred_rf) -> None:

    compare_df = pd.DataFrame(
        {
            "realite": y_test.reset_index(drop=True),
            "prediction_rf": pd.Series(y_pred_rf),
        }
    )

    compare_df["correct_rf"] = compare_df["realite"] == compare_df["prediction_rf"]

    print("\n=== Prédiction vs Réalité (20 premières lignes) ===")
    print(compare_df.head(20).to_string(index=False))
    print(f"Taux de bonnes prédictions (Random Forest): {compare_df['correct_rf'].mean():.4f}")


def save_model_bundle(metrics: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_name": metrics["model"],
        "pipeline": metrics["pipeline"],
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "metrics": {
            "validation_accuracy": metrics["validation_accuracy"],
            "validation_f1_score_weighted": metrics["validation_f1_score_weighted"],
            "test_accuracy": metrics["test_accuracy"],
            "test_f1_score_weighted": metrics["test_f1_score_weighted"],
        },
        "best_params": metrics.get("best_params"),
    }
    joblib.dump(bundle, output_path)
    print(f"\nModèle sauvegardé dans: {output_path}")


def main() -> None:
    dataset_path = get_dataset_path()
    model_output_path = get_model_output_path()
    print(f"Dataset utilisé: {dataset_path}")

    df = load_and_validate_data(dataset_path)
    plot_data_overview(df)

    X_train, X_val, X_test, y_train, y_val, y_test, y, preprocessor = prepare_data(df)
    metrics, labels, y_pred_rf = train_random_forest(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        y,
        preprocessor,
    )

    plot_confusion_matrix(metrics["confusion_matrix"], labels)
    compare_predictions_vs_reality(y_test, y_pred_rf)
    save_model_bundle(metrics, model_output_path)


if __name__ == "__main__":
    main()
