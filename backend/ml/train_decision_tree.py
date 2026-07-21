"""
Árbol de Decisión — clasifica ERS_status (NORMAL / ALERTA / CRÍTICO).

Objetivo estratégico: interpretabilidad para decisiones de ingeniería de pista.
Entrenamiento local (fallback) o invocado desde Azure ML vía run_azure_training.py.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

from model_pipeline import prepare_data

# Features seleccionadas por interpretabilidad en sustentación académica
DT_FEATURES = [
    "Normalized_TyreLife",
    "RaceProgress",
    "LapTime_Delta",
    "TyreLife",
    "Compound_encoded",
]
DT_TARGET = "ERS_status"
MAX_DEPTH = 5  # rango 4-6: balance entre precisión e interpretabilidad


def train(
    csv_path: str = "data/f1_strategy.csv",
    ml_dir: str = "ml",
    data_dir: str = "data",
    training_source: str = "local",
    azure_run_id: str | None = None,
) -> dict:
    """Entrena el árbol de decisión y persiste artefactos."""
    os.makedirs(ml_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    df, raw_count = prepare_data(csv_path)
    df_model = df[DT_FEATURES + [DT_TARGET]].dropna().copy()
    df_model[DT_TARGET] = df_model[DT_TARGET].astype(str)

    X = df_model[DT_FEATURES]
    y = df_model[DT_TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = round(float(accuracy_score(y_test, y_pred)), 4)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    classes = list(clf.classes_)

    # Visualización del árbol (interpretabilidad para la sustentación)
    plt.figure(figsize=(20, 10))
    plot_tree(
        clf,
        feature_names=DT_FEATURES,
        class_names=classes,
        filled=True,
        rounded=True,
        fontsize=9,
    )
    plt.title("Árbol de Decisión — Clasificación ERS_status")
    plt.tight_layout()
    plt.savefig(
        os.path.join(data_dir, "decision_tree.png"),
        dpi=150,
        facecolor="#0a0a0a",
    )
    plt.close()

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(
        ax=ax, cmap="Reds", colorbar=False
    )
    ax.set_title("Matriz de confusión — ERS_status")
    plt.tight_layout()
    plt.savefig(
        os.path.join(data_dir, "confusion_matrix_dt.png"),
        dpi=150,
        facecolor="#0a0a0a",
    )
    plt.close()

    joblib.dump(clf, os.path.join(ml_dir, "decision_tree_model.pkl"))

    trained_at = datetime.now(timezone.utc).isoformat()
    metrics = {
        "algorithm": "DecisionTreeClassifier",
        "target": DT_TARGET,
        "features": DT_FEATURES,
        "max_depth": MAX_DEPTH,
        "accuracy": accuracy,
        "classification_report": report,
        "classes": classes,
        "n_total": len(df_model),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_removed": raw_count - len(df),
        "training_source": training_source,
        "trained_at": trained_at,
        "objective": "Clasificar riesgo ERS para decisiones interpretables en pista",
    }
    if azure_run_id:
        metrics["azure_run_id"] = azure_run_id

    metrics_path = os.path.join(ml_dir, "metrics_classification.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"=== ÁRBOL DE DECISIÓN ===")
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Guardado: {ml_dir}/decision_tree_model.pkl, {data_dir}/decision_tree.png")

    return metrics


def log_mlflow_metrics(metrics: dict) -> None:
    """Registra métricas del clasificador en MLflow (Azure ML)."""
    import mlflow

    mlflow.log_metric("dt_accuracy", metrics["accuracy"])
    mlflow.log_param("dt_max_depth", metrics["max_depth"])
    mlflow.log_param("dt_target", metrics["target"])
    for label, scores in metrics["classification_report"].items():
        if isinstance(scores, dict) and "f1-score" in scores:
            safe = label.replace(" ", "_").replace("Í", "I")
            mlflow.log_metric(f"dt_f1_{safe}", scores["f1-score"])


if __name__ == "__main__":
    train()
