"""
SVM — detección de eventos críticos de seguridad (superclipping / ERS).

Objetivo estratégico: maximizar recall en clase crítica para alertas tempranas.
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
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from model_pipeline import prepare_data

SVM_FEATURES = ["LapTime_Delta", "ERS_risk", "RaceProgress", "TyreLife"]


def _create_evento_critico(df: pd.DataFrame) -> pd.Series:
    """
    Aproximación de evento crítico mientras no existan etiquetas reales de F1.

    Regla: evento_critico = 1 si
      (LapTime_Delta está en percentil >95 o <5) Y ERS_risk > 7
    Esto captura vueltas anómalas con alto riesgo energético simultáneo.
    """
    p95 = df["LapTime_Delta"].quantile(0.95)
    p5 = df["LapTime_Delta"].quantile(0.05)
    delta_extremo = (df["LapTime_Delta"] > p95) | (df["LapTime_Delta"] < p5)
    return (delta_extremo & (df["ERS_risk"] > 7)).astype(int)


def train(
    csv_path: str = "data/f1_strategy.csv",
    ml_dir: str = "ml",
    data_dir: str = "data",
    training_source: str = "local",
    azure_run_id: str | None = None,
) -> dict:
    """Entrena SVM (linear vs rbf) y guarda el modelo con mejor recall."""
    os.makedirs(ml_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    df, raw_count = prepare_data(csv_path)
    df["evento_critico"] = _create_evento_critico(df)

    df_model = df[SVM_FEATURES + ["evento_critico"]].dropna()
    X = df_model[SVM_FEATURES]
    y = df_model["evento_critico"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Muestra estratificada si el dataset es muy grande (SVM rbf es O(n²) en memoria)
    max_train = 15_000
    if len(X_train) > max_train:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=max_train, random_state=42, stratify=y_train
        )
        print(f"SVM: entrenando con muestra estratificada de {max_train} filas (dataset completo: {len(X)})")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    best_model = None
    best_kernel = None
    best_recall = -1.0
    best_pred = None

    for kernel in ("linear", "rbf"):
        svc = SVC(
            kernel=kernel,
            class_weight="balanced",  # priorizar recall en clase crítica
            probability=True,
            random_state=42,
        )
        svc.fit(X_train_scaled, y_train)
        y_pred = svc.predict(X_test_scaled)

        # Recall de clase crítica (label=1)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        results[kernel] = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
        }

        # Matriz de confusión por kernel
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Normal", "Crítico"]
        ).plot(ax=ax, cmap="Reds", colorbar=False)
        ax.set_title(f"SVM kernel={kernel}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(data_dir, f"confusion_matrix_svm_{kernel}.png"),
            dpi=150,
            facecolor="#0a0a0a",
        )
        plt.close()

        print(f"SVM {kernel}: acc={results[kernel]['accuracy']} "
              f"prec={results[kernel]['precision']} rec={results[kernel]['recall']} "
              f"f1={results[kernel]['f1']}")

        if rec >= best_recall:
            best_recall = rec
            best_model = svc
            best_kernel = kernel
            best_pred = y_pred

    joblib.dump(best_model, os.path.join(ml_dir, "svm_model.pkl"))
    joblib.dump(scaler, os.path.join(ml_dir, "svm_scaler.pkl"))

    trained_at = datetime.now(timezone.utc).isoformat()
    metrics = {
        "algorithm": "SVC",
        "target": "evento_critico",
        "target_rule": (
            "(LapTime_Delta percentil >95 o <5) AND ERS_risk > 7 — "
            "aproximación sin etiquetas reales de F1"
        ),
        "features": SVM_FEATURES,
        "best_kernel": best_kernel,
        "best_recall": round(float(best_recall), 4),
        "kernels": results,
        "class_balance": {
            "critical_pct": round(float(y.mean()) * 100, 2),
            "n_critical": int(y.sum()),
            "n_normal": int(len(y) - y.sum()),
        },
        "n_total": len(df_model),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_removed": raw_count - len(df),
        "training_source": training_source,
        "trained_at": trained_at,
        "objective": "Detectar eventos críticos de seguridad con alto recall",
    }
    if azure_run_id:
        metrics["azure_run_id"] = azure_run_id

    with open(os.path.join(ml_dir, "metrics_svm.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Mejor kernel: {best_kernel} (recall={best_recall:.4f})")
    print(f"Guardado: {ml_dir}/svm_model.pkl")

    return metrics


def log_mlflow_metrics(metrics: dict) -> None:
    import mlflow

    mlflow.log_metric("svm_best_recall", metrics["best_recall"])
    mlflow.log_param("svm_best_kernel", metrics["best_kernel"])
    for kernel, scores in metrics["kernels"].items():
        mlflow.log_metric(f"svm_{kernel}_accuracy", scores["accuracy"])
        mlflow.log_metric(f"svm_{kernel}_recall", scores["recall"])
        mlflow.log_metric(f"svm_{kernel}_f1", scores["f1"])


if __name__ == "__main__":
    train()
