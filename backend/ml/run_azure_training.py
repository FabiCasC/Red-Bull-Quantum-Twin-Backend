"""Entry point for Azure ML Command Job — entrena los 4 modelos en compute remoto."""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import mlflow

sys.path.insert(0, os.path.dirname(__file__))

from model_pipeline import log_mlflow_metrics, run_pipeline, save_artifacts
from train_decision_tree import log_mlflow_metrics as log_dt_metrics
from train_decision_tree import train as train_dt
from train_kmeans import log_mlflow_metrics as log_km_metrics
from train_kmeans import train as train_km
from train_svm import log_mlflow_metrics as log_svm_metrics
from train_svm import train as train_svm


def _write_models_source(ml_dir: str, azure_run_id: str | None) -> None:
    """Consolida origen y fecha de entrenamiento de los 4 modelos."""
    sources = {}

    mapping = [
        ("regression", "metrics.json", "r2"),
        ("decision_tree", "metrics_classification.json", "accuracy"),
        ("kmeans", "metrics_clustering.json", "silhouette_score"),
        ("svm", "metrics_svm.json", "best_recall"),
    ]
    for name, fname, metric_key in mapping:
        path = os.path.join(ml_dir, fname)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                m = json.load(f)
            sources[name] = {
                "source": m.get("training_source", "azure_ml"),
                "trained_at": m.get("trained_at"),
                "metric": m.get(metric_key),
                "azure_run_id": m.get("azure_run_id", azure_run_id),
            }

    with open(os.path.join(ml_dir, "models_source.json"), "w", encoding="utf-8") as f:
        json.dump(sources, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to f1_strategy.csv")
    parser.add_argument("--output", required=True, help="Output directory (uri_folder)")
    args = parser.parse_args()

    ml_dir = os.path.join(args.output, "ml")
    data_dir = os.path.join(args.output, "data")
    os.makedirs(ml_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    run_id = None
    print("=== AZURE ML — ENTRENAMIENTO DE 4 MODELOS ===")
    print(f"Data: {args.data} | Output: {args.output}")

    mlflow.start_run()
    try:
        run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None

        # 1. Regresión (LR vs RF)
        print("\n--- [1/4] Regresión ---")
        result = run_pipeline(args.data)
        log_mlflow_metrics(result)
        save_artifacts(
            result,
            output_dir=ml_dir,
            data_dir=data_dir,
            training_source="azure_ml",
            azure_run_id=run_id,
        )

        # 2. Árbol de Decisión
        print("\n--- [2/4] Árbol de Decisión ---")
        dt_metrics = train_dt(
            args.data, ml_dir, data_dir, training_source="azure_ml", azure_run_id=run_id
        )
        log_dt_metrics(dt_metrics)

        # 3. K-Means
        print("\n--- [3/4] K-Means ---")
        km_metrics = train_km(
            args.data, ml_dir, data_dir, training_source="azure_ml", azure_run_id=run_id
        )
        log_km_metrics(km_metrics)

        # 4. SVM
        print("\n--- [4/4] SVM ---")
        svm_metrics = train_svm(
            args.data, ml_dir, data_dir, training_source="azure_ml", azure_run_id=run_id
        )
        log_svm_metrics(svm_metrics)

        _write_models_source(ml_dir, run_id)
        print(f"\n✅ 4 modelos entrenados en Azure ML — run_id={run_id}")
    finally:
        mlflow.end_run()


if __name__ == "__main__":
    main()
