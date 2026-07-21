"""
K-Means — segmentación de estrategia de neumáticos y degradación.

Objetivo estratégico: descubrir perfiles de stint para recomendaciones de pit stop.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from model_pipeline import prepare_data

KMEANS_FEATURES = ["Cumulative_Degradation", "TyreLife", "LapTime_normalized"]
K_RANGE = range(2, 9)  # método del codo: k=2 a 8


def _name_cluster(profile: dict) -> str:
    """Genera etiqueta interpretable a partir del perfil promedio del cluster."""
    compound = profile.get("Compound_encoded_mean", 1)
    tyre = profile.get("TyreLife_mean", 0)
    lap = profile.get("LapTime_normalized_mean", 0)
    deg = profile.get("Cumulative_Degradation_mean", 0)

    compound_names = {0: "SOFT", 1: "MEDIUM", 2: "HARD", 3: "INTER", 4: "WET"}
    compound_label = compound_names.get(int(round(compound)), "MIXED")

    if deg > 15 and tyre > 25:
        return f"Degradación alta ({compound_label})"
    if lap < -0.5:
        return f"Ritmo competitivo ({compound_label})"
    if tyre < 10:
        return f"Stint temprano ({compound_label})"
    if deg > 8:
        return f"Desgaste moderado ({compound_label})"
    return f"Stint estable ({compound_label})"


def train(
    csv_path: str = "data/f1_strategy.csv",
    ml_dir: str = "ml",
    data_dir: str = "data",
    training_source: str = "local",
    azure_run_id: str | None = None,
) -> dict:
    """Entrena K-Means con método del codo y persiste artefactos."""
    os.makedirs(ml_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    df, raw_count = prepare_data(csv_path)
    df_model = df[KMEANS_FEATURES + ["Compound_encoded"]].dropna()

    X = df_model[KMEANS_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Muestra para método del codo (99k filas — acelera búsqueda de k sin perder representatividad)
    sample_size = min(20_000, len(X_scaled))
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X_scaled), sample_size, replace=False)
    X_elbow = X_scaled[sample_idx]

    inertias = []
    silhouettes = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_elbow)
        inertias.append(km.inertia_)
        silhouettes[k] = round(float(silhouette_score(X_elbow, labels)), 4)

    optimal_k = max(silhouettes, key=silhouettes.get)
    print(f"K óptimo (silhouette): k={optimal_k}, score={silhouettes[optimal_k]}")

    plt.figure(figsize=(8, 5))
    plt.plot(list(K_RANGE), inertias, "o-", color="#cc0000", linewidth=2)
    plt.axvline(optimal_k, color="#ffcc00", linestyle="--", label=f"k óptimo={optimal_k}")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inercia")
    plt.title("Método del codo — K-Means")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "elbow_method.png"), dpi=150, facecolor="#0a0a0a")
    plt.close()

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouettes[optimal_k]

    # Perfiles por cluster para nombrarlos en la sustentación
    df_model = df_model.copy()
    df_model["cluster"] = labels
    profiles = {}
    cluster_labels = {}
    for cid in sorted(df_model["cluster"].unique()):
        subset = df_model[df_model["cluster"] == cid]
        profile = {
            "Compound_encoded_mean": round(float(subset["Compound_encoded"].mean()), 3),
            "TyreLife_mean": round(float(subset["TyreLife"].mean()), 2),
            "LapTime_normalized_mean": round(float(subset["LapTime_normalized"].mean()), 4),
            "Cumulative_Degradation_mean": round(float(subset["Cumulative_Degradation"].mean()), 3),
            "count": int(len(subset)),
        }
        label = _name_cluster(profile)
        profiles[int(cid)] = profile
        cluster_labels[int(cid)] = label
        print(f"Cluster {cid} ({label}): {profile}")

    # Proyección PCA 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1], c=labels, cmap="RdYlGn_r", alpha=0.4, s=8
    )
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"K-Means k={optimal_k} — proyección PCA 2D")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "kmeans_clusters.png"), dpi=150, facecolor="#0a0a0a")
    plt.close()

    joblib.dump(kmeans, os.path.join(ml_dir, "kmeans_model.pkl"))
    joblib.dump(scaler, os.path.join(ml_dir, "kmeans_scaler.pkl"))
    joblib.dump(pca, os.path.join(ml_dir, "kmeans_pca.pkl"))

    trained_at = datetime.now(timezone.utc).isoformat()
    metrics = {
        "algorithm": "KMeans",
        "features": KMEANS_FEATURES,
        "optimal_k": optimal_k,
        "silhouette_score": sil_score,
        "inertias": {str(k): round(v, 2) for k, v in zip(K_RANGE, inertias)},
        "silhouettes": {str(k): v for k, v in silhouettes.items()},
        "cluster_profiles": profiles,
        "cluster_labels": cluster_labels,
        "n_total": len(df_model),
        "n_removed": raw_count - len(df),
        "training_source": training_source,
        "trained_at": trained_at,
        "objective": "Segmentar estrategias de stint para recomendaciones de pit stop",
    }
    if azure_run_id:
        metrics["azure_run_id"] = azure_run_id

    with open(os.path.join(ml_dir, "metrics_clustering.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Silhouette score: {sil_score}")
    print(f"Guardado: {ml_dir}/kmeans_model.pkl, {data_dir}/elbow_method.png")

    return metrics


def log_mlflow_metrics(metrics: dict) -> None:
    import mlflow

    mlflow.log_metric("kmeans_silhouette", metrics["silhouette_score"])
    mlflow.log_param("kmeans_optimal_k", metrics["optimal_k"])


if __name__ == "__main__":
    train()
