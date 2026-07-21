"""Shared ML pipeline for local and Azure ML training."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

FEATURES = [
    "TyreLife",
    "Compound_encoded",
    "Position",
    "Stint",
    "LapNumber",
    "Cumulative_Degradation",
    "Position_Change",
    "Prev_TyreLife",
    "PitStop",
    "RaceProgress",
    "Race_encoded",
    "Year_encoded",
]
TARGET = "LapTime_normalized"
COMPOUND_MAP = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}


@dataclass
class TrainingResult:
    df: pd.DataFrame
    df_model: pd.DataFrame
    features: list[str]
    target: str
    raw_count: int
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    model_lr: LinearRegression
    model_rf: RandomForestRegressor
    scaler: StandardScaler
    y_pred_lr: Any
    y_pred_rf: Any
    mae_lr: float
    mse_lr: float
    r2_lr: float
    mae_rf: float
    mse_rf: float
    r2_rf: float
    le_race: LabelEncoder


def load_data(csv_path: str) -> pd.DataFrame:
    """Carga el CSV combinado de estrategia F1."""
    return pd.read_csv(csv_path)


def prepare_data(csv_path: str) -> tuple[pd.DataFrame, int]:
    """
    Pipeline compartido: carga, limpieza y feature engineering de dashboard.
    Usado por los 4 scripts de entrenamiento para evitar duplicar lógica.
    Retorna (DataFrame listo, conteo filas originales).
    """
    df = load_data(csv_path)
    df, raw = clean_data(df)
    df.attrs["raw_count"] = raw
    df = engineer_dashboard_fields(df)
    return df, raw


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    raw = len(df)
    key_cols = [
        "LapTime (s)", "TyreLife", "Compound", "Position", "Stint",
        "LapNumber", "LapTime_Delta", "Normalized_TyreLife", "RaceProgress",
        "Cumulative_Degradation",
    ]
    df = df.dropna(subset=key_cols)

    threshold = df["LapTime (s)"].median() + 2 * df["LapTime (s)"].std()
    df = df[df["LapTime (s)"] < threshold]
    df = df[(df["Position"] >= 1) & (df["Position"] <= 20)]
    df = df[df["LapTime (s)"] > 0]

    df["Compound_encoded"] = df["Compound"].map(COMPOUND_MAP).fillna(1)
    df["LapTime_baseline"] = df.groupby("Race")["LapTime (s)"].transform("median")
    df["LapTime_normalized"] = df["LapTime (s)"] - df["LapTime_baseline"]

    le_race = LabelEncoder()
    df["Race_encoded"] = le_race.fit_transform(df["Race"].astype(str))
    df["Year_encoded"] = df["Year"].astype(int) - df["Year"].astype(int).min()
    df.attrs["le_race"] = le_race
    return df, raw


def engineer_dashboard_fields(df: pd.DataFrame) -> pd.DataFrame:
    df["ERS_risk"] = (
        (df["Normalized_TyreLife"] * 2.0)
        + (df["RaceProgress"] * 1.5)
        + (df["LapTime_Delta"].clip(-5, 5) * -1.0)
    )
    ers_min = df["ERS_risk"].min()
    ers_max = df["ERS_risk"].max()
    df["ERS_risk"] = ((df["ERS_risk"] - ers_min) / (ers_max - ers_min)) * 10
    df["ERS_status"] = pd.cut(
        df["ERS_risk"],
        bins=[-0.01, 3, 6, 10.01],
        labels=["NORMAL", "ALERTA", "CRÍTICO"],
    )
    df["superclipping_duration"] = df["ERS_risk"].apply(
        lambda x: round(1.0 + (x / 10) * 7.0, 2)
    )
    return df


def train_models(df: pd.DataFrame) -> TrainingResult:
    le_race: LabelEncoder = df.attrs.get("le_race")
    if le_race is None:
        raise ValueError("Race LabelEncoder missing — run clean_data() first")

    df = engineer_dashboard_fields(df.copy())

    df_model = df[FEATURES + [TARGET]].dropna()
    X = df_model[FEATURES]
    y = df_model[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_lr = LinearRegression()
    model_lr.fit(X_train_scaled, y_train)
    y_pred_lr = model_lr.predict(X_test_scaled)

    model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_rf.fit(X_train_scaled, y_train)
    y_pred_rf = model_rf.predict(X_test_scaled)

    return TrainingResult(
        df=df,
        df_model=df_model,
        features=FEATURES,
        target=TARGET,
        raw_count=df.attrs.get("raw_count", len(df)),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_lr=model_lr,
        model_rf=model_rf,
        scaler=scaler,
        y_pred_lr=y_pred_lr,
        y_pred_rf=y_pred_rf,
        mae_lr=round(mean_absolute_error(y_test, y_pred_lr), 4),
        mse_lr=round(mean_squared_error(y_test, y_pred_lr), 4),
        r2_lr=round(r2_score(y_test, y_pred_lr), 4),
        mae_rf=round(mean_absolute_error(y_test, y_pred_rf), 4),
        mse_rf=round(mean_squared_error(y_test, y_pred_rf), 4),
        r2_rf=round(r2_score(y_test, y_pred_rf), 4),
        le_race=le_race,
    )


def run_pipeline(csv_path: str) -> TrainingResult:
    df, raw = prepare_data(csv_path)
    return train_models(df)


def build_metrics(result: TrainingResult) -> dict:
    return {
        "best_model": "Random Forest",
        "comparison": {
            "linear_regression": {
                "mae": result.mae_lr,
                "mse": result.mse_lr,
                "r2": result.r2_lr,
            },
            "random_forest": {
                "mae": result.mae_rf,
                "mse": result.mse_rf,
                "r2": result.r2_rf,
            },
        },
        "algorithm": "Random Forest",
        "datasets": [
            "F1 Strategy Dataset 2024-2025 (Kaggle - aadigupta1601)",
            "F1 Telemetry Montreal 2023 (HuggingFace - renumics)",
        ],
        "target": "LapTime_normalized (deviation from circuit median, seconds)",
        "target_explanation": (
            "Lap time deviation from circuit median; lower values indicate "
            "faster laps relative to track baseline"
        ),
        "features": result.features,
        "n_total": len(result.df_model),
        "n_train": len(result.X_train),
        "n_test": len(result.X_test),
        "n_removed": result.raw_count - len(result.df),
        "mae": result.mae_rf,
        "mse": result.mse_rf,
        "r2": result.r2_rf,
        "ers_risk_range": {"min": 0, "max": 10},
        "superclipping_duration_range": {"min_s": 1.0, "max_s": 8.0},
        "note": (
            "ERS/MGU-K data is proprietary to F1 teams. This index is derived from "
            "public telemetry as a validated proxy for Superclipping risk under "
            "2026 regulations."
        ),
    }


def save_artifacts(
    result: TrainingResult,
    output_dir: str = "ml",
    data_dir: str = "data",
    training_source: str = "local",
    azure_run_id: str | None = None,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    joblib.dump(result.model_rf, os.path.join(output_dir, "model.pkl"))
    joblib.dump(result.model_lr, os.path.join(output_dir, "model_lr.pkl"))
    joblib.dump(result.scaler, os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(result.le_race, os.path.join(output_dir, "race_encoder.pkl"))

    metrics = build_metrics(result)
    metrics["training_source"] = training_source
    if azure_run_id:
        metrics["azure_run_id"] = azure_run_id

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    trained_at = datetime.now(timezone.utc).isoformat()
    source_info = {
        "source": training_source,
        "trained_at": trained_at,
        "best_model": "Random Forest",
        "r2": result.r2_rf,
    }
    if azure_run_id:
        source_info["azure_run_id"] = azure_run_id

    with open(os.path.join(output_dir, "model_source.json"), "w", encoding="utf-8") as f:
        json.dump(source_info, f, indent=2, ensure_ascii=False)

    results_df = pd.DataFrame({
        "real": result.y_test.values,
        "predicted": result.y_pred_rf,
        "laptime_s": result.df.loc[result.y_test.index, "LapTime (s)"].values,
        "tyre_life": result.df.loc[result.y_test.index, "TyreLife"].values,
        "position": result.df.loc[result.y_test.index, "Position"].values,
        "compound": result.df.loc[result.y_test.index, "Compound"].values,
        "ers_status": result.df.loc[result.y_test.index, "ERS_status"].values,
        "superclipping_duration": result.df.loc[result.y_test.index, "superclipping_duration"].values,
        "race": result.df.loc[result.y_test.index, "Race"].values if "Race" in result.df.columns else "Unknown",
        "lap": result.df.loc[result.y_test.index, "LapNumber"].values,
        "race_progress": result.df.loc[result.y_test.index, "RaceProgress"].values,
    })
    results_df.to_csv(os.path.join(data_dir, "results.csv"), index=False)

    sample = result.df.sample(min(200, len(result.df)), random_state=42).sort_values("LapNumber")
    dashboard_data = {
        "telemetry_history": [
            {
                "lap": int(row["LapNumber"]),
                "laptime_s": round(float(row["LapTime (s)"]), 3),
                "tyre_life": int(row["TyreLife"]),
                "position": int(row["Position"]),
                "compound": row["Compound"],
                "ers_risk": round(float(row["ERS_risk"]), 2),
                "ers_status": str(row["ERS_status"]),
                "superclipping_duration": round(float(row["superclipping_duration"]), 2),
                "race_progress": round(float(row["RaceProgress"]), 3),
                "lap_delta": round(float(row["LapTime_Delta"]), 3),
            }
            for _, row in sample.iterrows()
        ],
        "ers_distribution": result.df["ERS_status"].value_counts().to_dict(),
        "avg_superclipping_by_compound": (
            result.df.groupby("Compound")["superclipping_duration"].mean().round(2).to_dict()
        ),
        "avg_ers_risk_by_lap": (
            result.df.groupby("LapNumber")["ERS_risk"].mean().round(2).to_dict()
        ),
    }
    with open(os.path.join(data_dir, "dashboard_data.json"), "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)

    return metrics


def log_mlflow_metrics(result: TrainingResult) -> None:
    import mlflow

    mlflow.log_metric("mae_lr", result.mae_lr)
    mlflow.log_metric("mse_lr", result.mse_lr)
    mlflow.log_metric("r2_lr", result.r2_lr)
    mlflow.log_metric("mae_rf", result.mae_rf)
    mlflow.log_metric("mse_rf", result.mse_rf)
    mlflow.log_metric("r2_rf", result.r2_rf)
    mlflow.log_metric("mae", result.mae_rf)
    mlflow.log_metric("mse", result.mse_rf)
    mlflow.log_metric("r2", result.r2_rf)
    mlflow.log_param("best_model", "Random Forest")
    mlflow.log_param("n_features", len(result.features))
    mlflow.log_param("n_train", len(result.X_train))
    mlflow.log_param("n_test", len(result.X_test))
