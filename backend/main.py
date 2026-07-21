import base64
import json
import math
import os
import random
import subprocess
import threading
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from database import (
    get_db_status,
    get_recent_alerts,
    get_recent_predictions,
    init_db,
    insert_classification,
    insert_cluster_assignment,
    insert_safety_event,
)

BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
RENDER_OUTPUT = "blender/render_output.png"
RENDER_SCRIPT = "blender/render.py"

app = FastAPI(
    title="Red Bull Quantum-Twin 2026",
    description="F1 Digital Twin — ML predictions & superclipping API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
scaler = None
metrics_data = None
available_features: list[str] = []

# Modelos adicionales — carga independiente (un fallo no tumba el servidor)
decision_tree_model = None
dt_metrics = None
kmeans_model = None
kmeans_scaler = None
kmeans_metrics = None
svm_model = None
svm_scaler = None
svm_metrics = None

COMPOUND_MAP = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}


class RaceState:
    def __init__(self):
        self.soc = 68.2
        self.lap = 42

    def tick(self):
        self.soc = max(30, self.soc - random.uniform(0, 0.01))
        return self


race_state = RaceState()


def _calc_superclipping(row) -> tuple[float, str]:
    """Derive superclipping duration and ERS status from lap strategy data."""
    tyre = float(row.get('Normalized_TyreLife', 0) or 0)
    progress = float(row.get('RaceProgress', 0) or 0)
    delta = float(row.get('LapTime_Delta', 0) or 0)
    tyre_life = float(row.get('TyreLife', 1) or 1)

    risk = tyre * 0.35 + progress * 0.30 + max(0.0, -delta) * 0.20 + (tyre_life / 40.0) * 0.15
    duration = round(2.0 + risk * 4.5, 2)
    duration = max(1.5, min(7.5, duration))

    if 2.0 <= duration <= 4.0:
        status = "NORMAL"
    elif 4.0 < duration <= 5.5:
        status = "ALERTA"
    else:
        status = "CRÍTICO"
    return duration, status


def _ers_risk_from_row(row) -> float:
    """Calcula ERS_risk con la misma fórmula del pipeline (aprox. para inferencia en vivo)."""
    tyre = float(row.get("Normalized_TyreLife", 0) or 0)
    progress = float(row.get("RaceProgress", 0) or 0)
    delta = float(row.get("LapTime_Delta", 0) or 0)
    raw = (tyre * 2.0) + (progress * 1.5) + (max(-5, min(5, delta)) * -1.0)
    return round(max(0.0, min(10.0, raw * 0.65 + 2.5)), 2)


def _laptime_normalized(row) -> float:
    """Desviación aproximada respecto a baseline de circuito (~78s Monaco)."""
    laptime = float(row.get("LapTime (s)", 78) or 78)
    return round(laptime - 78.0, 4)


def _compound_encoded(row) -> int:
    return COMPOUND_MAP.get(str(row.get("Compound", "MEDIUM")), 1)


def _load_model_safe(path: str, label: str):
    """Carga un .pkl con manejo de errores individual."""
    try:
        if os.path.exists(path):
            return joblib.load(path)
        print(f"  [{label}] No encontrado: {path}")
    except Exception as exc:
        print(f"  [{label}] Error al cargar: {exc}")
    return None


def _load_json_safe(path: str, label: str) -> dict | None:
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    except Exception as exc:
        print(f"  [{label}] Error al leer JSON: {exc}")
    return None


def _ers_color(status: str) -> str:
    return {
        "NORMAL": "#00ff88",
        "ALERTA": "#ffcc00",
        "CRÍTICO": "#cc0000",
    }.get(status, "#cc0000")


def _load_simulation_laps() -> pd.DataFrame:
    strategy_path = "data/f1_strategy.csv"
    telemetry_path = "data/f1_telemetry.csv"

    if not os.path.exists(strategy_path):
        raise FileNotFoundError(f"Missing CSV: {strategy_path}")

    strategy = pd.read_csv(strategy_path)
    ver = strategy[strategy["Driver"] == "VER"].copy()
    if ver.empty:
        raise ValueError("No VER (Verstappen) data found in f1_strategy.csv")

    monaco = ver[ver["Race"].str.contains("Monaco", case=False, na=False)]
    race_name = monaco["Race"].iloc[0] if not monaco.empty else ver["Race"].value_counts().idxmax()
    laps = ver[ver["Race"] == race_name].sort_values("LapNumber")
    laps = laps.groupby("LapNumber", as_index=False).first()

    if os.path.exists(telemetry_path):
        telem = pd.read_csv(telemetry_path)
        telem = telem[(telem["Team"] == "Red Bull Racing") & (telem["Driver"] == "VER")]
        if not telem.empty:
            telem_lap = telem.groupby("LapNumber", as_index=False).agg({
                "SpeedFL": "mean",
                "SpeedST": "mean",
            })
            laps = laps.merge(telem_lap, on="LapNumber", how="left")

    durations, statuses = [], []
    for _, row in laps.iterrows():
        duration, status = _calc_superclipping(row)
        durations.append(duration)
        statuses.append(status)
    laps["superclipping_duration"] = durations
    laps["ers_status"] = statuses
    laps["race_name"] = race_name
    return laps


class SimulationState:
    def __init__(self):
        self._lock = threading.Lock()
        self.running = False
        self.current_lap_index = 0
        self.laps_df: pd.DataFrame | None = None
        self.total_laps = 0
        self.last_lap_snapshot: dict | None = None

    def load(self):
        self.laps_df = _load_simulation_laps()
        self.total_laps = int(len(self.laps_df))

    def start(self) -> str:
        with self._lock:
            if self.running:
                return "already running"
            if self.laps_df is None or self.laps_df.empty:
                raise RuntimeError("Simulation laps not loaded")
            self.current_lap_index = 0
            self.last_lap_snapshot = None
            self.running = True
            return "started"

    def stop(self):
        with self._lock:
            self.running = False
            self.current_lap_index = 0
            self.last_lap_snapshot = None

    def get_lap(self) -> dict:
        with self._lock:
            if not self.running or self.laps_df is None or self.laps_df.empty:
                return {"running": False}

            if self.current_lap_index >= len(self.laps_df):
                self.running = False
                return {"running": False}

            row = self.laps_df.iloc[self.current_lap_index]
            lap_number = int(row["LapNumber"])
            compound = str(row.get("Compound", "MEDIUM"))
            position = int(row.get("Position", 1))
            ers_status = str(row.get("ers_status", "NORMAL"))
            superclipping_duration = float(row.get("superclipping_duration", 3.0))

            payload = {
                "running": True,
                "lap_number": lap_number,
                "total_laps": self.total_laps,
                "position": position,
                "compound": compound,
                "ers_status": ers_status,
                "superclipping_duration": superclipping_duration,
            }
            self.last_lap_snapshot = {**payload, "_row": row}
            self.current_lap_index += 1

            if self.current_lap_index >= len(self.laps_df):
                self.running = False

            return payload

    def is_active(self) -> bool:
        with self._lock:
            return self.running or self.last_lap_snapshot is not None

    def current_row(self):
        with self._lock:
            if self.last_lap_snapshot and "_row" in self.last_lap_snapshot:
                return self.last_lap_snapshot["_row"]
            if self.laps_df is not None and not self.laps_df.empty:
                idx = min(self.current_lap_index, len(self.laps_df) - 1)
                return self.laps_df.iloc[idx]
            return None


simulation = SimulationState()


@app.on_event("startup")
async def startup():
    global model, scaler, metrics_data, available_features
    global decision_tree_model, dt_metrics
    global kmeans_model, kmeans_scaler, kmeans_metrics
    global svm_model, svm_scaler, svm_metrics

    init_db()

    try:
        simulation.load()
        print(f"Simulation loaded — {simulation.total_laps} laps ({simulation.laps_df['race_name'].iloc[0]})")
    except Exception as exc:
        print(f"Simulation data not loaded: {exc}")

    # ── Modelo 1: Regresión (RF) ──────────────────────────
    model_path = "ml/model.pkl"
    scaler_path = "ml/scaler.pkl"
    metrics_path = "ml/metrics.json"

    if all(os.path.exists(p) for p in [model_path, scaler_path, metrics_path]):
        model = _load_model_safe(model_path, "Regresión RF")
        scaler = _load_model_safe(scaler_path, "Scaler RF")
        metrics_data = _load_json_safe(metrics_path, "metrics.json")
        if metrics_data:
            available_features = metrics_data.get("features", [])
            print(f"Regresión cargada — {len(available_features)} features, R²={metrics_data.get('r2')}")
    else:
        print("Regresión no disponible — ejecuta: python ml/train_model.py")

    # ── Modelo 2: Árbol de Decisión ───────────────────────
    decision_tree_model = _load_model_safe("ml/decision_tree_model.pkl", "Árbol de Decisión")
    dt_metrics = _load_json_safe("ml/metrics_classification.json", "metrics_classification")
    if decision_tree_model:
        acc = dt_metrics.get("accuracy") if dt_metrics else "?"
        print(f"Árbol de Decisión cargado — accuracy={acc}")

    # ── Modelo 3: K-Means ─────────────────────────────────
    kmeans_model = _load_model_safe("ml/kmeans_model.pkl", "K-Means")
    kmeans_scaler = _load_model_safe("ml/kmeans_scaler.pkl", "K-Means Scaler")
    kmeans_metrics = _load_json_safe("ml/metrics_clustering.json", "metrics_clustering")
    if kmeans_model:
        k = kmeans_metrics.get("optimal_k") if kmeans_metrics else "?"
        print(f"K-Means cargado — k={k}")

    # ── Modelo 4: SVM ─────────────────────────────────────
    svm_model = _load_model_safe("ml/svm_model.pkl", "SVM")
    svm_scaler = _load_model_safe("ml/svm_scaler.pkl", "SVM Scaler")
    svm_metrics = _load_json_safe("ml/metrics_svm.json", "metrics_svm")
    if svm_model:
        kernel = svm_metrics.get("best_kernel") if svm_metrics else "?"
        print(f"SVM cargado — kernel={kernel}")


@app.get("/")
def root():
    return {
        "project": "Red Bull Quantum-Twin 2026",
        "status": "online",
        "model_loaded": model is not None,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/api/db-status")
def db_status():
    return {"database": get_db_status()}


@app.get("/api/model/source")
def model_source():
    """Origen y fecha del último entrenamiento de cada uno de los 4 modelos."""
    consolidated = _load_json_safe("ml/models_source.json", "models_source")
    if consolidated:
        return consolidated

    result = {}
    specs = [
        ("regression", "ml/metrics.json", "model_source.json", "r2", "training_source"),
        ("decision_tree", "ml/metrics_classification.json", None, "accuracy", "training_source"),
        ("kmeans", "ml/metrics_clustering.json", None, "silhouette_score", "training_source"),
        ("svm", "ml/metrics_svm.json", None, "best_recall", "training_source"),
    ]
    for name, metrics_file, legacy_file, metric_key, source_key in specs:
        m = _load_json_safe(metrics_file, name)
        if m:
            result[name] = {
                "source": m.get(source_key, "local"),
                "trained_at": m.get("trained_at"),
                "metric": m.get(metric_key),
            }
        elif legacy_file and name == "regression" and os.path.exists(f"ml/{legacy_file}"):
            with open(f"ml/{legacy_file}", encoding="utf-8") as f:
                legacy = json.load(f)
            result[name] = {
                "source": legacy.get("source", "local"),
                "trained_at": legacy.get("trained_at"),
                "metric": legacy.get("r2"),
            }

    if not result:
        raise HTTPException(
            status_code=503,
            detail="Información de modelos no disponible. Ejecuta los scripts en ml/.",
        )
    return result


@app.get("/api/metrics")
def get_metrics():
    if metrics_data is None:
        raise HTTPException(
            status_code=503,
            detail="Metrics not available. Run python ml/train_model.py first",
        )
    return metrics_data


@app.get("/api/results")
def get_results():
    results_path = 'data/results.csv'
    if not os.path.exists(results_path):
        raise HTTPException(
            status_code=503,
            detail="Results not available. Run python ml/train_model.py first",
        )
    df = pd.read_csv(results_path).head(50)
    return df.to_dict(orient='records')


@app.get("/api/history/predictions")
def history_predictions():
    return get_recent_predictions(10)


@app.get("/api/history/alerts")
def history_alerts():
    return get_recent_alerts(10)


@app.post("/api/simulation/start")
def simulation_start():
    if simulation.laps_df is None:
        raise HTTPException(
            status_code=503,
            detail="Simulation data unavailable. Ensure data/f1_strategy.csv exists with VER driver data.",
        )
    try:
        status = simulation.start()
        return {"status": status}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/simulation/stop")
def simulation_stop():
    simulation.stop()
    return {"status": "stopped"}


@app.get("/api/simulation/lap")
def simulation_lap():
    if simulation.laps_df is None:
        raise HTTPException(
            status_code=503,
            detail="Simulation data unavailable. Ensure data/f1_strategy.csv exists.",
        )
    return simulation.get_lap()


def _sim_row():
    if not simulation.running:
        return None
    row = simulation.current_row()
    if row is None and simulation.laps_df is not None and not simulation.laps_df.empty:
        return simulation.laps_df.iloc[min(simulation.current_lap_index, len(simulation.laps_df) - 1)]
    return row


@app.get("/api/telemetry")
def get_telemetry():
    row = _sim_row()
    if row is not None:
        speed = float(row.get("SpeedFL") or row.get("SpeedST") or 280)
        throttle = max(0, min(100, 55 + (speed - 200) * 0.25))
        brake = max(0, min(100, 100 - throttle * 0.8))
        progress = float(row.get("RaceProgress", 0.5) or 0.5)
        soc = max(30, 85 - progress * 40)
        t = time.time()
        return {
            "timestamp": round(t, 3),
            "current": {
                "velocidad_kmh": round(speed, 1),
                "soc_bateria": round(soc, 1),
                "throttle_pct": round(throttle, 1),
                "brake_pressure": round(brake, 1),
                "mguk_kw": round(280 + throttle * 0.8, 1),
                "aero_mode": "STRAIGHT MODE" if speed > 250 else "CORNER MODE",
                "temp_bateria": round(38 + float(row.get("TyreLife", 10)) * 0.4, 1),
            },
            "history": {
                "velocidad": [{"t": i, "v": round(speed + random.uniform(-10, 10), 1)} for i in range(20)],
                "soc": [{"t": i, "v": round(soc + random.uniform(-1, 1), 1)} for i in range(20)],
                "throttle": [{"t": i, "v": round(max(0, min(100, throttle + random.uniform(-8, 8))), 1)} for i in range(20)],
                "brake": [{"t": i, "v": round(max(0, brake + random.uniform(-3, 3)), 1)} for i in range(20)],
            }
        }

    state = race_state.tick()
    t = time.time()
    base_speed = 280 + 60 * abs(math.sin(t * 0.3))
    throttle = max(0, min(100, 80 + 20 * math.sin(t * 0.5)))
    brake = max(0, 40 * abs(math.sin(t * 0.4 + 1)) if throttle < 50 else 0)
    mguk = 300 + 50 * abs(math.sin(t * 0.2))
    return {
        "timestamp": round(t, 3),
        "current": {
            "velocidad_kmh": round(base_speed, 1),
            "soc_bateria": round(state.soc, 1),
            "throttle_pct": round(throttle, 1),
            "brake_pressure": round(brake, 1),
            "mguk_kw": round(mguk, 1),
            "aero_mode": "STRAIGHT MODE" if base_speed > 250 else "CORNER MODE",
            "temp_bateria": round(35 + 20 * abs(math.sin(t * 0.1)), 1),
        },
        "history": {
            "velocidad": [{"t": i, "v": round(base_speed + random.uniform(-30, 30), 1)} for i in range(20)],
            "soc": [{"t": i, "v": round(state.soc + (20 - i) * 0.15 + random.uniform(-1, 1), 1)} for i in range(20)],
            "throttle": [{"t": i, "v": round(max(0, min(100, throttle + random.uniform(-15, 15))), 1)} for i in range(20)],
            "brake": [{"t": i, "v": round(max(0, brake + random.uniform(-5, 5)), 1)} for i in range(20)],
        }
    }


@app.get("/api/superclipping")
def get_superclipping():
    row = _sim_row()
    if row is not None:
        predicted = float(row.get("superclipping_duration", 3.0))
        status = str(row.get("ers_status", "NORMAL"))
        return {
            "predicted_duration_s": predicted,
            "status": status,
            "color": _ers_color(status),
            "log": [
                {"time": f"LAP {int(row.get('LapNumber', 0))}", "event": "SIM_LAP_REPLAY", "type": "normal"},
                {"time": "T-12.8s", "event": "CLIPPING_NOMINAL", "type": "normal"},
                {"time": "T-25.1s", "event": "AERO_BALANCE_ADJ", "type": "warning"},
            ]
        }

    t = time.time()
    predicted = round(2.5 + abs(math.sin(t * 0.2)) * 3.5, 2)
    if 2.0 <= predicted <= 4.0:
        status = "NORMAL"
    elif 4.0 < predicted <= 5.5:
        status = "ALERTA"
    else:
        status = "CRÍTICO"
    return {
        "predicted_duration_s": predicted,
        "status": status,
        "color": _ers_color(status),
        "log": [
            {"time": "T-04.2s", "event": "MGU-H SYNC_ERR", "type": "error"},
            {"time": "T-12.8s", "event": "CLIPPING_NOMINAL", "type": "normal"},
            {"time": "T-25.1s", "event": "AERO_BALANCE_ADJ", "type": "warning"},
            {"time": "T-48.0s", "event": "ERS_SPIKE_DET", "type": "error"},
        ]
    }


@app.get("/api/race")
def get_race():
    row = _sim_row()
    if row is not None:
        lap = int(row.get("LapNumber", 1))
        position = int(row.get("Position", 1))
        gap_ahead = round(0.5 + float(row.get("LapTime_Delta", 0) or 0) * 0.1, 3)
        gap_behind = round(0.8 + abs(float(row.get("Position_Change", 0) or 0)) * 0.05, 3)
        return {
            "lap": lap,
            "total_laps": simulation.total_laps,
            "position": position,
            "sector": (lap % 3) + 1,
            "gap_ahead": f"+{max(0.001, gap_ahead)}s",
            "gap_behind": f"-{max(0.001, gap_behind)}s",
            "leaderboard": [
                {"pos": 1, "driver": "VERSTAPPEN", "team": "redbull", "gap": "—"},
                {"pos": position, "driver": "QUANTUM-26", "team": "redbull", "gap": f"+{gap_ahead}", "highlight": True},
                {"pos": position + 1, "driver": "LECLERC", "team": "ferrari", "gap": f"+{round(gap_ahead + gap_behind, 3)}"},
                {"pos": position + 2, "driver": "NORRIS", "team": "mclaren", "gap": f"+{round(gap_ahead + gap_behind + 1.2, 3)}"},
                {"pos": position + 3, "driver": "HAMILTON", "team": "mercedes", "gap": f"+{round(gap_ahead + gap_behind + 3.1, 3)}"},
            ]
        }

    t = time.time()
    gap_ahead = round(1.243 + math.sin(t * 0.1) * 0.2, 3)
    gap_behind = round(0.891 + math.sin(t * 0.15) * 0.15, 3)
    return {
        "lap": 42, "total_laps": 78, "position": 2, "sector": 2,
        "gap_ahead": f"+{gap_ahead}s",
        "gap_behind": f"-{gap_behind}s",
        "leaderboard": [
            {"pos": 1, "driver": "VERSTAPPEN", "team": "redbull", "gap": "—"},
            {"pos": 2, "driver": "QUANTUM-26", "team": "redbull", "gap": f"+{gap_ahead}", "highlight": True},
            {"pos": 3, "driver": "LECLERC", "team": "ferrari", "gap": f"+{round(gap_ahead + gap_behind, 3)}"},
            {"pos": 4, "driver": "NORRIS", "team": "mclaren", "gap": f"+{round(gap_ahead + gap_behind + 1.2, 3)}"},
            {"pos": 5, "driver": "HAMILTON", "team": "mercedes", "gap": f"+{round(gap_ahead + gap_behind + 3.1, 3)}"},
        ]
    }


@app.get("/api/tires")
def get_tires():
    row = _sim_row()
    if row is not None:
        t = time.time()
        tyre_life = int(float(row.get("TyreLife", 10)))
        wear = min(95, max(5, int(tyre_life * 2.5)))
        compound = str(row.get("Compound", "MEDIUM"))
        return {
            "compound": compound,
            "lap_on": tyre_life,
            "tires": {
                "FL": {"temp": round(92 + math.sin(t * 0.2) * 3, 1), "wear": wear},
                "FR": {"temp": round(104 + math.sin(t * 0.3) * 2, 1), "wear": max(5, wear - 4)},
                "RL": {"temp": round(88 + math.sin(t * 0.15) * 2, 1), "wear": min(95, wear + 6)},
                "RR": {"temp": round(89 + math.sin(t * 0.25) * 3, 1), "wear": min(95, wear + 5)},
            }
        }

    t = time.time()
    return {
        "compound": "MEDIUM", "lap_on": 28,
        "tires": {
            "FL": {"temp": round(92 + math.sin(t * 0.2) * 3, 1), "wear": 64},
            "FR": {"temp": round(104 + math.sin(t * 0.3) * 2, 1), "wear": 58},
            "RL": {"temp": round(88 + math.sin(t * 0.15) * 2, 1), "wear": 72},
            "RR": {"temp": round(89 + math.sin(t * 0.25) * 3, 1), "wear": 71},
        }
    }


@app.get("/api/recommendations")
def get_recommendations():
    return {"recommendations": [
        {"severity": "CRÍTICO", "title": "SOC CRITICAL — DEEP DISCHARGE RISK",
         "description": "Immediate deployment reduction suggested for Sector 3.", "time_ago": "T-10s"},
        {"severity": "ALERTA", "title": "TIRE WEAR THRESHOLD REACHED",
         "description": "Front-left graining detected. Adjust brake bias +2%.", "time_ago": "T-45s"},
        {"severity": "INFO", "title": "MGU-K REDUCTION OPTIMIZED",
         "description": "Efficiency gain +0.4s/lap predicted with Map 4 profile.", "time_ago": "T-2m"},
    ]}


@app.get("/api/strategy")
def get_strategy():
    return {
        "pit_window": {"lap_start": 45, "lap_end": 48},
        "predicted_finish": ["P2", "P3", "P1"],
        "model": "UNDERCUT_MODEL_V4",
        "stints": [
            {"label": "STINT 1", "compound": "MED", "laps": 42},
            {"label": "PIT", "compound": None, "laps": None},
            {"label": "STINT 2", "compound": "SOFT", "laps": 33},
        ]
    }


def _inference_row():
    """Fila de telemetría actual para inferencia ML (simulación o valores demo)."""
    row = _sim_row()
    if row is not None:
        return row
    return pd.Series({
        "LapNumber": race_state.lap,
        "Normalized_TyreLife": 0.65,
        "RaceProgress": 0.54,
        "LapTime_Delta": 0.12,
        "TyreLife": 28,
        "Compound": "MEDIUM",
        "LapTime (s)": 78.5,
        "Cumulative_Degradation": 12.0,
        "Race": "Monaco Grand Prix",
    })


@app.get("/api/risk-classification")
def risk_classification():
    """Clasificación ERS_status del Árbol de Decisión con probabilidades."""
    if decision_tree_model is None:
        raise HTTPException(
            status_code=503,
            detail="Árbol de Decisión no cargado. Ejecuta: python ml/train_decision_tree.py",
        )

    row = _inference_row()
    features = [[
        float(row.get("Normalized_TyreLife", 0)),
        float(row.get("RaceProgress", 0)),
        float(row.get("LapTime_Delta", 0)),
        float(row.get("TyreLife", 0)),
        float(_compound_encoded(row)),
    ]]

    pred = decision_tree_model.predict(features)[0]
    prob_normal, prob_alerta, prob_critico = 0.0, 0.0, 0.0
    if hasattr(decision_tree_model, "predict_proba"):
        class_prob_map = {"NORMAL": "normal", "ALERTA": "alerta", "CRÍTICO": "critico"}
        for cls, prob in zip(decision_tree_model.classes_, decision_tree_model.predict_proba(features)[0]):
            if cls == "NORMAL":
                prob_normal = round(float(prob), 4)
            elif cls == "ALERTA":
                prob_alerta = round(float(prob), 4)
            elif cls == "CRÍTICO":
                prob_critico = round(float(prob), 4)

    response = {
        "predicted_class": str(pred),
        "probabilities": {
            "NORMAL": prob_normal,
            "ALERTA": prob_alerta,
            "CRÍTICO": prob_critico,
        },
        "color": _ers_color(str(pred)),
        "lap": int(row.get("LapNumber", 0)),
        "accuracy_training": dt_metrics.get("accuracy") if dt_metrics else None,
    }

    try:
        insert_classification({
            "ers_status_predicted": str(pred),
            "prob_normal": prob_normal,
            "prob_alerta": prob_alerta,
            "prob_critico": prob_critico,
            "lap": int(row.get("LapNumber", 0)),
            "race": str(row.get("Race", "Unknown")),
        })
    except Exception as exc:
        print(f"WARNING: No se pudo persistir clasificación en DB: {exc}")

    return response


@app.get("/api/strategy-clusters")
def strategy_clusters():
    """Cluster K-Means actual + interpretación textual del perfil de stint."""
    if kmeans_model is None or kmeans_scaler is None:
        raise HTTPException(
            status_code=503,
            detail="K-Means no cargado. Ejecuta: python ml/train_kmeans.py",
        )

    row = _inference_row()
    features = [[
        float(row.get("Cumulative_Degradation", 0)),
        float(row.get("TyreLife", 0)),
        float(_laptime_normalized(row)),
    ]]
    scaled = kmeans_scaler.transform(features)
    cluster_id = int(kmeans_model.predict(scaled)[0])

    labels = (kmeans_metrics or {}).get("cluster_labels", {})
    profiles = (kmeans_metrics or {}).get("cluster_profiles", {})
    label = labels.get(str(cluster_id), labels.get(cluster_id, f"Cluster {cluster_id}"))
    profile = profiles.get(str(cluster_id), profiles.get(cluster_id, {}))

    interpretation = (
        f"Vuelta {int(row.get('LapNumber', 0))}: perfil '{label}'. "
        f"TyreLife promedio del cluster: {profile.get('TyreLife_mean', 'N/A')}, "
        f"degradación acumulada: {profile.get('Cumulative_Degradation_mean', 'N/A')}."
    )

    response = {
        "cluster_id": cluster_id,
        "cluster_label": label,
        "interpretation": interpretation,
        "profile": profile,
        "silhouette_score": kmeans_metrics.get("silhouette_score") if kmeans_metrics else None,
        "optimal_k": kmeans_metrics.get("optimal_k") if kmeans_metrics else None,
    }

    try:
        insert_cluster_assignment({
            "cluster_id": cluster_id,
            "cluster_label": label,
            "tyre_life": float(row.get("TyreLife", 0)),
            "race_progress": float(row.get("RaceProgress", 0)),
            "lap": int(row.get("LapNumber", 0)),
        })
    except Exception as exc:
        print(f"WARNING: No se pudo persistir cluster en DB: {exc}")

    return response


@app.get("/api/safety")
def safety():
    """Detección de evento crítico vía SVM (prioriza recall en clase crítica)."""
    if svm_model is None or svm_scaler is None:
        raise HTTPException(
            status_code=503,
            detail="SVM no cargado. Ejecuta: python ml/train_svm.py",
        )

    row = _inference_row()
    ers_risk = _ers_risk_from_row(row)
    features = [[
        float(row.get("LapTime_Delta", 0)),
        ers_risk,
        float(row.get("RaceProgress", 0)),
        float(row.get("TyreLife", 0)),
    ]]
    scaled = svm_scaler.transform(features)
    pred = int(svm_model.predict(scaled)[0])
    prob_crit = 0.0
    if hasattr(svm_model, "predict_proba"):
        prob_crit = round(float(svm_model.predict_proba(scaled)[0][1]), 4)

    status = "CRÍTICO" if pred == 1 else "NORMAL"
    response = {
        "status": status,
        "is_critical": bool(pred),
        "probability_critical": prob_crit,
        "kernel": svm_metrics.get("best_kernel") if svm_metrics else None,
        "lap": int(row.get("LapNumber", 0)),
        "ers_risk": ers_risk,
        "note": (svm_metrics or {}).get(
            "target_rule",
            "Aproximación sin etiquetas reales de F1",
        ),
    }

    try:
        insert_safety_event({
            "is_critical": pred,
            "probability_critical": prob_crit,
            "kernel": svm_metrics.get("best_kernel", "unknown") if svm_metrics else "unknown",
            "lap": int(row.get("LapNumber", 0)),
            "ers_risk": ers_risk,
        })
    except Exception as exc:
        print(f"WARNING: No se pudo persistir evento SVM en DB: {exc}")

    return response


@app.get("/api/render")
def get_render(ers_status: str = "NORMAL"):
    try:
        os.makedirs("blender", exist_ok=True)
        result = subprocess.run([
            BLENDER_PATH,
            "--background",
            "--python", RENDER_SCRIPT,
            "--", ers_status, RENDER_OUTPUT
        ], capture_output=True, text=True, timeout=60)

        if not os.path.exists(RENDER_OUTPUT):
            return {"error": "Render failed", "details": result.stderr[-500:]}

        with open(RENDER_OUTPUT, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()

        return {
            "image": f"data:image/png;base64,{img_data}",
            "ers_status": ers_status,
            "timestamp": time.time()
        }
    except subprocess.TimeoutExpired:
        return {"error": "Render timeout — Blender took too long"}
    except Exception as e:
        return {"error": str(e)}
