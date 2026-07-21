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
    get_recent_alerts,
    get_recent_predictions,
    init_db,
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

    init_db()

    try:
        simulation.load()
        print(f"Simulation loaded — {simulation.total_laps} laps ({simulation.laps_df['race_name'].iloc[0]})")
    except Exception as exc:
        print(f"Simulation data not loaded: {exc}")

    model_path = 'ml/model.pkl'
    scaler_path = 'ml/scaler.pkl'
    metrics_path = 'ml/metrics.json'

    if not all(os.path.exists(p) for p in [model_path, scaler_path, metrics_path]):
        print("Run python ml/train_model.py first")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(metrics_path, encoding='utf-8') as f:
        metrics_data = json.load(f)
    available_features = metrics_data.get('features', [])
    print(f"Model loaded — {len(available_features)} features, R²={metrics_data.get('r2')}")


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
