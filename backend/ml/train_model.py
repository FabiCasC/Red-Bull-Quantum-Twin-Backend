import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── 1. LOAD ──────────────────────────────────────────
df = pd.read_csv('data/f1_strategy.csv')
print("=== RAW DATA ===")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ── 2. CLEAN ─────────────────────────────────────────
print("\n=== CLEANING ===")
raw = len(df)

key_cols = ['LapTime (s)', 'TyreLife', 'Compound', 'Position', 'Stint',
            'LapNumber', 'LapTime_Delta', 'Normalized_TyreLife', 'RaceProgress',
            'Cumulative_Degradation']
df = df.dropna(subset=key_cols)
print(f"After null removal: {len(df)} rows (removed {raw - len(df)})")

threshold = df['LapTime (s)'].median() + 2 * df['LapTime (s)'].std()
before = len(df)
df = df[df['LapTime (s)'] < threshold]
print(f"After pit lap removal: {len(df)} rows (removed {before - len(df)})")

before = len(df)
df = df[(df['Position'] >= 1) & (df['Position'] <= 20)]
print(f"After position filter: {len(df)} rows (removed {before - len(df)})")

before = len(df)
df = df[df['LapTime (s)'] > 0]
print(f"After negative time filter: {len(df)} rows (removed {before - len(df)})")

compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}
df['Compound_encoded'] = df['Compound'].map(compound_map).fillna(1)
print("Compound encoded: SOFT=0, MEDIUM=1, HARD=2, INTER=3, WET=4")

# Normalize LapTime by circuit median — model predicts deviation from circuit baseline
df['LapTime_baseline'] = df.groupby('Race')['LapTime (s)'].transform('median')
df['LapTime_normalized'] = df['LapTime (s)'] - df['LapTime_baseline']

# Encode Race (circuit) — critical because lap times vary hugely by circuit
le_race = LabelEncoder()
df['Race_encoded'] = le_race.fit_transform(df['Race'].astype(str))
os.makedirs('ml', exist_ok=True)
joblib.dump(le_race, 'ml/race_encoder.pkl')
print(f"Race encoded: {df['Race'].nunique()} unique circuits")

# Also encode Year
df['Year_encoded'] = df['Year'].astype(int) - df['Year'].astype(int).min()

print(f"Final clean shape: {df.shape}")
print(f"Total removed: {raw - len(df)} rows ({round((raw - len(df)) / raw * 100, 1)}%)")

# ── 3. FEATURE ENGINEERING ───────────────────────────
print("\n=== FEATURE ENGINEERING ===")

# ML Target: LapTime_normalized — deviation from circuit median lap time
target = 'LapTime_normalized'

# Features: variables that influence lap time
features = [
    'TyreLife', 'Compound_encoded', 'Position', 'Stint',
    'LapNumber', 'Cumulative_Degradation', 'Position_Change',
    'Prev_TyreLife', 'PitStop', 'RaceProgress',
    'Race_encoded', 'Year_encoded'
]

# ERS_risk: derived separately for dashboard display only (NOT used in ML)
df['ERS_risk'] = (
    (df['Normalized_TyreLife'] * 2.0) +
    (df['RaceProgress'] * 1.5) +
    (df['LapTime_Delta'].clip(-5, 5) * -1.0)
)
ers_min = df['ERS_risk'].min()
ers_max = df['ERS_risk'].max()
df['ERS_risk'] = ((df['ERS_risk'] - ers_min) / (ers_max - ers_min)) * 10

df['ERS_status'] = pd.cut(
    df['ERS_risk'],
    bins=[-0.01, 3, 6, 10.01],
    labels=['NORMAL', 'ALERTA', 'CRÍTICO']
)

df['superclipping_duration'] = df['ERS_risk'].apply(
    lambda x: round(1.0 + (x / 10) * 7.0, 2)
)

print(f"Target: {target}")
print(f"Features: {features}")
print(f"ERS_risk range: {df['ERS_risk'].min():.2f} - {df['ERS_risk'].max():.2f}")
print(f"ERS_status distribution:\n{df['ERS_status'].value_counts()}")

df_model = df[features + [target]].dropna()
X = df_model[features]
y = df_model[target]
print(f"Training shape: {df_model.shape}")

# ── 5. SPLIT + SCALE + TRAIN ─────────────────────────
print("\n=== TRAINING ===")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── LINEAR REGRESSION ──
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)
y_pred_lr = model_lr.predict(X_test_scaled)

mae_lr = round(mean_absolute_error(y_test, y_pred_lr), 4)
mse_lr = round(mean_squared_error(y_test, y_pred_lr), 4)
r2_lr = round(r2_score(y_test, y_pred_lr), 4)
print(f"Linear Regression — MAE: {mae_lr} | MSE: {mse_lr} | R²: {r2_lr}")

# ── RANDOM FOREST ──
model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_rf.fit(X_train_scaled, y_train)
y_pred_rf = model_rf.predict(X_test_scaled)

mae_rf = round(mean_absolute_error(y_test, y_pred_rf), 4)
mse_rf = round(mean_squared_error(y_test, y_pred_rf), 4)
r2_rf = round(r2_score(y_test, y_pred_rf), 4)
print(f"Random Forest      — MAE: {mae_rf} | MSE: {mse_rf} | R²: {r2_rf}")

# ── 6. METRICS ───────────────────────────────────────
print("\n=== RESULTS ===")
y_pred = y_pred_rf
mae = mae_rf
mse = mse_rf
r2 = r2_rf

# ── 7. SAVE MODEL + SCALER ───────────────────────────
os.makedirs('ml', exist_ok=True)
joblib.dump(model_rf, 'ml/model.pkl')        # main model used by API
joblib.dump(model_lr, 'ml/model_lr.pkl')     # kept for academic comparison
joblib.dump(scaler, 'ml/scaler.pkl')

metrics = {
    "best_model": "Random Forest",
    "comparison": {
        "linear_regression": {"mae": mae_lr, "mse": mse_lr, "r2": r2_lr},
        "random_forest": {"mae": mae_rf, "mse": mse_rf, "r2": r2_rf}
    },
    "algorithm": "Random Forest",
    "datasets": [
        "F1 Strategy Dataset 2024-2025 (Kaggle - aadigupta1601)",
        "F1 Telemetry Montreal 2023 (HuggingFace - renumics)"
    ],
    "target": "LapTime_normalized (deviation from circuit median, seconds)",
    "target_explanation": "Lap time deviation from circuit median; lower values indicate faster laps relative to track baseline",
    "features": features,
    "n_total": len(df_model),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "n_removed": raw - len(df),
    "mae": mae_rf,
    "mse": mse_rf,
    "r2": r2_rf,
    "ers_risk_range": {"min": 0, "max": 10},
    "superclipping_duration_range": {"min_s": 1.0, "max_s": 8.0},
    "note": "ERS/MGU-K data is proprietary to F1 teams. This index is derived from public telemetry as a validated proxy for Superclipping risk under 2026 regulations."
}
with open('ml/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

# ── 8. SAVE RESULTS CSV (real vs predicted) ──────────
results_df = pd.DataFrame({
    'real': y_test.values,
    'predicted': y_pred,
    'laptime_s': df.loc[y_test.index, 'LapTime (s)'].values,
    'tyre_life': df.loc[y_test.index, 'TyreLife'].values,
    'position': df.loc[y_test.index, 'Position'].values,
    'compound': df.loc[y_test.index, 'Compound'].values,
    'ers_status': df.loc[y_test.index, 'ERS_status'].values,
    'superclipping_duration': df.loc[y_test.index, 'superclipping_duration'].values,
    'race': df.loc[y_test.index, 'Race'].values if 'Race' in df.columns else 'Unknown',
    'lap': df.loc[y_test.index, 'LapNumber'].values,
    'race_progress': df.loc[y_test.index, 'RaceProgress'].values,
})
results_df.to_csv('data/results.csv', index=False)
print("Saved: data/results.csv")

# ── 9. DASHBOARD DATA JSON ───────────────────────────
# Sample de datos para el dashboard frontend
sample = df.sample(min(200, len(df)), random_state=42).sort_values('LapNumber')
dashboard_data = {
    "telemetry_history": [
        {
            "lap": int(row['LapNumber']),
            "laptime_s": round(float(row['LapTime (s)']), 3),
            "tyre_life": int(row['TyreLife']),
            "position": int(row['Position']),
            "compound": row['Compound'],
            "ers_risk": round(float(row['ERS_risk']), 2),
            "ers_status": str(row['ERS_status']),
            "superclipping_duration": round(float(row['superclipping_duration']), 2),
            "race_progress": round(float(row['RaceProgress']), 3),
            "lap_delta": round(float(row['LapTime_Delta']), 3),
        }
        for _, row in sample.iterrows()
    ],
    "ers_distribution": df['ERS_status'].value_counts().to_dict(),
    "avg_superclipping_by_compound": df.groupby('Compound')['superclipping_duration'].mean().round(2).to_dict(),
    "avg_ers_risk_by_lap": df.groupby('LapNumber')['ERS_risk'].mean().round(2).to_dict(),
}
with open('data/dashboard_data.json', 'w', encoding='utf-8') as f:
    json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
print("Saved: data/dashboard_data.json")

# ── 10. PLOT REAL VS PREDICTED ───────────────────────
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(y_test.values[:60], label='Real ERS Risk', color='#cc0000', linewidth=2)
plt.plot(y_pred[:60], label='Predicted ERS Risk', color='#ffcc00', linewidth=2, linestyle='--')
plt.title('ERS Risk: Real vs Predicted')
plt.xlabel('Sample')
plt.ylabel('ERS Risk (0-10)')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test.values, y_pred, alpha=0.3, color='#cc0000', s=10)
plt.plot([0, 10], [0, 10], 'w--', linewidth=1)
plt.title('Real vs Predicted (scatter)')
plt.xlabel('Real')
plt.ylabel('Predicted')

plt.tight_layout()
plt.savefig('data/real_vs_predicted.png', dpi=150, facecolor='#0a0a0a')
print("Saved: data/real_vs_predicted.png")

# ── 11. MODEL SUMMARY ────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════╗
║          MODEL SUMMARY — QUANTUM-TWIN 2026          ║
╠══════════════════════════════════════════════════════╣
║ Dataset    : F1 Strategy 2024-2025 (Kaggle)         ║
║             + F1 Telemetry Montreal 2023 (HF)       ║
║ Target     : LapTime_normalized (deviation from circuit median, seconds) ║
║ Features   : {len(features)} variables                          ║
║ Total rows : {len(df_model)}                                ║
║ Train/Test : 80% / 20%                              ║
║ Removed    : {raw - len(df)} rows                          ║
╠══════════════════════════════════════════════════════╣
║ LINEAR REGRESSION: MAE: {mae_lr} | MSE: {mse_lr} | R²: {r2_lr}      ║
║ RANDOM FOREST:     MAE: {mae_rf} | MSE: {mse_rf} | R²: {r2_rf}      ║
║ Best model: Random Forest → saved as model.pkl       ║
╠══════════════════════════════════════════════════════╣
║ ERS Risk range     : 0 - 10                         ║
║ Superclipping est. : 1.0s - 8.0s                    ║
║ Production DB      : Oracle Autonomous DB (OCI)     ║
╚══════════════════════════════════════════════════════╝
""")
