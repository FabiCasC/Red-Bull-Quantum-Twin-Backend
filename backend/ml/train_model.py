"""Local training pipeline — 100% offline fallback when Azure ML is unavailable."""
import os

import matplotlib.pyplot as plt

from model_pipeline import run_pipeline, save_artifacts

# ── 1. LOAD + TRAIN (shared pipeline) ─────────────────
csv_path = "data/f1_strategy.csv"
print("=== RAW DATA ===")
result = run_pipeline(csv_path)
print(f"Shape after pipeline: {result.df.shape}")
print(f"Columns: {list(result.df.columns)}")

print("\n=== CLEANING ===")
print(f"Total removed: {result.raw_count - len(result.df)} rows")
print(f"Final clean shape: {result.df.shape}")

print("\n=== FEATURE ENGINEERING ===")
print(f"Target: {result.target}")
print(f"Features: {result.features}")
print(f"ERS_risk range: {result.df['ERS_risk'].min():.2f} - {result.df['ERS_risk'].max():.2f}")
print(f"ERS_status distribution:\n{result.df['ERS_status'].value_counts()}")
print(f"Training shape: {result.df_model.shape}")

# ── 5. TRAINING RESULTS ───────────────────────────────
print("\n=== TRAINING ===")
print(f"Train: {len(result.X_train)} rows | Test: {len(result.X_test)} rows")
print(f"Linear Regression — MAE: {result.mae_lr} | MSE: {result.mse_lr} | R²: {result.r2_lr}")
print(f"Random Forest      — MAE: {result.mae_rf} | MSE: {result.mse_rf} | R²: {result.r2_rf}")

# ── 7. SAVE ───────────────────────────────────────────
print("\n=== RESULTS ===")
metrics = save_artifacts(result, training_source="local")
print("Saved: ml/model.pkl, ml/scaler.pkl, ml/metrics.json, ml/model_source.json")
print("Saved: data/results.csv, data/dashboard_data.json")

y_pred = result.y_pred_rf
y_test = result.y_test

# ── 10. PLOT REAL VS PREDICTED ───────────────────────
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(y_test.values[:60], label="Real LapTime norm.", color="#cc0000", linewidth=2)
plt.plot(y_pred[:60], label="Predicted", color="#ffcc00", linewidth=2, linestyle="--")
plt.title("LapTime_normalized: Real vs Predicted")
plt.xlabel("Sample")
plt.ylabel("Deviation (s)")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test.values, y_pred, alpha=0.3, color="#cc0000", s=10)
mn, mx = float(y_test.min()), float(y_test.max())
plt.plot([mn, mx], [mn, mx], "w--", linewidth=1)
plt.title("Real vs Predicted (scatter)")
plt.xlabel("Real")
plt.ylabel("Predicted")

plt.tight_layout()
os.makedirs("data", exist_ok=True)
plt.savefig("data/real_vs_predicted.png", dpi=150, facecolor="#0a0a0a")
print("Saved: data/real_vs_predicted.png")

# ── 11. MODEL SUMMARY ────────────────────────────────
print(
    f"\n{'=' * 54}\n"
    f"  MODEL SUMMARY — QUANTUM-TWIN 2026\n"
    f"{'=' * 54}\n"
    f"  Source     : local (train_model.py)\n"
    f"  Target     : LapTime_normalized\n"
    f"  Features   : {len(result.features)} variables\n"
    f"  Total rows : {len(result.df_model)}\n"
    f"  Removed    : {result.raw_count - len(result.df)} rows\n"
    f"  LINEAR REG : MAE={result.mae_lr} MSE={result.mse_lr} R2={result.r2_lr}\n"
    f"  RANDOM FOREST: MAE={result.mae_rf} MSE={result.mse_rf} R2={result.r2_rf}\n"
    f"  Best model : Random Forest -> ml/model.pkl\n"
    f"{'=' * 54}\n"
)
