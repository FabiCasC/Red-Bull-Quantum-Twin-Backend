# Red Bull Quantum-Twin 2026 — Backend

FastAPI backend con pipeline de 4 algoritmos ML para gemelo digital F1.

## Setup

### 1. Instalar dependencias
```bash
cd backend
pip install -r requirements.txt
```

### 2. Oracle Autonomous Database (OCI) — persistencia en producción

#### Wallet
1. Descarga el wallet desde OCI Console → Autonomous Database → DB Connection → Download Wallet.
2. Descomprime el ZIP y copia **todos** los archivos dentro de `backend/wallet/`:
   ```
   backend/wallet/
   ├── cwallet.sso
   ├── ewallet.pem
   ├── tnsnames.ora
   ├── sqlnet.ora
   └── ...
   ```

#### Variables de entorno
1. Copia el template:
   ```bash
   cp .env.example .env
   ```
2. Edita `.env` con tus credenciales:
   ```env
   ORACLE_USER=admin
   ORACLE_PASSWORD=tu_contraseña
   ORACLE_DSN=quantumtwind_high
   ORACLE_WALLET_DIR=wallet
   ORACLE_WALLET_PASSWORD=   # solo si el wallet está cifrado
   ```
3. El DSN debe coincidir con una entrada de `wallet/tnsnames.ora` que termine en `_high`.

> **Fallback automático:** si Oracle no está configurado o la conexión falla, la app usa SQLite en `db/quantum_twin.db` sin detener el servidor. Verifica el motor activo en `GET /api/db-status`.

### 3. Obtener datos
```bash
# Kaggle:
kaggle datasets download -d aadigupta1601/f1-strategy-dataset-pit-stop-prediction
unzip *.zip -d data/

# HuggingFace:
python data/export_huggingface.py
```

### 4. Entrenar los 4 modelos

**Opción A — Local (fallback 100%, sin Azure):**
```bash
python ml/train_model.py           # Regresión: LR vs RF → LapTime_normalized
python ml/train_decision_tree.py   # Clasificación: ERS_status
python ml/train_kmeans.py          # Clustering: perfiles de stint
python ml/train_svm.py             # SVM: eventos críticos de seguridad
```

**Opción B — Azure ML (entrena los 4 modelos en la nube):**

1. Instala [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) o usa login por navegador.
2. Agrega a tu `.env`:
   ```env
   AZURE_SUBSCRIPTION_ID=tu-subscription-id
   AZURE_RESOURCE_GROUP=quantumtwin-ml
   AZURE_ML_WORKSPACE=nombre-de-tu-workspace
   AZURE_TENANT_ID=tu-tenant-id
   ```
3. Ejecuta:
   ```bash
   python ml/azure_train.py
   ```
   El script:
   - Sube `data/f1_strategy.csv` como Data Asset versionado
   - Crea compute instance `quantum-twin-ci` (Standard_DS2_v2) si no existe
   - Ejecuta los 4 scripts de entrenamiento como Command Job
   - Registra métricas en MLflow (Azure ML nativo)
   - Descarga todos los `.pkl` y métricas a `ml/` localmente

4. Verifica origen de los modelos:
   ```bash
   curl http://localhost:8000/api/model/source
   ```

> ⚠️ **Después de entrenar en Azure:** ve a Azure Portal → Machine Learning → Compute → Instances → `quantum-twin-ci` → **Stop** o **Delete** para no consumir créditos extra.

Si Azure falla, `azure_train.py` indica en qué paso ocurrió el error. Usa los 4 scripts locales como respaldo.

### 5. Iniciar API
```bash
uvicorn main:app --reload --port 8000
```

### 6. Documentación interactiva
http://localhost:8000/docs

---

## Arquitectura

```
┌─────────────┐     HTTP/REST      ┌──────────────────┐
│   Frontend  │ ◄────────────────► │  FastAPI (main)  │
│   React     │                    │  Inferencia local│
└─────────────┘                    └────────┬─────────┘
                                            │
                         ┌──────────────────┼──────────────────┐
                         ▼                  ▼                  ▼
              ┌──────────────────┐  ┌─────────────┐  ┌─────────────────┐
              │ Oracle Autonomous│  │ SQLite      │  │ Azure ML        │
              │ DB (OCI)         │  │ (fallback)  │  │ (entrenamiento) │
              │ wallet mTLS      │  │ demo/offline│  │ 4 Command Jobs  │
              └──────────────────┘  └─────────────┘  └─────────────────┘
```

- **Inferencia:** siempre local (`.pkl` en `ml/`) por latencia en tiempo de carrera.
- **Entrenamiento:** local (`train_*.py`) o cloud (`azure_train.py` → `run_azure_training.py`).
- **Persistencia:** Oracle en producción; SQLite si Oracle no conecta.

---

## Los 4 modelos de Machine Learning

| # | Objetivo estratégico | Algoritmo | Target | Script | Métricas | Endpoint API |
|---|---------------------|-----------|--------|--------|----------|--------------|
| 1 | Predecir tiempo de vuelta normalizado | Linear Regression + **Random Forest** | `LapTime_normalized` | `train_model.py` | `ml/metrics.json` | `/api/metrics`, `/api/results` |
| 2 | Clasificar riesgo ERS (interpretable) | **Decision Tree** (max_depth 5) | `ERS_status` | `train_decision_tree.py` | `ml/metrics_classification.json` | `GET /api/risk-classification` |
| 3 | Segmentar estrategias de stint | **K-Means** + PCA | Clusters de degradación | `train_kmeans.py` | `ml/metrics_clustering.json` | `GET /api/strategy-clusters` |
| 4 | Detectar eventos críticos de seguridad | **SVM** (linear/rbf, class_weight balanced) | `evento_critico` (regla proxy) | `train_svm.py` | `ml/metrics_svm.json` | `GET /api/safety` |

**Pipeline compartido:** `ml/model_pipeline.py` — carga, limpieza, encoding y feature engineering usados por los 4 scripts.

**Dataset:** 99,457 registros — F1 Strategy Dataset 2024-2025 (Kaggle) + F1 Telemetry Montreal 2023 (HuggingFace).

---

## Estructura del proyecto

```
backend/
├── wallet/                      ← Oracle wallet (NO en git)
├── .env                         ← credenciales (NO en git)
├── .env.example
├── data/
│   ├── f1_strategy.csv
│   ├── f1_telemetry.csv
│   ├── decision_tree.png
│   ├── elbow_method.png
│   └── kmeans_clusters.png
├── ml/
│   ├── model_pipeline.py        ← lógica compartida
│   ├── train_model.py           ← [1] Regresión
│   ├── train_decision_tree.py   ← [2] Árbol de Decisión
│   ├── train_kmeans.py          ← [3] K-Means
│   ├── train_svm.py             ← [4] SVM
│   ├── azure_train.py           ← orquestador Azure ML
│   ├── run_azure_training.py    ← corre en compute Azure
│   ├── .amlignore               ← excluye .pkl del upload
│   └── *.pkl, metrics*.json
├── db/quantum_twin.db           ← SQLite fallback
├── main.py
├── database.py
└── requirements.txt
```

---

## API Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Info del proyecto |
| GET | `/api/health` | Health check |
| GET | `/api/db-status` | Motor DB activo (`oracle` o `sqlite_fallback`) |
| GET | `/api/model/source` | Origen y fecha de entrenamiento de los 4 modelos |
| GET | `/api/metrics` | Métricas regresión (MAE, MSE, R²) |
| GET | `/api/risk-classification` | Clase ERS + probabilidades (Árbol de Decisión) |
| GET | `/api/strategy-clusters` | Cluster K-Means + interpretación |
| GET | `/api/safety` | Evento crítico SVM (normal/crítico) |
| GET | `/api/results` | Real vs predicted (50 filas) |
| GET | `/api/history/predictions` | Últimas 10 predicciones (Oracle/SQLite) |
| GET | `/api/history/alerts` | Últimas 10 alertas |
| GET | `/api/telemetry` | Telemetría en vivo |
| GET | `/api/superclipping` | Estado superclipping |
| GET | `/api/race` | Estado de carrera |
| GET | `/api/tires` | Datos de neumáticos |
| GET | `/api/recommendations` | Recomendaciones AI |
| GET | `/api/strategy` | Estrategia de pits |
| GET | `/api/render` | Render 3D Blender |
| POST | `/api/simulation/start` | Iniciar simulación |
| POST | `/api/simulation/stop` | Detener simulación |
| GET | `/api/simulation/lap` | Vuelta actual simulación |
