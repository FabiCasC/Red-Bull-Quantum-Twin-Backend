"""
Submit F1 model training to Azure ML and download artifacts locally.

Prerequisites:
  1. az login   (or set AZURE credentials for DefaultAzureCredential)
  2. Fill AZURE_* vars in .env
  3. pip install -r requirements.txt

After training: STOP or DELETE the compute instance in Azure Portal
to avoid extra charges — see reminder at end of this script.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")
except ImportError:
    pass

ML_DIR = BACKEND_DIR / "ml"
DATA_DIR = BACKEND_DIR / "data"
COMPUTE_NAME = "quantum-twin-ci"
COMPUTE_SIZE = "Standard_DS2_v2"
DATA_ASSET_NAME = "f1-strategy-csv"
EXPERIMENT_NAME = "quantum-twin-f1-training"


class AzureTrainError(Exception):
    """Raised when a specific Azure ML step fails."""

    def __init__(self, step: str, message: str):
        super().__init__(f"[{step}] {message}")
        self.step = step


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise AzureTrainError(
            "configuration",
            f"Missing environment variable {name}. Copy .env.example → .env and fill Azure settings.",
        )
    return value


def _get_credential():
    """Authenticate via Azure CLI if available, otherwise open browser login."""
    step = "authentication"
    try:
        from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

        tenant_id = os.getenv("AZURE_TENANT_ID")

        try:
            credential = DefaultAzureCredential(
                exclude_shared_token_cache_credential=True,
                exclude_visual_studio_code_credential=True,
                additionally_allowed_tenants=[tenant_id] if tenant_id else ["*"],
            )
            credential.get_token("https://management.azure.com/.default")
            print("Authenticated via Azure CLI / existing credentials.")
            return credential
        except Exception:
            if not tenant_id:
                raise AzureTrainError(
                    step,
                    "AZURE_TENANT_ID missing in .env. Add your Azure AD tenant ID "
                    "(Azure Portal → Microsoft Entra ID → Overview → Tenant ID).",
                ) from None

            print(
                "Azure CLI not found or not logged in.\n"
                f"Opening browser for login (tenant: {tenant_id})..."
            )
            credential = InteractiveBrowserCredential(tenant_id=tenant_id)
            credential.get_token("https://management.azure.com/.default")
            print("Browser authentication successful.")
            return credential
    except AzureTrainError:
        raise
    except Exception as exc:
        raise AzureTrainError(
            step,
            f"Authentication failed. Install Azure CLI (`az login`) or complete browser login: {exc}",
        ) from exc


def _get_ml_client():
    step = "workspace_connection"
    try:
        from azure.ai.ml import MLClient

        credential = _get_credential()
        subscription_id = _require_env("AZURE_SUBSCRIPTION_ID")
        resource_group = _require_env("AZURE_RESOURCE_GROUP")
        workspace_name = _require_env("AZURE_ML_WORKSPACE")

        client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )
        # Test connection
        client.workspaces.get(workspace_name)
        print(f"Connected to Azure ML workspace: {workspace_name}")
        return client
    except AzureTrainError:
        raise
    except Exception as exc:
        raise AzureTrainError(
            step,
            f"Could not connect to workspace '{os.getenv('AZURE_ML_WORKSPACE')}': {exc}",
        ) from exc


def _ensure_compute(ml_client):
    step = "compute_setup"
    try:
        from azure.ai.ml.entities import ComputeInstance

        try:
            compute = ml_client.compute.get(COMPUTE_NAME)
            print(f"Compute instance '{COMPUTE_NAME}' already exists (state: {compute.provisioning_state})")
            return COMPUTE_NAME
        except Exception:
            pass

        print(f"Creating compute instance '{COMPUTE_NAME}' ({COMPUTE_SIZE})...")
        compute = ComputeInstance(name=COMPUTE_NAME, size=COMPUTE_SIZE)
        poller = ml_client.compute.begin_create_or_update(compute)
        poller.result()
        print(f"Compute instance '{COMPUTE_NAME}' ready.")
        return COMPUTE_NAME
    except AzureTrainError:
        raise
    except Exception as exc:
        raise AzureTrainError(step, f"Failed to create/get compute instance: {exc}") from exc


def _upload_data_asset(ml_client):
    step = "data_upload"
    csv_path = DATA_DIR / "f1_strategy.csv"
    if not csv_path.exists():
        raise AzureTrainError(step, f"CSV not found: {csv_path}")

    try:
        from azure.ai.ml.constants import AssetTypes
        from azure.ai.ml.entities import Data

        version = time.strftime("%Y%m%d%H%M%S")
        data = Data(
            name=DATA_ASSET_NAME,
            version=version,
            path=str(csv_path),
            type=AssetTypes.URI_FILE,
            description="F1 Strategy Dataset — Kaggle aadigupta1601",
        )
        asset = ml_client.data.create_or_update(data)
        print(f"Data asset uploaded: {asset.name} v{asset.version}")
        return asset
    except AzureTrainError:
        raise
    except Exception as exc:
        raise AzureTrainError(step, f"Failed to upload data asset: {exc}") from exc


def _submit_job(ml_client, data_asset, compute_name: str):
    step = "job_submission"
    try:
        from azure.ai.ml import Input, Output, command
        from azure.ai.ml.constants import AssetTypes

        job = command(
            code=str(ML_DIR),
            command=(
                "python run_azure_training.py "
                "--data ${{inputs.strategy_data}} "
                "--output ${{outputs.model_dir}}"
            ),
            inputs={
                "strategy_data": Input(type=AssetTypes.URI_FILE, path=data_asset.id),
            },
            outputs={
                "model_dir": Output(type=AssetTypes.URI_FOLDER),
            },
            environment="AzureML-sklearn-1.5@latest",
            compute=compute_name,
            experiment_name=EXPERIMENT_NAME,
            display_name="quantum-twin-4models-training",
        )
        returned = ml_client.jobs.create_or_update(job)
        print(f"Job submitted: {returned.name} — status: {returned.status}")
        return returned
    except AzureTrainError:
        raise
    except Exception as exc:
        raise AzureTrainError(step, f"Failed to submit training job: {exc}") from exc


def _wait_for_job(ml_client, job_name: str):
    step = "job_execution"
    try:
        print(f"Waiting for job '{job_name}' to complete (this may take several minutes)...")
        ml_client.jobs.stream(job_name)
        job = ml_client.jobs.get(job_name)
        if job.status != "Completed":
            raise AzureTrainError(
                step,
                f"Job finished with status '{job.status}'. Check Azure ML Studio for logs.",
            )
        print(f"Job '{job_name}' completed successfully.")
        return job
    except AzureTrainError:
        raise
    except Exception as exc:
        raise AzureTrainError(step, f"Job monitoring failed: {exc}") from exc


def _download_artifacts(ml_client, job_name: str):
    step = "artifact_download"
    try:
        download_root = BACKEND_DIR / "azure_download"
        if download_root.exists():
            shutil.rmtree(download_root)
        download_root.mkdir(parents=True)

        ml_client.jobs.download(
            name=job_name,
            download_path=str(download_root),
            output_name="model_dir",
        )

        # Find downloaded ml/ folder (path varies by SDK version)
        model_pkl = list(download_root.rglob("model.pkl"))
        if not model_pkl:
            raise FileNotFoundError("model.pkl not found in job outputs")

        remote_ml = model_pkl[0].parent
        remote_data = remote_ml.parent / "data"
        if not remote_data.exists():
            remote_data = model_pkl[0].parent.parent / "data"

        ML_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        for fname in [
            "model.pkl", "model_lr.pkl", "scaler.pkl", "race_encoder.pkl",
            "metrics.json", "model_source.json", "models_source.json",
            "decision_tree_model.pkl", "metrics_classification.json",
            "kmeans_model.pkl", "kmeans_scaler.pkl", "kmeans_pca.pkl",
            "metrics_clustering.json",
            "svm_model.pkl", "svm_scaler.pkl", "metrics_svm.json",
        ]:
            src = remote_ml / fname
            if src.exists():
                shutil.copy2(src, ML_DIR / fname)
                print(f"  → ml/{fname}")

        if remote_data.exists():
            for fname in [
                "results.csv", "dashboard_data.json",
                "decision_tree.png", "confusion_matrix_dt.png",
                "elbow_method.png", "kmeans_clusters.png",
                "confusion_matrix_svm_linear.png", "confusion_matrix_svm_rbf.png",
            ]:
                src = remote_data / fname
                if src.exists():
                    shutil.copy2(src, DATA_DIR / fname)
                    print(f"  → data/{fname}")

        print("Artifacts saved locally — main.py will load them on next restart.")
    except AzureTrainError:
        raise
    except Exception as exc:
        raise AzureTrainError(step, f"Failed to download model artifacts: {exc}") from exc


def main():
    os.chdir(BACKEND_DIR)
    print("=" * 60)
    print("  Red Bull Quantum-Twin — Azure ML Training")
    print("=" * 60)

    try:
        ml_client = _get_ml_client()
        compute_name = _ensure_compute(ml_client)
        data_asset = _upload_data_asset(ml_client)
        job = _submit_job(ml_client, data_asset, compute_name)
        _wait_for_job(ml_client, job.name)
        _download_artifacts(ml_client, job.name)

        print("\n✅ Azure ML training complete.")
        print("   Restart uvicorn to load the new model.")
        print("   GET /api/model/source → should show source: azure_ml")

    except AzureTrainError as exc:
        print(f"\n❌ Azure ML training failed at step: {exc.step}", file=sys.stderr)
        print(f"   {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n❌ Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        "\n" + "!" * 60 + "\n"
        "  ⚠️  IMPORTANTE — APAGA O ELIMINA EL COMPUTE INSTANCE\n"
        "  en Azure Portal → Machine Learning → Compute → Instances\n"
        f"  → selecciona '{COMPUTE_NAME}' → Stop o Delete\n"
        "  Si lo dejas encendido seguirá consumiendo créditos de Azure.\n"
        + "!" * 60
    )


if __name__ == "__main__":
    main()
