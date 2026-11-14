"""Container/local launcher: create a synthetic model only when explicitly in demo mode."""
import os
import sys
from pathlib import Path

from healthcare_sepsis_ai.data import synthetic_patients
from healthcare_sepsis_ai.training import train


def main():
    model_path = Path(os.environ.get("HEALTHCARE_MODEL_PATH", "artifacts/demo_model.joblib"))
    if not model_path.exists() and os.environ.get("HEALTHCARE_ALLOW_DEMO", "true").lower() == "true":
        train(synthetic_patients(), model_path)
    os.execv(sys.executable, [sys.executable, "-m", "uvicorn", "apps.backend.main:app",
                            "--host", os.environ.get("HOST", "127.0.0.1"),
                            "--port", os.environ.get("PORT", "8000")])


if __name__ == "__main__":
    main()
