"""Explicit experiment matrix. Installs optional research dependencies separately."""
import argparse
import json
from pathlib import Path

from healthcare_sepsis_ai.data import read_physionet, synthetic_patients
from healthcare_sepsis_ai.training import train

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path)
parser.add_argument("--families", nargs="+", choices=["logistic", "catboost", "xgboost"], default=["logistic"])
parser.add_argument("--windows", nargs="+", type=int, choices=[6, 12, 24], default=[6])
args = parser.parse_args()
data = read_physionet(args.data) if args.data else synthetic_patients()
results = []
for family in args.families:
    for window in args.windows:
        bundle = train(data, Path(f"artifacts/{family}-w{window}.joblib"), family, window)
        results.append({k: v for k, v in bundle["report"].items() if k != "splits"})
Path("artifacts/benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
