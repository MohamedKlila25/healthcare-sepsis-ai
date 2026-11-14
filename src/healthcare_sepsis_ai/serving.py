"""Research-only model inference and model-independent occlusion sensitivity."""
import os
from pathlib import Path

import joblib
import numpy as np

from .data import VITALS, features, synthetic_patients


class ResearchService:
    def __init__(self):
        self.records = synthetic_patients(n_patients=12)
        self.bundle = None
        path = Path(os.environ.get("HEALTHCARE_MODEL_PATH", "artifacts/demo_model.joblib"))
        # Only load artifacts controlled by the server owner; never accept uploaded pickle/joblib files.
        if path.is_file():
            self.bundle = joblib.load(path)
            if not self.bundle["synthetic"]:
                raise ValueError("Public synthetic demo requires a synthetic-trained model artifact")

    def timeline(self, patient_id, hour=None):
        df = self.records[self.records.patient_id == patient_id]
        if df.empty:
            raise KeyError(patient_id)
        if hour is not None:
            df = df[df.hour <= hour]
        if df.empty:
            raise ValueError("No observations at the requested hour")
        return df.copy()

    def predict(self, patient_id, hour=None):
        df = self.timeline(patient_id, hour)
        missing = float(df[VITALS].tail(6).isna().mean().mean())
        latest_missing = float(df[VITALS].iloc[-1].isna().mean())
        reason = None
        if self.bundle is None:
            reason = "Model unavailable. Train the synthetic demo model first."
        elif missing > 0.5 or latest_missing > 0.75:
            reason = "Insufficient measurements: prediction withheld."
        common = {"patient_id": patient_id, "hour": int(df.hour.iloc[-1]), "synthetic": True,
                  "missing_fraction_6h": round(missing, 4), "clinical_validation": False,
                  "requires_human_review": True, "abstained": reason is not None, "reason": reason,
                  "model_version": self.bundle["version"] if self.bundle else None}
        if reason:
            return {**common, "risk_score": None, "trajectory": [], "contributions": []}
        x = features(df, self.bundle["window"])[self.bundle["columns"]]
        scores = self.bundle["model"].predict_proba(x)[:, 1]
        trajectory = []
        for i, (_, row) in enumerate(df.iterrows()):
            # Quality gating applies at each historical prediction, not just the final hour.
            history_missing = float(df[VITALS].iloc[max(0, i - 5):i + 1].isna().mean().mean())
            valid = history_missing <= 0.5 and row[VITALS].isna().mean() <= 0.75
            trajectory.append({"hour": int(row.hour), "risk": round(float(scores[i]), 5) if valid else None})
        last = x.iloc[[-1]].copy()
        medians = self.bundle["base_model"].steps[0][1].statistics_
        contributions = []
        for vital in VITALS:
            changed = last.copy()
            for i, column in enumerate(x.columns):
                if column.startswith(vital + "_"):
                    changed.loc[:, column] = medians[i]
            delta = float(scores[-1] - self.bundle["model"].predict_proba(changed)[0, 1])
            contributions.append({"feature": vital, "risk_delta": round(delta, 5)})
        contributions.sort(key=lambda item: abs(item["risk_delta"]), reverse=True)
        return {**common, "risk_score": round(float(scores[-1]), 5), "trajectory": trajectory,
                "contributions": contributions, "explanation_method": "grouped_feature_occlusion",
                "uncertainty": "Quality-based abstention only; no validated confidence interval"}

    def patients(self):
        return [{"patient_id": patient, "hours": len(frame), "synthetic": True}
                for patient, frame in self.records.groupby("patient_id")]

    def json_timeline(self, patient_id):
        df = self.timeline(patient_id).drop(columns=["SepsisLabel"])
        return df.replace({np.nan: None}).to_dict(orient="records")

