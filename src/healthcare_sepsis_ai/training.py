"""Patient-disjoint fitting, calibration and held-out evaluation."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import features, read_physionet, synthetic_patients


def split_patients(df, seed=42):
    labels = df.groupby("patient_id").SepsisLabel.max()
    train_ids, remaining = train_test_split(labels.index, test_size=0.4, random_state=seed,
                                           stratify=labels)
    calibration_ids, test_ids = train_test_split(remaining, test_size=0.5, random_state=seed,
                                                stratify=labels.loc[remaining])
    return {"train": list(train_ids), "calibration": list(calibration_ids), "test": list(test_ids)}


def build_model(family):
    if family == "logistic":
        estimator = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)
    elif family == "xgboost":
        from xgboost import XGBClassifier
        estimator = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  eval_metric="logloss", random_state=42, n_jobs=2)
    elif family == "catboost":
        from catboost import CatBoostClassifier
        estimator = CatBoostClassifier(iterations=200, depth=5, verbose=False,
                                       random_seed=42, allow_writing_files=False, thread_count=2)
    else:
        raise ValueError(f"Unknown model family: {family}")
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), estimator)


def metrics(y, scores):
    return {"auprc": float(average_precision_score(y, scores)),
            "auroc": float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else None,
            "brier": float(brier_score_loss(y, scores))}


def patient_bootstrap(y, scores, ids, iterations=200):
    rng = np.random.default_rng(42)
    unique = np.unique(ids)
    values = []
    groups = {p: np.flatnonzero(ids == p) for p in unique}
    for _ in range(iterations):
        idx = np.concatenate([groups[p] for p in rng.choice(unique, len(unique), replace=True)])
        if len(np.unique(y[idx])) == 2:
            values.append(average_precision_score(y[idx], scores[idx]))
    return [float(x) for x in np.quantile(values, [0.025, 0.975])] if values else None


def train(df, output: Path, family="logistic", window=6):
    splits = split_patients(df)
    x = features(df, window)
    masks = {key: df.patient_id.isin(ids).to_numpy() for key, ids in splits.items()}
    base = build_model(family)
    base.fit(x.loc[masks["train"]], df.loc[masks["train"], "SepsisLabel"])
    # FrozenEstimator prevents calibration from refitting training preprocessing.
    from sklearn.frozen import FrozenEstimator
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    calibrated.fit(x.loc[masks["calibration"]], df.loc[masks["calibration"], "SepsisLabel"])
    test = df.loc[masks["test"]]
    scores = calibrated.predict_proba(x.loc[masks["test"]])[:, 1]
    y = test.SepsisLabel.to_numpy()
    report = metrics(y, scores)
    report["auprc_patient_bootstrap_95ci"] = patient_bootstrap(y, scores, test.patient_id.to_numpy())
    # Count alert episodes (threshold crossings), not every positive hour.
    alerts = test[["patient_id", "hour", "SepsisLabel"]].copy()
    alerts["alert"] = scores >= 0.5
    alerts["crossing"] = alerts.alert & ~alerts.groupby("patient_id").alert.shift(fill_value=False)
    negative = alerts.groupby("patient_id").SepsisLabel.transform("max") == 0
    negative_days = max(float(negative.sum()) / 24, 1e-9)
    report["false_alert_episodes_per_nonseptic_patient_day"] = float(alerts.loc[negative, "crossing"].sum() / negative_days)
    report.update({"threshold": 0.5, "rows": len(df), "patients": df.patient_id.nunique(),
                   "test_prevalence": float(y.mean()), "family": family, "window_hours": window,
                   "synthetic": bool(df.synthetic.all()), "target": "supplied_shifted_SepsisLabel",
                   "clinical_validation": False, "splits": splits})
    bundle = {"model": calibrated, "base_model": base, "columns": list(x.columns),
              "window": window, "family": family, "synthetic": report["synthetic"],
              "version": f"{family}-w{window}-seed42-v1", "report": report}
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, help="Authorized local PhysioNet PSV directory")
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo_model.joblib"))
    parser.add_argument("--family", choices=["logistic", "xgboost", "catboost"], default="logistic")
    parser.add_argument("--window", type=int, choices=[6, 12, 24], default=6)
    args = parser.parse_args()
    df = read_physionet(args.data) if args.data else synthetic_patients()
    bundle = train(df, args.output, args.family, args.window)
    print(json.dumps({k: v for k, v in bundle["report"].items() if k != "splits"}, indent=2))


if __name__ == "__main__":
    main()
