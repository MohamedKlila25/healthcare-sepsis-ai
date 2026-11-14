import numpy as np
import pandas as pd
import pytest

from healthcare_sepsis_ai.data import VITALS, features, read_physionet, synthetic_patients
from healthcare_sepsis_ai.training import split_patients


def test_future_values_and_other_patients_do_not_affect_past():
    df = synthetic_patients(8)
    original = features(df)
    modified = df.copy()
    modified.loc[(modified.patient_id == "DEMO-001") & (modified.hour > 12), VITALS] = 99999
    modified.loc[modified.patient_id == "DEMO-002", VITALS] = -99999
    result = features(modified)
    idx = (df.patient_id == "DEMO-001") & (df.hour <= 12)
    pd.testing.assert_frame_equal(original.loc[idx], result.loc[idx])


def test_patient_splits_are_disjoint():
    df = synthetic_patients(40)
    splits = split_patients(df)
    assert not set(splits["train"]) & set(splits["test"])
    assert not set(splits["train"]) & set(splits["calibration"])
    assert not set(splits["test"]) & set(splits["calibration"])
    assert set().union(*map(set, splits.values())) == set(df.patient_id)


def test_duplicate_and_gapped_hours_rejected():
    df = synthetic_patients(1)
    with pytest.raises(ValueError, match="unique"):
        features(pd.concat([df, df.iloc[[0]]]))
    df.loc[1:, "hour"] += 1
    with pytest.raises(ValueError, match="consecutive"):
        features(df)


def test_physionet_preserves_challenge_target(tmp_path):
    df = synthetic_patients(1)
    df[[*VITALS, "SepsisLabel"]].to_csv(tmp_path / "patient.psv", sep="|", index=False)
    loaded = read_physionet(tmp_path)
    assert loaded.SepsisLabel.tolist() == df.SepsisLabel.tolist()
    assert not loaded.synthetic.any()


def test_metrics_are_real_and_calibrator_uses_frozen_training_model(trained_bundle):
    report = trained_bundle["report"]
    assert report["synthetic"] and report["clinical_validation"] is False
    assert 0 <= report["auprc"] <= 1
    assert report["auprc_patient_bootstrap_95ci"] is not None
    calibrator = trained_bundle["model"]
    assert calibrator.estimator.estimator is trained_bundle["base_model"]
    assert np.isfinite(trained_bundle["base_model"].steps[0][1].statistics_).all()
