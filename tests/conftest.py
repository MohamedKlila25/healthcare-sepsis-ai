import pytest

from healthcare_sepsis_ai.data import synthetic_patients
from healthcare_sepsis_ai.training import train


@pytest.fixture(scope="session")
def trained_bundle(tmp_path_factory):
    return train(synthetic_patients(n_patients=40), tmp_path_factory.mktemp("models") / "test.joblib")
