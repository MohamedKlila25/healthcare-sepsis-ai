import pytest

torch = pytest.importorskip("torch")
from healthcare_sepsis_ai.temporal import GRUD, TCN  # noqa: E402


@pytest.mark.parametrize("kind", ["grud", "tcn"])
def test_temporal_models_cannot_see_future(kind):
    torch.manual_seed(42)
    values = torch.randn(2, 12, 8)
    mask = torch.ones_like(values)
    deltas = torch.ones_like(values)
    model = GRUD(8) if kind == "grud" else TCN(8)
    def run(x):
        return model(x, mask, deltas, torch.zeros(8)) if kind == "grud" else model(x, mask)
    before = run(values)
    changed = values.clone()
    changed[:, 6:] = 100
    after = run(changed)
    assert before.shape == (2, 12)
    assert torch.allclose(before[:, :6], after[:, :6])
