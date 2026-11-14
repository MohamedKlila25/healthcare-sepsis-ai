"""Data-quality summaries: distribution shift does not measure clinical performance."""
import numpy as np

from .data import VITALS


def drift_report(reference, current):
    report = {}
    for vital in VITALS:
        ref = reference[vital].dropna().to_numpy()
        now = current[vital].dropna().to_numpy()
        if len(ref) < 2 or len(now) < 2:
            report[vital] = {"status": "insufficient_data"}
            continue
        bounds = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
        if len(bounds) < 3:
            report[vital] = {"status": "constant_reference"}
            continue
        bounds[0], bounds[-1] = -np.inf, np.inf
        r = np.maximum(np.histogram(ref, bounds)[0] / len(ref), 1e-6)
        c = np.maximum(np.histogram(now, bounds)[0] / len(now), 1e-6)
        report[vital] = {"psi": float(np.sum((c - r) * np.log(c / r))),
                         "current_missing_fraction": float(current[vital].isna().mean())}
    return {"features": report, "clinical_performance_inferred": False}
