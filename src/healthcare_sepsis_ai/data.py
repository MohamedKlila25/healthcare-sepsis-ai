"""Synthetic examples and causal features; no patient data is bundled."""
from pathlib import Path

import numpy as np
import pandas as pd

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp", "Lactate", "WBC"]
CENTERS = dict(zip(VITALS, [78, 97, 36.8, 120, 80, 17, 1.2, 7]))
SCALES = dict(zip(VITALS, [12, 2, 0.5, 16, 10, 4, 0.7, 3]))


def synthetic_patients(n_patients: int = 180, hours: int = 36, seed: int = 42) -> pd.DataFrame:
    """Generate a seeded simulation, NOT a clinically validated disease mechanism."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        positive = p % 4 == 0
        onset = int(rng.integers(20, hours)) if positive else None
        offsets = {key: rng.normal(0, SCALES[key] * 0.35) for key in VITALS}
        for hour in range(hours):
            deterioration = max(0, (hour - (onset - 12)) / 12) if positive else 0
            changes = [24, -3, 1.3, -30, -22, 11, 3.5, 8]
            row = {"patient_id": f"DEMO-{p + 1:03d}", "hour": hour,
                   "SepsisLabel": int(positive and hour >= onset - 6), "synthetic": True}
            for key, change in zip(VITALS, changes):
                value = CENTERS[key] + offsets[key] + rng.normal(0, SCALES[key] * 0.45)
                value += deterioration * change
                row[key] = round(value, 2) if rng.random() > 0.12 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def read_physionet(directory: Path) -> pd.DataFrame:
    """Read challenge PSV files without changing the supplied shifted SepsisLabel."""
    files = sorted(directory.rglob("*.psv"))
    if not files:
        raise ValueError("No PhysioNet .psv records found")
    frames = []
    for file in files:
        df = pd.read_csv(file, sep="|")
        required = set(VITALS + ["SepsisLabel"])
        if not required.issubset(df.columns):
            raise ValueError(f"Invalid columns in {file.name}")
        if df.empty or not df.SepsisLabel.isin([0, 1]).all():
            raise ValueError(f"Invalid labels or empty record: {file.name}")
        # Relative path prevents collisions across hospitals with repeated filenames.
        df["patient_id"] = file.relative_to(directory).as_posix().removesuffix(".psv")
        df["hour"] = np.arange(len(df))
        df["synthetic"] = False
        frames.append(df[["patient_id", "hour", *VITALS, "SepsisLabel", "synthetic"]])
    return pd.concat(frames, ignore_index=True)


def validate_timeline(df: pd.DataFrame) -> None:
    if df.empty or df.duplicated(["patient_id", "hour"]).any():
        raise ValueError("Timeline must be nonempty with unique patient/hour pairs")
    if df.hour.isna().any() or (df.hour < 0).any():
        raise ValueError("Hours must be nonnegative")
    for _, group in df.groupby("patient_id", sort=False):
        if not group.hour.is_monotonic_increasing:
            raise ValueError("Patient hours must be increasing")
        if (group.hour % 1 != 0).any() or (group.hour.diff().dropna() != 1).any():
            raise ValueError("Timeline must use consecutive integer hourly observations")


def features(df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Every feature at hour t uses only rows at or before t within its patient."""
    validate_timeline(df)
    if window < 1:
        raise ValueError("Window must be positive")
    result = pd.DataFrame(index=df.index)
    for _, group in df.groupby("patient_id", sort=False):
        for vital in VITALS:
            values = group[vital]
            observed_hour = group.hour.where(values.notna()).ffill()
            last = values.ffill()
            result.loc[group.index, f"{vital}_last"] = last
            result.loc[group.index, f"{vital}_missing"] = values.isna().astype(float)
            result.loc[group.index, f"{vital}_age"] = group.hour - observed_hour
            rolling = values.rolling(window, min_periods=1)
            for name in ["mean", "std", "min", "max", "count"]:
                result.loc[group.index, f"{vital}_{name}"] = getattr(rolling, name)()
            result.loc[group.index, f"{vital}_change"] = last - last.shift(window - 1)
        sbp = group.SBP.ffill().replace(0, np.nan)
        result.loc[group.index, "shock_index"] = group.HR.ffill() / sbp
    return result.replace([np.inf, -np.inf], np.nan)

