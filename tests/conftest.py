"""
Shared fixtures.

The real METABRIC extract (data/processed/clinical_clean.csv) is not tracked in git,
so tests run against a synthetic cohort with the same schema. Survival times depend
on grade and lymph nodes so the Cox and ML tools have real signal to find.
"""

import os

import numpy as np
import pandas as pd
import pytest

# The agent module builds a ChatAnthropic client at import time; tests never call it.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

SUBTYPES = ["LumA", "LumB", "Her2", "Basal", "Normal", "claudin-low"]


def make_synthetic_cohort(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grade = rng.integers(1, 4, n)
    lymph_nodes = rng.poisson(2, n)
    tumor_size = rng.gamma(4, 6, n).round(1)
    hazard = 0.01 * np.exp(0.5 * (grade - 2) + 0.15 * lymph_nodes)
    event_time = rng.exponential(1 / hazard)
    censor_time = rng.uniform(20, 300, n)
    return pd.DataFrame({
        "age": rng.normal(60, 12, n).round(1),
        "grade": grade,
        "tumor_size": tumor_size,
        "lymph_nodes": lymph_nodes,
        "npi": (0.2 * tumor_size / 10 + grade + np.minimum(lymph_nodes, 3)).round(2),
        "subtype": rng.choice(SUBTYPES, n),
        "er": rng.choice(["Positive", "Negative"], n, p=[0.75, 0.25]),
        "her2": rng.choice(["Positive", "Negative"], n, p=[0.15, 0.85]),
        "chemo": rng.choice(["YES", "NO"], n, p=[0.3, 0.7]),
        "hormone_tx": rng.choice(["YES", "NO"], n, p=[0.6, 0.4]),
        "os_months": np.minimum(event_time, censor_time).round(1),
        "os_event": (event_time <= censor_time).astype(int),
    })


@pytest.fixture
def cohort_csv(tmp_path, monkeypatch):
    """Write a synthetic cohort and point the tools at it."""
    import src.tools as tools

    path = tmp_path / "clinical_clean.csv"
    make_synthetic_cohort().to_csv(path, index=False)
    monkeypatch.setattr(tools, "DATA_PATH", str(path))
    return path
