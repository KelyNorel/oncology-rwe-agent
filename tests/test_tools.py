"""Unit tests for the survival-analysis tools (no network, no LLM)."""

import base64
import json

import pytest

import src.tools as tools


def _is_png(b64: str) -> bool:
    return base64.b64decode(b64)[:8] == b"\x89PNG\r\n\x1a\n"


def _json_roundtrip(result: dict) -> dict:
    """Tool outputs travel to the LLM as JSON; they must serialize cleanly."""
    return json.loads(json.dumps(result))


# ── describe_dataset ─────────────────────────────────────────────────────────

def test_describe_dataset_counts(cohort_csv):
    out = tools.describe_dataset.invoke({})
    assert out["n_patients"] == 400
    assert 0 < out["os_event_rate"] < 1
    assert sum(out["subtype_counts"].values()) == 400
    assert _is_png(out["plot_base64"])
    _json_roundtrip(out)


# ── kaplan_meier_analysis ────────────────────────────────────────────────────

@pytest.mark.parametrize("var", ["subtype", "er", "her2", "hormone_tx", "chemo", "npi_group"])
def test_km_valid_variables(cohort_csv, var):
    out = tools.kaplan_meier_analysis.invoke({"stratify_by": var})
    assert "error" not in out
    assert 0 <= out["logrank_p"] <= 1
    assert out["significant"] == (out["logrank_p"] < 0.05)
    assert sum(g["n"] for g in out["groups"].values()) + sum(
        s["n"] for s in out["skipped_groups"].values()
    ) <= 400
    assert _is_png(out["plot_base64"])
    _json_roundtrip(out)


def test_km_unknown_variable_returns_error(cohort_csv):
    out = tools.kaplan_meier_analysis.invoke({"stratify_by": "race"})
    assert "error" in out
    assert "subtype" in out["error"]


def test_km_reports_groups_too_small_to_fit(cohort_csv, tmp_path, monkeypatch):
    import pandas as pd

    df = pd.read_csv(cohort_csv)
    df.loc[df.index[:3], "subtype"] = "rare"
    path = tmp_path / "small_group.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(tools, "DATA_PATH", str(path))

    out = tools.kaplan_meier_analysis.invoke({"stratify_by": "subtype"})
    assert "rare" not in out["groups"]
    assert out["skipped_groups"]["rare"]["n"] == 3


# ── cox_model ────────────────────────────────────────────────────────────────

def test_cox_recovers_true_risk_factors(cohort_csv):
    out = tools.cox_model.invoke({"features": ["grade", "lymph_nodes", "age"]})
    hr = out["hazard_ratios"]["exp(coef)"]
    p = out["hazard_ratios"]["p"]
    # Synthetic hazard increases with grade and lymph nodes, not with age.
    assert hr["grade"] > 1 and p["grade"] < 0.05
    assert hr["lymph_nodes"] > 1 and p["lymph_nodes"] < 0.05
    assert 0.5 < out["c_index"] <= 1
    assert out["ignored_features"] == []
    _json_roundtrip(out)


def test_cox_reports_ignored_features(cohort_csv):
    out = tools.cox_model.invoke({"features": ["grade", "subtype", "bmi"]})
    assert list(out["hazard_ratios"]["exp(coef)"]) == ["grade"]
    assert sorted(out["ignored_features"]) == ["bmi", "subtype"]


def test_cox_no_valid_features_returns_error(cohort_csv):
    out = tools.cox_model.invoke({"features": ["bmi"]})
    assert "error" in out


# ── ml_prediction ────────────────────────────────────────────────────────────

def test_ml_prediction_auc_range_and_serializable(cohort_csv):
    out = tools.ml_prediction.invoke({"cutoff_months": 60})
    assert set(out["auc_scores"]) == {"Logistic Regression", "Random Forest"}
    assert all(0 <= v <= 1 for v in out["auc_scores"].values())
    assert out["best_auc"] == max(out["auc_scores"].values())
    _json_roundtrip(out)


def test_ml_prediction_is_deterministic(cohort_csv):
    a = tools.ml_prediction.invoke({"cutoff_months": 36})
    b = tools.ml_prediction.invoke({"cutoff_months": 36})
    assert a == b


@pytest.mark.parametrize("cutoff", [0, -12])
def test_ml_prediction_rejects_nonpositive_cutoff(cohort_csv, cutoff):
    out = tools.ml_prediction.invoke({"cutoff_months": cutoff})
    assert "error" in out


def test_ml_prediction_rejects_cutoff_with_too_few_events(cohort_csv):
    out = tools.ml_prediction.invoke({"cutoff_months": 1})
    assert "error" in out
