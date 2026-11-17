"""
Live check that the configured Claude model still exists.

Skipped unless a real ANTHROPIC_API_KEY is available (shell environment or .env).
The app and eval/run_eval.py run the same check (check_model_available) at startup.
Uses the Models API: no tokens are spent.
"""

import os

import pytest
from dotenv import dotenv_values

PLACEHOLDER = "test-key-not-used"  # set by conftest for offline tests


def _real_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key and key != PLACEHOLDER:
        return key
    return dotenv_values(".env").get("ANTHROPIC_API_KEY")


@pytest.mark.live
def test_configured_model_exists(monkeypatch):
    key = _real_api_key()
    if not key:
        pytest.skip("no ANTHROPIC_API_KEY; live model check skipped")

    monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    from src.agent import check_model_available

    ok, message = check_model_available()
    assert ok, message
