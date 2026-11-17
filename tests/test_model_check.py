"""Offline tests for the startup model check (the Anthropic client is faked)."""

import types

import anthropic
import httpx
import pytest

import src.agent as agent


def _fake_client(exc=None):
    def retrieve(model_id):
        if exc is not None:
            raise exc
        return types.SimpleNamespace(id=model_id)
    return lambda api_key=None: types.SimpleNamespace(models=types.SimpleNamespace(retrieve=retrieve))


def _status_error(cls, code):
    resp = httpx.Response(code, request=httpx.Request("GET", "https://api.anthropic.com"))
    return cls("error", response=resp, body=None)


def test_model_available(monkeypatch):
    monkeypatch.setattr(anthropic, "Anthropic", _fake_client())
    ok, msg = agent.check_model_available()
    assert ok and agent.MODEL_NAME in msg


def test_retired_model_is_reported(monkeypatch):
    monkeypatch.setattr(anthropic, "Anthropic", _fake_client(_status_error(anthropic.NotFoundError, 404)))
    ok, msg = agent.check_model_available()
    assert not ok and "retired" in msg and agent.MODEL_NAME in msg


def test_bad_key_is_reported(monkeypatch):
    monkeypatch.setattr(anthropic, "Anthropic", _fake_client(_status_error(anthropic.AuthenticationError, 401)))
    ok, msg = agent.check_model_available()
    assert not ok and "ANTHROPIC_API_KEY" in msg


def test_eval_stops_before_running_when_model_missing(monkeypatch):
    from eval import run_eval

    monkeypatch.setattr(agent, "check_model_available", lambda: (False, "Model gone"))
    monkeypatch.setattr(run_eval, "trace_run", lambda *a, **k: pytest.fail("should not run the agent"))
    monkeypatch.setattr("sys.argv", ["run_eval", "--runs", "1", "--out", "/tmp/should_not_exist.jsonl"])
    with pytest.raises(SystemExit):
        run_eval.main()
