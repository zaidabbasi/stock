"""Tests for the local credential setup flow (no network, no real secrets)."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from src.credentials import (
    CredentialValidationError,
    interactive_setup,
    mask,
    validate_alpaca_key_id,
    write_env_values,
)


def test_live_alpaca_key_prefix_rejected() -> None:
    with pytest.raises(CredentialValidationError, match="LIVE"):
        validate_alpaca_key_id("AKABCDEFGHIJKLMNOPQR")


def test_paper_key_accepted_and_masked() -> None:
    v = validate_alpaca_key_id("  PKABCDEFGHIJKLMNWXYZ ")
    assert v == "PKABCDEFGHIJKLMNWXYZ"
    assert mask(v) == "PK…WXYZ" and "ABCDEFGH" not in mask(v)


def test_write_env_preserves_other_lines_and_sets_0600(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("# comment\nTRADING_MODE=PAPER\nAPCA_API_KEY_ID=old\nMAX_POSITION_PCT=0.05\n")
    write_env_values({"APCA_API_KEY_ID": "PKNEWKEY1234567890", "GEMINI_API_KEY": "AIzaTESTKEY_1234567"}, env_path=env)
    text = env.read_text()
    assert "APCA_API_KEY_ID=PKNEWKEY1234567890" in text and "old" not in text
    assert "MAX_POSITION_PCT=0.05" in text and "# comment" in text
    assert "GEMINI_API_KEY=AIzaTESTKEY_1234567" in text
    assert stat.S_IMODE(env.stat().st_mode) == 0o600


def test_interactive_setup_never_prints_secrets(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    answers = iter(["PKSECRETKEYID123WXYZ", "SUPERSECRETVALUE0987654321", ""])
    env = interactive_setup(env_path=tmp_path / ".env", prompt_fn=lambda _p: next(answers))
    out = capsys.readouterr().out
    assert "PKSECRETKEYID123WXYZ" not in out and "SUPERSECRETVALUE0987654321" not in out
    assert "APCA_API_SECRET_KEY=SUPERSECRETVALUE0987654321" in env.read_text()
