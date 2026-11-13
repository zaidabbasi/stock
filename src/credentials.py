"""Local, interactive credential setup and verification.

Secrets are typed into hidden prompts (getpass), validated, and written to `.env`
with owner-only permissions (0600). Nothing secret is ever printed; only masked hints.
"""

from __future__ import annotations

import getpass
import os
import re
import tempfile
from pathlib import Path
from typing import Callable, Mapping, Optional

ENV_PATH = Path(".env")
ENV_TEMPLATE_PATH = Path(".env.example")

_SAFE_VALUE = re.compile(r"^[A-Za-z0-9_\-]{16,200}$")


class CredentialValidationError(ValueError):
    """Raised when a supplied credential fails local format checks."""


def mask(value: str) -> str:
    """Return a non-reversible hint such as 'PK…WXYZ' for confirmation output."""
    if len(value) <= 6:
        return "****"
    return f"{value[:2]}…{value[-4:]}"


def validate_alpaca_key_id(value: str) -> str:
    v = value.strip()
    if v.startswith("AK"):
        raise CredentialValidationError(
            "This looks like a LIVE Alpaca key (prefix 'AK'). Only PAPER keys (prefix 'PK') are accepted."
        )
    if not v.startswith("PK"):
        raise CredentialValidationError("Alpaca paper key IDs start with 'PK'. Check you copied the Paper key.")
    if not _SAFE_VALUE.match(v):
        raise CredentialValidationError("Key ID contains unexpected characters or has an invalid length.")
    return v


def validate_generic_secret(value: str, label: str) -> str:
    v = value.strip()
    if not _SAFE_VALUE.match(v):
        raise CredentialValidationError(f"{label} contains unexpected characters or has an invalid length.")
    return v


def write_env_values(updates: Mapping[str, str], env_path: Path = ENV_PATH,
                     template_path: Path = ENV_TEMPLATE_PATH) -> Path:
    """Atomically upsert KEY=VALUE pairs into `.env`, preserving other lines, with 0600 permissions."""
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    elif template_path.exists():
        lines = template_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = ["TRADING_MODE=PAPER", "ALPACA_PAPER_BASE_URL=https://paper-api.alpaca.markets"]

    remaining = dict(updates)
    out: list[str] = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
        if key in remaining:
            out.append(f"{key}={remaining.pop(key)}")
        else:
            out.append(line)
    out.extend(f"{k}={v}" for k, v in remaining.items())

    env_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(env_path.parent), prefix=".env.", suffix=".tmp")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("\n".join(out) + "\n")
        os.replace(tmp, env_path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    os.chmod(env_path, 0o600)
    return env_path


def _prompt(label: str, validator: Callable[[str], str], prompt_fn: Callable[[str], str],
            optional: bool = False, attempts: int = 3) -> Optional[str]:
    for _ in range(attempts):
        raw = prompt_fn(f"{label}{' (press Enter to skip)' if optional else ''} [input hidden]: ")
        if optional and not raw.strip():
            return None
        try:
            value = validator(raw)
            print(f"  ✓ accepted {mask(value)}")
            return value
        except CredentialValidationError as exc:
            print(f"  ✗ {exc}")
    raise SystemExit(f"Too many invalid attempts for {label}.")


def interactive_setup(env_path: Path = ENV_PATH, prompt_fn: Callable[[str], str] = getpass.getpass) -> Path:
    print("alpaca-gemini-trader credential setup — values are hidden and stored only in", env_path.resolve())
    key_id = _prompt("Alpaca PAPER API Key ID", validate_alpaca_key_id, prompt_fn)
    secret = _prompt("Alpaca PAPER Secret Key",
                     lambda v: validate_generic_secret(v, "Alpaca secret"), prompt_fn)
    gemini = _prompt("Gemini API Key", lambda v: validate_generic_secret(v, "Gemini key"),
                     prompt_fn, optional=True)

    updates = {"TRADING_MODE": "PAPER", "APCA_API_KEY_ID": key_id or "", "APCA_API_SECRET_KEY": secret or ""}
    if gemini:
        updates["GEMINI_API_KEY"] = gemini
    path = write_env_values(updates, env_path=env_path)
    print(f"Saved to {path} (permissions 600). Next: python -m src.cli verify")
    return path


def verify_connections() -> int:
    """Call Alpaca paper (and optionally Gemini) and print only non-secret account facts."""
    from src.config import SecurityConfigurationError, load_settings

    try:
        settings = load_settings()
        settings.require_broker_credentials()
    except SecurityConfigurationError as exc:
        print(f"✗ Configuration error: {exc}")
        return 2

    rc = 0
    try:
        from alpaca.trading.client import TradingClient

        client = TradingClient(settings.APCA_API_KEY_ID, settings.APCA_API_SECRET_KEY,
                               paper=True, url_override=settings.ALPACA_PAPER_BASE_URL)
        acct = client.get_account()
        clock = client.get_clock()
        print("✓ Alpaca PAPER connection OK")
        print(f"  key: {mask(settings.APCA_API_KEY_ID)} | status: {acct.status} | equity: ${float(acct.equity):,.2f}")
        print(f"  non-marginable BP: ${float(acct.non_marginable_buying_power):,.2f} | "
              f"options level: {getattr(acct, 'options_trading_level', 'n/a')} | market open: {clock.is_open}")
        level = getattr(acct, "options_trading_level", None)
        if level is not None and int(level) < 1:
            print("  ! Options trading level < 1: the Wheel strategy (CSP/CC) will be rejected by Alpaca.")
    except Exception as exc:  # never echo request headers / secrets
        print(f"✗ Alpaca PAPER connection failed: {type(exc).__name__}: {str(exc)[:160]}")
        rc = 1

    if settings.GEMINI_API_KEY and "YOUR_GEMINI" not in settings.GEMINI_API_KEY:
        try:
            from google import genai

            genai.Client(api_key=settings.GEMINI_API_KEY).models.get(model=settings.GEMINI_MODEL)
            print(f"✓ Gemini key OK and model '{settings.GEMINI_MODEL}' is available")
        except Exception as exc:
            print(f"✗ Gemini check failed for model '{settings.GEMINI_MODEL}': {type(exc).__name__}: {str(exc)[:160]}")
            rc = rc or 1
    else:
        print("- Gemini key not set (only needed once AI-gated strategies are enabled)")
    return rc
