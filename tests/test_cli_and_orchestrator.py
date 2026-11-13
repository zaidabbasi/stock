"""Unit tests for operator CLI (`src/cli.py`) and orchestrator (`src/main.py`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import Settings
from src.execution.broker import AlpacaBroker
from src.execution.risk_gateway import RiskGateway
from src.intelligence.analyst import GeminiAnalyst
from src.main import TradingOrchestrator
from src.persistence.ledger import LedgerRepository
from tests.conftest import FakeTradingClient
from tests.test_scenarios import FakeGenAIClient


def test_cli_unlock_and_status_commands(
    test_settings: Settings,
    ledger: LedgerRepository,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from src import cli

    monkeypatch.setattr(cli, "load_settings", lambda: test_settings)

    # Trigger a halt first
    ledger.trigger_halt(reason="Simulated -3% intraday flash crash", policy_id="TRD-POL-006", current_equity=97_000.0)
    assert ledger.get_system_state().trading_status == "HALTED"  # type: ignore[union-attr]

    # Run `python -m src.cli unlock --baseline-equity 97000`
    rc = cli.main(["unlock", "--baseline-equity", "97000"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["event"] == "circuit_breaker_unlocked"
    assert out["trading_status"] == "ACTIVE"

    state = ledger.get_system_state()
    assert state is not None
    assert state.trading_status == "ACTIVE"

    # Run `python -m src.cli status`
    rc_status = cli.main(["status"])
    assert rc_status == 0
    status_payload = json.loads(capsys.readouterr().out.strip())
    assert status_payload["trading_mode"] == "PAPER"
    assert status_payload["system_state"]["trading_status"] == "ACTIVE"


def test_orchestrator_single_cycle_active_and_halted(
    test_settings: Settings,
    ledger: LedgerRepository,
    broker: AlpacaBroker,
    risk_gateway: RiskGateway,
    fake_trading_client: FakeTradingClient,
) -> None:
    analyst = GeminiAnalyst(settings=test_settings, genai_client=FakeGenAIClient({}))
    orchestrator = TradingOrchestrator(
        settings=test_settings,
        ledger=ledger,
        broker=broker,
        risk_gateway=risk_gateway,
        analyst=analyst,
    )

    # 1. Active cycle
    res_active = orchestrator.run_single_cycle()
    assert res_active["status"] == "ACTIVE"
    assert res_active["equity"] == pytest.approx(100_000.0)

    # 2. Simulate -3% flash drawdown on account -> next cycle triggers HALT_TRIGGERED
    fake_trading_client.account.equity = 97_000.0
    res_halt = orchestrator.run_single_cycle()
    assert res_halt["status"] == "HALT_TRIGGERED"
    assert fake_trading_client.cancel_orders_called == 1

    # 3. Subsequent cycle skips while HALTED
    res_skipped = orchestrator.run_single_cycle()
    assert res_skipped["status"] == "HALTED"
