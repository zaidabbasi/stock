"""Unit tests for fail-closed Paper Trading locks and execution boundary invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import SecurityConfigurationError, Settings, validate_broker_endpoint
from src.execution.broker import AlpacaBroker
from src.intelligence.schemas import StrategyType, TradeAction, TradeProposal
from tests.conftest import FakeTradingClient


def test_trading_mode_not_paper_raises_fatal_error() -> None:
    with pytest.raises(SecurityConfigurationError, match="TRADING_MODE='LIVE' is forbidden"):
        Settings(TRADING_MODE="LIVE")


def test_live_alpaca_endpoint_raises_fatal_error() -> None:
    with pytest.raises(SecurityConfigurationError, match="Live Alpaca endpoint detected"):
        Settings(
            TRADING_MODE="PAPER",
            ALPACA_PAPER_BASE_URL="https://api.alpaca.markets",
        )

    with pytest.raises(SecurityConfigurationError, match="Live Alpaca endpoint detected"):
        validate_broker_endpoint("https://api.alpaca.markets/v2/orders")


def test_injected_non_paper_trading_client_raises_fatal_error(test_settings: Settings) -> None:
    live_client = FakeTradingClient(paper=False, base_url="https://paper-api.alpaca.markets")
    with pytest.raises(SecurityConfigurationError, match="paper=False"):
        AlpacaBroker(settings=test_settings, trading_client=live_client)


def test_injected_live_url_trading_client_raises_fatal_error(test_settings: Settings) -> None:
    live_url_client = FakeTradingClient(paper=True, base_url="https://api.alpaca.markets")
    with pytest.raises(SecurityConfigurationError, match="Live Alpaca endpoint detected"):
        AlpacaBroker(settings=test_settings, trading_client=live_url_client)


def test_secrets_excluded_from_settings_repr(test_settings: Settings) -> None:
    rendered = repr(test_settings)
    assert "PK_TEST_PAPER_KEY" not in rendered
    assert "SK_TEST_PAPER_SECRET" not in rendered
    assert "TEST_GEMINI_KEY" not in rendered


def test_trade_proposal_forbids_sizing_and_broker_fields() -> None:
    """Invariant #3: Gemini TradeProposal schema rejects share qty, strikes, or stop prices."""
    with pytest.raises(ValidationError):
        TradeProposal(
            ticker="NVDA",
            strategy=StrategyType.TRAILING_MOMENTUM,
            action=TradeAction.BUY_EQUITY,
            conviction_score=0.88,
            rationale="Strong semiconductor momentum and positive guidance.",
            risk_flags=[],
            qty=100,  # type: ignore[call-arg]
        )
