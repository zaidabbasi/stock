"""Unit tests verifying TRD-POL-001 through TRD-POL-006 enforcement, SQLite logging, and warnings."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import pytest

from src.execution.risk_gateway import RiskGateway
from src.intelligence.schemas import (
    OptionRight,
    OrderIntent,
    PolicyId,
    PortfolioSnapshot,
    RiskDecisionStatus,
    StrategyType,
    TradeAction,
)
from src.persistence.ledger import LedgerRepository
from tests.conftest import FakeTradingClient


def test_trd_pol_001_single_ticker_concentration_cap(
    risk_gateway: RiskGateway,
    ledger: LedgerRepository,
    fake_trading_client: FakeTradingClient,
    default_portfolio: PortfolioSnapshot,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    TRD-POL-001: Reject any entry that pushes total notional exposure in a single underlying
    above MAX_POSITION_PCT (5% of $100,000 = $5,000).
    """
    # Existing AAPL exposure is $3,500; adding $2,000 pushes total to $5,500 (> $5,000 limit)
    portfolio = default_portfolio.model_copy(
        update={"ticker_notional_exposure": {"AAPL": 3_500.0}}
    )
    intent = OrderIntent(
        strategy=StrategyType.TRAILING_MOMENTUM,
        ticker="AAPL",
        action=TradeAction.BUY_EQUITY,
        qty=10,
        notional_exposure_delta=2_000.0,
        limit_price=200.0,
        trailing_stop_pct=0.05,
        date_bucket="2026-11-12T10:30",
    )

    with caplog.at_level(logging.WARNING):
        decision, order_res = risk_gateway.submit(intent, portfolio)

    assert decision.status == RiskDecisionStatus.REJECTED
    assert decision.policy_id == PolicyId.TRD_POL_001
    assert order_res is None
    assert len(fake_trading_client.submitted_orders) == 0
    assert "TRD-POL-001" in caplog.text

    logs = ledger.list_audit_logs(policy_id="TRD-POL-001")
    assert len(logs) == 1
    assert logs[0].ticker == "AAPL"
    assert logs[0].risk_decision == "REJECTED"


def test_trd_pol_002_cash_secured_put_collateral_lock(
    risk_gateway: RiskGateway,
    ledger: LedgerRepository,
    fake_trading_client: FakeTradingClient,
    default_portfolio: PortfolioSnapshot,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    TRD-POL-002: Verify non_marginable_buying_power minus existing reserved collateral
    covers 100 * strike_price before selling any CSP.
    """
    # Existing open CSP reserves $18,000; non_marginable_buying_power is $20,000 -> only $2,000 free
    ledger.record_csp_open(
        ticker="MSFT",
        option_symbol="MSFT261218P00180000",
        strike_price=180.0,
        expiration_date="2026-12-18",
        premium_per_share=2.50,
    )
    portfolio = default_portfolio.model_copy(
        update={
            "total_equity": 100_000.0,
            "non_marginable_buying_power": 20_000.0,
        }
    )
    # Selling 1 AMD $45 put requires $4,500 collateral (<= 5% concentration cap, but > $2,000 free cash)
    intent = OrderIntent(
        strategy=StrategyType.WHEEL,
        ticker="AMD",
        action=TradeAction.SELL_CSP,
        qty=1,
        notional_exposure_delta=4_500.0,
        limit_price=1.25,
        option_symbol="AMD261218P00045000",
        option_right=OptionRight.PUT,
        strike_price=45.0,
        expiration_date=date(2026, 12, 18),
        date_bucket="2026-11-12T10:35",
    )

    with caplog.at_level(logging.WARNING):
        decision, order_res = risk_gateway.submit(intent, portfolio)

    assert decision.status == RiskDecisionStatus.REJECTED
    assert decision.policy_id == PolicyId.TRD_POL_002
    assert order_res is None
    assert len(fake_trading_client.submitted_orders) == 0
    assert "TRD-POL-002" in caplog.text

    logs = ledger.list_audit_logs(policy_id="TRD-POL-002")
    assert len(logs) == 1
    assert logs[0].ticker == "AMD"


def test_trd_pol_003_covered_call_cost_basis_floor(
    risk_gateway: RiskGateway,
    ledger: LedgerRepository,
    fake_trading_client: FakeTradingClient,
    default_portfolio: PortfolioSnapshot,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    TRD-POL-003: Never sell a covered call with a strike price below the recorded average
    cost basis of the underlying 100-share lot (adjusted for prior collected premiums).
    """
    # Assigned 100 shares at $50.00 strike after collecting $2.00/sh premium -> net cost basis = $48.00
    ledger.record_csp_assignment(
        ticker="INTC",
        assignment_strike=50.0,
        shares_qty=100,
        prior_premium_per_share=2.0,
    )
    cycle = ledger.get_wheel_cycle("INTC")
    assert cycle.net_cost_basis_per_share == pytest.approx(48.0)

    # Attempt to sell a Covered Call at strike $47.00 (< $48.00 net cost basis)
    below_floor_intent = OrderIntent(
        strategy=StrategyType.WHEEL,
        ticker="INTC",
        action=TradeAction.SELL_CC,
        qty=1,
        notional_exposure_delta=0.0,
        limit_price=1.10,
        option_symbol="INTC261218C00047000",
        option_right=OptionRight.CALL,
        strike_price=47.0,
        expiration_date=date(2026, 12, 18),
        date_bucket="2026-11-12T10:40",
    )

    with caplog.at_level(logging.WARNING):
        decision, order_res = risk_gateway.submit(below_floor_intent, default_portfolio)

    assert decision.status == RiskDecisionStatus.REJECTED
    assert decision.policy_id == PolicyId.TRD_POL_003
    assert order_res is None
    assert len(fake_trading_client.submitted_orders) == 0
    assert "TRD-POL-003" in caplog.text

    logs = ledger.list_audit_logs(policy_id="TRD-POL-003")
    assert len(logs) == 1
    assert logs[0].ticker == "INTC"


def test_trd_pol_004_bounded_dip_laddering_cap_and_volatility_pause(
    risk_gateway: RiskGateway,
    ledger: LedgerRepository,
    fake_trading_client: FakeTradingClient,
    default_portfolio: PortfolioSnapshot,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    TRD-POL-004: Limit scaled dip buys to a maximum of 2 add-on tranches per position,
    strictly subject to TRD-POL-001 and paused if 20-day realized volatility exceeds threshold.
    """
    # Seed 2 already-used add-on tranches for NVDA
    ledger.upsert_dip_ladder(
        ticker="NVDA",
        price=120.0,
        add_on_tranches_used=2,
        realized_vol_20d=0.35,
    )

    # 3rd add-on tranche must be blocked by TRD-POL-004
    third_tranche_intent = OrderIntent(
        strategy=StrategyType.TRAILING_MOMENTUM,
        ticker="NVDA",
        action=TradeAction.ADD_DIP_TRANCHE,
        qty=5,
        notional_exposure_delta=575.0,
        limit_price=115.0,
        dip_tranche_index=3,
        realized_vol_20d=0.35,
        date_bucket="2026-11-12T11:00",
    )

    with caplog.at_level(logging.WARNING):
        decision_tranche, _ = risk_gateway.submit(third_tranche_intent, default_portfolio)

    assert decision_tranche.status == RiskDecisionStatus.REJECTED
    assert decision_tranche.policy_id == PolicyId.TRD_POL_004

    # High 20-day realized volatility (0.75 > 0.60 max) on 1st tranche for TSLA must also be blocked by TRD-POL-004
    high_vol_intent = OrderIntent(
        strategy=StrategyType.TRAILING_MOMENTUM,
        ticker="TSLA",
        action=TradeAction.ADD_DIP_TRANCHE,
        qty=5,
        notional_exposure_delta=1_000.0,
        limit_price=200.0,
        dip_tranche_index=1,
        realized_vol_20d=0.75,
        date_bucket="2026-11-12T11:05",
    )

    decision_vol, _ = risk_gateway.submit(high_vol_intent, default_portfolio)
    assert decision_vol.status == RiskDecisionStatus.REJECTED
    assert decision_vol.policy_id == PolicyId.TRD_POL_004

    logs = ledger.list_audit_logs(policy_id="TRD-POL-004")
    assert len(logs) == 2


def test_trd_pol_005_idempotency_and_per_ticker_cooldown(
    risk_gateway: RiskGateway,
    ledger: LedgerRepository,
    fake_trading_client: FakeTradingClient,
    default_portfolio: PortfolioSnapshot,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    TRD-POL-005: Deterministic client_order_id deduplication and 15-minute per-ticker cooldown window.
    """
    intent_1 = OrderIntent(
        strategy=StrategyType.SMART_MONEY,
        ticker="GOOG",
        action=TradeAction.BUY_EQUITY,
        qty=10,
        notional_exposure_delta=1_800.0,
        limit_price=180.0,
        trailing_stop_pct=0.05,
        date_bucket="2026-11-12T10:30",
    )

    # First submission succeeds
    decision_1, order_1 = risk_gateway.submit(intent_1, default_portfolio)
    assert decision_1.status == RiskDecisionStatus.APPROVED
    assert decision_1.policy_id == PolicyId.PASS
    assert order_1 is not None
    assert len(fake_trading_client.submitted_orders) == 1

    # Exact duplicate intent (same date_bucket -> identical client_order_id) is blocked
    with caplog.at_level(logging.WARNING):
        decision_dup, order_dup = risk_gateway.submit(intent_1, default_portfolio)

    assert decision_dup.status == RiskDecisionStatus.REJECTED
    assert decision_dup.policy_id == PolicyId.TRD_POL_005
    assert order_dup is None

    # New date_bucket only 5 minutes later (< 15m cooldown) for same ticker is also blocked by TRD-POL-005
    portfolio_5m_later = default_portfolio.model_copy(
        update={"as_of_ny": default_portfolio.as_of_ny + timedelta(minutes=5)}
    )
    intent_5m = intent_1.model_copy(update={"date_bucket": "2026-11-12T10:35"})
    decision_cooldown, _ = risk_gateway.submit(intent_5m, portfolio_5m_later)

    assert decision_cooldown.status == RiskDecisionStatus.REJECTED
    assert decision_cooldown.policy_id == PolicyId.TRD_POL_005

    logs = ledger.list_audit_logs(policy_id="TRD-POL-005")
    assert len(logs) == 2


def test_trd_pol_006_intraday_circuit_breaker_and_manual_unlock_lockout(
    risk_gateway: RiskGateway,
    ledger: LedgerRepository,
    fake_trading_client: FakeTradingClient,
    default_portfolio: PortfolioSnapshot,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    TRD-POL-006: If combined daily drawdown exceeds MAX_DAILY_DRAWDOWN_PCT (2%), immediately:
      1. Cancel all open orders (`cancel_orders()`)
      2. Write HALTED state to SQLite
      3. Block all further execution until manually cleared (`manual_unlock()`).
    """
    # Establish intraday peak of $100,000, then simulate a -3% drop to $97,000
    ledger.sync_equity(current_equity=100_000.0, daily_starting_equity=100_000.0, as_of_ny=default_portfolio.as_of_ny)
    crash_portfolio = default_portfolio.model_copy(
        update={
            "total_equity": 97_000.0,
            "daily_starting_equity": 100_000.0,
            "intraday_peak_equity": 100_000.0,
        }
    )

    intent = OrderIntent(
        strategy=StrategyType.TRAILING_MOMENTUM,
        ticker="MSFT",
        action=TradeAction.BUY_EQUITY,
        qty=5,
        notional_exposure_delta=2_000.0,
        limit_price=400.0,
        trailing_stop_pct=0.05,
        date_bucket="2026-11-12T11:15",
    )

    with caplog.at_level(logging.WARNING):
        decision, order_res = risk_gateway.submit(intent, crash_portfolio)

    assert decision.status == RiskDecisionStatus.HALT_TRIGGERED
    assert decision.policy_id == PolicyId.TRD_POL_006
    assert order_res is None
    assert fake_trading_client.cancel_orders_called == 1
    assert len(fake_trading_client.submitted_orders) == 0

    sys_state = ledger.get_system_state()
    assert sys_state is not None
    assert sys_state.trading_status == "HALTED"
    assert sys_state.halt_policy_id == "TRD-POL-006"

    # Even if equity recovers to $100,000, subsequent orders remain blocked until manual unlock
    recovered_portfolio = default_portfolio.model_copy(update={"total_equity": 100_000.0})
    intent_after = intent.model_copy(update={"ticker": "AMZN", "date_bucket": "2026-11-12T11:30"})
    decision_locked, _ = risk_gateway.submit(intent_after, recovered_portfolio)
    assert decision_locked.status == RiskDecisionStatus.REJECTED
    assert decision_locked.policy_id == PolicyId.TRD_POL_006

    # Manual unlock clears HALTED state and permits compliant orders
    ledger.manual_unlock(reset_baseline_equity=100_000.0, as_of_ny=recovered_portfolio.as_of_ny)
    intent_unlocked = intent.model_copy(update={"ticker": "AMZN", "date_bucket": "2026-11-12T11:45"})
    decision_ok, order_ok = risk_gateway.submit(intent_unlocked, recovered_portfolio)
    assert decision_ok.status == RiskDecisionStatus.APPROVED
    assert order_ok is not None
