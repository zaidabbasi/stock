"""Unit & integration tests for OrderClass.MLEG Iron Condors, Deeper Bull Put Spreads, and zero-cost RULES mode."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from alpaca.trading.enums import OrderClass, PositionIntent
from src.config import NY_TZ, Settings
from src.execution.broker import AlpacaBroker
from src.execution.risk_gateway import RiskGateway, compute_defined_risk_spread_metrics
from src.intelligence.analyst import GeminiAnalyst
from src.intelligence.schemas import (
    MarketContextPacket,
    OptionContractQuote,
    OptionLegSpec,
    OptionRight,
    OrderIntent,
    PolicyId,
    PortfolioSnapshot,
    RiskDecisionStatus,
    StrategyType,
    TradeAction,
)
from src.main import TradingOrchestrator
from src.persistence.ledger import LedgerRepository
from src.strategies.credit_spreads import CreditSpreadStrategy
from tests.conftest import FakeTradingClient


def _make_googl_packet(as_of_ny: datetime) -> MarketContextPacket:
    return MarketContextPacket(
        ticker="GOOGL",
        strategy_candidate=StrategyType.CREDIT_SPREAD,
        as_of_ny=as_of_ny,
        price=298.70,
        atr_14d=11.13,
        rsi_14d=37.2,
        realized_vol_20d=0.54,
        return_20d=-0.10,
        avg_daily_dollar_volume=12_200_000_000.0,
        spread_width_pct=0.001,
        catalyst_summary="Google Cloud quarterly growth steady",
    )


def _make_googl_chains(exp: date, as_of_ny: datetime) -> tuple[list[OptionContractQuote], list[OptionContractQuote]]:
    puts = [
        OptionContractQuote(
            option_symbol=f"GOOGL{exp:%y%m%d}P00255000", underlying_ticker="GOOGL", right=OptionRight.PUT,
            strike_price=255.0, expiration_date=exp, bid=0.14, ask=0.17,
            delta=-0.018, theta=-0.02, implied_volatility=0.38,
        ),
        OptionContractQuote(
            option_symbol=f"GOOGL{exp:%y%m%d}P00260000", underlying_ticker="GOOGL", right=OptionRight.PUT,
            strike_price=260.0, expiration_date=exp, bid=0.28, ask=0.32,
            delta=-0.042, theta=-0.04, implied_volatility=0.39,
        ),
        OptionContractQuote(
            option_symbol=f"GOOGL{exp:%y%m%d}P00265000", underlying_ticker="GOOGL", right=OptionRight.PUT,
            strike_price=265.0, expiration_date=exp, bid=0.44, ask=0.50,
            delta=-0.068, theta=-0.06, implied_volatility=0.39,
        ),
        OptionContractQuote(
            option_symbol=f"GOOGL{exp:%y%m%d}P00270000", underlying_ticker="GOOGL", right=OptionRight.PUT,
            strike_price=270.0, expiration_date=exp, bid=0.29, ask=0.37,
            delta=-0.095, theta=-0.07, implied_volatility=0.40,
        ),
        OptionContractQuote(
            option_symbol=f"GOOGL{exp:%y%m%d}P00275000", underlying_ticker="GOOGL", right=OptionRight.PUT,
            strike_price=275.0, expiration_date=exp, bid=0.58, ask=0.66,
            delta=-0.105, theta=-0.09, implied_volatility=0.41,
        ),
    ]
    calls = [
        OptionContractQuote(
            option_symbol=f"GOOGL{exp:%y%m%d}C00325000", underlying_ticker="GOOGL", right=OptionRight.CALL,
            strike_price=325.0, expiration_date=exp, bid=0.64, ask=0.72,
            delta=0.092, theta=-0.08, implied_volatility=0.39,
        ),
        OptionContractQuote(
            option_symbol=f"GOOGL{exp:%y%m%d}C00330000", underlying_ticker="GOOGL", right=OptionRight.CALL,
            strike_price=330.0, expiration_date=exp, bid=0.41, ask=0.45,
            delta=0.060, theta=-0.06, implied_volatility=0.39,
        ),
    ]
    return puts, calls


def test_iron_condor_and_deeper_bull_spread_non_overlapping(
    test_settings: Settings,
    ledger: LedgerRepository,
    default_portfolio: PortfolioSnapshot,
    base_ny_time: datetime,
) -> None:
    """Iron Condor (~10 delta) + Deeper Bull Spread (~4-5 delta) avoid strike collisions."""
    exp = base_ny_time.date() + timedelta(days=8)
    packet = _make_googl_packet(base_ny_time)
    puts, calls = _make_googl_chains(exp, base_ny_time)

    strategy = CreditSpreadStrategy(settings=test_settings, ledger=ledger)
    ic_intent = strategy.build_iron_condor_intent(packet, puts, calls, default_portfolio)
    assert ic_intent is not None
    assert ic_intent.action == TradeAction.OPEN_IRON_CONDOR
    assert len(ic_intent.legs) == 4
    assert ic_intent.limit_price is not None and ic_intent.limit_price > 0.20
    assert ic_intent.notional_exposure_delta <= 5_000.0

    # Persist IC as OPEN so the deeper bull spread sees the held IC strikes
    ic_short_put = next(l for l in ic_intent.legs if l.right == OptionRight.PUT and l.side == "SELL")
    ic_long_put = next(l for l in ic_intent.legs if l.right == OptionRight.PUT and l.side == "BUY")
    assert ic_short_put.strike_price == 275.0
    assert ic_long_put.strike_price == 270.0

    import json

    ledger.record_spread_pending_open(
        spread_id=ic_intent.spread_id or "ic-test",
        ticker="GOOGL",
        structure="IRON_CONDOR",
        expiration_date=exp.isoformat(),
        qty=ic_intent.qty,
        entry_credit_per_spread=ic_intent.limit_price,
        max_risk_usd=ic_intent.notional_exposure_delta,
        profit_target_debit=round(ic_intent.limit_price * 0.5, 2),
        stop_loss_debit=round(ic_intent.limit_price * 2.5, 2),
        legs_json=json.dumps([l.model_dump(mode="json") for l in ic_intent.legs]),
        open_order_id="ord-ic-1",
        as_of_ny=base_ny_time,
    )
    ledger.mark_spread_open(
        ic_intent.spread_id or "ic-test",
        filled_credit_per_spread=ic_intent.limit_price,
        profit_take_ratio=0.5,
        stop_loss_multiple=2.5,
        as_of_ny=base_ny_time,
    )

    updated_portfolio = default_portfolio.model_copy(
        update={"ticker_notional_exposure": {"GOOGL": ic_intent.notional_exposure_delta}}
    )
    bps_intent = strategy.build_bull_put_spread_intent(
        packet, puts, updated_portfolio, deeper_otm=True
    )
    assert bps_intent is not None
    assert bps_intent.action == TradeAction.OPEN_BULL_PUT_SPREAD
    assert len(bps_intent.legs) == 2

    bps_short_put = next(l for l in bps_intent.legs if l.side == "SELL")
    bps_long_put = next(l for l in bps_intent.legs if l.side == "BUY")
    assert bps_short_put.strike_price <= ic_long_put.strike_price - 2.5
    assert bps_long_put.strike_price < bps_short_put.strike_price
    ic_syms = {l.option_symbol for l in ic_intent.legs}
    bps_syms = {l.option_symbol for l in bps_intent.legs}
    assert ic_syms.isdisjoint(bps_syms)
    weekly_expected_credit = (ic_intent.limit_price * ic_intent.qty + (bps_intent.limit_price or 0.0) * bps_intent.qty) * 100.0
    assert weekly_expected_credit >= 30.0


def test_mleg_order_request_signed_limit_and_snapshot_netting(
    test_settings: Settings,
    broker: AlpacaBroker,
    fake_trading_client: FakeTradingClient,
) -> None:
    """Verify MLEG negative limit on credit open, positive limit on debit close, and spread netting."""
    exp = date(2026, 11, 20)
    ic_legs = (
        OptionLegSpec(option_symbol="GOOGL261120P00275000", right=OptionRight.PUT, strike_price=275.0, expiration_date=exp, side="SELL", position_intent="sell_to_open"),
        OptionLegSpec(option_symbol="GOOGL261120P00270000", right=OptionRight.PUT, strike_price=270.0, expiration_date=exp, side="BUY", position_intent="buy_to_open"),
        OptionLegSpec(option_symbol="GOOGL261120C00325000", right=OptionRight.CALL, strike_price=325.0, expiration_date=exp, side="SELL", position_intent="sell_to_open"),
        OptionLegSpec(option_symbol="GOOGL261120C00330000", right=OptionRight.CALL, strike_price=330.0, expiration_date=exp, side="BUY", position_intent="buy_to_open"),
    )
    open_intent = OrderIntent(
        strategy=StrategyType.CREDIT_SPREAD,
        ticker="GOOGL",
        action=TradeAction.OPEN_IRON_CONDOR,
        qty=1,
        notional_exposure_delta=450.0,
        limit_price=0.50,
        legs=ic_legs,
        spread_id="ic-test",
        date_bucket="2026-11-12",
    )
    req_open = broker.build_order_request(open_intent, "cid-open-1")
    assert req_open.order_class == OrderClass.MLEG
    assert req_open.symbol is None
    assert req_open.side is None
    assert req_open.limit_price == pytest.approx(-0.50)
    assert len(req_open.legs) == 4
    assert req_open.legs[0].position_intent == PositionIntent.SELL_TO_OPEN

    close_legs = tuple(
        l.model_copy(
            update={
                "side": "BUY" if l.side == "SELL" else "SELL",
                "position_intent": "buy_to_close" if l.side == "SELL" else "sell_to_close",
            }
        )
        for l in ic_legs
    )
    close_intent = OrderIntent(
        strategy=StrategyType.CREDIT_SPREAD,
        ticker="GOOGL",
        action=TradeAction.CLOSE_IRON_CONDOR,
        qty=1,
        notional_exposure_delta=0.0,
        limit_price=0.24,
        legs=close_legs,
        spread_id="ic-test",
        date_bucket="2026-11-12:11",
    )
    req_close = broker.build_order_request(close_intent, "cid-close-1")
    assert req_close.order_class == OrderClass.MLEG
    assert req_close.limit_price == pytest.approx(0.24)
    assert req_close.legs[0].position_intent == PositionIntent.BUY_TO_CLOSE

    # Verify get_portfolio_snapshot pairs 4 IC option positions as $500 max risk (not 100 * 275 = $27,500!)
    fake_trading_client.positions = [
        SimpleNamespace(symbol="GOOGL261120P00275000", asset_class="us_option", qty="-1", market_value="-60.0"),
        SimpleNamespace(symbol="GOOGL261120P00270000", asset_class="us_option", qty="1", market_value="35.0"),
        SimpleNamespace(symbol="GOOGL261120C00325000", asset_class="us_option", qty="-1", market_value="-65.0"),
        SimpleNamespace(symbol="GOOGL261120C00330000", asset_class="us_option", qty="1", market_value="40.0"),
    ]
    fake_trading_client.get_orders = lambda filter=None: [
        SimpleNamespace(
            symbol=None,
            qty="1",
            filled_qty="0",
            side=None,
            legs=[
                SimpleNamespace(symbol="GOOGL261120P00260000", side="sell", position_intent="sell_to_open"),
                SimpleNamespace(symbol="GOOGL261120P00255000", side="buy", position_intent="buy_to_open"),
            ],
        )
    ]
    snap = broker.get_portfolio_snapshot()
    # Held IC wing = $500 + open pending $5-wide bull put spread = $500 -> $1,000 total GOOGL exposure
    assert snap.ticker_notional_exposure["GOOGL"] == pytest.approx(1000.0)


def test_risk_gateway_spread_validation_and_close_bypasses_cooldown(
    risk_gateway: RiskGateway,
    ledger: LedgerRepository,
    fake_trading_client: FakeTradingClient,
    default_portfolio: PortfolioSnapshot,
) -> None:
    """Ensure defined-risk metrics reject naked legs under TRD-POL-002 and cooldown allows exits."""
    exp = date(2026, 11, 20)
    naked_intent = OrderIntent(
        strategy=StrategyType.CREDIT_SPREAD,
        ticker="GOOGL",
        action=TradeAction.OPEN_BULL_PUT_SPREAD,
        qty=1,
        notional_exposure_delta=500.0,
        limit_price=0.30,
        legs=(
            OptionLegSpec(
                option_symbol="GOOGL261120P00270000", right=OptionRight.PUT,
                strike_price=270.0, expiration_date=exp, side="SELL", position_intent="sell_to_open",
            ),
        ),
        spread_id="bad-bps",
        date_bucket="2026-11-12",
    )
    ok, _w, _m, reason = compute_defined_risk_spread_metrics(naked_intent)
    assert not ok
    dec_bad, _ = risk_gateway.submit(naked_intent, default_portfolio)
    assert dec_bad.status == RiskDecisionStatus.REJECTED
    assert dec_bad.policy_id == PolicyId.TRD_POL_002
    assert len(fake_trading_client.submitted_orders) == 0

    # Valid defined-risk 5-wide bull put spread: $500 width - $35 credit = $465 max loss (< 5% of $100k)
    valid_bps = OrderIntent(
        strategy=StrategyType.CREDIT_SPREAD,
        ticker="GOOGL",
        action=TradeAction.OPEN_BULL_PUT_SPREAD,
        qty=1,
        notional_exposure_delta=465.0,
        limit_price=0.35,
        legs=(
            OptionLegSpec(
                option_symbol="GOOGL261120P00265000", right=OptionRight.PUT,
                strike_price=265.0, expiration_date=exp, side="SELL", position_intent="sell_to_open",
            ),
            OptionLegSpec(
                option_symbol="GOOGL261120P00260000", right=OptionRight.PUT,
                strike_price=260.0, expiration_date=exp, side="BUY", position_intent="buy_to_open",
            ),
        ),
        spread_id="bps-good-1",
        date_bucket="2026-11-12",
    )
    dec_ok, order_res = risk_gateway.submit(valid_bps, default_portfolio)
    assert dec_ok.status == RiskDecisionStatus.APPROVED
    assert order_res is not None
    assert len(fake_trading_client.submitted_orders) == 1

    # Immediate 2nd entry on GOOGL blocked by TRD-POL-005 15m cooldown
    second_entry = valid_bps.model_copy(update={"spread_id": "bps-good-2", "date_bucket": "2026-11-12-2"})
    dec_cd, _ = risk_gateway.submit(second_entry, default_portfolio)
    assert dec_cd.status == RiskDecisionStatus.REJECTED
    assert dec_cd.policy_id == PolicyId.TRD_POL_005

    # Closing exit bypasses cooldown cleanly
    ledger.mark_spread_open("bps-good-1", filled_credit_per_spread=0.35, profit_take_ratio=0.5, stop_loss_multiple=2.5, as_of_ny=default_portfolio.as_of_ny)
    close_bps = OrderIntent(
        strategy=StrategyType.CREDIT_SPREAD,
        ticker="GOOGL",
        action=TradeAction.CLOSE_BULL_PUT_SPREAD,
        qty=1,
        notional_exposure_delta=0.0,
        limit_price=0.16,
        legs=(
            OptionLegSpec(
                option_symbol="GOOGL261120P00265000", right=OptionRight.PUT,
                strike_price=265.0, expiration_date=exp, side="BUY", position_intent="buy_to_close",
            ),
            OptionLegSpec(
                option_symbol="GOOGL261120P00260000", right=OptionRight.PUT,
                strike_price=260.0, expiration_date=exp, side="SELL", position_intent="sell_to_close",
            ),
        ),
        spread_id="bps-good-1",
        date_bucket="2026-11-12:exit",
    )
    dec_close, close_res = risk_gateway.submit(close_bps, default_portfolio)
    assert dec_close.status == RiskDecisionStatus.APPROVED
    assert close_res is not None
    sp = ledger.get_spread_position("bps-good-1")
    assert sp is not None and sp.state == "PENDING_CLOSE"


class _FakeBar:
    def __init__(self, close: float) -> None:
        self.high = close * 1.01
        self.low = close * 0.99
        self.close = close
        self.volume = 50_000_000.0


def test_orchestrator_mleg_e2e_rules_mode_zero_gemini_calls(
    tmp_path: Any,
    base_ny_time: datetime,
) -> None:
    """End-to-end RULES mode ($0 Gemini cost): opens IC -> waits cooldown -> opens deeper BPS -> closes IC at 50% profit."""
    exp = base_ny_time.date() + timedelta(days=8)
    puts, calls = _make_googl_chains(exp, base_ny_time)

    settings = Settings(
        TRADING_MODE="PAPER",
        ANALYST_MODE="RULES",
        GEMINI_API_KEY="DO_NOT_CALL",
        MOMENTUM_WATCHLIST="",
        WHEEL_WATCHLIST="",
        CREDIT_SPREAD_WATCHLIST="GOOGL",
        WEEKLY_CREDIT_TARGET_USD=30.0,
        ORDER_COOLDOWN_MINUTES=15,
        SQLITE_DB_PATH=tmp_path / "mleg_e2e.db",
    )
    ledger = LedgerRepository(db_path=settings.SQLITE_DB_PATH)
    client = FakeTradingClient()
    client.clock.timestamp = base_ny_time

    orders_by_id: dict[str, Any] = {}

    def _submit(order_data: Any) -> Any:
        res = FakeTradingClient.submit_order(client, order_data)
        orders_by_id[res.id] = SimpleNamespace(
            id=res.id,
            status="filled",
            filled_avg_price=str(order_data.limit_price),
            filled_qty=str(order_data.qty),
        )
        return res

    client.submit_order = _submit  # type: ignore[method-assign]
    client.get_order_by_id = lambda oid: orders_by_id[oid]  # type: ignore[attr-defined]
    client.get_orders = lambda filter=None: []  # type: ignore[attr-defined]

    broker = AlpacaBroker(
        settings=settings,
        trading_client=client,
        stock_data_client=SimpleNamespace(get_stock_bars=lambda *a, **kw: None),
        option_data_client=SimpleNamespace(
            get_option_chain=lambda *a, **kw: {},
            get_option_latest_quote=lambda *a, **kw: {},
        ),
    )
    broker.get_stock_daily_bars = lambda ticker, days=60: [_FakeBar(298.0 + (i % 3) * 0.5) for i in range(30)]  # type: ignore[method-assign]
    broker.get_stock_quote_snapshot = lambda ticker: (298.60, 298.80, 298.70)  # type: ignore[method-assign]
    broker.get_recent_news_headlines = lambda ticker: []  # type: ignore[method-assign]

    quotes_state = {
        OptionRight.PUT: list(puts),
        OptionRight.CALL: list(calls),
    }
    broker.get_option_chain_quotes = lambda ticker, right, min_dte=1, max_dte=45, as_of_date=None: quotes_state[right]  # type: ignore[method-assign]

    def _get_option_quote(sym: str) -> Optional[OptionContractQuote]:
        for q in quotes_state[OptionRight.PUT] + quotes_state[OptionRight.CALL]:
            if q.option_symbol == sym:
                return q
        return None

    broker.get_option_quote = _get_option_quote  # type: ignore[method-assign]

    analyst = GeminiAnalyst(settings=settings, genai_client=None)
    gateway = RiskGateway(settings=settings, ledger=ledger, broker=broker)
    orch = TradingOrchestrator(settings=settings, ledger=ledger, broker=broker, risk_gateway=gateway, analyst=analyst)

    # Cycle 1: Opens IC (post-submit reconcile immediately fills IC -> OPEN); TRD-POL-005 blocks immediate BPS
    c1 = orch.run_single_cycle()
    assert c1["status"] == "ACTIVE"
    assert c1["actions_evaluated"] == ["CREDIT_SPREAD:OPEN_IRON_CONDOR:GOOGL"]
    assert len(client.submitted_orders) == 1
    assert len(client.submitted_orders[0].legs) == 4
    assert c1["reconciled"]["fills_reconciled"] == 1
    assert len(ledger.list_audit_logs(policy_id="TRD-POL-005")) == 1

    # Cycle 2 (+16 minutes): Cooldown expired -> opens Deeper Bull Put Spread -> reconciles to OPEN!
    client.clock.timestamp = base_ny_time + timedelta(minutes=16)
    c2 = orch.run_single_cycle()
    assert "CREDIT_SPREAD:OPEN_BULL_PUT_SPREAD:GOOGL" in c2["actions_evaluated"]
    assert len(client.submitted_orders) == 2
    assert len(client.submitted_orders[1].legs) == 2
    assert len(ledger.list_spread_positions(ticker="GOOGL", states=("OPEN",))) == 2

    # Cycle 3 (+17 minutes): Decays option prices so IC & BPS hit 50% profit target -> emits CLOSE MLEG orders & closes both!
    client.clock.timestamp = base_ny_time + timedelta(minutes=33)
    quotes_state[OptionRight.PUT] = [
        replace(q, bid=round(q.bid * 0.20, 2), ask=round(q.ask * 0.20, 2))
        for q in quotes_state[OptionRight.PUT]
    ]
    quotes_state[OptionRight.CALL] = [
        replace(q, bid=round(q.bid * 0.20, 2), ask=round(q.ask * 0.20, 2))
        for q in quotes_state[OptionRight.CALL]
    ]
    c3 = orch.run_single_cycle()
    close_evals = [a for a in c3["actions_evaluated"] if ":CLOSE_" in a]
    assert len(close_evals) == 2
    assert len(client.submitted_orders) == 4
    assert client.submitted_orders[2].limit_price > 0  # positive debit to close!
    assert client.submitted_orders[3].limit_price > 0

    closed_spreads = ledger.list_spread_positions(states=("CLOSED",))
    assert len(closed_spreads) == 2
    assert all(s.realized_pnl_usd > 0.0 for s in closed_spreads)


def test_check_spread_exit_stop_loss_trigger(
    test_settings: Settings,
    ledger: LedgerRepository,
    base_ny_time: datetime,
) -> None:
    """Verify check_spread_exit triggers CLOSE_BULL_PUT_SPREAD when debit reaches stop_loss_debit."""
    import json

    exp = base_ny_time.date() + timedelta(days=7)
    short_sym = f"GOOGL{exp:%y%m%d}P00265000"
    long_sym = f"GOOGL{exp:%y%m%d}P00260000"
    legs = [
        OptionLegSpec(
            option_symbol=short_sym, right=OptionRight.PUT, strike_price=265.0,
            expiration_date=exp, side="SELL", position_intent="sell_to_open",
        ).model_dump(mode="json"),
        OptionLegSpec(
            option_symbol=long_sym, right=OptionRight.PUT, strike_price=260.0,
            expiration_date=exp, side="BUY", position_intent="buy_to_open",
        ).model_dump(mode="json"),
    ]
    ledger.record_spread_pending_open(
        spread_id="bps-stop-1",
        ticker="GOOGL",
        structure="BULL_PUT_SPREAD",
        expiration_date=exp.isoformat(),
        qty=1,
        entry_credit_per_spread=0.40,
        max_risk_usd=460.0,
        profit_target_debit=0.20,
        stop_loss_debit=1.00,
        legs_json=json.dumps(legs),
        open_order_id="ord-stop-open",
        as_of_ny=base_ny_time,
    )
    ledger.mark_spread_open(
        "bps-stop-1",
        filled_credit_per_spread=0.40,
        profit_take_ratio=0.50,
        stop_loss_multiple=2.5,
        as_of_ny=base_ny_time,
    )
    rec = ledger.get_spread_position("bps-stop-1")
    assert rec is not None

    # Widened quotes producing ~1.10 debit (>= 1.00 stop_loss_debit)
    quotes = {
        short_sym: OptionContractQuote(
            option_symbol=short_sym, underlying_ticker="GOOGL", right=OptionRight.PUT,
            strike_price=265.0, expiration_date=exp, bid=1.70, ask=1.80,
        ),
        long_sym: OptionContractQuote(
            option_symbol=long_sym, underlying_ticker="GOOGL", right=OptionRight.PUT,
            strike_price=260.0, expiration_date=exp, bid=0.60, ask=0.70,
        ),
    }
    strategy = CreditSpreadStrategy(settings=test_settings, ledger=ledger)
    exit_intent = strategy.check_spread_exit(rec, quotes, base_ny_time + timedelta(days=1))
    assert exit_intent is not None
    assert exit_intent.action == TradeAction.CLOSE_BULL_PUT_SPREAD
    assert exit_intent.limit_price is not None and exit_intent.limit_price >= 1.00

