"""Headless market-hours orchestrator loop with structured JSON logging and graceful SIGINT/SIGTERM shutdown."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime, time as dt_time
from types import FrameType
from typing import Any, Optional, Sequence

import numpy as np

from src.config import NY_TZ, SecurityConfigurationError, Settings, load_settings
from src.execution.broker import AlpacaBroker
from src.execution.risk_gateway import RiskGateway
from src.intelligence.analyst import GeminiAnalyst
from src.intelligence.schemas import (
    MarketContextPacket,
    OptionRight,
    OrderIntent,
    PortfolioSnapshot,
    RiskDecisionStatus,
    StrategyType,
    TradeAction,
    WheelState,
)
from src.persistence.ledger import LedgerRepository
from src.strategies.indicators import (
    compute_atr_14,
    compute_bid_ask_spread_pct,
    compute_realized_vol_20d,
    compute_rsi_14,
)
from src.strategies.smart_money_tracker import (
    SmartMoneyTrackerStrategy,
    load_disclosures_from_file,
)
from src.strategies.trailing_momentum import TrailingMomentumStrategy
from src.strategies.wheel import WheelStrategy


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line structured JSON anchored to America/New_York."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp_ny": datetime.now(tz=NY_TZ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for extra_key in (
            "policy_id",
            "ticker",
            "strategy",
            "action",
            "client_order_id",
            "reason",
            "drawdown_pct",
            "current_equity",
            "high_water_mark",
            "operation",
        ):
            if hasattr(record, extra_key):
                payload[extra_key] = getattr(record, extra_key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_json_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredJsonFormatter())
    root.handlers = [handler]


class TradingOrchestrator:
    """
    Coordinates market-session verification, portfolio drawdown monitoring, order/position
    reconciliation, two-step trailing-stop attachment, and all three strategy engines.
    """

    def __init__(
        self,
        settings: Settings,
        ledger: LedgerRepository,
        broker: AlpacaBroker,
        risk_gateway: RiskGateway,
        analyst: GeminiAnalyst,
    ) -> None:
        self.settings = settings
        self.ledger = ledger
        self.broker = broker
        self.risk_gateway = risk_gateway
        self.analyst = analyst
        self.trailing_momentum = TrailingMomentumStrategy(
            settings=settings,
            analyst=analyst,
            ledger=ledger,
        )
        self.smart_money = SmartMoneyTrackerStrategy(
            settings=settings,
            analyst=analyst,
        )
        self.wheel = WheelStrategy(
            settings=settings,
            ledger=ledger,
            analyst=analyst,
        )
        self._shutdown_requested = False
        self._logger = logging.getLogger(__name__)

    def request_shutdown(self, signum: int, _frame: Optional[FrameType] = None) -> None:
        sig_name = signal.Signals(signum).name if signum in signal.Signals._value2member_map_ else str(signum)
        self._logger.info(f"Received {sig_name}; initiating clean orchestrator shutdown.")
        self._shutdown_requested = True

    def _is_in_entry_blackout(self, now_ny: datetime, next_close_ny: datetime) -> bool:
        """Avoid opening new positions immediately after open or right before close."""
        market_open_dt = datetime.combine(now_ny.date(), dt_time(9, 30), tzinfo=NY_TZ)
        mins_since_open = (now_ny - market_open_dt).total_seconds() / 60.0
        mins_to_close = (next_close_ny - now_ny).total_seconds() / 60.0
        if 0.0 <= mins_since_open < self.settings.ENTRY_BLACKOUT_MINUTES_AFTER_OPEN:
            return True
        if 0.0 <= mins_to_close < self.settings.ENTRY_BLACKOUT_MINUTES_BEFORE_CLOSE:
            return True
        return False

    def reconcile_orders_and_positions(self, as_of_ny: datetime, dry_run: bool = False) -> dict[str, int]:
        """
        Reconcile open tracked orders with Alpaca:
          1. Mark filled / canceled / expired orders in SQLite `tracked_orders` and `audit_log`.
          2. Advance Wheel FSM on option order fills.
          3. Submit standalone GTC `TrailingStopOrderRequest` once a `BUY_EQUITY` entry fills.
          4. Reconcile Wheel assignments / expirations against live Alpaca positions.
        """
        summary = {"fills_reconciled": 0, "trailing_stops_placed": 0, "wheel_cycles_reconciled": 0}

        for order in self.ledger.list_tracked_orders(status="OPEN"):
            try:
                status_res = self.broker.get_order_status(order.alpaca_order_id)
                if status_res is None:
                    continue
                if status_res.status == "filled":
                    filled_qty = status_res.filled_qty if status_res.filled_qty is not None else order.qty
                    fill_price = status_res.filled_avg_price
                    self.ledger.update_tracked_order(
                        order.alpaca_order_id,
                        status="FILLED",
                        filled_qty=filled_qty,
                        filled_avg_price=fill_price,
                    )
                    if order.audit_log_id is not None:
                        self.ledger.update_audit_execution(
                            audit_id=order.audit_log_id,
                            alpaca_order_id=order.alpaca_order_id,
                            fill_price=fill_price,
                            filled_qty=filled_qty,
                        )
                    updated_order = next(
                        (
                            o
                            for o in self.ledger.list_tracked_orders(status="FILLED")
                            if o.alpaca_order_id == order.alpaca_order_id
                        ),
                        order,
                    )
                    if order.action in (
                        TradeAction.SELL_CSP.value,
                        TradeAction.BUY_TO_CLOSE_CSP.value,
                        TradeAction.SELL_CC.value,
                        TradeAction.BUY_TO_CLOSE_CC.value,
                    ):
                        self.wheel.reconcile_filled_order(updated_order, as_of_ny=as_of_ny)
                    summary["fills_reconciled"] += 1
                elif status_res.status in ("canceled", "cancelled", "expired", "rejected", "done_for_day"):
                    self.ledger.update_tracked_order(
                        order.alpaca_order_id,
                        status="CLOSED_UNFILLED",
                        last_error=f"Broker order ended with status={status_res.status}",
                    )
            except Exception as exc:
                self._logger.warning(
                    f"Order reconciliation error for {order.alpaca_order_id}: {exc}",
                    extra={"ticker": order.ticker},
                )

        if not dry_run:
            pending_stops = self.ledger.list_tracked_orders(
                status="FILLED",
                protection_status=("PENDING", "FAILED"),
            )
            for order in pending_stops:
                if order.action != TradeAction.BUY_EQUITY.value or not order.trailing_stop_pct:
                    continue
                qty_to_protect = int(round(order.filled_qty or order.qty))
                if qty_to_protect <= 0:
                    continue
                try:
                    trail_client_id = f"trail-{order.client_order_id[:28]}"
                    res = self.broker.submit_trailing_stop(
                        ticker=order.ticker,
                        qty=qty_to_protect,
                        trailing_stop_pct=order.trailing_stop_pct,
                        client_order_id=trail_client_id,
                    )
                    self.ledger.update_tracked_order(
                        order.alpaca_order_id,
                        protection_status="PLACED",
                        protection_order_id=res.alpaca_order_id,
                        last_error=None,
                    )
                    summary["trailing_stops_placed"] += 1
                    self._logger.info(
                        f"Placed {order.trailing_stop_pct:.1%} GTC trailing stop for filled {order.ticker} position",
                        extra={"ticker": order.ticker, "action": "TRAILING_STOP_PLACED"},
                    )
                except Exception as exc:
                    self.ledger.update_tracked_order(
                        order.alpaca_order_id,
                        protection_status="FAILED",
                        last_error=str(exc)[:300],
                    )
                    self._logger.warning(
                        f"Failed placing trailing stop for {order.ticker}: {exc}",
                        extra={"ticker": order.ticker},
                    )

        try:
            positions = self.broker.get_positions_raw()
            for tkr in self.settings.wheel_tickers:
                self.wheel.reconcile_wheel_positions(ticker=tkr, positions=positions, as_of_ny=as_of_ny)
                summary["wheel_cycles_reconciled"] += 1
        except Exception as exc:
            self._logger.warning(f"Wheel position reconciliation error: {exc}")

        return summary

    def _build_market_context(
        self,
        ticker: str,
        strategy: StrategyType,
        as_of_ny: datetime,
    ) -> Optional[MarketContextPacket]:
        """Fetch bars, quote, and news from Alpaca data clients and compute deterministic indicators."""
        if not hasattr(self.broker.stock_data_client, "get_stock_bars"):
            return None

        bars = self.broker.get_stock_daily_bars(ticker, days=60)
        if len(bars) < 15:
            self._logger.info(f"Insufficient bar history ({len(bars)}) for {ticker}; skipping.", extra={"ticker": ticker})
            return None

        highs = [float(b.high) for b in bars]
        lows = [float(b.low) for b in bars]
        closes = [float(b.close) for b in bars]
        volumes = [float(b.volume) for b in bars]

        atr_14 = compute_atr_14(highs, lows, closes)
        rsi_14 = compute_rsi_14(closes)
        vol_20d = compute_realized_vol_20d(closes)
        dollar_vols = [c * v for c, v in zip(closes[-20:], volumes[-20:])]
        avg_dollar_vol = float(np.mean(dollar_vols)) if dollar_vols else 0.0
        lookback = min(20, len(closes) - 1)
        return_20d = (closes[-1] - closes[-1 - lookback]) / closes[-1 - lookback] if closes[-1 - lookback] > 0 else 0.0

        bid, ask, last_trade = self.broker.get_stock_quote_snapshot(ticker)
        if bid > 0.0 and ask > 0.0 and ask >= bid:
            spread_pct = compute_bid_ask_spread_pct(bid, ask)
            price = 0.5 * (bid + ask) if spread_pct <= 0.05 else (last_trade or closes[-1])
        else:
            spread_pct = 0.005
            price = last_trade or closes[-1]

        if price <= 0.0:
            return None

        news_summary = self.broker.get_recent_news_summary(ticker, limit=5)
        catalyst = news_summary or f"Quantitative technical snapshot only for {ticker}; no breaking headlines."

        return MarketContextPacket(
            ticker=ticker,
            as_of_ny=as_of_ny,
            strategy_candidate=strategy,
            price=round(price, 4),
            atr_14d=round(atr_14, 4),
            rsi_14d=round(rsi_14, 2),
            realized_vol_20d=round(vol_20d, 4),
            spread_width_pct=round(spread_pct, 4),
            avg_daily_dollar_volume=round(avg_dollar_vol, 2),
            return_20d=round(return_20d, 4),
            catalyst_summary=catalyst,
        )

    def _submit_intent(
        self,
        intent: OrderIntent,
        portfolio: PortfolioSnapshot,
        dry_run: bool,
        submitted_counter: list[str],
    ) -> PortfolioSnapshot:
        """Pass `intent` through `RiskGateway.submit()` and track intra-cycle exposure updates."""
        try:
            decision, _order_res = self.risk_gateway.submit(
                intent=intent,
                portfolio=portfolio,
                require_market_open=True,
                dry_run=dry_run,
            )
            if decision.status == RiskDecisionStatus.APPROVED:
                submitted_counter.append(f"{intent.strategy.value}:{intent.action.value}:{intent.ticker}")
                if intent.action in (
                    TradeAction.BUY_EQUITY,
                    TradeAction.ADD_DIP_TRANCHE,
                    TradeAction.SELL_CSP,
                ):
                    return portfolio.with_added_exposure(intent.ticker, intent.notional_exposure_delta)
        except Exception as exc:
            self._logger.error(
                f"Order submission failed for {intent.ticker} ({intent.action.value}): {exc}",
                extra={"ticker": intent.ticker, "strategy": intent.strategy.value, "action": intent.action.value},
            )
        return portfolio

    def run_single_cycle(self, dry_run: bool = False) -> dict[str, Any]:
        """
        Execute one full deterministic trading cycle:
          1. Verify system_state is not HALTED.
          2. Check Alpaca market clock (`America/New_York`).
          3. Sync portfolio equity, record `equity_snapshots`, and evaluate `TRD-POL-006`.
          4. Reconcile tracked orders, place trailing stops on filled equity buys, and update Wheel FSM.
          5. If market is open, execute Wheel, Trailing Momentum, and Smart Money Tracker strategies.
        """
        sys_state = self.ledger.get_system_state()
        if sys_state is not None and sys_state.trading_status == "HALTED":
            self._logger.warning(
                "Orchestrator cycle skipped: system is locked in HALTED state.",
                extra={"policy_id": sys_state.halt_policy_id, "reason": sys_state.halt_reason},
            )
            return {"status": "HALTED", "reason": sys_state.halt_reason}

        clock = self.broker.get_market_clock()
        today_ny = clock.timestamp_ny.strftime("%Y-%m-%d")
        daily_start_override = (
            sys_state.daily_starting_equity
            if (sys_state is not None and sys_state.trading_date_ny == today_ny)
            else None
        )
        portfolio = self.broker.get_portfolio_snapshot(daily_starting_equity=daily_start_override)

        cb_reason = self.risk_gateway.check_intraday_circuit_breaker(portfolio)
        synced_after = self.ledger.get_system_state()
        if synced_after is not None:
            self.ledger.record_equity_snapshot(
                equity=portfolio.total_equity,
                daily_starting_equity=synced_after.daily_starting_equity,
                intraday_peak_equity=synced_after.intraday_peak_equity,
                as_of_ny=clock.timestamp_ny,
            )

        if cb_reason is not None:
            return {"status": "HALT_TRIGGERED", "reason": cb_reason}

        reconcile_summary = self.reconcile_orders_and_positions(as_of_ny=clock.timestamp_ny, dry_run=dry_run)

        if not clock.is_open:
            self._logger.info(
                f"Market closed (as_of_ny={clock.timestamp_ny.isoformat()}, "
                f"next_open_ny={clock.next_open_ny.isoformat()}). Standing by."
            )
            return {
                "status": "MARKET_CLOSED",
                "as_of_ny": clock.timestamp_ny.isoformat(),
                "next_open_ny": clock.next_open_ny.isoformat(),
                "equity": portfolio.total_equity,
                "reconciled": reconcile_summary,
            }

        submitted_actions: list[str] = []
        in_blackout = self._is_in_entry_blackout(clock.timestamp_ny, clock.next_close_ny)
        has_option_data = hasattr(self.broker.option_data_client, "get_option_chain")
        has_stock_data = hasattr(self.broker.stock_data_client, "get_stock_bars")

        # ---------------------------------------------------------------------
        # Strategy 1: Options Wheel (`WHEEL_WATCHLIST`)
        # ---------------------------------------------------------------------
        for tkr in self.settings.wheel_tickers:
            if self.ledger.count_open_tracked(tkr) > 0:
                continue
            cycle = self.ledger.get_wheel_cycle(tkr)
            try:
                if cycle.state in (WheelState.CSP_OPEN, WheelState.CC_OPEN) and cycle.active_option_symbol:
                    if hasattr(self.broker.option_data_client, "get_option_latest_quote"):
                        opt_quote = self.broker.get_option_quote(cycle.active_option_symbol)
                        if opt_quote is not None:
                            btc_intent = self.wheel.check_early_profit_taking(
                                ticker=tkr,
                                current_quote=opt_quote,
                                as_of_ny=clock.timestamp_ny,
                            )
                            if btc_intent is not None:
                                portfolio = self._submit_intent(btc_intent, portfolio, dry_run, submitted_actions)

                elif cycle.state == WheelState.ASSIGNED_SHARES and not in_blackout and has_option_data and has_stock_data:
                    packet = self._build_market_context(tkr, StrategyType.WHEEL, clock.timestamp_ny)
                    if packet is None:
                        continue
                    call_chain = self.broker.get_option_chain_quotes(
                        ticker=tkr,
                        right=OptionRight.CALL,
                        min_dte=self.wheel.min_dte,
                        max_dte=self.wheel.max_dte,
                        as_of_date=clock.timestamp_ny.date(),
                    )
                    selected_call = self.wheel.select_contract(
                        spot_price=packet.price,
                        right=OptionRight.CALL,
                        chain=call_chain,
                        as_of_date=clock.timestamp_ny.date(),
                        min_strike_floor=cycle.net_cost_basis_per_share,
                        strict_floor=True,
                    )
                    if selected_call is not None:
                        cc_intent = self.wheel.build_cc_entry_intent(
                            ticker=tkr,
                            spot_price=packet.price,
                            as_of_ny=clock.timestamp_ny,
                            contract=selected_call,
                        )
                        if cc_intent is not None:
                            portfolio = self._submit_intent(cc_intent, portfolio, dry_run, submitted_actions)

                elif cycle.state == WheelState.IDLE and not in_blackout and has_option_data and has_stock_data:
                    packet = self._build_market_context(tkr, StrategyType.WHEEL, clock.timestamp_ny)
                    if packet is None:
                        continue
                    put_chain = self.broker.get_option_chain_quotes(
                        ticker=tkr,
                        right=OptionRight.PUT,
                        min_dte=self.wheel.min_dte,
                        max_dte=self.wheel.max_dte,
                        as_of_date=clock.timestamp_ny.date(),
                    )
                    csp_intent = self.wheel.build_csp_entry_intent(packet=packet, chain=put_chain)
                    if csp_intent is not None:
                        portfolio = self._submit_intent(csp_intent, portfolio, dry_run, submitted_actions)
            except Exception as exc:
                self._logger.warning(f"Wheel strategy cycle error for {tkr}: {exc}", extra={"ticker": tkr})

        # ---------------------------------------------------------------------
        # Strategy 2: Trailing Momentum & Dip-Laddering (`MOMENTUM_WATCHLIST`)
        # ---------------------------------------------------------------------
        if has_stock_data and not in_blackout:
            for tkr in self.settings.momentum_tickers:
                if self.ledger.count_open_tracked(tkr) > 0:
                    continue
                try:
                    packet = self._build_market_context(tkr, StrategyType.TRAILING_MOMENTUM, clock.timestamp_ny)
                    if packet is None:
                        continue
                    has_exposure = portfolio.ticker_notional_exposure.get(tkr, 0.0) > 0.0
                    ladder = self.ledger.get_dip_ladder(tkr)

                    if has_exposure and ladder is not None:
                        if ladder.add_on_tranches_used < self.settings.MAX_DIP_TRANCHES:
                            dip_intent = self.trailing_momentum.build_dip_ladder_intent(
                                packet=packet,
                                portfolio=portfolio,
                            )
                            if dip_intent is not None:
                                portfolio = self._submit_intent(dip_intent, portfolio, dry_run, submitted_actions)
                    elif not has_exposure:
                        entry_intent = self.trailing_momentum.evaluate_entry(
                            packet=packet,
                            portfolio=portfolio,
                        )
                        if entry_intent is not None:
                            portfolio = self._submit_intent(entry_intent, portfolio, dry_run, submitted_actions)
                except Exception as exc:
                    self._logger.warning(f"Momentum strategy error for {tkr}: {exc}", extra={"ticker": tkr})

        # ---------------------------------------------------------------------
        # Strategy 3: Smart Money Tracker (`DISCLOSURE_FEED_PATH`)
        # ---------------------------------------------------------------------
        if not in_blackout:
            for disclosure in load_disclosures_from_file(self.settings.DISCLOSURE_FEED_PATH):
                if (
                    portfolio.ticker_notional_exposure.get(disclosure.ticker, 0.0) > 0.0
                    or self.ledger.count_open_tracked(disclosure.ticker) > 0
                ):
                    continue
                try:
                    res = self.smart_money.evaluate_disclosure(
                        disclosure=disclosure,
                        portfolio=portfolio,
                        as_of_ny=clock.timestamp_ny,
                    )
                    if res.passed and res.intent is not None:
                        portfolio = self._submit_intent(res.intent, portfolio, dry_run, submitted_actions)
                except Exception as exc:
                    self._logger.warning(
                        f"SmartMoney strategy error for {disclosure.ticker}: {exc}",
                        extra={"ticker": disclosure.ticker},
                    )

        # Immediately attach trailing stops / advance Wheel state if any newly submitted orders filled fast
        if submitted_actions and not dry_run:
            post_reconcile = self.reconcile_orders_and_positions(as_of_ny=clock.timestamp_ny, dry_run=dry_run)
            for k, v in post_reconcile.items():
                reconcile_summary[k] = reconcile_summary.get(k, 0) + v

        self._logger.info(
            "Market session cycle complete.",
            extra={"current_equity": portfolio.total_equity},
        )
        return {
            "status": "ACTIVE",
            "dry_run": dry_run,
            "as_of_ny": clock.timestamp_ny.isoformat(),
            "equity": portfolio.total_equity,
            "non_marginable_buying_power": portfolio.non_marginable_buying_power,
            "in_entry_blackout": in_blackout,
            "actions_evaluated": submitted_actions,
            "reconciled": reconcile_summary,
        }

    def run_loop(
        self,
        poll_interval_seconds: float = 60.0,
        max_cycles: Optional[int] = None,
        dry_run: bool = False,
    ) -> None:
        cycles = 0
        while not self._shutdown_requested:
            self.run_single_cycle(dry_run=dry_run)
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            if not self._shutdown_requested:
                time.sleep(poll_interval_seconds)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Headless orchestrator for alpaca-gemini-trader.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single orchestration cycle and exit cleanly.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run data fetch, analyst, and RiskGateway checks without submitting live paper orders.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Polling interval in seconds during continuous execution (defaults to POLL_INTERVAL_SECONDS).",
    )
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
        configure_json_logging(settings.LOG_LEVEL)
        settings.require_broker_credentials()
    except SecurityConfigurationError as exc:
        logging.getLogger(__name__).critical(f"Security configuration check failed: {exc}")
        return 2

    ledger = LedgerRepository(db_path=settings.SQLITE_DB_PATH)
    broker = AlpacaBroker(settings=settings)
    analyst = GeminiAnalyst(settings=settings)
    risk_gateway = RiskGateway(settings=settings, ledger=ledger, broker=broker)

    orchestrator = TradingOrchestrator(
        settings=settings,
        ledger=ledger,
        broker=broker,
        risk_gateway=risk_gateway,
        analyst=analyst,
    )

    signal.signal(signal.SIGINT, orchestrator.request_shutdown)
    signal.signal(signal.SIGTERM, orchestrator.request_shutdown)

    if args.once:
        result = orchestrator.run_single_cycle(dry_run=args.dry_run)
        print(json.dumps(result, indent=2, default=str))
        return 0

    interval = args.interval if args.interval is not None else float(settings.POLL_INTERVAL_SECONDS)
    orchestrator.run_loop(poll_interval_seconds=interval, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
