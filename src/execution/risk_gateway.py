"""Mandatory synchronous RiskGateway enforcing TRD-POL-001 through TRD-POL-006 before order execution."""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from src.config import NY_TZ, Settings
from src.execution.broker import AlpacaBroker, BrokerOrderResult
from src.intelligence.schemas import (
    OrderIntent,
    PolicyId,
    PortfolioSnapshot,
    RiskDecision,
    RiskDecisionStatus,
    TradeAction,
    WheelState,
)
from src.persistence.ledger import LedgerRepository

logger = logging.getLogger(__name__)

_CLOSING_ACTIONS = frozenset({TradeAction.BUY_TO_CLOSE_CSP, TradeAction.BUY_TO_CLOSE_CC})


def generate_deterministic_client_order_id(
    strategy_id: str,
    ticker: str,
    action: str,
    date_bucket: str,
) -> str:
    """
    TRD-POL-005: Deterministic `client_order_id` hash from `strategy_id + ticker + action + date_bucket`.
    """
    canonical = f"{strategy_id.strip().upper()}|{ticker.strip().upper()}|{action.strip().upper()}|{date_bucket.strip()}"
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return f"agt-{ticker.strip().upper()[:5]}-{digest}"


class RiskGateway:
    """
    Synchronous pre-trade risk gate that intercepts every `OrderIntent` prior to `TradingClient.submit_order()`.
    Enforces:
      - TRD-POL-001: Single-Ticker Concentration Cap (<= MAX_POSITION_PCT of total equity)
      - TRD-POL-002: Cash-Secured Put Collateral Lock (available cash covers 100 * strike)
      - TRD-POL-003: Covered Call Cost-Basis Floor (CC strike >= net_cost_basis_per_share)
      - TRD-POL-004: Bounded Dip-Laddering Cap (max 2 add-on tranches, subject to TRD-POL-001 & 20d vol ceiling)
      - TRD-POL-005: Idempotency & Rate Cooldown (deterministic client_order_id + 15-min per-ticker entry cooldown)
      - TRD-POL-006: Portfolio Intraday Circuit Breaker (max daily loss from daily_starting_equity -> cancel_orders() + HALTED)
    """

    def __init__(
        self,
        settings: Settings,
        ledger: LedgerRepository,
        broker: AlpacaBroker,
    ) -> None:
        self.settings = settings
        self.ledger = ledger
        self.broker = broker

    def check_intraday_circuit_breaker(self, portfolio: PortfolioSnapshot) -> Optional[str]:
        """
        Evaluate TRD-POL-006 against `portfolio` and SQLite `system_state`.
        Persists intraday peak equity while measuring daily realized + unrealized loss
        against `daily_starting_equity`. If breached or already HALTED, cancels all open
        orders, writes `HALTED` state to SQLite, and returns the reason string.
        """
        existing_state = self.ledger.get_system_state()
        if existing_state is not None and existing_state.trading_status == "HALTED":
            return (
                f"Portfolio circuit breaker is already HALTED "
                f"(reason='{existing_state.halt_reason}'). Run `python -m src.cli unlock` to clear."
            )

        today_ny = portfolio.as_of_ny.astimezone(NY_TZ).strftime("%Y-%m-%d")
        baseline_for_sync = (
            existing_state.daily_starting_equity
            if (existing_state is not None and existing_state.trading_date_ny == today_ny)
            else portfolio.daily_starting_equity
        )

        synced = self.ledger.sync_equity(
            current_equity=portfolio.total_equity,
            daily_starting_equity=baseline_for_sync,
            as_of_ny=portfolio.as_of_ny,
        )
        daily_start = synced.daily_starting_equity
        if daily_start <= 0:
            return None

        daily_loss_pct = (daily_start - portfolio.total_equity) / daily_start
        if daily_loss_pct > self.settings.MAX_DAILY_DRAWDOWN_PCT + 1e-9:
            reason = (
                f"Daily loss {daily_loss_pct:.2%} exceeded MAX_DAILY_DRAWDOWN_PCT "
                f"({self.settings.MAX_DAILY_DRAWDOWN_PCT:.2%}): equity=${portfolio.total_equity:,.2f}, "
                f"daily_starting_equity=${daily_start:,.2f}."
            )
            logger.warning(
                "TRD-POL-006 circuit breaker triggered: halting system and cancelling all open orders",
                extra={
                    "policy_id": PolicyId.TRD_POL_006.value,
                    "drawdown_pct": round(daily_loss_pct, 6),
                    "current_equity": portfolio.total_equity,
                    "high_water_mark": synced.intraday_peak_equity,
                },
            )
            self.broker.cancel_all_open_orders()
            self.ledger.trigger_halt(
                reason=reason,
                policy_id=PolicyId.TRD_POL_006.value,
                current_equity=portfolio.total_equity,
                as_of_ny=portfolio.as_of_ny,
            )
            return reason

        return None

    def evaluate(
        self,
        intent: OrderIntent,
        portfolio: PortfolioSnapshot,
        execution_status: str = "DRY_RUN",
    ) -> RiskDecision:
        """
        Synchronously validate `intent` against TRD-POL-001 .. TRD-POL-006 and log decision to `audit_log`.
        Does NOT mutate `order_cooldowns` or `dip_ladders` — those are persisted ONLY after confirmed
        broker order acceptance inside `submit()`.
        """
        client_order_id = generate_deterministic_client_order_id(
            strategy_id=intent.strategy.value,
            ticker=intent.ticker,
            action=intent.action.value,
            date_bucket=intent.date_bucket,
        )

        # ---------------------------------------------------------------------
        # 1. TRD-POL-006: Portfolio Intraday Circuit Breaker
        # ---------------------------------------------------------------------
        existing = self.ledger.get_system_state()
        was_already_halted = existing is not None and existing.trading_status == "HALTED"
        cb_reason = self.check_intraday_circuit_breaker(portfolio)
        if cb_reason is not None:
            decision_status = (
                RiskDecisionStatus.REJECTED if was_already_halted else RiskDecisionStatus.HALT_TRIGGERED
            )
            return self._reject(
                intent=intent,
                client_order_id=client_order_id,
                policy_id=PolicyId.TRD_POL_006,
                reason=cb_reason,
                portfolio=portfolio,
                status=decision_status,
            )

        # ---------------------------------------------------------------------
        # 2. TRD-POL-005: Idempotency & Rate Cooldown
        # ---------------------------------------------------------------------
        if self.ledger.has_client_order_id(client_order_id):
            return self._reject(
                intent=intent,
                client_order_id=client_order_id,
                policy_id=PolicyId.TRD_POL_005,
                reason=f"Duplicate idempotent client_order_id '{client_order_id}' already submitted.",
                portfolio=portfolio,
            )

        if intent.action not in _CLOSING_ACTIONS:
            last_epoch = self.ledger.get_last_order_epoch_for_ticker(intent.ticker)
            if last_epoch is not None:
                cooldown_seconds = self.settings.ORDER_COOLDOWN_MINUTES * 60.0
                elapsed_seconds = portfolio.as_of_ny.timestamp() - last_epoch
                if elapsed_seconds < cooldown_seconds:
                    remaining_min = (cooldown_seconds - elapsed_seconds) / 60.0
                    return self._reject(
                        intent=intent,
                        client_order_id=client_order_id,
                        policy_id=PolicyId.TRD_POL_005,
                        reason=(
                            f"Ticker '{intent.ticker}' is within {self.settings.ORDER_COOLDOWN_MINUTES}m "
                            f"order cooldown ({remaining_min:.1f}m remaining)."
                        ),
                        portfolio=portfolio,
                    )

        # ---------------------------------------------------------------------
        # 3. TRD-POL-004: Bounded Dip-Laddering Cap (Max 2 tranches & Vol Gate)
        # ---------------------------------------------------------------------
        if intent.action == TradeAction.ADD_DIP_TRANCHE or intent.dip_tranche_index > 0:
            ladder = self.ledger.get_dip_ladder(intent.ticker)
            recorded_tranches = ladder.add_on_tranches_used if ladder is not None else 0
            target_tranche = max(recorded_tranches + 1, intent.dip_tranche_index)

            if target_tranche > self.settings.MAX_DIP_TRANCHES:
                return self._reject(
                    intent=intent,
                    client_order_id=client_order_id,
                    policy_id=PolicyId.TRD_POL_004,
                    reason=(
                        f"Dip-ladder tranche #{target_tranche} for '{intent.ticker}' exceeds "
                        f"MAX_DIP_TRANCHES ({self.settings.MAX_DIP_TRANCHES})."
                    ),
                    portfolio=portfolio,
                )

            effective_vol = max(
                intent.realized_vol_20d,
                ladder.realized_vol_20d if ladder is not None else 0.0,
            )
            is_paused = (ladder is not None and ladder.paused_high_vol) or (
                effective_vol > self.settings.MAX_REALIZED_VOL_20D
            )
            if is_paused:
                return self._reject(
                    intent=intent,
                    client_order_id=client_order_id,
                    policy_id=PolicyId.TRD_POL_004,
                    reason=(
                        f"Dip-laddering paused for '{intent.ticker}': 20-day realized volatility "
                        f"({effective_vol:.2%}) exceeds MAX_REALIZED_VOL_20D ({self.settings.MAX_REALIZED_VOL_20D:.2%})."
                    ),
                    portfolio=portfolio,
                )

        # ---------------------------------------------------------------------
        # 4. TRD-POL-001: Single-Ticker Concentration Cap (<= 5% Total Equity)
        # ---------------------------------------------------------------------
        exposure_increasing_actions = {
            TradeAction.BUY_EQUITY,
            TradeAction.ADD_DIP_TRANCHE,
            TradeAction.SELL_CSP,
        }
        if intent.action in exposure_increasing_actions:
            existing_exposure = portfolio.ticker_notional_exposure.get(intent.ticker, 0.0)
            if intent.action == TradeAction.SELL_CSP:
                csp_assignment_notional = 100.0 * (intent.strike_price or 0.0) * intent.qty
                added_exposure = max(intent.notional_exposure_delta, csp_assignment_notional)
            else:
                added_exposure = intent.notional_exposure_delta

            projected_exposure = existing_exposure + added_exposure
            max_allowed_exposure = self.settings.MAX_POSITION_PCT * portfolio.total_equity

            if projected_exposure > max_allowed_exposure + 1e-6:
                return self._reject(
                    intent=intent,
                    client_order_id=client_order_id,
                    policy_id=PolicyId.TRD_POL_001,
                    reason=(
                        f"Projected notional exposure for '{intent.ticker}' (${projected_exposure:,.2f}) "
                        f"exceeds MAX_POSITION_PCT ({self.settings.MAX_POSITION_PCT:.2%} of "
                        f"${portfolio.total_equity:,.2f} = ${max_allowed_exposure:,.2f})."
                    ),
                    portfolio=portfolio,
                )

        # ---------------------------------------------------------------------
        # 5. TRD-POL-002: Cash-Secured Put Collateral Lock
        # ---------------------------------------------------------------------
        if intent.action == TradeAction.SELL_CSP:
            if intent.strike_price is None or intent.strike_price <= 0:
                return self._reject(
                    intent=intent,
                    client_order_id=client_order_id,
                    policy_id=PolicyId.TRD_POL_002,
                    reason="SELL_CSP intent missing valid positive strike_price.",
                    portfolio=portfolio,
                )
            required_collateral = 100.0 * intent.strike_price * intent.qty
            ledger_reserved = self.ledger.get_total_reserved_csp_collateral()
            # Avoid double-subtracting CSP collateral already deducted by Alpaca in buying power
            uncounted_reserved = max(0.0, ledger_reserved - portfolio.broker_reserved_csp_collateral)
            available_buying_power = portfolio.non_marginable_buying_power - uncounted_reserved
            if portfolio.options_buying_power is not None:
                available_buying_power = min(
                    available_buying_power,
                    portfolio.options_buying_power - uncounted_reserved,
                )

            if available_buying_power < required_collateral - 1e-6:
                return self._reject(
                    intent=intent,
                    client_order_id=client_order_id,
                    policy_id=PolicyId.TRD_POL_002,
                    reason=(
                        f"Insufficient cash collateral for '{intent.ticker}' CSP: required=${required_collateral:,.2f}, "
                        f"available=${available_buying_power:,.2f} (non_marginable_bp=${portfolio.non_marginable_buying_power:,.2f}, "
                        f"uncounted_reserved=${uncounted_reserved:,.2f})."
                    ),
                    portfolio=portfolio,
                )

        # ---------------------------------------------------------------------
        # 6. TRD-POL-003: Covered Call Cost-Basis Floor
        # ---------------------------------------------------------------------
        if intent.action == TradeAction.SELL_CC:
            wheel_cycle = self.ledger.get_wheel_cycle(intent.ticker)
            if wheel_cycle.state != WheelState.ASSIGNED_SHARES or wheel_cycle.assigned_shares_qty < 100 * intent.qty:
                return self._reject(
                    intent=intent,
                    client_order_id=client_order_id,
                    policy_id=PolicyId.TRD_POL_003,
                    reason=(
                        f"Cannot sell Covered Call on '{intent.ticker}': Wheel state is '{wheel_cycle.state.value}' "
                        f"with {wheel_cycle.assigned_shares_qty} assigned shares (need >= {100 * intent.qty})."
                    ),
                    portfolio=portfolio,
                )
            if intent.strike_price is None or intent.strike_price < wheel_cycle.net_cost_basis_per_share - 1e-6:
                return self._reject(
                    intent=intent,
                    client_order_id=client_order_id,
                    policy_id=PolicyId.TRD_POL_003,
                    reason=(
                        f"Covered Call strike (${intent.strike_price or 0.0:.2f}) for '{intent.ticker}' "
                        f"is below net cost-basis floor (${wheel_cycle.net_cost_basis_per_share:.2f}/share)."
                    ),
                    portfolio=portfolio,
                )

        # ---------------------------------------------------------------------
        # All 6 policies passed -> Record APPROVED audit row (state mutated on broker confirm)
        # ---------------------------------------------------------------------
        audit_id = self.ledger.append_audit_log(
            client_order_id=client_order_id,
            intent=intent,
            risk_decision=RiskDecisionStatus.APPROVED,
            policy_id=PolicyId.PASS,
            rejection_reason=None,
            as_of_ny=portfolio.as_of_ny,
            execution_status=execution_status,
        )
        return RiskDecision(
            status=RiskDecisionStatus.APPROVED,
            policy_id=PolicyId.PASS,
            client_order_id=client_order_id,
            reason="Order intent passed TRD-POL-001 through TRD-POL-006.",
            audit_log_id=audit_id,
        )

    def submit(
        self,
        intent: OrderIntent,
        portfolio: PortfolioSnapshot,
        require_market_open: bool = True,
        dry_run: bool = False,
    ) -> tuple[RiskDecision, Optional[BrokerOrderResult]]:
        """
        Intercept `intent` via `evaluate()` and, if `APPROVED` and not `dry_run`, submit to `AlpacaBroker`.
        Cooldowns, dip-ladder tranches, and `tracked_orders` are persisted ONLY after confirmed broker submission.
        """
        decision = self.evaluate(
            intent=intent,
            portfolio=portfolio,
            execution_status="DRY_RUN" if dry_run else "PENDING_SUBMIT",
        )
        if decision.status != RiskDecisionStatus.APPROVED or dry_run:
            return decision, None

        try:
            order_result = self.broker.submit_order_intent(
                intent=intent,
                client_order_id=decision.client_order_id,
                require_market_open=require_market_open,
            )
        except Exception as exc:
            self.ledger.update_audit_execution(
                audit_id=decision.audit_log_id,
                execution_status="SUBMIT_FAILED",
                execution_error=str(exc),
            )
            raise

        self.ledger.update_audit_execution(
            audit_id=decision.audit_log_id,
            alpaca_order_id=order_result.alpaca_order_id,
            fill_price=order_result.filled_avg_price,
            filled_qty=order_result.filled_qty,
            execution_status="SUBMITTED",
        )
        self.ledger.record_order_cooldown(
            client_order_id=decision.client_order_id,
            ticker=intent.ticker,
            strategy=intent.strategy.value,
            action=intent.action.value,
            date_bucket=intent.date_bucket,
            as_of_ny=portfolio.as_of_ny,
        )
        if intent.action == TradeAction.BUY_EQUITY:
            self.ledger.upsert_dip_ladder(
                ticker=intent.ticker,
                price=intent.limit_price or (intent.notional_exposure_delta / max(intent.qty, 1)),
                add_on_tranches_used=0,
                realized_vol_20d=intent.realized_vol_20d,
                paused_high_vol=False,
                as_of_ny=portfolio.as_of_ny,
                reset=True,
            )
        elif intent.action == TradeAction.ADD_DIP_TRANCHE:
            existing_ladder = self.ledger.get_dip_ladder(intent.ticker)
            used = (existing_ladder.add_on_tranches_used if existing_ladder else 0) + 1
            self.ledger.upsert_dip_ladder(
                ticker=intent.ticker,
                price=intent.limit_price or (intent.notional_exposure_delta / max(intent.qty, 1)),
                add_on_tranches_used=used,
                realized_vol_20d=intent.realized_vol_20d,
                paused_high_vol=False,
                as_of_ny=portfolio.as_of_ny,
            )

        self.ledger.track_order(
            intent=intent,
            alpaca_order_id=order_result.alpaca_order_id,
            client_order_id=decision.client_order_id,
            audit_log_id=decision.audit_log_id,
            trailing_stop_pct=intent.trailing_stop_pct,
            as_of_ny=portfolio.as_of_ny,
        )
        if intent.action == TradeAction.BUY_EQUITY and intent.trailing_stop_pct:
            self.ledger.update_tracked_order(
                order_result.alpaca_order_id,
                protection_status="PENDING",
            )

        return decision, order_result

    def _reject(
        self,
        intent: OrderIntent,
        client_order_id: str,
        policy_id: PolicyId,
        reason: str,
        portfolio: PortfolioSnapshot,
        status: RiskDecisionStatus = RiskDecisionStatus.REJECTED,
    ) -> RiskDecision:
        logger.warning(
            f"RiskGateway blocked order [{policy_id.value}]: {reason}",
            extra={
                "policy_id": policy_id.value,
                "ticker": intent.ticker,
                "strategy": intent.strategy.value,
                "action": intent.action.value,
                "client_order_id": client_order_id,
                "reason": reason,
            },
        )
        audit_id = self.ledger.append_audit_log(
            client_order_id=client_order_id,
            intent=intent,
            risk_decision=status,
            policy_id=policy_id,
            rejection_reason=reason,
            as_of_ny=portfolio.as_of_ny,
        )
        return RiskDecision(
            status=status,
            policy_id=policy_id,
            client_order_id=client_order_id,
            reason=reason,
            audit_log_id=audit_id,
        )
