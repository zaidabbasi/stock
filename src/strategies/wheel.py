"""Deterministic Options Wheel finite-state machine strategy (`IDLE` -> `CSP_OPEN` -> `ASSIGNED_SHARES` -> `CC_OPEN`)."""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Optional, Sequence

from src.config import Settings
from src.execution.broker import parse_occ_option_symbol
from src.intelligence.analyst import GeminiAnalyst
from src.intelligence.schemas import (
    MarketContextPacket,
    OptionContractQuote,
    OptionRight,
    OrderIntent,
    StrategyType,
    TradeAction,
    TradeProposal,
    WheelState,
)
from src.persistence.ledger import LedgerRepository, TrackedOrderRecord, WheelCycleRecord

logger = logging.getLogger(__name__)

_PENNY = Decimal("0.01")


def _round_tick(value: float) -> float:
    return float(Decimal(str(value)).quantize(_PENNY, rounding=ROUND_HALF_UP))


class WheelStrategy:
    """
    Deterministic Options Wheel Engine:
      - Sells ~10% OTM Cash-Secured Puts (14-30 DTE) on approved tickers when `IDLE`.
      - Enforces a 25% bid-ask spread haircut check (`spread_pct <= 0.25`) before submitting option limit orders.
      - Monitors open contracts (`CSP_OPEN`, `CC_OPEN`) for 50% early profit-taking (`mark <= 0.50 * entry_premium`).
      - Reconciles option fills, expirations, assignments, and call-aways from Alpaca positions & orders.
      - Sells ~10% OTM Covered Calls (14-30 DTE) when `ASSIGNED_SHARES`, subject to `TRD-POL-003` cost-basis floor.
    """

    def __init__(
        self,
        settings: Settings,
        ledger: LedgerRepository,
        analyst: Optional[GeminiAnalyst] = None,
        target_otm_pct: float = 0.10,
        min_dte: int = 14,
        max_dte: int = 30,
        early_profit_take_ratio: float = 0.50,
    ) -> None:
        self.settings = settings
        self.ledger = ledger
        self.analyst = analyst
        self.target_otm_pct = target_otm_pct
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.early_profit_take_ratio = early_profit_take_ratio

    @staticmethod
    def compute_haircut_limit_price(quote: OptionContractQuote, side: str = "SELL") -> float:
        """
        Apply a 25% bid-ask spread haircut to the mid price using exact decimal penny rounding
        to improve fill probability without crossing the entire spread.
        """
        width = max(0.0, quote.ask - quote.bid)
        if side.upper() == "SELL":
            raw = quote.mid - 0.25 * width
            return _round_tick(max(quote.bid, raw))
        raw = quote.mid + 0.25 * width
        return _round_tick(min(quote.ask, raw))

    def is_spread_acceptable(self, quote: OptionContractQuote) -> bool:
        """Verify bid-ask spread width does not exceed MAX_OPTION_SPREAD_PCT (default 25% of mid)."""
        return quote.spread_pct <= self.settings.MAX_OPTION_SPREAD_PCT + 1e-9

    def select_contract(
        self,
        spot_price: float,
        right: OptionRight,
        chain: Sequence[OptionContractQuote],
        as_of_date: date,
        min_strike_floor: Optional[float] = None,
        strict_floor: bool = False,
    ) -> Optional[OptionContractQuote]:
        """
        Deterministically select a 14-30 DTE option closest to ~10% OTM that passes the 25% spread haircut check.
        When `strict_floor=True` on CALLs, contracts below `min_strike_floor` are filtered out cleanly.
        """
        target_strike = (
            spot_price * (1.0 - self.target_otm_pct)
            if right == OptionRight.PUT
            else spot_price * (1.0 + self.target_otm_pct)
        )
        if min_strike_floor is not None and right == OptionRight.CALL:
            target_strike = max(target_strike, min_strike_floor)

        eligible: list[OptionContractQuote] = []
        for contract in chain:
            if contract.right != right:
                continue
            dte = (contract.expiration_date - as_of_date).days
            if not (self.min_dte <= dte <= self.max_dte):
                continue
            if strict_floor and min_strike_floor is not None and right == OptionRight.CALL:
                if contract.strike_price < min_strike_floor - 1e-6:
                    continue
            if not self.is_spread_acceptable(contract):
                logger.info(
                    "Rejecting option contract due to >25% bid-ask spread haircut gate",
                    extra={
                        "option_symbol": contract.option_symbol,
                        "spread_pct": round(contract.spread_pct, 4),
                        "max_allowed": self.settings.MAX_OPTION_SPREAD_PCT,
                    },
                )
                continue
            eligible.append(contract)

        if not eligible:
            return None

        eligible.sort(key=lambda c: abs(c.strike_price - target_strike))
        return eligible[0]

    def build_csp_entry_intent(
        self,
        packet: MarketContextPacket,
        chain: Sequence[OptionContractQuote],
        proposal: Optional[TradeProposal] = None,
    ) -> Optional[OrderIntent]:
        """
        When ticker is in `IDLE` state, evaluate via Gemini (or supplied `proposal`), pick ~10% OTM 14-30 DTE Put,
        verify <=25% spread width, and emit `SELL_CSP` OrderIntent.
        """
        cycle = self.ledger.get_wheel_cycle(packet.ticker)
        if cycle.state != WheelState.IDLE:
            return None

        evaluated_proposal = proposal
        if evaluated_proposal is None and self.analyst is not None:
            evaluated_proposal = self.analyst.evaluate_context(packet)

        if evaluated_proposal is not None:
            if (
                evaluated_proposal.action not in (TradeAction.SELL_CSP, TradeAction.BUY_EQUITY)
                or evaluated_proposal.conviction_score < self.settings.MIN_CONVICTION_SCORE
            ):
                return None

        contract = self.select_contract(
            spot_price=packet.price,
            right=OptionRight.PUT,
            chain=chain,
            as_of_date=packet.as_of_ny.date(),
        )
        if contract is None:
            return None

        limit_price = self.compute_haircut_limit_price(contract, side="SELL")
        if limit_price <= 0.0:
            return None
        assignment_notional = round(100.0 * contract.strike_price, 2)

        return OrderIntent(
            strategy=StrategyType.WHEEL,
            ticker=packet.ticker,
            action=TradeAction.SELL_CSP,
            qty=1,
            notional_exposure_delta=assignment_notional,
            limit_price=limit_price,
            option_symbol=contract.option_symbol,
            option_right=OptionRight.PUT,
            strike_price=contract.strike_price,
            expiration_date=contract.expiration_date,
            option_spread_pct=round(contract.spread_pct, 4),
            date_bucket=packet.as_of_ny.strftime("%Y-%m-%dT%H:%M"),
            proposal=evaluated_proposal,
        )

    def build_cc_entry_intent(
        self,
        ticker: str,
        spot_price: float,
        as_of_ny: datetime,
        contract: OptionContractQuote,
    ) -> Optional[OrderIntent]:
        """
        Construct a Covered Call (`SELL_CC`) intent for an `ASSIGNED_SHARES` position.
        Enforces the 25% bid-ask spread gate; passes strike validation to `RiskGateway` (`TRD-POL-003`).
        """
        if not self.is_spread_acceptable(contract):
            logger.warning(
                "Blocked CC order construction: option spread exceeds 25% threshold",
                extra={"option_symbol": contract.option_symbol, "spread_pct": contract.spread_pct},
            )
            return None

        dte = (contract.expiration_date - as_of_ny.date()).days
        if not (self.min_dte <= dte <= self.max_dte):
            return None

        limit_price = self.compute_haircut_limit_price(contract, side="SELL")
        if limit_price <= 0.0:
            return None
        return OrderIntent(
            strategy=StrategyType.WHEEL,
            ticker=ticker.strip().upper(),
            action=TradeAction.SELL_CC,
            qty=1,
            notional_exposure_delta=0.0,
            limit_price=limit_price,
            option_symbol=contract.option_symbol,
            option_right=OptionRight.CALL,
            strike_price=contract.strike_price,
            expiration_date=contract.expiration_date,
            option_spread_pct=round(contract.spread_pct, 4),
            date_bucket=as_of_ny.strftime("%Y-%m-%dT%H:%M"),
        )

    def check_early_profit_taking(
        self,
        ticker: str,
        current_quote: OptionContractQuote,
        as_of_ny: datetime,
    ) -> Optional[OrderIntent]:
        """
        Monitor an open CSP or CC contract and emit a buy-to-close (`BUY_TO_CLOSE_CSP` / `BUY_TO_CLOSE_CC`)
        intent when current option mark price drops to <= 50% of collected entry premium.
        """
        cycle = self.ledger.get_wheel_cycle(ticker)
        if cycle.state not in (WheelState.CSP_OPEN, WheelState.CC_OPEN):
            return None
        if cycle.entry_premium_per_share <= 0.0 or not cycle.active_option_symbol:
            return None

        target_buyback = self.early_profit_take_ratio * cycle.entry_premium_per_share
        if current_quote.mid <= target_buyback + 1e-6 and self.is_spread_acceptable(current_quote):
            action = (
                TradeAction.BUY_TO_CLOSE_CSP
                if cycle.state == WheelState.CSP_OPEN
                else TradeAction.BUY_TO_CLOSE_CC
            )
            limit_price = self.compute_haircut_limit_price(current_quote, side="BUY")
            if limit_price <= 0.0:
                return None
            return OrderIntent(
                strategy=StrategyType.WHEEL,
                ticker=cycle.ticker,
                action=action,
                qty=1,
                notional_exposure_delta=0.0,
                limit_price=limit_price,
                option_symbol=cycle.active_option_symbol,
                option_right=current_quote.right,
                strike_price=cycle.strike_price,
                expiration_date=current_quote.expiration_date,
                option_spread_pct=round(current_quote.spread_pct, 4),
                date_bucket=as_of_ny.strftime("%Y-%m-%dT%H:%M"),
            )
        return None

    def handle_csp_assignment(
        self,
        ticker: str,
        assignment_strike: Optional[float] = None,
        shares_qty: int = 100,
        as_of_ny: Optional[datetime] = None,
    ) -> WheelCycleRecord:
        """Transition ticker from `CSP_OPEN` to `ASSIGNED_SHARES` in SQLite ledger."""
        return self.ledger.record_csp_assignment(
            ticker=ticker,
            assignment_strike=assignment_strike,
            shares_qty=shares_qty,
            as_of_ny=as_of_ny,
        )

    def reconcile_filled_order(self, order: TrackedOrderRecord, as_of_ny: datetime) -> Optional[WheelCycleRecord]:
        """Advance Wheel FSM in SQLite when a tracked option order fills on Alpaca."""
        price = order.filled_avg_price if order.filled_avg_price is not None else (order.limit_price or 0.0)
        if order.action == TradeAction.SELL_CSP.value:
            return self.ledger.record_csp_open(
                ticker=order.ticker,
                option_symbol=order.symbol,
                strike_price=float(order.strike_price or 0.0),
                expiration_date=order.expiration_date or as_of_ny.date().isoformat(),
                premium_per_share=price,
                alpaca_order_id=order.alpaca_order_id,
                as_of_ny=as_of_ny,
            )
        if order.action == TradeAction.BUY_TO_CLOSE_CSP.value:
            return self.ledger.record_csp_closed_or_expired(
                ticker=order.ticker,
                buyback_debit_per_share=price,
                as_of_ny=as_of_ny,
            )
        if order.action == TradeAction.SELL_CC.value:
            return self.ledger.record_cc_open(
                ticker=order.ticker,
                option_symbol=order.symbol,
                strike_price=float(order.strike_price or 0.0),
                expiration_date=order.expiration_date or as_of_ny.date().isoformat(),
                premium_per_share=price,
                alpaca_order_id=order.alpaca_order_id,
                as_of_ny=as_of_ny,
            )
        if order.action == TradeAction.BUY_TO_CLOSE_CC.value:
            return self.ledger.record_cc_closed_or_expired(
                ticker=order.ticker,
                buyback_debit_per_share=price,
                as_of_ny=as_of_ny,
            )
        return None

    def reconcile_wheel_positions(
        self,
        ticker: str,
        positions: Sequence[Any],
        as_of_ny: datetime,
    ) -> WheelCycleRecord:
        """
        Reconcile SQLite `wheel_cycles` against live Alpaca positions to automatically detect:
          - Put assignment (`CSP_OPEN` -> `ASSIGNED_SHARES`)
          - Put expiration OTM (`CSP_OPEN` -> `IDLE`)
          - Call exercise / called away (`CC_OPEN` -> `IDLE`)
          - Call expiration OTM (`CC_OPEN` -> `ASSIGNED_SHARES`)
          - External equity close (`ASSIGNED_SHARES` -> `IDLE`)
        """
        tkr = ticker.strip().upper()
        cycle = self.ledger.get_wheel_cycle(tkr)

        equity_shares = 0
        equity_avg_entry = 0.0
        short_puts: list[tuple[str, date, float]] = []
        short_calls: list[tuple[str, date, float]] = []

        for pos in positions:
            sym = str(getattr(pos, "symbol", "")).strip().upper()
            qty_val = float(getattr(pos, "qty", 0.0) or 0.0)
            parsed = parse_occ_option_symbol(sym)
            if parsed is None:
                if sym == tkr and qty_val > 0:
                    equity_shares += int(round(qty_val))
                    equity_avg_entry = float(getattr(pos, "avg_entry_price", 0.0) or 0.0)
            else:
                underlying, exp_dt, right, strike = parsed
                if underlying != tkr:
                    continue
                if qty_val < 0 and right == OptionRight.PUT:
                    short_puts.append((sym, exp_dt, strike))
                elif qty_val < 0 and right == OptionRight.CALL:
                    short_calls.append((sym, exp_dt, strike))

        today = as_of_ny.date()

        if cycle.state == WheelState.CSP_OPEN:
            has_put = any(sym == cycle.active_option_symbol for sym, _, _ in short_puts) or bool(short_puts)
            if not has_put:
                if equity_shares >= 100:
                    logger.info(f"Detected CSP assignment for {tkr}: {equity_shares} shares held.")
                    return self.ledger.record_csp_assignment(
                        ticker=tkr,
                        assignment_strike=cycle.strike_price or equity_avg_entry,
                        shares_qty=equity_shares,
                        as_of_ny=as_of_ny,
                    )
                if cycle.expiration_date and today > date.fromisoformat(cycle.expiration_date):
                    logger.info(f"Detected OTM CSP expiration for {tkr}; releasing collateral.")
                    return self.ledger.record_csp_closed_or_expired(
                        ticker=tkr,
                        buyback_debit_per_share=0.0,
                        as_of_ny=as_of_ny,
                    )
        elif cycle.state == WheelState.CC_OPEN:
            has_call = any(sym == cycle.active_option_symbol for sym, _, _ in short_calls) or bool(short_calls)
            if not has_call:
                if equity_shares < 100:
                    logger.info(f"Detected Covered Call exercise (called away) for {tkr}; resetting cycle to IDLE.")
                    return self.ledger.record_cc_called_away(
                        ticker=tkr,
                        reset_to_idle=True,
                        as_of_ny=as_of_ny,
                    )
                if cycle.expiration_date and today > date.fromisoformat(cycle.expiration_date):
                    logger.info(f"Detected OTM Covered Call expiration for {tkr}; returning to ASSIGNED_SHARES.")
                    return self.ledger.record_cc_closed_or_expired(
                        ticker=tkr,
                        buyback_debit_per_share=0.0,
                        as_of_ny=as_of_ny,
                    )
        elif cycle.state == WheelState.ASSIGNED_SHARES:
            if equity_shares < 100:
                logger.info(f"Assigned shares no longer present on Alpaca for {tkr}; resetting to IDLE.")
                return self.ledger.reset_wheel_to_idle(tkr, as_of_ny=as_of_ny)
        elif cycle.state == WheelState.IDLE:
            # Self-heal if pre-existing positions already exist on the Paper account
            if short_puts:
                sym, exp_dt, strike = short_puts[0]
                return self.ledger.record_csp_open(
                    ticker=tkr,
                    option_symbol=sym,
                    strike_price=strike,
                    expiration_date=exp_dt.isoformat(),
                    premium_per_share=0.0,
                    as_of_ny=as_of_ny,
                )
            if equity_shares >= 100 and equity_avg_entry > 0:
                rec = self.ledger.record_csp_assignment(
                    ticker=tkr,
                    assignment_strike=equity_avg_entry,
                    shares_qty=equity_shares,
                    prior_premium_per_share=0.0,
                    as_of_ny=as_of_ny,
                )
                if short_calls:
                    sym, exp_dt, strike = short_calls[0]
                    rec = self.ledger.record_cc_open(
                        ticker=tkr,
                        option_symbol=sym,
                        strike_price=strike,
                        expiration_date=exp_dt.isoformat(),
                        premium_per_share=0.0,
                        as_of_ny=as_of_ny,
                    )
                return rec

        return cycle
