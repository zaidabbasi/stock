"""Defined-risk weekly multi-leg option credit strategies (Iron Condor + Deeper Bull Put Spread)."""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Sequence

from src.config import Settings
from src.intelligence.analyst import GeminiAnalyst
from src.intelligence.schemas import (
    MarketContextPacket,
    OptionContractQuote,
    OptionLegSpec,
    OptionRight,
    OrderIntent,
    PortfolioSnapshot,
    StrategyType,
    TradeAction,
    TradeProposal,
)
from src.persistence.ledger import LedgerRepository, SpreadPositionRecord

logger = logging.getLogger(__name__)

_PENNY = Decimal("0.01")


def _round_tick(value: float) -> float:
    return float(Decimal(str(value)).quantize(_PENNY, rounding=ROUND_HALF_UP))


class CreditSpreadStrategy:
    """
    Deterministic multi-leg (`OrderClass.MLEG`) credit strategy engine:
      - Selects 5-14 DTE contracts using live Black-Scholes delta (~8-12 delta for Iron Condor,
        ~3-8 delta for Deeper Bull Put Spread) and fallback % OTM when Greeks are unavailable.
      - Strictly enforces per-leg liquidity (`spread_pct <= MAX_OPTION_SPREAD_PCT`) and zero
        strike overlap against already-open spread legs on the same ticker/expiration.
      - Computes a 25% spread-slippage haircut net credit limit price and sizes contracts to aim
        for `WEEKLY_CREDIT_TARGET_USD` strictly inside `TRD-POL-001` (`5%` account risk cap).
      - Monitors open spreads for 50% early profit-taking (`profit_target_debit`) or stop-loss
        exit (`stop_loss_debit`).
    """

    def __init__(
        self,
        settings: Settings,
        ledger: LedgerRepository,
        analyst: Optional[GeminiAnalyst] = None,
    ) -> None:
        self.settings = settings
        self.ledger = ledger
        self.analyst = analyst

    def _vet_market_context(
        self, packet: MarketContextPacket, proposal: Optional[TradeProposal]
    ) -> Optional[TradeProposal]:
        if packet.avg_daily_dollar_volume < self.settings.MIN_DAILY_DOLLAR_VOLUME:
            return None
        evaluated = proposal or (self.analyst.evaluate_context(packet) if self.analyst is not None else None)
        if evaluated is None:
            return TradeProposal(
                ticker=packet.ticker,
                strategy=StrategyType.CREDIT_SPREAD,
                action=TradeAction.OPEN_IRON_CONDOR,
                conviction_score=0.80,
                rationale="Deterministic liquidity/volatility pre-check passed for defined-risk spread.",
                risk_flags=[],
                reject_reason=None,
            )
        if (
            evaluated.action in (TradeAction.HOLD, TradeAction.REJECT)
            or evaluated.conviction_score < self.settings.MIN_CONVICTION_SCORE
        ):
            return None
        return evaluated

    def _open_leg_symbols(self, ticker: str) -> tuple[set[str], Optional[float]]:
        """Return currently held spread leg symbols + lowest short-put strike in open ICs for `ticker`."""
        used: set[str] = set()
        lowest_ic_long_put: Optional[float] = None
        for rec in self.ledger.list_spread_positions(
            ticker=ticker,
            states=("PENDING_OPEN", "OPEN", "PENDING_CLOSE"),
        ):
            try:
                legs = json.loads(rec.legs_json)
            except Exception:
                continue
            for leg in legs:
                sym = str(leg.get("option_symbol", ""))
                if sym:
                    used.add(sym)
                if rec.structure == "IRON_CONDOR" and leg.get("right") == "PUT" and leg.get("side") == "BUY":
                    strike = float(leg.get("strike_price", 0.0))
                    if strike > 0 and (lowest_ic_long_put is None or strike < lowest_ic_long_put):
                        lowest_ic_long_put = strike
        return used, lowest_ic_long_put

    def _is_liquid(self, quote: OptionContractQuote) -> bool:
        return quote.bid > 0.0 and quote.ask >= quote.bid and quote.spread_pct <= self.settings.MAX_OPTION_SPREAD_PCT + 1e-9

    @staticmethod
    def _pick_wing(
        short_quote: OptionContractQuote,
        same_exp_quotes: Sequence[OptionContractQuote],
        is_put: bool,
        used_symbols: set[str],
        preferred_widths: tuple[float, ...] = (5.0, 2.5, 10.0),
    ) -> Optional[OptionContractQuote]:
        candidates = [
            q
            for q in same_exp_quotes
            if q.option_symbol not in used_symbols
            and (
                (is_put and q.strike_price < short_quote.strike_price - 0.5)
                or (not is_put and q.strike_price > short_quote.strike_price + 0.5)
            )
        ]
        if not candidates:
            return None
        for width in preferred_widths:
            target = short_quote.strike_price - width if is_put else short_quote.strike_price + width
            match = min(candidates, key=lambda q: abs(q.strike_price - target))
            actual_width = abs(short_quote.strike_price - match.strike_price)
            if 2.0 <= actual_width <= 12.5 and abs(match.strike_price - target) <= 0.6:
                return match
        candidates.sort(key=lambda q: abs(abs(short_quote.strike_price - q.strike_price) - 5.0))
        best = candidates[0]
        actual_width = abs(short_quote.strike_price - best.strike_price)
        return best if 2.0 <= actual_width <= 15.0 else None

    def _select_short_leg(
        self,
        quotes: Sequence[OptionContractQuote],
        spot: float,
        is_put: bool,
        target_abs_delta: float,
        min_abs_delta: float,
        max_abs_delta: float,
        fallback_otm_pct: float,
        used_symbols: set[str],
        max_put_strike: Optional[float] = None,
    ) -> list[OptionContractQuote]:
        eligible: list[OptionContractQuote] = []
        for q in quotes:
            if q.option_symbol in used_symbols or not self._is_liquid(q):
                continue
            if is_put and max_put_strike is not None and q.strike_price > max_put_strike + 1e-6:
                continue
            if is_put and q.strike_price >= spot * 0.96:
                continue
            if not is_put and q.strike_price <= spot * 1.04:
                continue
            if q.delta is not None:
                ad = abs(q.delta)
                if not (min_abs_delta <= ad <= max_abs_delta):
                    continue
            eligible.append(q)

        if not eligible:
            return []

        if all(q.delta is not None for q in eligible):
            eligible.sort(key=lambda q: abs(abs(float(q.delta)) - target_abs_delta))
        else:
            target_strike = spot * (1.0 - fallback_otm_pct if is_put else 1.0 + fallback_otm_pct)
            eligible.sort(key=lambda q: abs(q.strike_price - target_strike))
        return eligible

    def _compute_credit_and_qty(
        self,
        short_legs: Sequence[OptionContractQuote],
        long_legs: Sequence[OptionContractQuote],
        max_wing_width: float,
        portfolio: PortfolioSnapshot,
        ticker: str,
        target_usd: float,
        min_unit_credit: float,
    ) -> Optional[tuple[float, int, float]]:
        net_mid = sum(q.mid for q in short_legs) - sum(q.mid for q in long_legs)
        net_natural = sum(q.bid for q in short_legs) - sum(q.ask for q in long_legs)
        if net_mid <= 0.0:
            return None
        slippage = max(0.0, net_mid - net_natural)
        haircut_credit = _round_tick(max(0.01, net_mid - 0.25 * slippage))
        if haircut_credit < min_unit_credit - 1e-9 or haircut_credit >= max_wing_width - 0.10:
            return None

        per_unit_max_loss = (max_wing_width - haircut_credit) * 100.0
        if per_unit_max_loss <= 0.0:
            return None

        existing_exp = portfolio.ticker_notional_exposure.get(ticker, 0.0)
        budget = max(0.0, self.settings.MAX_POSITION_PCT * portfolio.total_equity - existing_exp)
        max_qty_by_budget = math.floor(budget / per_unit_max_loss)
        if max_qty_by_budget <= 0:
            return None

        desired_qty = max(1, math.ceil(target_usd / (haircut_credit * 100.0)))
        qty = min(desired_qty, max_qty_by_budget, 5)
        max_loss_total = round(per_unit_max_loss * qty, 2)
        return haircut_credit, qty, max_loss_total

    def build_iron_condor_intent(
        self,
        packet: MarketContextPacket,
        put_chain: Sequence[OptionContractQuote],
        call_chain: Sequence[OptionContractQuote],
        portfolio: PortfolioSnapshot,
        proposal: Optional[TradeProposal] = None,
    ) -> Optional[OrderIntent]:
        """Construct a ~8-12 delta defined-risk Iron Condor (`7-14 DTE`, `OrderClass.MLEG`)."""
        vetted = self._vet_market_context(packet, proposal)
        if vetted is None:
            return None

        as_of_date = packet.as_of_ny.date()
        used_symbols, _ = self._open_leg_symbols(packet.ticker)

        expiries = sorted(
            {
                q.expiration_date
                for q in put_chain
                if self.settings.SPREAD_MIN_DTE <= (q.expiration_date - as_of_date).days <= self.settings.SPREAD_MAX_DTE
            }
            & {
                q.expiration_date
                for q in call_chain
                if self.settings.SPREAD_MIN_DTE <= (q.expiration_date - as_of_date).days <= self.settings.SPREAD_MAX_DTE
            }
        )
        for exp in expiries:
            exp_puts = [q for q in put_chain if q.expiration_date == exp and self._is_liquid(q)]
            exp_calls = [q for q in call_chain if q.expiration_date == exp and self._is_liquid(q)]
            short_puts = self._select_short_leg(
                exp_puts, packet.price, True, target_abs_delta=0.10, min_abs_delta=0.05, max_abs_delta=0.16,
                fallback_otm_pct=0.08, used_symbols=used_symbols,
            )
            short_calls = self._select_short_leg(
                exp_calls, packet.price, False, target_abs_delta=0.09, min_abs_delta=0.04, max_abs_delta=0.15,
                fallback_otm_pct=0.09, used_symbols=used_symbols,
            )
            for sp in short_puts[:4]:
                lp = self._pick_wing(sp, exp_puts, is_put=True, used_symbols=used_symbols)
                if lp is None:
                    continue
                for sc in short_calls[:4]:
                    lc = self._pick_wing(sc, exp_calls, is_put=False, used_symbols=used_symbols)
                    if lc is None:
                        continue
                    put_width = sp.strike_price - lp.strike_price
                    call_width = lc.strike_price - sc.strike_price
                    worst_width = max(put_width, call_width)
                    sized = self._compute_credit_and_qty(
                        short_legs=(sp, sc),
                        long_legs=(lp, lc),
                        max_wing_width=worst_width,
                        portfolio=portfolio,
                        ticker=packet.ticker,
                        target_usd=self.settings.WEEKLY_CREDIT_TARGET_USD,
                        min_unit_credit=0.20,
                    )
                    if sized is None:
                        continue
                    limit_credit, qty, max_loss_total = sized
                    spread_id = f"ic-{packet.ticker}-{exp.isoformat()}-{int(sp.strike_price)}-{int(sc.strike_price)}"
                    date_bucket = f"{packet.as_of_ny:%Y-%m-%d}:{spread_id}"
                    legs = (
                        OptionLegSpec(
                            option_symbol=sp.option_symbol, right=OptionRight.PUT,
                            strike_price=sp.strike_price, expiration_date=exp,
                            side="SELL", position_intent="sell_to_open", ratio_qty=1,
                        ),
                        OptionLegSpec(
                            option_symbol=lp.option_symbol, right=OptionRight.PUT,
                            strike_price=lp.strike_price, expiration_date=exp,
                            side="BUY", position_intent="buy_to_open", ratio_qty=1,
                        ),
                        OptionLegSpec(
                            option_symbol=sc.option_symbol, right=OptionRight.CALL,
                            strike_price=sc.strike_price, expiration_date=exp,
                            side="SELL", position_intent="sell_to_open", ratio_qty=1,
                        ),
                        OptionLegSpec(
                            option_symbol=lc.option_symbol, right=OptionRight.CALL,
                            strike_price=lc.strike_price, expiration_date=exp,
                            side="BUY", position_intent="buy_to_open", ratio_qty=1,
                        ),
                    )
                    return OrderIntent(
                        strategy=StrategyType.CREDIT_SPREAD,
                        ticker=packet.ticker,
                        action=TradeAction.OPEN_IRON_CONDOR,
                        qty=qty,
                        notional_exposure_delta=max_loss_total,
                        limit_price=limit_credit,
                        legs=legs,
                        spread_id=spread_id,
                        date_bucket=date_bucket,
                        proposal=vetted.model_copy(update={"action": TradeAction.OPEN_IRON_CONDOR}),
                    )
        return None

    def build_bull_put_spread_intent(
        self,
        packet: MarketContextPacket,
        put_chain: Sequence[OptionContractQuote],
        portfolio: PortfolioSnapshot,
        proposal: Optional[TradeProposal] = None,
        deeper_otm: bool = True,
    ) -> Optional[OrderIntent]:
        """Construct a deeper OTM Bull Put Credit Spread (`~4-8 delta`, avoiding open IC strikes)."""
        vetted = self._vet_market_context(packet, proposal)
        if vetted is None:
            return None

        as_of_date = packet.as_of_ny.date()
        used_symbols, lowest_ic_long_put = self._open_leg_symbols(packet.ticker)
        # Ensure deeper bull spread sits cleanly below any open IC wing when both exist on the same ticker
        max_put_strike = (lowest_ic_long_put - 2.5) if (deeper_otm and lowest_ic_long_put) else None

        expiries = sorted(
            {
                q.expiration_date
                for q in put_chain
                if self.settings.SPREAD_MIN_DTE <= (q.expiration_date - as_of_date).days <= self.settings.SPREAD_MAX_DTE
            }
        )
        target_d = 0.05 if deeper_otm else 0.12
        min_d, max_d = (0.025, 0.085) if deeper_otm else (0.07, 0.18)
        otm_fallback = 0.095 if deeper_otm else 0.07

        for exp in expiries:
            exp_puts = [q for q in put_chain if q.expiration_date == exp and self._is_liquid(q)]
            short_puts = self._select_short_leg(
                exp_puts, packet.price, True, target_abs_delta=target_d, min_abs_delta=min_d, max_abs_delta=max_d,
                fallback_otm_pct=otm_fallback, used_symbols=used_symbols, max_put_strike=max_put_strike,
            )
            for sp in short_puts[:5]:
                lp = self._pick_wing(sp, exp_puts, is_put=True, used_symbols=used_symbols)
                if lp is None:
                    continue
                width = sp.strike_price - lp.strike_price
                sized = self._compute_credit_and_qty(
                    short_legs=(sp,),
                    long_legs=(lp,),
                    max_wing_width=width,
                    portfolio=portfolio,
                    ticker=packet.ticker,
                    target_usd=self.settings.WEEKLY_CREDIT_TARGET_USD,
                    min_unit_credit=0.10,
                )
                if sized is None:
                    continue
                limit_credit, qty, max_loss_total = sized
                spread_id = f"bps-{packet.ticker}-{exp.isoformat()}-{int(sp.strike_price)}-{int(lp.strike_price)}"
                date_bucket = f"{packet.as_of_ny:%Y-%m-%d}:{spread_id}"
                legs = (
                    OptionLegSpec(
                        option_symbol=sp.option_symbol, right=OptionRight.PUT,
                        strike_price=sp.strike_price, expiration_date=exp,
                        side="SELL", position_intent="sell_to_open", ratio_qty=1,
                    ),
                    OptionLegSpec(
                        option_symbol=lp.option_symbol, right=OptionRight.PUT,
                        strike_price=lp.strike_price, expiration_date=exp,
                        side="BUY", position_intent="buy_to_open", ratio_qty=1,
                    ),
                )
                return OrderIntent(
                    strategy=StrategyType.CREDIT_SPREAD,
                    ticker=packet.ticker,
                    action=TradeAction.OPEN_BULL_PUT_SPREAD,
                    qty=qty,
                    notional_exposure_delta=max_loss_total,
                    limit_price=limit_credit,
                    legs=legs,
                    spread_id=spread_id,
                    date_bucket=date_bucket,
                    proposal=vetted.model_copy(update={"action": TradeAction.OPEN_BULL_PUT_SPREAD}),
                )
        return None

    def check_spread_exit(
        self,
        spread: SpreadPositionRecord,
        quotes_by_symbol: dict[str, OptionContractQuote],
        as_of_ny: datetime,
    ) -> Optional[OrderIntent]:
        """Emit a closing MLEG debit order if spread reaches 50% profit target or stop-loss ceiling."""
        if spread.state != "OPEN":
            return None
        try:
            raw_legs = json.loads(spread.legs_json)
        except Exception:
            return None

        short_quotes: list[OptionContractQuote] = []
        long_quotes: list[OptionContractQuote] = []
        close_legs: list[OptionLegSpec] = []
        for leg in raw_legs:
            sym = str(leg["option_symbol"])
            q = quotes_by_symbol.get(sym)
            if q is None:
                return None
            was_sell = str(leg["side"]).upper() == "SELL"
            if was_sell:
                short_quotes.append(q)
                close_legs.append(
                    OptionLegSpec(
                        option_symbol=sym,
                        right=OptionRight(leg["right"]),
                        strike_price=float(leg["strike_price"]),
                        expiration_date=date.fromisoformat(str(leg["expiration_date"])),
                        side="BUY",
                        position_intent="buy_to_close",
                        ratio_qty=int(leg.get("ratio_qty", 1)),
                    )
                )
            else:
                long_quotes.append(q)
                close_legs.append(
                    OptionLegSpec(
                        option_symbol=sym,
                        right=OptionRight(leg["right"]),
                        strike_price=float(leg["strike_price"]),
                        expiration_date=date.fromisoformat(str(leg["expiration_date"])),
                        side="SELL",
                        position_intent="sell_to_close",
                        ratio_qty=int(leg.get("ratio_qty", 1)),
                    )
                )

        # Cost to close = buy back shorts - sell longs
        mid_debit = sum(q.mid for q in short_quotes) - sum(q.mid for q in long_quotes)
        natural_debit = sum(q.ask for q in short_quotes) - sum(q.bid for q in long_quotes)
        is_profit_take = mid_debit <= spread.profit_target_debit + 1e-9
        is_stop_loss = mid_debit >= spread.stop_loss_debit - 1e-9
        if not (is_profit_take or is_stop_loss):
            return None

        slip = max(0.0, natural_debit - mid_debit)
        limit_debit = _round_tick(max(0.01, mid_debit + 0.25 * slip))
        action = (
            TradeAction.CLOSE_IRON_CONDOR
            if spread.structure == "IRON_CONDOR"
            else TradeAction.CLOSE_BULL_PUT_SPREAD
        )
        return OrderIntent(
            strategy=StrategyType.CREDIT_SPREAD,
            ticker=spread.ticker,
            action=action,
            qty=spread.qty,
            notional_exposure_delta=0.0,
            limit_price=limit_debit,
            legs=tuple(close_legs),
            spread_id=spread.spread_id,
            date_bucket=f"{as_of_ny:%Y-%m-%d:%H}:{spread.spread_id}",
        )
