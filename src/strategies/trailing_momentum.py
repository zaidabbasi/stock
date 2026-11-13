"""Trailing Momentum strategy: catalyst-driven equity entry + native OTO trailing stop + bounded dip-laddering."""

from __future__ import annotations

import math
from typing import Optional

from src.config import Settings
from src.intelligence.analyst import GeminiAnalyst
from src.intelligence.schemas import (
    MarketContextPacket,
    OrderIntent,
    PortfolioSnapshot,
    StrategyType,
    TradeAction,
    TradeProposal,
)
from src.persistence.ledger import LedgerRepository


class TrailingMomentumStrategy:
    """
    1. Evaluates liquid equities via `GeminiAnalyst` (`MarketContextPacket` -> `TradeProposal`).
    2. On high-conviction approval (`BUY_EQUITY`), computes deterministic share quantity and attaches
       a broker-native trailing stop (`DEFAULT_TRAILING_STOP_PCT=0.05`).
    3. Manages bounded dip-ladder limit orders (`ADD_DIP_TRANCHE`, capped at 2 add-on tranches by TRD-POL-004).
    """

    def __init__(
        self,
        settings: Settings,
        analyst: GeminiAnalyst,
        ledger: LedgerRepository,
        initial_allocation_pct: float = 0.02,
        dip_tranche_allocation_pct: float = 0.0125,
        min_dip_pullback_pct: float = 0.03,
    ) -> None:
        self.settings = settings
        self.analyst = analyst
        self.ledger = ledger
        self.initial_allocation_pct = initial_allocation_pct
        self.dip_tranche_allocation_pct = dip_tranche_allocation_pct
        self.min_dip_pullback_pct = min_dip_pullback_pct

    def evaluate_entry(
        self,
        packet: MarketContextPacket,
        portfolio: PortfolioSnapshot,
        proposal: Optional[TradeProposal] = None,
    ) -> Optional[OrderIntent]:
        """
        Evaluate a candidate ticker for initial momentum entry.
         Gemini determines qualitative/quantitative conviction; Python deterministically computes shares and stops.
        """
        if packet.avg_daily_dollar_volume < self.settings.MIN_DAILY_DOLLAR_VOLUME:
            return None

        evaluated_proposal = proposal or self.analyst.evaluate_context(packet)
        if (
            evaluated_proposal.action != TradeAction.BUY_EQUITY
            or evaluated_proposal.conviction_score < self.settings.MIN_CONVICTION_SCORE
        ):
            return None

        # Deterministic sizing: target `initial_allocation_pct` of equity, strictly within remaining TRD-POL-001 room
        existing_exposure = portfolio.ticker_notional_exposure.get(packet.ticker, 0.0)
        max_ticker_budget = self.settings.MAX_POSITION_PCT * portfolio.total_equity
        remaining_budget = max(0.0, max_ticker_budget - existing_exposure)

        target_dollars = min(self.initial_allocation_pct * portfolio.total_equity, remaining_budget)
        qty = math.floor(target_dollars / packet.price)
        if qty <= 0:
            return None

        notional = round(qty * packet.price, 2)
        date_bucket = packet.as_of_ny.strftime("%Y-%m-%dT%H:%M")

        return OrderIntent(
            strategy=StrategyType.TRAILING_MOMENTUM,
            ticker=packet.ticker,
            action=TradeAction.BUY_EQUITY,
            qty=qty,
            notional_exposure_delta=notional,
            limit_price=round(packet.price, 2),
            trailing_stop_pct=self.settings.DEFAULT_TRAILING_STOP_PCT,
            dip_tranche_index=0,
            realized_vol_20d=packet.realized_vol_20d,
            date_bucket=date_bucket,
            proposal=evaluated_proposal,
        )

    def build_dip_ladder_intent(
        self,
        packet: MarketContextPacket,
        portfolio: PortfolioSnapshot,
        override_tranche_index: Optional[int] = None,
    ) -> Optional[OrderIntent]:
        """
        Construct a deterministic dip-ladder limit order intent when price pulls back below the last tranche price.
        Note: `RiskGateway` remains the ultimate authority enforcing `TRD-POL-004` and `TRD-POL-001`.
        """
        ladder = self.ledger.get_dip_ladder(packet.ticker)
        if ladder is None and override_tranche_index is None:
            return None

        reference_price = ladder.last_tranche_price if ladder is not None else packet.price * 1.05
        pullback_pct = (reference_price - packet.price) / reference_price
        if pullback_pct < self.min_dip_pullback_pct and override_tranche_index is None:
            return None

        next_tranche = (
            override_tranche_index
            if override_tranche_index is not None
            else ((ladder.add_on_tranches_used if ladder else 0) + 1)
        )

        target_dollars = self.dip_tranche_allocation_pct * portfolio.total_equity
        qty = max(1, math.floor(target_dollars / packet.price))
        notional = round(qty * packet.price, 2)
        date_bucket = packet.as_of_ny.strftime("%Y-%m-%dT%H:%M")

        return OrderIntent(
            strategy=StrategyType.TRAILING_MOMENTUM,
            ticker=packet.ticker,
            action=TradeAction.ADD_DIP_TRANCHE,
            qty=qty,
            notional_exposure_delta=notional,
            limit_price=round(packet.price, 2),
            dip_tranche_index=next_tranche,
            realized_vol_20d=packet.realized_vol_20d,
            date_bucket=date_bucket,
        )
