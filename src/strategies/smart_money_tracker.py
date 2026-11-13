"""Smart Money Tracker strategy: ingests congressional & 13F disclosures with strict 14-day filing-date lag gate."""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from src.config import Settings
from src.intelligence.analyst import GeminiAnalyst
from src.intelligence.schemas import (
    MarketContextPacket,
    OrderIntent,
    PortfolioSnapshot,
    StrategyType,
    TradeAction,
)

logger = logging.getLogger(__name__)


class DisclosureSource(StrEnum):
    CONGRESSIONAL = "CONGRESSIONAL"
    FORM_13F = "13F"


class SmartMoneyDisclosure(BaseModel):
    """Normalized public disclosure filing record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str = Field(..., min_length=1, max_length=12, pattern=r"^[A-Z.]+$")
    filer_name: str = Field(..., min_length=2, max_length=120)
    source: DisclosureSource
    transaction_date: date
    public_disclosure_date: date
    estimated_notional_usd: float = Field(..., gt=0.0)
    price: float = Field(..., gt=0.0)
    atr_14d: float = Field(default=2.0, ge=0.0)
    rsi_14d: float = Field(default=55.0, ge=0.0, le=100.0)
    realized_vol_20d: float = Field(default=0.25, ge=0.0)
    spread_width_pct: float = Field(default=0.001, ge=0.0)
    avg_daily_dollar_volume: float = Field(..., ge=0.0)
    raw_filing_summary: str = Field(..., max_length=3500)


class DisclosureFilterResult(BaseModel):
    """Result of deterministic pre-Gemini filing eligibility checks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    disclosure_lag_days: int
    reject_reason: Optional[str] = None
    intent: Optional[OrderIntent] = None


class SmartMoneyTrackerStrategy:
    """
    Filters public congressional / 13F disclosures by:
      1. Public disclosure / filing date lag (`<= MAX_DISCLOSURE_LAG_DAYS`, default 14 days).
      2. Minimum daily dollar liquidity (`>= MIN_DAILY_DOLLAR_VOLUME`).
      3. Bounded Gemini catalyst & risk synthesis (`GeminiAnalyst.evaluate_context()`).
    """

    def __init__(
        self,
        settings: Settings,
        analyst: GeminiAnalyst,
        allocation_pct: float = 0.025,
    ) -> None:
        self.settings = settings
        self.analyst = analyst
        self.allocation_pct = allocation_pct

    def evaluate_disclosure(
        self,
        disclosure: SmartMoneyDisclosure,
        portfolio: PortfolioSnapshot,
        as_of_ny: Optional[datetime] = None,
    ) -> DisclosureFilterResult:
        """
        Deterministically reject stale (>14 days since public disclosure date) or illiquid filings
        BEFORE invoking Gemini or emitting an OrderIntent.
        """
        eval_dt = as_of_ny or portfolio.as_of_ny
        eval_date = eval_dt.date()

        # Use public disclosure/filing date (and also guard against backdated transaction lag if disclosure lag > 14d)
        disclosure_lag_days = (eval_date - disclosure.public_disclosure_date).days
        if disclosure_lag_days < 0:
            return DisclosureFilterResult(
                passed=False,
                disclosure_lag_days=disclosure_lag_days,
                reject_reason="Invalid future public_disclosure_date.",
            )

        if disclosure_lag_days > self.settings.MAX_DISCLOSURE_LAG_DAYS:
            reason = (
                f"Rejected stale {disclosure.source.value} filing for '{disclosure.ticker}': "
                f"public disclosure lag is {disclosure_lag_days} days "
                f"(exceeds MAX_DISCLOSURE_LAG_DAYS={self.settings.MAX_DISCLOSURE_LAG_DAYS})."
            )
            logger.info(
                reason,
                extra={
                    "ticker": disclosure.ticker,
                    "public_disclosure_date": disclosure.public_disclosure_date.isoformat(),
                    "disclosure_lag_days": disclosure_lag_days,
                },
            )
            return DisclosureFilterResult(
                passed=False,
                disclosure_lag_days=disclosure_lag_days,
                reject_reason=reason,
            )

        if disclosure.avg_daily_dollar_volume < self.settings.MIN_DAILY_DOLLAR_VOLUME:
            reason = (
                f"Rejected '{disclosure.ticker}' due to insufficient liquidity: "
                f"avg_daily_dollar_volume=${disclosure.avg_daily_dollar_volume:,.0f} < "
                f"${self.settings.MIN_DAILY_DOLLAR_VOLUME:,.0f}."
            )
            return DisclosureFilterResult(
                passed=False,
                disclosure_lag_days=disclosure_lag_days,
                reject_reason=reason,
            )

        packet = MarketContextPacket(
            ticker=disclosure.ticker,
            as_of_ny=eval_dt,
            strategy_candidate=StrategyType.SMART_MONEY,
            price=disclosure.price,
            atr_14d=disclosure.atr_14d,
            rsi_14d=disclosure.rsi_14d,
            realized_vol_20d=disclosure.realized_vol_20d,
            spread_width_pct=disclosure.spread_width_pct,
            avg_daily_dollar_volume=disclosure.avg_daily_dollar_volume,
            disclosure_filing_date=disclosure.public_disclosure_date,
            disclosure_lag_days=disclosure_lag_days,
            catalyst_summary=(
                f"[{disclosure.source.value} FILING by {disclosure.filer_name}, "
                f"disclosed {disclosure.public_disclosure_date.isoformat()}] {disclosure.raw_filing_summary}"
            ),
        )

        proposal = self.analyst.evaluate_context(packet)
        if (
            proposal.action != TradeAction.BUY_EQUITY
            or proposal.conviction_score < self.settings.MIN_CONVICTION_SCORE
        ):
            return DisclosureFilterResult(
                passed=False,
                disclosure_lag_days=disclosure_lag_days,
                reject_reason=proposal.reject_reason or "Gemini conviction below threshold or action != BUY_EQUITY.",
            )

        existing_exposure = portfolio.ticker_notional_exposure.get(disclosure.ticker, 0.0)
        max_budget = max(0.0, self.settings.MAX_POSITION_PCT * portfolio.total_equity - existing_exposure)
        target_notional = min(self.allocation_pct * portfolio.total_equity, max_budget)
        qty = math.floor(target_notional / disclosure.price)
        if qty <= 0:
            return DisclosureFilterResult(
                passed=False,
                disclosure_lag_days=disclosure_lag_days,
                reject_reason="Zero shares permitted within single-ticker concentration budget.",
            )

        intent = OrderIntent(
            strategy=StrategyType.SMART_MONEY,
            ticker=disclosure.ticker,
            action=TradeAction.BUY_EQUITY,
            qty=qty,
            notional_exposure_delta=round(qty * disclosure.price, 2),
            limit_price=round(disclosure.price, 2),
            trailing_stop_pct=self.settings.DEFAULT_TRAILING_STOP_PCT,
            realized_vol_20d=disclosure.realized_vol_20d,
            date_bucket=eval_dt.strftime("%Y-%m-%dT%H:%M"),
            proposal=proposal,
        )
        return DisclosureFilterResult(
            passed=True,
            disclosure_lag_days=disclosure_lag_days,
            intent=intent,
        )


def load_disclosures_from_file(path: Path) -> list[SmartMoneyDisclosure]:
    """Load and validate SmartMoneyDisclosure records from a local JSON file if it exists."""
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        logger.warning(f"Ignoring malformed disclosure feed at {path}: expected JSON array.")
        return []
    disclosures: list[SmartMoneyDisclosure] = []
    for item in raw:
        try:
            disclosures.append(SmartMoneyDisclosure.model_validate(item))
        except Exception as exc:
            logger.warning(f"Skipping invalid disclosure record in {path}: {exc}")
    return disclosures
