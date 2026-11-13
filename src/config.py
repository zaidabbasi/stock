"""Configuration and fail-closed security invariants for alpaca-gemini-trader."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

NY_TZ = ZoneInfo("America/New_York")
FORBIDDEN_LIVE_HOSTS = frozenset({"api.alpaca.markets"})
ALLOWED_PAPER_HOSTS = frozenset({"paper-api.alpaca.markets"})


class SecurityConfigurationError(RuntimeError):
    """Raised immediately when paper-trading safety locks or endpoint invariants are violated."""


def validate_broker_endpoint(url: object) -> None:
    """
    Fail-closed verification that a trading endpoint is Alpaca's PAPER host.
    Accepts strings or alpaca-py `BaseURL` enums (uses `.value`). Path suffixes such as `/v2` are ignored.
    """
    if url is None or url == "":
        return
    raw = getattr(url, "value", url)
    normalized = str(raw).strip().lower()
    parsed = urlparse(normalized if "://" in normalized else f"https://{normalized}")
    host = parsed.hostname or normalized

    if host in FORBIDDEN_LIVE_HOSTS:
        raise SecurityConfigurationError(
            f"FATAL: Live Alpaca endpoint detected ('{raw}'). "
            "This system is strictly locked to Paper Trading (paper-api.alpaca.markets)."
        )
    if host not in ALLOWED_PAPER_HOSTS:
        raise SecurityConfigurationError(
            f"FATAL: Untrusted or non-paper Alpaca endpoint '{raw}' (host='{host}'). "
            "Only 'https://paper-api.alpaca.markets' is permitted."
        )


def _parse_tickers(raw: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(t.strip().upper() for t in raw.split(",") if t.strip()))


class Settings(BaseSettings):
    """
    Typed application configuration loaded exclusively from `.env` or environment variables.
    Enforces fail-closed Paper Trading locks at instantiation time.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- Security & Mode Locks ---
    TRADING_MODE: str = Field(default="PAPER", description="Must strictly equal 'PAPER'")
    ALPACA_PAPER_BASE_URL: str = Field(default="https://paper-api.alpaca.markets")

    # --- Credentials (repr=False prevents accidental secret leakage in logs/tracebacks) ---
    APCA_API_KEY_ID: str = Field(default="", repr=False)
    APCA_API_SECRET_KEY: str = Field(default="", repr=False)
    GEMINI_API_KEY: str = Field(default="", repr=False)

    # --- Analyst Configuration ---
    ANALYST_MODE: Literal["GEMINI", "RULES"] = Field(
        default="GEMINI",
        description="GEMINI = google-genai co-pilot (fails closed to REJECT without key); RULES = deterministic filters",
    )
    GEMINI_MODEL: str = Field(default="gemini-3.8-flash")
    ANALYST_CACHE_MINUTES: int = Field(default=60, ge=1, le=1440)

    # --- Universe ---
    MOMENTUM_WATCHLIST: str = Field(default="AAPL,MSFT,NVDA,AMZN,GOOGL,META")
    WHEEL_WATCHLIST: str = Field(default="F,PFE,T,SOFI,INTC")
    DISCLOSURE_FEED_PATH: Path = Field(default=Path("data/disclosures.json"))

    # --- Market Data Feeds ---
    ALPACA_BARS_FEED: Literal["sip", "iex"] = Field(default="sip")
    ALPACA_QUOTE_FEED: Literal["sip", "iex"] = Field(default="iex")

    # --- Deterministic Risk Gateway Thresholds (TRD-POL-001 .. TRD-POL-006) ---
    MAX_POSITION_PCT: float = Field(default=0.05, gt=0.0, le=0.25)
    MAX_DIP_TRANCHES: int = Field(default=2, ge=1, le=2)
    MAX_REALIZED_VOL_20D: float = Field(default=0.60, gt=0.0, le=2.0)
    ORDER_COOLDOWN_MINUTES: int = Field(default=15, ge=1, le=1440)
    MAX_DAILY_DRAWDOWN_PCT: float = Field(default=0.02, gt=0.0, le=0.10)

    # --- Strategy Execution Parameters ---
    DEFAULT_TRAILING_STOP_PCT: float = Field(default=0.05, gt=0.0, le=0.20)
    MIN_CONVICTION_SCORE: float = Field(default=0.70, ge=0.0, le=1.0)
    MAX_DISCLOSURE_LAG_DAYS: int = Field(default=14, ge=1, le=30)
    MIN_DAILY_DOLLAR_VOLUME: float = Field(default=5_000_000.0, gt=0.0)
    MAX_OPTION_SPREAD_PCT: float = Field(default=0.25, gt=0.0, le=0.50)
    ENTRY_BLACKOUT_MINUTES_AFTER_OPEN: int = Field(default=5, ge=0, le=120)
    ENTRY_BLACKOUT_MINUTES_BEFORE_CLOSE: int = Field(default=15, ge=0, le=120)

    # --- Orchestrator, Persistence & Observability ---
    POLL_INTERVAL_SECONDS: int = Field(default=60, ge=10, le=3600)
    SQLITE_DB_PATH: Path = Field(default=Path("data/state.db"))
    LOG_LEVEL: str = Field(default="INFO")

    @model_validator(mode="after")
    def _enforce_invariants(self) -> "Settings":
        if self.TRADING_MODE != "PAPER":
            raise SecurityConfigurationError(
                f"FATAL: TRADING_MODE='{self.TRADING_MODE}' is forbidden. "
                "TRADING_MODE must be strictly 'PAPER'."
            )
        validate_broker_endpoint(self.ALPACA_PAPER_BASE_URL)
        overlap = set(self.momentum_tickers) & set(self.wheel_tickers)
        if overlap:
            raise ValueError(
                f"MOMENTUM_WATCHLIST and WHEEL_WATCHLIST must be disjoint (overlap: {sorted(overlap)}); "
                "Wheel assignment detection relies on share counts being owned solely by the Wheel."
            )
        return self

    @property
    def momentum_tickers(self) -> tuple[str, ...]:
        return _parse_tickers(self.MOMENTUM_WATCHLIST)

    @property
    def wheel_tickers(self) -> tuple[str, ...]:
        return _parse_tickers(self.WHEEL_WATCHLIST)

    @property
    def has_gemini_key(self) -> bool:
        return bool(self.GEMINI_API_KEY) and "YOUR_GEMINI" not in self.GEMINI_API_KEY

    def require_broker_credentials(self) -> None:
        """Verify that non-placeholder Alpaca credentials exist before live network calls."""
        if (
            not self.APCA_API_KEY_ID
            or not self.APCA_API_SECRET_KEY
            or "YOUR_PAPER" in self.APCA_API_KEY_ID
            or "YOUR_PAPER" in self.APCA_API_SECRET_KEY
        ):
            raise SecurityConfigurationError(
                "Missing or placeholder APCA_API_KEY_ID / APCA_API_SECRET_KEY in .env."
            )
        if self.APCA_API_KEY_ID.startswith("AK"):
            raise SecurityConfigurationError("LIVE Alpaca key (prefix 'AK') detected. Only PAPER keys are allowed.")

    def require_gemini_credentials(self) -> None:
        if not self.has_gemini_key:
            raise SecurityConfigurationError("Missing or placeholder GEMINI_API_KEY in .env.")


def load_settings(**overrides: object) -> Settings:
    """Instantiate and validate Settings with optional overrides."""
    return Settings(**overrides)  # type: ignore[arg-type]
