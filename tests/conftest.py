"""Shared deterministic fixtures for alpaca-gemini-trader tests (zero network calls)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pytest

from src.config import NY_TZ, Settings
from src.execution.broker import AlpacaBroker
from src.execution.risk_gateway import RiskGateway
from src.intelligence.schemas import PortfolioSnapshot
from src.persistence.ledger import LedgerRepository


@dataclass
class FakeClock:
    is_open: bool = True
    timestamp: datetime = field(
        default_factory=lambda: datetime(2026, 11, 12, 10, 30, 0, tzinfo=NY_TZ)
    )
    next_open: datetime = field(
        default_factory=lambda: datetime(2026, 11, 13, 9, 30, 0, tzinfo=NY_TZ)
    )
    next_close: datetime = field(
        default_factory=lambda: datetime(2026, 11, 12, 16, 0, 0, tzinfo=NY_TZ)
    )


@dataclass
class FakeAccount:
    equity: float = 100_000.0
    last_equity: float = 100_000.0
    non_marginable_buying_power: float = 80_000.0
    cash: float = 80_000.0


@dataclass
class FakeOrderResponse:
    id: str = "alpaca-paper-order-001"
    status: str = "accepted"
    filled_avg_price: Optional[float] = 100.0
    filled_qty: Optional[float] = 10.0


class FakeTradingClient:
    """In-memory fake of Alpaca TradingClient strictly locked to paper=True."""

    def __init__(self, paper: bool = True, base_url: str = "https://paper-api.alpaca.markets") -> None:
        self.paper = paper
        self._base_url = base_url
        self.clock = FakeClock()
        self.account = FakeAccount()
        self.positions: list[Any] = []
        self.submitted_orders: list[Any] = []
        self.cancel_orders_called: int = 0

    def get_clock(self) -> FakeClock:
        return self.clock

    def get_account(self) -> FakeAccount:
        return self.account

    def get_all_positions(self) -> list[Any]:
        return self.positions

    def cancel_orders(self) -> list[Any]:
        self.cancel_orders_called += 1
        return []

    def submit_order(self, order_data: Any) -> FakeOrderResponse:
        self.submitted_orders.append(order_data)
        return FakeOrderResponse(id=f"paper-ord-{len(self.submitted_orders)}")


@pytest.fixture
def base_ny_time() -> datetime:
    return datetime(2026, 11, 12, 10, 30, 0, tzinfo=NY_TZ)


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    db_file = tmp_path / "test_state.db"
    return Settings(
        TRADING_MODE="PAPER",
        ALPACA_PAPER_BASE_URL="https://paper-api.alpaca.markets",
        APCA_API_KEY_ID="PK_TEST_PAPER_KEY",
        APCA_API_SECRET_KEY="SK_TEST_PAPER_SECRET",
        GEMINI_API_KEY="TEST_GEMINI_KEY",
        GEMINI_MODEL="gemini-3.8-flash",
        MAX_POSITION_PCT=0.05,
        MAX_DIP_TRANCHES=2,
        MAX_REALIZED_VOL_20D=0.60,
        ORDER_COOLDOWN_MINUTES=15,
        MAX_DAILY_DRAWDOWN_PCT=0.02,
        DEFAULT_TRAILING_STOP_PCT=0.05,
        SQLITE_DB_PATH=db_file,
    )


@pytest.fixture
def ledger(test_settings: Settings) -> LedgerRepository:
    return LedgerRepository(db_path=test_settings.SQLITE_DB_PATH)


@pytest.fixture
def fake_trading_client() -> FakeTradingClient:
    return FakeTradingClient()


@pytest.fixture
def broker(test_settings: Settings, fake_trading_client: FakeTradingClient) -> AlpacaBroker:
    return AlpacaBroker(
        settings=test_settings,
        trading_client=fake_trading_client,
        stock_data_client=object(),
        option_data_client=object(),
    )


@pytest.fixture
def risk_gateway(
    test_settings: Settings,
    ledger: LedgerRepository,
    broker: AlpacaBroker,
) -> RiskGateway:
    return RiskGateway(settings=test_settings, ledger=ledger, broker=broker)


@pytest.fixture
def default_portfolio(base_ny_time: datetime) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        as_of_ny=base_ny_time,
        total_equity=100_000.0,
        non_marginable_buying_power=80_000.0,
        daily_starting_equity=100_000.0,
        intraday_peak_equity=100_000.0,
        ticker_notional_exposure={},
    )
