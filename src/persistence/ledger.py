"""SQLite ledger managing system_state, wheel_cycles, dip_ladders, order_cooldowns, and audit_log."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from src.config import NY_TZ
from src.intelligence.schemas import (
    OrderIntent,
    PolicyId,
    RiskDecisionStatus,
    TradeProposal,
    WheelState,
)


def now_ny() -> datetime:
    """Return current timezone-aware datetime anchored to America/New_York."""
    return datetime.now(tz=NY_TZ)


@dataclass(frozen=True)
class SystemStateRecord:
    trading_status: str  # 'ACTIVE' | 'HALTED'
    halt_reason: Optional[str]
    halt_policy_id: Optional[str]
    trading_date_ny: str
    daily_starting_equity: float
    intraday_peak_equity: float
    current_equity: float
    realized_pnl_today: float
    unrealized_pnl_today: float
    updated_at_ny: str


@dataclass(frozen=True)
class WheelCycleRecord:
    ticker: str
    state: WheelState
    active_option_symbol: Optional[str]
    active_alpaca_order_id: Optional[str]
    option_right: Optional[str]
    strike_price: Optional[float]
    expiration_date: Optional[str]
    entry_premium_per_share: float
    reserved_csp_collateral: float
    assigned_shares_qty: int
    gross_cost_basis_per_share: float
    cumulative_premium_per_share: float
    net_cost_basis_per_share: float
    cycle_count: int
    updated_at_ny: str


@dataclass(frozen=True)
class DipLadderRecord:
    ticker: str
    initial_entry_price: float
    add_on_tranches_used: int
    last_tranche_price: float
    realized_vol_20d: float
    paused_high_vol: bool
    updated_at_ny: str


@dataclass(frozen=True)
class AuditLogRecord:
    id: int
    timestamp_ny: str
    client_order_id: str
    ticker: str
    strategy: str
    action: str
    proposal_json: Optional[str]
    intent_json: str
    risk_decision: str
    policy_id: str
    rejection_reason: Optional[str]
    alpaca_order_id: Optional[str]
    fill_price: Optional[float]
    filled_qty: Optional[float]
    execution_status: Optional[str] = None
    execution_error: Optional[str] = None


SCHEMA_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS system_state (
    id                      INTEGER PRIMARY KEY CHECK (id = 1),
    trading_status          TEXT NOT NULL DEFAULT 'ACTIVE' CHECK (trading_status IN ('ACTIVE', 'HALTED')),
    halt_reason             TEXT,
    halt_policy_id          TEXT,
    trading_date_ny         TEXT NOT NULL,
    daily_starting_equity   REAL NOT NULL CHECK (daily_starting_equity > 0),
    intraday_peak_equity    REAL NOT NULL CHECK (intraday_peak_equity > 0),
    current_equity          REAL NOT NULL CHECK (current_equity >= 0),
    realized_pnl_today      REAL NOT NULL DEFAULT 0.0,
    unrealized_pnl_today    REAL NOT NULL DEFAULT 0.0,
    updated_at_ny           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS wheel_cycles (
    ticker                          TEXT PRIMARY KEY,
    state                           TEXT NOT NULL DEFAULT 'IDLE' CHECK (
                                        state IN ('IDLE', 'CSP_OPEN', 'ASSIGNED_SHARES', 'CC_OPEN', 'CALLED_AWAY')
                                    ),
    active_option_symbol            TEXT,
    active_alpaca_order_id          TEXT,
    option_right                    TEXT CHECK (option_right IN ('PUT', 'CALL') OR option_right IS NULL),
    strike_price                    REAL CHECK (strike_price > 0 OR strike_price IS NULL),
    expiration_date                 TEXT,
    entry_premium_per_share         REAL NOT NULL DEFAULT 0.0,
    reserved_csp_collateral         REAL NOT NULL DEFAULT 0.0,
    assigned_shares_qty             INTEGER NOT NULL DEFAULT 0 CHECK (assigned_shares_qty >= 0),
    gross_cost_basis_per_share      REAL NOT NULL DEFAULT 0.0,
    cumulative_premium_per_share    REAL NOT NULL DEFAULT 0.0,
    net_cost_basis_per_share        REAL NOT NULL DEFAULT 0.0,
    cycle_count                     INTEGER NOT NULL DEFAULT 0,
    updated_at_ny                   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dip_ladders (
    ticker                  TEXT PRIMARY KEY,
    initial_entry_price     REAL NOT NULL CHECK (initial_entry_price > 0),
    add_on_tranches_used    INTEGER NOT NULL DEFAULT 0 CHECK (add_on_tranches_used >= 0),
    last_tranche_price      REAL NOT NULL CHECK (last_tranche_price > 0),
    realized_vol_20d        REAL NOT NULL DEFAULT 0.0,
    paused_high_vol         INTEGER NOT NULL DEFAULT 0 CHECK (paused_high_vol IN (0, 1)),
    updated_at_ny           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS order_cooldowns (
    client_order_id         TEXT PRIMARY KEY,
    ticker                  TEXT NOT NULL,
    strategy                TEXT NOT NULL,
    action                  TEXT NOT NULL,
    date_bucket             TEXT NOT NULL,
    submitted_at_epoch      REAL NOT NULL,
    submitted_at_ny         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cooldown_ticker_time ON order_cooldowns (ticker, submitted_at_epoch DESC);

CREATE TABLE IF NOT EXISTS audit_log (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ny            TEXT NOT NULL,
    client_order_id         TEXT NOT NULL,
    ticker                  TEXT NOT NULL,
    strategy                TEXT NOT NULL,
    action                  TEXT NOT NULL,
    proposal_json           TEXT,
    intent_json             TEXT NOT NULL,
    risk_decision           TEXT NOT NULL CHECK (risk_decision IN ('APPROVED', 'REJECTED', 'HALT_TRIGGERED')),
    policy_id               TEXT NOT NULL,
    rejection_reason        TEXT,
    alpaca_order_id         TEXT,
    fill_price              REAL,
    filled_qty              REAL
);
CREATE INDEX IF NOT EXISTS idx_audit_ticker_ts ON audit_log (ticker, timestamp_ny DESC);
CREATE INDEX IF NOT EXISTS idx_audit_policy ON audit_log (policy_id, risk_decision);

CREATE TABLE IF NOT EXISTS tracked_orders (
    alpaca_order_id         TEXT PRIMARY KEY,
    client_order_id         TEXT NOT NULL,
    audit_log_id            INTEGER,
    ticker                  TEXT NOT NULL,
    strategy                TEXT NOT NULL,
    action                  TEXT NOT NULL,
    symbol                  TEXT NOT NULL,
    qty                     REAL NOT NULL,
    limit_price             REAL,
    strike_price            REAL,
    expiration_date         TEXT,
    trailing_stop_pct       REAL,
    status                  TEXT NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'FILLED', 'CLOSED_UNFILLED')),
    filled_qty              REAL NOT NULL DEFAULT 0.0,
    filled_avg_price        REAL,
    protection_status       TEXT NOT NULL DEFAULT 'NONE' CHECK (protection_status IN ('NONE', 'PENDING', 'PLACED', 'FAILED')),
    protection_order_id     TEXT,
    last_error              TEXT,
    submitted_at_ny         TEXT NOT NULL,
    updated_at_ny           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tracked_status ON tracked_orders (status, ticker);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ny            TEXT NOT NULL,
    equity                  REAL NOT NULL,
    daily_starting_equity   REAL NOT NULL,
    intraday_peak_equity    REAL NOT NULL,
    daily_pnl_pct           REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS spread_positions (
    spread_id               TEXT PRIMARY KEY,
    ticker                  TEXT NOT NULL,
    structure               TEXT NOT NULL CHECK (structure IN ('BULL_PUT_SPREAD', 'IRON_CONDOR')),
    expiration_date         TEXT NOT NULL,
    qty                     INTEGER NOT NULL CHECK (qty > 0),
    legs_json               TEXT NOT NULL,
    entry_credit_per_spread REAL NOT NULL,
    max_risk_usd            REAL NOT NULL,
    profit_target_debit     REAL NOT NULL,
    stop_loss_debit         REAL NOT NULL,
    state                   TEXT NOT NULL CHECK (
                                state IN ('PENDING_OPEN', 'OPEN', 'PENDING_CLOSE', 'CLOSED', 'CANCELED')
                            ),
    open_order_id           TEXT,
    close_order_id          TEXT,
    realized_pnl_usd        REAL,
    opened_at_ny            TEXT NOT NULL,
    closed_at_ny            TEXT,
    updated_at_ny           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spread_ticker_state ON spread_positions (ticker, state);
"""

AUDIT_MIGRATION_COLUMNS = {
    "execution_status": "TEXT",   # PENDING_SUBMIT | SUBMITTED | SUBMIT_FAILED | DRY_RUN | NULL (rejected)
    "execution_error": "TEXT",
}
TRACKED_MIGRATION_COLUMNS = {
    "spread_id": "TEXT",
}


@dataclass(frozen=True)
class TrackedOrderRecord:
    alpaca_order_id: str
    client_order_id: str
    audit_log_id: Optional[int]
    ticker: str
    strategy: str
    action: str
    symbol: str
    qty: float
    limit_price: Optional[float]
    strike_price: Optional[float]
    expiration_date: Optional[str]
    trailing_stop_pct: Optional[float]
    status: str
    filled_qty: float
    filled_avg_price: Optional[float]
    protection_status: str
    protection_order_id: Optional[str]
    last_error: Optional[str]
    submitted_at_ny: str
    updated_at_ny: str
    spread_id: Optional[str] = None


@dataclass(frozen=True)
class SpreadPositionRecord:
    spread_id: str
    ticker: str
    structure: str
    expiration_date: str
    qty: int
    legs_json: str
    entry_credit_per_spread: float
    max_risk_usd: float
    profit_target_debit: float
    stop_loss_debit: float
    state: str
    open_order_id: Optional[str]
    close_order_id: Optional[str]
    realized_pnl_usd: Optional[float]
    opened_at_ny: str
    closed_at_ny: Optional[str]
    updated_at_ny: str


class LedgerRepository:
    """
    Synchronous SQLite repository backing RiskGateway, Wheel FSM, and compliance audit logging.
    """

    def __init__(self, db_path: Path | str = Path("data/state.db")) -> None:
        self.db_path = Path(db_path)
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        """Create all required tables and indexes idempotently and apply additive migrations."""
        with self._connect() as conn:
            conn.executescript(SCHEMA_DDL)
            existing = {r["name"] for r in conn.execute("PRAGMA table_info(audit_log)").fetchall()}
            for col, col_type in AUDIT_MIGRATION_COLUMNS.items():
                if col not in existing:
                    conn.execute(f"ALTER TABLE audit_log ADD COLUMN {col} {col_type}")
            tracked_cols = {r["name"] for r in conn.execute("PRAGMA table_info(tracked_orders)").fetchall()}
            for col, col_type in TRACKED_MIGRATION_COLUMNS.items():
                if col not in tracked_cols:
                    conn.execute(f"ALTER TABLE tracked_orders ADD COLUMN {col} {col_type}")

    # =========================================================================
    # 1. System State & Circuit Breaker (TRD-POL-006)
    # =========================================================================

    def sync_equity(
        self,
        current_equity: float,
        daily_starting_equity: Optional[float] = None,
        as_of_ny: Optional[datetime] = None,
    ) -> SystemStateRecord:
        """
        Persist or update daily starting equity and intraday high-water-mark equity.
        If the system is HALTED, it remains HALTED until manual_unlock() is invoked.
        """
        ts = (as_of_ny or now_ny()).astimezone(NY_TZ)
        today_str = ts.strftime("%Y-%m-%d")
        ts_iso = ts.isoformat()

        with self._connect() as conn:
            row = conn.execute("SELECT * FROM system_state WHERE id = 1").fetchone()
            if row is None:
                start_eq = daily_starting_equity if daily_starting_equity is not None else current_equity
                peak_eq = max(start_eq, current_equity)
                conn.execute(
                    """
                    INSERT INTO system_state (
                        id, trading_status, halt_reason, halt_policy_id,
                        trading_date_ny, daily_starting_equity, intraday_peak_equity,
                        current_equity, realized_pnl_today, unrealized_pnl_today, updated_at_ny
                    ) VALUES (1, 'ACTIVE', NULL, NULL, ?, ?, ?, ?, 0.0, ?, ?)
                    """,
                    (today_str, start_eq, peak_eq, current_equity, current_equity - start_eq, ts_iso),
                )
            else:
                status = row["trading_status"]
                if row["trading_date_ny"] != today_str and status != "HALTED":
                    # New trading session rollover
                    start_eq = daily_starting_equity if daily_starting_equity is not None else current_equity
                    peak_eq = max(start_eq, current_equity)
                else:
                    start_eq = (
                        daily_starting_equity
                        if daily_starting_equity is not None
                        else float(row["daily_starting_equity"])
                    )
                    peak_eq = max(float(row["intraday_peak_equity"]), start_eq, current_equity)

                conn.execute(
                    """
                    UPDATE system_state
                       SET trading_date_ny = ?,
                           daily_starting_equity = ?,
                           intraday_peak_equity = ?,
                           current_equity = ?,
                           unrealized_pnl_today = ?,
                           updated_at_ny = ?
                     WHERE id = 1
                    """,
                    (
                        today_str if status != "HALTED" else row["trading_date_ny"],
                        start_eq,
                        peak_eq,
                        current_equity,
                        current_equity - start_eq,
                        ts_iso,
                    ),
                )

        state = self.get_system_state()
        assert state is not None
        return state

    def get_system_state(self) -> Optional[SystemStateRecord]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM system_state WHERE id = 1").fetchone()
            if row is None:
                return None
            return SystemStateRecord(
                trading_status=row["trading_status"],
                halt_reason=row["halt_reason"],
                halt_policy_id=row["halt_policy_id"],
                trading_date_ny=row["trading_date_ny"],
                daily_starting_equity=float(row["daily_starting_equity"]),
                intraday_peak_equity=float(row["intraday_peak_equity"]),
                current_equity=float(row["current_equity"]),
                realized_pnl_today=float(row["realized_pnl_today"]),
                unrealized_pnl_today=float(row["unrealized_pnl_today"]),
                updated_at_ny=row["updated_at_ny"],
            )

    def trigger_halt(
        self,
        reason: str,
        policy_id: str = PolicyId.TRD_POL_006.value,
        current_equity: Optional[float] = None,
        as_of_ny: Optional[datetime] = None,
    ) -> SystemStateRecord:
        """Write HALTED lock state to SQLite immediately."""
        ts = (as_of_ny or now_ny()).astimezone(NY_TZ)
        ts_iso = ts.isoformat()
        today_str = ts.strftime("%Y-%m-%d")

        with self._connect() as conn:
            row = conn.execute("SELECT * FROM system_state WHERE id = 1").fetchone()
            if row is None:
                eq = current_equity if current_equity is not None else 100_000.0
                conn.execute(
                    """
                    INSERT INTO system_state (
                        id, trading_status, halt_reason, halt_policy_id,
                        trading_date_ny, daily_starting_equity, intraday_peak_equity,
                        current_equity, realized_pnl_today, unrealized_pnl_today, updated_at_ny
                    ) VALUES (1, 'HALTED', ?, ?, ?, ?, ?, ?, 0.0, 0.0, ?)
                    """,
                    (reason, policy_id, today_str, eq, eq, eq, ts_iso),
                )
            else:
                eq = current_equity if current_equity is not None else float(row["current_equity"])
                conn.execute(
                    """
                    UPDATE system_state
                       SET trading_status = 'HALTED',
                           halt_reason = ?,
                           halt_policy_id = ?,
                           current_equity = ?,
                           updated_at_ny = ?
                     WHERE id = 1
                    """,
                    (reason, policy_id, eq, ts_iso),
                )
        state = self.get_system_state()
        assert state is not None
        return state

    def manual_unlock(
        self,
        reset_baseline_equity: Optional[float] = None,
        as_of_ny: Optional[datetime] = None,
    ) -> SystemStateRecord:
        """Clear HALTED state via operator CLI (`python -m src.cli unlock`)."""
        ts = (as_of_ny or now_ny()).astimezone(NY_TZ)
        ts_iso = ts.isoformat()
        today_str = ts.strftime("%Y-%m-%d")

        with self._connect() as conn:
            row = conn.execute("SELECT * FROM system_state WHERE id = 1").fetchone()
            if row is None:
                eq = reset_baseline_equity if reset_baseline_equity is not None else 100_000.0
                conn.execute(
                    """
                    INSERT INTO system_state (
                        id, trading_status, halt_reason, halt_policy_id,
                        trading_date_ny, daily_starting_equity, intraday_peak_equity,
                        current_equity, realized_pnl_today, unrealized_pnl_today, updated_at_ny
                    ) VALUES (1, 'ACTIVE', NULL, NULL, ?, ?, ?, ?, 0.0, 0.0, ?)
                    """,
                    (today_str, eq, eq, eq, ts_iso),
                )
            else:
                eq = reset_baseline_equity if reset_baseline_equity is not None else float(row["current_equity"])
                conn.execute(
                    """
                    UPDATE system_state
                       SET trading_status = 'ACTIVE',
                           halt_reason = NULL,
                           halt_policy_id = NULL,
                           trading_date_ny = ?,
                           daily_starting_equity = ?,
                           intraday_peak_equity = ?,
                           current_equity = ?,
                           unrealized_pnl_today = 0.0,
                           updated_at_ny = ?
                     WHERE id = 1
                    """,
                    (today_str, eq, eq, eq, ts_iso),
                )
        state = self.get_system_state()
        assert state is not None
        return state

    # =========================================================================
    # 2. Wheel Strategy Finite-State Machine (TRD-POL-002 & TRD-POL-003)
    # =========================================================================

    def get_wheel_cycle(self, ticker: str) -> WheelCycleRecord:
        tkr = ticker.strip().upper()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM wheel_cycles WHERE ticker = ?", (tkr,)).fetchone()
            if row is None:
                return WheelCycleRecord(
                    ticker=tkr,
                    state=WheelState.IDLE,
                    active_option_symbol=None,
                    active_alpaca_order_id=None,
                    option_right=None,
                    strike_price=None,
                    expiration_date=None,
                    entry_premium_per_share=0.0,
                    reserved_csp_collateral=0.0,
                    assigned_shares_qty=0,
                    gross_cost_basis_per_share=0.0,
                    cumulative_premium_per_share=0.0,
                    net_cost_basis_per_share=0.0,
                    cycle_count=0,
                    updated_at_ny=now_ny().isoformat(),
                )
            return self._row_to_wheel_cycle(row)

    def list_wheel_cycles(self) -> list[WheelCycleRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM wheel_cycles ORDER BY ticker ASC").fetchall()
            return [self._row_to_wheel_cycle(r) for r in rows]

    @staticmethod
    def _row_to_wheel_cycle(row: sqlite3.Row) -> WheelCycleRecord:
        return WheelCycleRecord(
            ticker=row["ticker"],
            state=WheelState(row["state"]),
            active_option_symbol=row["active_option_symbol"],
            active_alpaca_order_id=row["active_alpaca_order_id"],
            option_right=row["option_right"],
            strike_price=float(row["strike_price"]) if row["strike_price"] is not None else None,
            expiration_date=row["expiration_date"],
            entry_premium_per_share=float(row["entry_premium_per_share"]),
            reserved_csp_collateral=float(row["reserved_csp_collateral"]),
            assigned_shares_qty=int(row["assigned_shares_qty"]),
            gross_cost_basis_per_share=float(row["gross_cost_basis_per_share"]),
            cumulative_premium_per_share=float(row["cumulative_premium_per_share"]),
            net_cost_basis_per_share=float(row["net_cost_basis_per_share"]),
            cycle_count=int(row["cycle_count"]),
            updated_at_ny=row["updated_at_ny"],
        )

    def get_total_reserved_csp_collateral(self) -> float:
        """Return sum of reserved collateral across all currently open CSPs (TRD-POL-002)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(reserved_csp_collateral), 0.0) AS total FROM wheel_cycles WHERE state = 'CSP_OPEN'"
            ).fetchone()
            return float(row["total"]) if row is not None else 0.0

    def record_csp_open(
        self,
        ticker: str,
        option_symbol: str,
        strike_price: float,
        expiration_date: str,
        premium_per_share: float,
        alpaca_order_id: Optional[str] = None,
        as_of_ny: Optional[datetime] = None,
    ) -> WheelCycleRecord:
        """Transition IDLE -> CSP_OPEN, locking 100 * strike_price collateral and recording premium."""
        tkr = ticker.strip().upper()
        current = self.get_wheel_cycle(tkr)
        ts_iso = (as_of_ny or now_ny()).astimezone(NY_TZ).isoformat()
        cum_prem = current.cumulative_premium_per_share + premium_per_share
        collateral = 100.0 * strike_price

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO wheel_cycles (
                    ticker, state, active_option_symbol, active_alpaca_order_id,
                    option_right, strike_price, expiration_date,
                    entry_premium_per_share, reserved_csp_collateral,
                    assigned_shares_qty, gross_cost_basis_per_share,
                    cumulative_premium_per_share, net_cost_basis_per_share,
                    cycle_count, updated_at_ny
                ) VALUES (?, 'CSP_OPEN', ?, ?, 'PUT', ?, ?, ?, ?, 0, 0.0, ?, 0.0, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    state = 'CSP_OPEN',
                    active_option_symbol = excluded.active_option_symbol,
                    active_alpaca_order_id = excluded.active_alpaca_order_id,
                    option_right = 'PUT',
                    strike_price = excluded.strike_price,
                    expiration_date = excluded.expiration_date,
                    entry_premium_per_share = excluded.entry_premium_per_share,
                    reserved_csp_collateral = excluded.reserved_csp_collateral,
                    cumulative_premium_per_share = excluded.cumulative_premium_per_share,
                    updated_at_ny = excluded.updated_at_ny
                """,
                (
                    tkr,
                    option_symbol,
                    alpaca_order_id,
                    strike_price,
                    expiration_date,
                    premium_per_share,
                    collateral,
                    cum_prem,
                    current.cycle_count,
                    ts_iso,
                ),
            )
        return self.get_wheel_cycle(tkr)

    def record_csp_closed_or_expired(
        self,
        ticker: str,
        buyback_debit_per_share: float = 0.0,
        as_of_ny: Optional[datetime] = None,
    ) -> WheelCycleRecord:
        """Transition CSP_OPEN -> IDLE upon 50% early profit-taking BTC or OTM expiration."""
        tkr = ticker.strip().upper()
        current = self.get_wheel_cycle(tkr)
        ts_iso = (as_of_ny or now_ny()).astimezone(NY_TZ).isoformat()
        cum_prem = max(0.0, current.cumulative_premium_per_share - buyback_debit_per_share)

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE wheel_cycles
                   SET state = 'IDLE',
                       active_option_symbol = NULL,
                       active_alpaca_order_id = NULL,
                       option_right = NULL,
                       strike_price = NULL,
                       expiration_date = NULL,
                       entry_premium_per_share = 0.0,
                       reserved_csp_collateral = 0.0,
                       cumulative_premium_per_share = ?,
                       cycle_count = cycle_count + 1,
                       updated_at_ny = ?
                 WHERE ticker = ?
                """,
                (cum_prem, ts_iso, tkr),
            )
        return self.get_wheel_cycle(tkr)

    def record_csp_assignment(
        self,
        ticker: str,
        assignment_strike: Optional[float] = None,
        shares_qty: int = 100,
        prior_premium_per_share: Optional[float] = None,
        as_of_ny: Optional[datetime] = None,
    ) -> WheelCycleRecord:
        """
        Transition CSP_OPEN -> ASSIGNED_SHARES upon put assignment.
        Computes net_cost_basis_per_share = gross_cost_basis_per_share - cumulative_premium_per_share.
        """
        tkr = ticker.strip().upper()
        current = self.get_wheel_cycle(tkr)
        ts_iso = (as_of_ny or now_ny()).astimezone(NY_TZ).isoformat()

        gross_basis = assignment_strike if assignment_strike is not None else (current.strike_price or 0.0)
        if gross_basis <= 0.0:
            raise ValueError(f"Cannot record CSP assignment for {tkr} without a positive strike price.")

        cum_prem = (
            prior_premium_per_share
            if prior_premium_per_share is not None
            else current.cumulative_premium_per_share
        )
        net_basis = round(max(0.0, gross_basis - cum_prem), 4)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO wheel_cycles (
                    ticker, state, active_option_symbol, active_alpaca_order_id,
                    option_right, strike_price, expiration_date,
                    entry_premium_per_share, reserved_csp_collateral,
                    assigned_shares_qty, gross_cost_basis_per_share,
                    cumulative_premium_per_share, net_cost_basis_per_share,
                    cycle_count, updated_at_ny
                ) VALUES (?, 'ASSIGNED_SHARES', NULL, NULL, NULL, NULL, NULL, 0.0, 0.0, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    state = 'ASSIGNED_SHARES',
                    active_option_symbol = NULL,
                    active_alpaca_order_id = NULL,
                    option_right = NULL,
                    strike_price = NULL,
                    expiration_date = NULL,
                    entry_premium_per_share = 0.0,
                    reserved_csp_collateral = 0.0,
                    assigned_shares_qty = excluded.assigned_shares_qty,
                    gross_cost_basis_per_share = excluded.gross_cost_basis_per_share,
                    cumulative_premium_per_share = excluded.cumulative_premium_per_share,
                    net_cost_basis_per_share = excluded.net_cost_basis_per_share,
                    updated_at_ny = excluded.updated_at_ny
                """,
                (tkr, shares_qty, gross_basis, cum_prem, net_basis, current.cycle_count, ts_iso),
            )
        return self.get_wheel_cycle(tkr)

    def record_cc_open(
        self,
        ticker: str,
        option_symbol: str,
        strike_price: float,
        expiration_date: str,
        premium_per_share: float,
        alpaca_order_id: Optional[str] = None,
        as_of_ny: Optional[datetime] = None,
    ) -> WheelCycleRecord:
        """Transition ASSIGNED_SHARES -> CC_OPEN and reduce net cost basis by newly collected CC premium."""
        tkr = ticker.strip().upper()
        current = self.get_wheel_cycle(tkr)
        ts_iso = (as_of_ny or now_ny()).astimezone(NY_TZ).isoformat()
        cum_prem = current.cumulative_premium_per_share + premium_per_share
        net_basis = round(max(0.0, current.gross_cost_basis_per_share - cum_prem), 4)

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE wheel_cycles
                   SET state = 'CC_OPEN',
                       active_option_symbol = ?,
                       active_alpaca_order_id = ?,
                       option_right = 'CALL',
                       strike_price = ?,
                       expiration_date = ?,
                       entry_premium_per_share = ?,
                       cumulative_premium_per_share = ?,
                       net_cost_basis_per_share = ?,
                       updated_at_ny = ?
                 WHERE ticker = ?
                """,
                (
                    option_symbol,
                    alpaca_order_id,
                    strike_price,
                    expiration_date,
                    premium_per_share,
                    cum_prem,
                    net_basis,
                    ts_iso,
                    tkr,
                ),
            )
        return self.get_wheel_cycle(tkr)

    def record_cc_closed_or_expired(
        self,
        ticker: str,
        buyback_debit_per_share: float = 0.0,
        as_of_ny: Optional[datetime] = None,
    ) -> WheelCycleRecord:
        """Transition CC_OPEN -> ASSIGNED_SHARES upon 50% early BTC or OTM expiration."""
        tkr = ticker.strip().upper()
        current = self.get_wheel_cycle(tkr)
        ts_iso = (as_of_ny or now_ny()).astimezone(NY_TZ).isoformat()
        cum_prem = max(0.0, current.cumulative_premium_per_share - buyback_debit_per_share)
        net_basis = round(max(0.0, current.gross_cost_basis_per_share - cum_prem), 4)

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE wheel_cycles
                   SET state = 'ASSIGNED_SHARES',
                       active_option_symbol = NULL,
                       active_alpaca_order_id = NULL,
                       option_right = NULL,
                       strike_price = NULL,
                       expiration_date = NULL,
                       entry_premium_per_share = 0.0,
                       cumulative_premium_per_share = ?,
                       net_cost_basis_per_share = ?,
                       updated_at_ny = ?
                 WHERE ticker = ?
                """,
                (cum_prem, net_basis, ts_iso, tkr),
            )
        return self.get_wheel_cycle(tkr)

    def record_cc_called_away(
        self,
        ticker: str,
        reset_to_idle: bool = False,
        as_of_ny: Optional[datetime] = None,
    ) -> WheelCycleRecord:
        """Transition CC_OPEN -> CALLED_AWAY (or IDLE if reset_to_idle=True) when shares are called away."""
        tkr = ticker.strip().upper()
        ts_iso = (as_of_ny or now_ny()).astimezone(NY_TZ).isoformat()
        target_state = WheelState.IDLE.value if reset_to_idle else WheelState.CALLED_AWAY.value

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE wheel_cycles
                   SET state = ?,
                       active_option_symbol = NULL,
                       active_alpaca_order_id = NULL,
                       option_right = NULL,
                       strike_price = NULL,
                       expiration_date = NULL,
                       entry_premium_per_share = 0.0,
                       reserved_csp_collateral = 0.0,
                       assigned_shares_qty = 0,
                       gross_cost_basis_per_share = 0.0,
                       cumulative_premium_per_share = 0.0,
                       net_cost_basis_per_share = 0.0,
                       cycle_count = cycle_count + 1,
                       updated_at_ny = ?
                 WHERE ticker = ?
                """,
                (target_state, ts_iso, tkr),
            )
        return self.get_wheel_cycle(tkr)

    # =========================================================================
    # 3. Bounded Dip-Ladder State (TRD-POL-004)
    # =========================================================================

    def get_dip_ladder(self, ticker: str) -> Optional[DipLadderRecord]:
        tkr = ticker.strip().upper()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM dip_ladders WHERE ticker = ?", (tkr,)).fetchone()
            if row is None:
                return None
            return DipLadderRecord(
                ticker=row["ticker"],
                initial_entry_price=float(row["initial_entry_price"]),
                add_on_tranches_used=int(row["add_on_tranches_used"]),
                last_tranche_price=float(row["last_tranche_price"]),
                realized_vol_20d=float(row["realized_vol_20d"]),
                paused_high_vol=bool(row["paused_high_vol"]),
                updated_at_ny=row["updated_at_ny"],
            )

    def upsert_dip_ladder(
        self,
        ticker: str,
        price: float,
        add_on_tranches_used: int,
        realized_vol_20d: float = 0.0,
        paused_high_vol: bool = False,
        as_of_ny: Optional[datetime] = None,
        reset: bool = False,
    ) -> DipLadderRecord:
        tkr = ticker.strip().upper()
        ts_iso = (as_of_ny or now_ny()).astimezone(NY_TZ).isoformat()
        existing = self.get_dip_ladder(tkr)
        initial_price = existing.initial_entry_price if (existing is not None and not reset) else price

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO dip_ladders (
                    ticker, initial_entry_price, add_on_tranches_used,
                    last_tranche_price, realized_vol_20d, paused_high_vol, updated_at_ny
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    initial_entry_price = excluded.initial_entry_price,
                    add_on_tranches_used = excluded.add_on_tranches_used,
                    last_tranche_price = excluded.last_tranche_price,
                    realized_vol_20d = excluded.realized_vol_20d,
                    paused_high_vol = excluded.paused_high_vol,
                    updated_at_ny = excluded.updated_at_ny
                """,
                (tkr, initial_price, add_on_tranches_used, price, realized_vol_20d, int(paused_high_vol), ts_iso),
            )
        rec = self.get_dip_ladder(tkr)
        assert rec is not None
        return rec

    # =========================================================================
    # 4. Idempotency & Cooldowns (TRD-POL-005)
    # =========================================================================

    def has_client_order_id(self, client_order_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM order_cooldowns WHERE client_order_id = ?",
                (client_order_id,),
            ).fetchone()
            return row is not None

    def get_last_order_epoch_for_ticker(self, ticker: str) -> Optional[float]:
        tkr = ticker.strip().upper()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT submitted_at_epoch FROM order_cooldowns WHERE ticker = ? ORDER BY submitted_at_epoch DESC LIMIT 1",
                (tkr,),
            ).fetchone()
            return float(row["submitted_at_epoch"]) if row is not None else None

    def record_order_cooldown(
        self,
        client_order_id: str,
        ticker: str,
        strategy: str,
        action: str,
        date_bucket: str,
        as_of_ny: Optional[datetime] = None,
    ) -> None:
        ts = (as_of_ny or now_ny()).astimezone(NY_TZ)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO order_cooldowns (
                    client_order_id, ticker, strategy, action,
                    date_bucket, submitted_at_epoch, submitted_at_ny
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    client_order_id,
                    ticker.strip().upper(),
                    strategy,
                    action,
                    date_bucket,
                    ts.timestamp(),
                    ts.isoformat(),
                ),
            )

    # =========================================================================
    # 5. Append-Only Audit Log
    # =========================================================================

    def append_audit_log(
        self,
        client_order_id: str,
        intent: OrderIntent,
        risk_decision: RiskDecisionStatus,
        policy_id: PolicyId,
        rejection_reason: Optional[str] = None,
        proposal: Optional[TradeProposal] = None,
        alpaca_order_id: Optional[str] = None,
        fill_price: Optional[float] = None,
        filled_qty: Optional[float] = None,
        as_of_ny: Optional[datetime] = None,
        execution_status: Optional[str] = None,
    ) -> int:
        ts_iso = (as_of_ny or now_ny()).astimezone(NY_TZ).isoformat()
        proposal_obj = proposal or intent.proposal
        proposal_json = proposal_obj.model_dump_json() if proposal_obj is not None else None
        intent_json = intent.model_dump_json()

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO audit_log (
                    timestamp_ny, client_order_id, ticker, strategy, action,
                    proposal_json, intent_json, risk_decision, policy_id,
                    rejection_reason, alpaca_order_id, fill_price, filled_qty, execution_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_iso,
                    client_order_id,
                    intent.ticker,
                    intent.strategy.value,
                    intent.action.value,
                    proposal_json,
                    intent_json,
                    risk_decision.value,
                    policy_id.value,
                    rejection_reason,
                    alpaca_order_id,
                    fill_price,
                    filled_qty,
                    execution_status,
                ),
            )
            return int(cursor.lastrowid or 0)

    def update_audit_execution(
        self,
        audit_id: int,
        alpaca_order_id: Optional[str] = None,
        fill_price: Optional[float] = None,
        filled_qty: Optional[float] = None,
        execution_status: Optional[str] = None,
        execution_error: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE audit_log
                   SET alpaca_order_id = COALESCE(?, alpaca_order_id),
                       fill_price = COALESCE(?, fill_price),
                       filled_qty = COALESCE(?, filled_qty),
                       execution_status = COALESCE(?, execution_status),
                       execution_error = COALESCE(?, execution_error)
                 WHERE id = ?
                """,
                (alpaca_order_id, fill_price, filled_qty, execution_status, execution_error, audit_id),
            )

    def list_audit_logs(
        self,
        limit: int = 100,
        policy_id: Optional[str] = None,
        ticker: Optional[str] = None,
    ) -> list[AuditLogRecord]:
        query = "SELECT * FROM audit_log WHERE 1=1"
        params: list[object] = []
        if policy_id:
            query += " AND policy_id = ?"
            params.append(policy_id)
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker.strip().upper())
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                AuditLogRecord(
                    id=int(r["id"]),
                    timestamp_ny=r["timestamp_ny"],
                    client_order_id=r["client_order_id"],
                    ticker=r["ticker"],
                    strategy=r["strategy"],
                    action=r["action"],
                    proposal_json=r["proposal_json"],
                    intent_json=r["intent_json"],
                    risk_decision=r["risk_decision"],
                    policy_id=r["policy_id"],
                    rejection_reason=r["rejection_reason"],
                    alpaca_order_id=r["alpaca_order_id"],
                    fill_price=float(r["fill_price"]) if r["fill_price"] is not None else None,
                    filled_qty=float(r["filled_qty"]) if r["filled_qty"] is not None else None,
                    execution_status=r["execution_status"],
                    execution_error=r["execution_error"],
                )
                for r in rows
            ]

    # =========================================================================
    # 6. Broker Order Tracking (fills -> state transitions, trailing-stop protection)
    # =========================================================================

    def track_order(
        self,
        intent: OrderIntent,
        alpaca_order_id: str,
        client_order_id: str,
        audit_log_id: Optional[int],
        trailing_stop_pct: Optional[float] = None,
        as_of_ny: Optional[datetime] = None,
    ) -> None:
        ts_iso = (as_of_ny or now_ny()).astimezone(NY_TZ).isoformat()
        symbol_str = (
            ",".join(leg.option_symbol for leg in intent.legs)
            if intent.legs
            else (intent.option_symbol or intent.ticker)
        )
        exp_str = (
            intent.expiration_date.isoformat()
            if intent.expiration_date
            else (intent.legs[0].expiration_date.isoformat() if intent.legs else None)
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tracked_orders (
                    alpaca_order_id, client_order_id, audit_log_id, ticker, strategy, action, symbol,
                    qty, limit_price, strike_price, expiration_date, trailing_stop_pct,
                    status, filled_qty, protection_status, submitted_at_ny, updated_at_ny, spread_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', 0.0, 'NONE', ?, ?, ?)
                """,
                (
                    alpaca_order_id,
                    client_order_id,
                    audit_log_id,
                    intent.ticker,
                    intent.strategy.value,
                    intent.action.value,
                    symbol_str,
                    float(intent.qty),
                    intent.limit_price,
                    intent.strike_price,
                    exp_str,
                    trailing_stop_pct,
                    ts_iso,
                    ts_iso,
                    intent.spread_id,
                ),
            )

    @staticmethod
    def _row_to_tracked(r: sqlite3.Row) -> TrackedOrderRecord:
        return TrackedOrderRecord(
            alpaca_order_id=r["alpaca_order_id"],
            client_order_id=r["client_order_id"],
            audit_log_id=r["audit_log_id"],
            ticker=r["ticker"],
            strategy=r["strategy"],
            action=r["action"],
            symbol=r["symbol"],
            qty=float(r["qty"]),
            limit_price=r["limit_price"],
            strike_price=r["strike_price"],
            expiration_date=r["expiration_date"],
            trailing_stop_pct=r["trailing_stop_pct"],
            status=r["status"],
            filled_qty=float(r["filled_qty"]),
            filled_avg_price=r["filled_avg_price"],
            protection_status=r["protection_status"],
            protection_order_id=r["protection_order_id"],
            last_error=r["last_error"],
            submitted_at_ny=r["submitted_at_ny"],
            updated_at_ny=r["updated_at_ny"],
            spread_id=r["spread_id"] if "spread_id" in r.keys() else None,
        )

    def list_tracked_orders(
        self, status: Optional[str] = None, protection_status: Optional[tuple[str, ...]] = None
    ) -> list[TrackedOrderRecord]:
        query = "SELECT * FROM tracked_orders WHERE 1=1"
        params: list[object] = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if protection_status:
            query += f" AND protection_status IN ({','.join('?' * len(protection_status))})"
            params.extend(protection_status)
        query += " ORDER BY submitted_at_ny ASC"
        with self._connect() as conn:
            return [self._row_to_tracked(r) for r in conn.execute(query, params).fetchall()]

    def count_open_tracked(self, ticker: str, action: Optional[str] = None) -> int:
        query = "SELECT COUNT(*) AS n FROM tracked_orders WHERE status = 'OPEN' AND ticker = ?"
        params: list[object] = [ticker.strip().upper()]
        if action:
            query += " AND action = ?"
            params.append(action)
        with self._connect() as conn:
            return int(conn.execute(query, params).fetchone()["n"])

    def update_tracked_order(self, alpaca_order_id: str, **fields: object) -> None:
        allowed = {
            "status", "filled_qty", "filled_avg_price", "protection_status",
            "protection_order_id", "last_error",
        }
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"Unsupported tracked_orders fields: {bad}")
        if not fields:
            return
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE tracked_orders SET {assignments}, updated_at_ny = ? WHERE alpaca_order_id = ?",
                (*fields.values(), now_ny().isoformat(), alpaca_order_id),
            )

    # =========================================================================
    # 7. Equity Snapshots (dashboard equity curve)
    # =========================================================================

    def record_equity_snapshot(
        self,
        equity: float,
        daily_starting_equity: float,
        intraday_peak_equity: float,
        as_of_ny: Optional[datetime] = None,
    ) -> None:
        ts_iso = (as_of_ny or now_ny()).astimezone(NY_TZ).isoformat()
        pnl_pct = (equity - daily_starting_equity) / daily_starting_equity if daily_starting_equity > 0 else 0.0
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO equity_snapshots (
                    timestamp_ny, equity, daily_starting_equity, intraday_peak_equity, daily_pnl_pct
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (ts_iso, equity, daily_starting_equity, intraday_peak_equity, pnl_pct),
            )

    # =========================================================================
    # 8. Additional Wheel transitions used by broker reconciliation
    # =========================================================================

    def reset_wheel_to_idle(self, ticker: str, as_of_ny: Optional[datetime] = None) -> WheelCycleRecord:
        """CALLED_AWAY -> IDLE (or ASSIGNED_SHARES with no shares left -> IDLE)."""
        tkr = ticker.strip().upper()
        ts_iso = (as_of_ny or now_ny()).astimezone(NY_TZ).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE wheel_cycles
                   SET state = 'IDLE',
                       active_option_symbol = NULL,
                       active_alpaca_order_id = NULL,
                       option_right = NULL,
                       strike_price = NULL,
                       expiration_date = NULL,
                       entry_premium_per_share = 0.0,
                       reserved_csp_collateral = 0.0,
                       assigned_shares_qty = 0,
                       gross_cost_basis_per_share = 0.0,
                       cumulative_premium_per_share = 0.0,
                       net_cost_basis_per_share = 0.0,
                       updated_at_ny = ?
                 WHERE ticker = ?
                """,
                (ts_iso, tkr),
            )
        return self.get_wheel_cycle(tkr)

    # =========================================================================
    # 9. Multi-Leg Defined-Risk Spread Tracking (Bull Put Spread / Iron Condor)
    # =========================================================================

    def record_spread_pending_open(
        self,
        spread_id: str,
        ticker: str,
        structure: str,
        expiration_date: str,
        qty: int,
        legs_json: str,
        entry_credit_per_spread: float,
        max_risk_usd: float,
        profit_target_debit: float,
        stop_loss_debit: float,
        open_order_id: str,
        as_of_ny: Optional[datetime] = None,
    ) -> None:
        ts_iso = (as_of_ny or now_ny()).astimezone(NY_TZ).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO spread_positions (
                    spread_id, ticker, structure, expiration_date, qty, legs_json,
                    entry_credit_per_spread, max_risk_usd, profit_target_debit, stop_loss_debit,
                    state, open_order_id, close_order_id, realized_pnl_usd, opened_at_ny, closed_at_ny, updated_at_ny
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING_OPEN', ?, NULL, NULL, ?, NULL, ?)
                """,
                (
                    spread_id,
                    ticker.strip().upper(),
                    structure,
                    expiration_date,
                    qty,
                    legs_json,
                    entry_credit_per_spread,
                    max_risk_usd,
                    profit_target_debit,
                    stop_loss_debit,
                    open_order_id,
                    ts_iso,
                    ts_iso,
                ),
            )

    def mark_spread_open(
        self,
        spread_id: str,
        filled_credit_per_spread: Optional[float] = None,
        profit_take_ratio: float = 0.50,
        stop_loss_multiple: float = 2.5,
        as_of_ny: Optional[datetime] = None,
    ) -> None:
        ts_iso = (as_of_ny or now_ny()).astimezone(NY_TZ).isoformat()
        rec = self.get_spread_position(spread_id)
        if rec is None:
            return
        credit = abs(filled_credit_per_spread) if filled_credit_per_spread else rec.entry_credit_per_spread
        pt = round(max(0.05, credit * (1.0 - profit_take_ratio)), 2)
        sl = round(max(credit + 0.10, credit * stop_loss_multiple), 2)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE spread_positions
                   SET state = 'OPEN',
                       entry_credit_per_spread = ?,
                       profit_target_debit = ?,
                       stop_loss_debit = ?,
                       updated_at_ny = ?
                 WHERE spread_id = ?
                """,
                (credit, pt, sl, ts_iso, spread_id),
            )

    def mark_spread_pending_close(
        self, spread_id: str, close_order_id: str, as_of_ny: Optional[datetime] = None
    ) -> None:
        ts_iso = (as_of_ny or now_ny()).astimezone(NY_TZ).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE spread_positions
                   SET state = 'PENDING_CLOSE',
                       close_order_id = ?,
                       updated_at_ny = ?
                 WHERE spread_id = ?
                """,
                (close_order_id, ts_iso, spread_id),
            )

    def mark_spread_closed(
        self,
        spread_id: str,
        close_debit_per_spread: float,
        as_of_ny: Optional[datetime] = None,
    ) -> None:
        ts_iso = (as_of_ny or now_ny()).astimezone(NY_TZ).isoformat()
        rec = self.get_spread_position(spread_id)
        if rec is None:
            return
        pnl = round((rec.entry_credit_per_spread - abs(close_debit_per_spread)) * 100.0 * rec.qty, 2)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE spread_positions
                   SET state = 'CLOSED',
                       realized_pnl_usd = ?,
                       closed_at_ny = ?,
                       updated_at_ny = ?
                 WHERE spread_id = ?
                """,
                (pnl, ts_iso, ts_iso, spread_id),
            )

    def mark_spread_canceled(self, spread_id: str, as_of_ny: Optional[datetime] = None) -> None:
        ts_iso = (as_of_ny or now_ny()).astimezone(NY_TZ).isoformat()
        rec = self.get_spread_position(spread_id)
        if rec is None:
            return
        new_state = "OPEN" if rec.state == "PENDING_CLOSE" else "CANCELED"
        with self._connect() as conn:
            conn.execute(
                "UPDATE spread_positions SET state = ?, updated_at_ny = ? WHERE spread_id = ?",
                (new_state, ts_iso, spread_id),
            )

    def get_spread_position(self, spread_id: str) -> Optional[SpreadPositionRecord]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM spread_positions WHERE spread_id = ?", (spread_id,)).fetchone()
            return self._row_to_spread(row) if row is not None else None

    def list_spread_positions(
        self, ticker: Optional[str] = None, states: Optional[tuple[str, ...]] = None
    ) -> list[SpreadPositionRecord]:
        query = "SELECT * FROM spread_positions WHERE 1=1"
        params: list[object] = []
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker.strip().upper())
        if states:
            query += f" AND state IN ({','.join('?' * len(states))})"
            params.extend(states)
        query += " ORDER BY opened_at_ny DESC"
        with self._connect() as conn:
            return [self._row_to_spread(r) for r in conn.execute(query, params).fetchall()]

    @staticmethod
    def _row_to_spread(r: sqlite3.Row) -> SpreadPositionRecord:
        return SpreadPositionRecord(
            spread_id=r["spread_id"],
            ticker=r["ticker"],
            structure=r["structure"],
            expiration_date=r["expiration_date"],
            qty=int(r["qty"]),
            legs_json=r["legs_json"],
            entry_credit_per_spread=float(r["entry_credit_per_spread"]),
            max_risk_usd=float(r["max_risk_usd"]),
            profit_target_debit=float(r["profit_target_debit"]),
            stop_loss_debit=float(r["stop_loss_debit"]),
            state=r["state"],
            open_order_id=r["open_order_id"],
            close_order_id=r["close_order_id"],
            realized_pnl_usd=float(r["realized_pnl_usd"]) if r["realized_pnl_usd"] is not None else None,
            opened_at_ny=r["opened_at_ny"],
            closed_at_ny=r["closed_at_ny"],
            updated_at_ny=r["updated_at_ny"],
        )
