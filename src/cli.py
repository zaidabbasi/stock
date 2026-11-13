"""Operator CLI utilities for alpaca-gemini-trader (`python -m src.cli unlock | status | init-db`)."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from src.config import NY_TZ, load_settings
from src.persistence.ledger import LedgerRepository, now_ny


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Operator CLI for alpaca-gemini-trader risk & state management.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    unlock_p = sub.add_parser(
        "unlock",
        help="Manually clear TRD-POL-006 HALTED circuit-breaker lock state in SQLite.",
    )
    unlock_p.add_argument(
        "--baseline-equity",
        type=float,
        default=None,
        help="Optional new daily starting/peak equity baseline ($).",
    )

    sub.add_parser("status", help="Display current system_state, active Wheel cycles, and recent audit logs.")
    sub.add_parser("init-db", help="Initialize SQLite schema in data/state.db.")
    sub.add_parser("setup", help="Interactively store Alpaca PAPER (and optional Gemini) keys in .env (hidden input).")
    sub.add_parser("verify", help="Verify .env credentials against Alpaca PAPER / Gemini without printing secrets.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "setup":
        from src.credentials import interactive_setup

        interactive_setup()
        return 0
    if args.command == "verify":
        from src.credentials import verify_connections

        return verify_connections()

    settings = load_settings()
    ledger = LedgerRepository(db_path=settings.SQLITE_DB_PATH)

    if args.command == "init-db":
        ledger.init_schema()
        print(
            json.dumps(
                {
                    "event": "db_initialized",
                    "db_path": str(settings.SQLITE_DB_PATH),
                    "timestamp_ny": now_ny().isoformat(),
                }
            )
        )
        return 0

    if args.command == "unlock":
        updated = ledger.manual_unlock(
            reset_baseline_equity=args.baseline_equity,
            as_of_ny=now_ny(),
        )
        print(
            json.dumps(
                {
                    "event": "circuit_breaker_unlocked",
                    "trading_status": updated.trading_status,
                    "daily_starting_equity": updated.daily_starting_equity,
                    "intraday_peak_equity": updated.intraday_peak_equity,
                    "current_equity": updated.current_equity,
                    "updated_at_ny": updated.updated_at_ny,
                }
            )
        )
        return 0

    if args.command == "status":
        sys_state = ledger.get_system_state()
        wheel_cycles = ledger.list_wheel_cycles()
        open_spreads = ledger.list_spread_positions(states=("PENDING_OPEN", "OPEN", "PENDING_CLOSE"))
        recent_logs = ledger.list_audit_logs(limit=10)
        open_tracked = ledger.list_tracked_orders(status="OPEN")
        payload = {
            "timestamp_ny": now_ny().astimezone(NY_TZ).isoformat(),
            "trading_mode": settings.TRADING_MODE,
            "analyst_mode": settings.ANALYST_MODE,
            "gemini_model": settings.GEMINI_MODEL,
            "has_gemini_key": settings.has_gemini_key,
            "system_state": sys_state.__dict__ if sys_state else None,
            "active_wheel_cycles": [c.__dict__ for c in wheel_cycles],
            "open_spreads": len(open_spreads),
            "open_tracked_orders": len(open_tracked),
            "recent_audit_count": len(recent_logs),
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
