"""Strictly read-only Streamlit observability dashboard (`streamlit run src/dashboard.py`)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest

from src.config import NY_TZ, SecurityConfigurationError, Settings, load_settings


def load_table_df(db_path: Path, query: str) -> pd.DataFrame:
    """Query SQLite strictly in read-only mode without creating or modifying the database."""
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        try:
            return pd.read_sql_query(query, conn)
        except Exception:
            return pd.DataFrame()


def load_system_state_ro(db_path: Path) -> Optional[dict[str, Any]]:
    df = load_table_df(db_path, "SELECT * FROM system_state WHERE id = 1")
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def fetch_alpaca_paper_overview(settings: Settings) -> dict[str, Any]:
    """Read-only fetch from Alpaca Paper Trading API (never submits orders or writes state)."""
    if (
        not settings.APCA_API_KEY_ID
        or not settings.APCA_API_SECRET_KEY
        or "YOUR_PAPER" in settings.APCA_API_KEY_ID
    ):
        return {"connected": False, "error": "Alpaca credentials not configured in .env."}

    try:
        client = TradingClient(
            api_key=settings.APCA_API_KEY_ID,
            secret_key=settings.APCA_API_SECRET_KEY,
            paper=True,
            url_override=settings.ALPACA_PAPER_BASE_URL,
        )
        account = client.get_account()
        clock = client.get_clock()
        positions = client.get_all_positions()
        open_orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100))
        history = client.get_portfolio_history(GetPortfolioHistoryRequest(period="1M", timeframe="1D"))

        hist_rows = []
        timestamps = getattr(history, "timestamp", None) or []
        equities = getattr(history, "equity", None) or []
        for ts_epoch, eq in zip(timestamps, equities):
            if eq is not None:
                dt = datetime.fromtimestamp(ts_epoch, tz=NY_TZ)
                hist_rows.append({"timestamp_ny": dt.strftime("%Y-%m-%d"), "equity": float(eq)})

        pos_rows = [
            {
                "symbol": str(p.symbol),
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
            }
            for p in positions
        ]
        order_rows = [
            {
                "id": str(o.id),
                "symbol": str(o.symbol),
                "side": str(getattr(o.side, "value", o.side)),
                "type": str(getattr(o.type, "value", o.type)),
                "qty": float(o.qty or 0),
                "limit_price": float(o.limit_price) if o.limit_price is not None else None,
                "trail_percent": float(o.trail_percent) if getattr(o, "trail_percent", None) is not None else None,
                "status": str(getattr(o.status, "value", o.status)),
            }
            for o in open_orders
        ]
        return {
            "connected": True,
            "status": str(getattr(account.status, "value", account.status)),
            "equity": float(account.equity),
            "last_equity": float(getattr(account, "last_equity", account.equity) or account.equity),
            "non_marginable_bp": float(account.non_marginable_buying_power),
            "options_bp": float(getattr(account, "options_buying_power", 0.0) or 0.0),
            "options_level": getattr(account, "options_trading_level", "n/a"),
            "market_open": bool(clock.is_open),
            "clock_ny": clock.timestamp.astimezone(NY_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"),
            "positions": pd.DataFrame(pos_rows),
            "open_orders": pd.DataFrame(order_rows),
            "history": pd.DataFrame(hist_rows),
        }
    except Exception as exc:
        return {"connected": False, "error": f"{type(exc).__name__}: {str(exc)[:180]}"}


def render_dashboard() -> None:
    st.set_page_config(
        page_title="Alpaca-Gemini Trader | Paper Observability",
        layout="wide",
    )
    st.title("Alpaca-Gemini Quantitative Paper Trader — Observability Dashboard")
    st.caption(
        "Strictly Read-Only Monitor | Deterministic RiskGateway (TRD-POL-001 .. TRD-POL-006), "
        "Alpaca Paper API & SQLite State Ledger"
    )

    try:
        settings = load_settings()
    except SecurityConfigurationError as exc:
        st.error(f"FATAL Security Configuration Violation: {exc}")
        st.stop()
        return

    db_path = settings.SQLITE_DB_PATH
    sys_state = load_system_state_ro(db_path)
    alpaca_info = fetch_alpaca_paper_overview(settings)

    # --- Top Status Banner ---
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Trading Mode Lock", f"{settings.TRADING_MODE} ({settings.ANALYST_MODE})")

    if sys_state is not None:
        col2.metric("Circuit Breaker (TRD-POL-006)", str(sys_state["trading_status"]))
        daily_start = float(sys_state["daily_starting_equity"])
        curr_eq = alpaca_info["equity"] if alpaca_info.get("connected") else float(sys_state["current_equity"])
        daily_loss_pct = (daily_start - curr_eq) / daily_start if daily_start > 0 else 0.0
        col3.metric("Paper Equity", f"${curr_eq:,.2f}")
        col4.metric("Daily Starting Equity", f"${daily_start:,.2f}")
        col5.metric(
            "Daily Loss vs Start",
            f"{daily_loss_pct:.2%}",
            delta=f"Limit: {settings.MAX_DAILY_DRAWDOWN_PCT:.2%}",
            delta_color="inverse",
        )
        if sys_state["trading_status"] == "HALTED":
            st.error(
                f"SYSTEM HALTED [{sys_state['halt_policy_id']}]: {sys_state['halt_reason']} — "
                "Clear manually via `python -m src.cli unlock`."
            )
    elif alpaca_info.get("connected"):
        col2.metric("Circuit Breaker (TRD-POL-006)", "READY (No Cycles Yet)")
        col3.metric("Paper Equity", f"${alpaca_info['equity']:,.2f}")
        col4.metric("Non-Marginable BP", f"${alpaca_info['non_marginable_bp']:,.2f}")
        col5.metric("Market Clock (NY)", "OPEN" if alpaca_info["market_open"] else "CLOSED")
    else:
        col2.metric("Circuit Breaker (TRD-POL-006)", "UNINITIALIZED")
        col3.metric("Paper Equity", "N/A")
        col4.metric("Daily Starting Equity", "N/A")
        col5.metric("Max Daily Loss Limit", f"{settings.MAX_DAILY_DRAWDOWN_PCT:.2%}")

    if alpaca_info.get("connected"):
        st.caption(
            f"Alpaca Paper Status: `{alpaca_info['status']}` | Options Level: `{alpaca_info['options_level']}` | "
            f"Options BP: `${alpaca_info['options_bp']:,.2f}` | Clock: `{alpaca_info['clock_ny']}`"
        )
    else:
        st.warning(f"Alpaca Paper API unavailable: {alpaca_info.get('error')}")

    st.divider()

    # --- Equity Curve ---
    st.subheader("Paper Portfolio Equity Curve")
    eq_df = load_table_df(
        db_path,
        "SELECT timestamp_ny, equity, daily_starting_equity, intraday_peak_equity FROM equity_snapshots ORDER BY id ASC",
    )
    if not eq_df.empty:
        st.line_chart(eq_df.set_index("timestamp_ny")[["equity", "daily_starting_equity"]])
    elif alpaca_info.get("connected") and not alpaca_info["history"].empty:
        st.line_chart(alpaca_info["history"].set_index("timestamp_ny")[["equity"]])
    else:
        st.info("No intraday snapshots or portfolio history recorded yet.")

    # --- Live Alpaca Paper Positions & Open Orders ---
    if alpaca_info.get("connected"):
        p_col, o_col = st.columns(2)
        with p_col:
            st.subheader("Live Alpaca Paper Positions")
            pos_df = alpaca_info["positions"]
            st.dataframe(pos_df, use_container_width=True) if not pos_df.empty else st.info("No open positions.")
        with o_col:
            st.subheader("Open Alpaca Paper Orders")
            ord_df = alpaca_info["open_orders"]
            st.dataframe(ord_df, use_container_width=True) if not ord_df.empty else st.info("No open orders.")

    st.divider()

    # --- Active Wheel Cycles & Dip Ladders ---
    w_col, d_col = st.columns(2)
    with w_col:
        st.subheader("Options Wheel State Machines (`wheel_cycles`)")
        wheel_df = load_table_df(db_path, "SELECT * FROM wheel_cycles ORDER BY ticker ASC")
        st.dataframe(wheel_df, use_container_width=True) if not wheel_df.empty else st.info("No Wheel cycles recorded.")

    with d_col:
        st.subheader("Bounded Dip Ladders (`dip_ladders` — TRD-POL-004)")
        dip_df = load_table_df(db_path, "SELECT * FROM dip_ladders ORDER BY ticker ASC")
        st.dataframe(dip_df, use_container_width=True) if not dip_df.empty else st.info("No active dip-ladder trackers.")

    st.subheader("Multi-Leg Defined-Risk Credit Spreads (`spread_positions`)")
    spread_df = load_table_df(
        db_path,
        "SELECT spread_id, ticker, structure, expiration_date, qty, entry_credit_per_spread, max_risk_usd, profit_target_debit, stop_loss_debit, state, realized_pnl_usd, updated_at_ny FROM spread_positions ORDER BY opened_at_ny DESC LIMIT 100",
    )
    st.dataframe(spread_df, use_container_width=True) if not spread_df.empty else st.info("No spread positions yet.")

    st.divider()

    st.subheader("Tracked Broker Orders & Trailing-Stop Protection (`tracked_orders`)")
    tracked_df = load_table_df(
        db_path,
        "SELECT alpaca_order_id, ticker, strategy, action, status, filled_qty, filled_avg_price, protection_status, protection_order_id, spread_id, updated_at_ny FROM tracked_orders ORDER BY updated_at_ny DESC LIMIT 100",
    )
    st.dataframe(tracked_df, use_container_width=True) if not tracked_df.empty else st.info("No tracked orders yet.")

    st.divider()

    # --- Audit Log & Gemini TradeProposal Rationales ---
    st.subheader("RiskGateway Compliance Audit Log & Analyst Proposals (`audit_log`)")
    audit_df = load_table_df(
        db_path,
        """
        SELECT id, timestamp_ny, client_order_id, ticker, strategy, action,
               risk_decision, policy_id, execution_status, rejection_reason,
               alpaca_order_id, fill_price, filled_qty, execution_error, proposal_json
          FROM audit_log
         ORDER BY id DESC
         LIMIT 200
        """,
    )
    if audit_df.empty:
        st.info("No order intents or policy evaluations logged yet.")
    else:
        st.dataframe(
            audit_df.drop(columns=["proposal_json"], errors="ignore"),
            use_container_width=True,
        )

        with st.expander("Inspect Analyst TradeProposal Rationales"):
            for _, row in audit_df.iterrows():
                raw_prop = row.get("proposal_json")
                if raw_prop:
                    try:
                        parsed: dict[str, Any] = json.loads(raw_prop)
                        st.markdown(
                            f"**#{row['id']} — {row['ticker']} ({row['strategy']}) | "
                            f"Conviction: `{parsed.get('conviction_score')}` | "
                            f"Gateway Decision: `{row['risk_decision']} ({row['policy_id']})`**"
                        )
                        st.write(parsed.get("rationale", ""))
                        if parsed.get("risk_flags"):
                            st.caption(f"Risk Flags: {', '.join(parsed['risk_flags'])}")
                    except Exception:
                        pass


if __name__ == "__main__":
    render_dashboard()
