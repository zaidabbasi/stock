"""Fail-closed Alpaca Paper Trading broker client wrapper with clock gating, market data, and backoff retries."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Optional, TypeVar

from alpaca.common.enums import BaseURL
from alpaca.data.enums import DataFeed
from alpaca.data.historical.news import NewsClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import (
    NewsRequest,
    OptionChainRequest,
    OptionLatestQuoteRequest,
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestTradeRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import ContractType, OrderClass, OrderSide, PositionIntent, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    OptionLegRequest,
    TrailingStopOrderRequest,
)

from src.config import (
    NY_TZ,
    SecurityConfigurationError,
    Settings,
    validate_broker_endpoint,
)
from src.intelligence.schemas import (
    OptionContractQuote,
    OptionRight,
    OrderIntent,
    PortfolioSnapshot,
    TradeAction,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")

_OCC_PATTERN = re.compile(r"^([A-Z.]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})$")


def parse_occ_option_symbol(symbol: str) -> Optional[tuple[str, date, OptionRight, float]]:
    """
    Parse an OCC option symbol such as 'F261127P00016500' into:
        (underlying_ticker, expiration_date, OptionRight, strike_price)
    Returns None if `symbol` is an equity symbol or non-OCC string.
    """
    clean = symbol.strip().upper()
    m = _OCC_PATTERN.match(clean)
    if not m:
        return None
    underlying, yy, mm, dd, cp, strike_raw = m.groups()
    exp = date(2000 + int(yy), int(mm), int(dd))
    right = OptionRight.PUT if cp == "P" else OptionRight.CALL
    strike = int(strike_raw) / 1000.0
    return underlying, exp, right, strike


@dataclass(frozen=True)
class MarketClockStatus:
    """Verified market session state anchored to America/New_York and Alpaca get_clock()."""

    is_open: bool
    timestamp_ny: datetime
    next_open_ny: datetime
    next_close_ny: datetime


@dataclass(frozen=True)
class BrokerOrderResult:
    """Normalized broker order submission / polling response."""

    alpaca_order_id: str
    client_order_id: str
    status: str
    filled_avg_price: Optional[float] = None
    filled_qty: Optional[float] = None


class AlpacaBroker:
    """
    Deterministic Alpaca Paper Trading wrapper.
    Hardcodes `paper=True` on TradingClient and verifies Paper Trading invariants prior to any network call.
    """

    def __init__(
        self,
        settings: Settings,
        trading_client: Optional[Any] = None,
        stock_data_client: Optional[Any] = None,
        option_data_client: Optional[Any] = None,
        news_client: Optional[Any] = None,
        max_retries: int = 3,
        base_backoff_seconds: float = 0.5,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._verify_security_locks(settings, trading_client)
        self.settings = settings
        self.max_retries = max_retries
        self.base_backoff_seconds = base_backoff_seconds
        self._sleep = sleep_fn

        if trading_client is not None:
            self.trading_client = trading_client
        else:
            # Invariant #2: Hardcode paper=True and explicitly validate URL override
            self.trading_client = TradingClient(
                api_key=settings.APCA_API_KEY_ID or "PAPER_PLACEHOLDER_KEY",
                secret_key=settings.APCA_API_SECRET_KEY or "PAPER_PLACEHOLDER_SECRET",
                paper=True,
                url_override=settings.ALPACA_PAPER_BASE_URL,
            )

        self._verify_trading_client_is_paper(self.trading_client)

        self.stock_data_client = stock_data_client or StockHistoricalDataClient(
            api_key=settings.APCA_API_KEY_ID or "PAPER_PLACEHOLDER_KEY",
            secret_key=settings.APCA_API_SECRET_KEY or "PAPER_PLACEHOLDER_SECRET",
        )
        self.option_data_client = option_data_client or OptionHistoricalDataClient(
            api_key=settings.APCA_API_KEY_ID or "PAPER_PLACEHOLDER_KEY",
            secret_key=settings.APCA_API_SECRET_KEY or "PAPER_PLACEHOLDER_SECRET",
        )
        self.news_client = news_client or NewsClient(
            api_key=settings.APCA_API_KEY_ID or "PAPER_PLACEHOLDER_KEY",
            secret_key=settings.APCA_API_SECRET_KEY or "PAPER_PLACEHOLDER_SECRET",
        )

    @staticmethod
    def _verify_security_locks(settings: Settings, trading_client: Optional[Any] = None) -> None:
        if settings.TRADING_MODE != "PAPER":
            raise SecurityConfigurationError(
                f"FATAL: AlpacaBroker refused initialization because TRADING_MODE='{settings.TRADING_MODE}' != 'PAPER'."
            )
        validate_broker_endpoint(settings.ALPACA_PAPER_BASE_URL)
        if trading_client is not None:
            AlpacaBroker._verify_trading_client_is_paper(trading_client)

    @staticmethod
    def _verify_trading_client_is_paper(client: Any) -> None:
        """Inspect the underlying TradingClient to guarantee paper mode and non-live base URL."""
        base_url = getattr(client, "_base_url", None)
        if base_url is not None:
            if base_url == BaseURL.TRADING_LIVE:
                raise SecurityConfigurationError(
                    "FATAL: Live Alpaca endpoint detected on TradingClient (BaseURL.TRADING_LIVE). "
                    "Only Paper Trading is permitted."
                )
            validate_broker_endpoint(base_url)
        if getattr(client, "paper", None) is False:
            raise SecurityConfigurationError(
                "FATAL: Injected TradingClient has paper=False. Live trading is strictly forbidden."
            )
        if getattr(client, "_sandbox", None) is False:
            raise SecurityConfigurationError(
                "FATAL: TradingClient has _sandbox=False. Live trading is strictly forbidden."
            )

    def _with_retry(self, operation_name: str, fn: Callable[[], T]) -> T:
        """Execute a broker API call with exponential backoff on transient failures."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn()
            except SecurityConfigurationError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    logger.error(
                        "Broker operation failed after max retries",
                        extra={"operation": operation_name, "attempts": attempt, "error": str(exc)},
                    )
                    raise
                delay = self.base_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Transient broker error; retrying with exponential backoff",
                    extra={"operation": operation_name, "attempt": attempt, "delay_sec": delay, "error": str(exc)},
                )
                self._sleep(delay)
        assert last_exc is not None
        raise last_exc

    def get_market_clock(self) -> MarketClockStatus:
        """
        Invariant #5: Anchor all market-session and scheduling checks to America/New_York
        verified against Alpaca's `get_clock()` API.
        """
        clock = self._with_retry("get_clock", lambda: self.trading_client.get_clock())
        ts_ny = clock.timestamp.astimezone(NY_TZ)
        next_open_ny = clock.next_open.astimezone(NY_TZ)
        next_close_ny = clock.next_close.astimezone(NY_TZ)
        return MarketClockStatus(
            is_open=bool(clock.is_open),
            timestamp_ny=ts_ny,
            next_open_ny=next_open_ny,
            next_close_ny=next_close_ny,
        )

    def get_positions_raw(self) -> list[Any]:
        return list(self._with_retry("get_all_positions", lambda: self.trading_client.get_all_positions()))

    def get_open_orders(self) -> list[Any]:
        if not hasattr(self.trading_client, "get_orders"):
            return []
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
        return list(self._with_retry("get_orders", lambda: self.trading_client.get_orders(filter=req)))

    def get_order_status(self, alpaca_order_id: str) -> Optional[BrokerOrderResult]:
        if not hasattr(self.trading_client, "get_order_by_id"):
            return None
        raw = self._with_retry(
            f"get_order_by_id:{alpaca_order_id}",
            lambda: self.trading_client.get_order_by_id(alpaca_order_id),
        )
        return self._normalize_order(raw)

    @staticmethod
    def _normalize_order(raw_order: Any, fallback_client_id: str = "") -> BrokerOrderResult:
        status_raw = getattr(raw_order, "status", "submitted")
        status_str = str(getattr(status_raw, "value", status_raw)).lower()
        filled_price_raw = getattr(raw_order, "filled_avg_price", None)
        filled_qty_raw = getattr(raw_order, "filled_qty", None)
        return BrokerOrderResult(
            alpaca_order_id=str(getattr(raw_order, "id", "")),
            client_order_id=str(getattr(raw_order, "client_order_id", fallback_client_id) or fallback_client_id),
            status=status_str,
            filled_avg_price=float(filled_price_raw) if filled_price_raw is not None else None,
            filled_qty=float(filled_qty_raw) if filled_qty_raw is not None else None,
        )

    def get_portfolio_snapshot(self, daily_starting_equity: Optional[float] = None) -> PortfolioSnapshot:
        """
        Fetch account equity, non-marginable buying power, options buying power, and per-ticker
        notional exposure including short-put assignment risk (`100 * strike * |qty|`) and open orders.
        """
        clock = self.get_market_clock()
        account = self._with_retry("get_account", lambda: self.trading_client.get_account())
        positions = self.get_positions_raw()
        open_orders = self.get_open_orders()

        total_equity = float(account.equity)
        last_equity = float(getattr(account, "last_equity", total_equity) or total_equity)
        non_marginable_bp = float(
            getattr(account, "non_marginable_buying_power", getattr(account, "cash", total_equity))
        )
        options_bp_raw = getattr(account, "options_buying_power", None)
        options_bp = float(options_bp_raw) if options_bp_raw is not None else None

        exposure_by_ticker: dict[str, float] = {}
        equity_shares_by_ticker: dict[str, float] = {}
        broker_reserved_csp_collateral = 0.0

        # Pass 1: equities and long options
        option_positions: list[tuple[Any, tuple[str, date, OptionRight, float]]] = []
        for pos in positions:
            symbol = str(pos.symbol).strip().upper()
            parsed_occ = parse_occ_option_symbol(symbol)
            if parsed_occ is None:
                underlying = self.extract_underlying_ticker(symbol)
                qty_val = float(getattr(pos, "qty", 0.0) or 0.0)
                mkt_val = abs(float(getattr(pos, "market_value", 0.0) or 0.0))
                equity_shares_by_ticker[underlying] = equity_shares_by_ticker.get(underlying, 0.0) + qty_val
                exposure_by_ticker[underlying] = exposure_by_ticker.get(underlying, 0.0) + mkt_val
            else:
                option_positions.append((pos, parsed_occ))

        # Pass 2: option positions (net defined-risk spreads at same expiry; value unpaired short puts at strike notional)
        option_buckets: dict[tuple[str, date], dict[str, list[list[float]]]] = {}
        for pos, (underlying, exp_dt, right, strike) in option_positions:
            qty_val = float(getattr(pos, "qty", 0.0) or 0.0)
            mkt_val = abs(float(getattr(pos, "market_value", 0.0) or 0.0))
            if qty_val == 0:
                continue
            bucket = option_buckets.setdefault(
                (underlying, exp_dt),
                {"short_put": [], "long_put": [], "short_call": [], "long_call": [], "long_mkt": []},
            )
            if right == OptionRight.PUT:
                if qty_val < 0:
                    bucket["short_put"].append([strike, abs(qty_val)])
                else:
                    bucket["long_put"].append([strike, qty_val])
                    bucket["long_mkt"].append([mkt_val])
            else:
                if qty_val < 0:
                    bucket["short_call"].append([strike, abs(qty_val)])
                else:
                    bucket["long_call"].append([strike, qty_val])
                    bucket["long_mkt"].append([mkt_val])

        for (underlying, _exp_dt), bkt in option_buckets.items():
            # Reduce short calls covered by 100-share equity lots
            covered_calls_avail = max(0.0, equity_shares_by_ticker.get(underlying, 0.0) / 100.0)
            short_calls = sorted(bkt["short_call"], key=lambda x: x[0])
            uncovered_short_calls: list[list[float]] = []
            for s_strike, s_qty in short_calls:
                cov = min(s_qty, covered_calls_avail)
                covered_calls_avail -= cov
                rem = s_qty - cov
                if rem > 1e-6:
                    uncovered_short_calls.append([s_strike, rem])

            # Pair short puts with protective lower-or-any long puts (descending strike order)
            sp_list = sorted(bkt["short_put"], key=lambda x: x[0], reverse=True)
            lp_list = sorted(bkt["long_put"], key=lambda x: x[0], reverse=True)
            put_spread_risk = 0.0
            unpaired_put_risk = 0.0
            lp_idx = 0
            for s_strike, s_qty in sp_list:
                rem = s_qty
                while rem > 1e-6 and lp_idx < len(lp_list):
                    l_strike, l_qty = lp_list[lp_idx]
                    if l_qty <= 1e-6:
                        lp_idx += 1
                        continue
                    paired = min(rem, l_qty)
                    put_spread_risk += max(0.0, s_strike - l_strike) * 100.0 * paired
                    rem -= paired
                    lp_list[lp_idx][1] -= paired
                    if lp_list[lp_idx][1] <= 1e-6:
                        lp_idx += 1
                if rem > 1e-6:
                    unpaired_risk = 100.0 * s_strike * rem
                    unpaired_put_risk += unpaired_risk
                    broker_reserved_csp_collateral += unpaired_risk

            # Pair uncovered short calls with protective higher long calls (ascending strike order)
            sc_list = sorted(uncovered_short_calls, key=lambda x: x[0])
            lc_list = sorted(bkt["long_call"], key=lambda x: x[0])
            call_spread_risk = 0.0
            unpaired_call_risk = 0.0
            lc_idx = 0
            for s_strike, s_qty in sc_list:
                rem = s_qty
                while rem > 1e-6 and lc_idx < len(lc_list):
                    l_strike, l_qty = lc_list[lc_idx]
                    if l_qty <= 1e-6:
                        lc_idx += 1
                        continue
                    paired = min(rem, l_qty)
                    call_spread_risk += max(0.0, l_strike - s_strike) * 100.0 * paired
                    rem -= paired
                    lc_list[lc_idx][1] -= paired
                    if lc_list[lc_idx][1] <= 1e-6:
                        lc_idx += 1
                if rem > 1e-6:
                    unpaired_call_risk += 100.0 * s_strike * rem

            # At expiration, worst case across put vs call wings cannot occur on both sides simultaneously
            condor_wing_risk = max(put_spread_risk, call_spread_risk)
            unpaired_long_val = sum(m[0] for m in bkt["long_mkt"]) if (unpaired_put_risk + unpaired_call_risk == 0 and condor_wing_risk == 0) else 0.0
            broker_reserved_csp_collateral += condor_wing_risk
            exposure_by_ticker[underlying] = (
                exposure_by_ticker.get(underlying, 0.0)
                + condor_wing_risk
                + unpaired_put_risk
                + unpaired_call_risk
                + unpaired_long_val
            )

        # Pass 3: pending open orders (open BUY equity, open SELL put orders, or open MLEG orders)
        for order in open_orders:
            qty_total = float(getattr(order, "qty", 0.0) or 0.0)
            filled = float(getattr(order, "filled_qty", 0.0) or 0.0)
            remaining = max(0.0, qty_total - filled)
            if remaining <= 0:
                continue

            order_legs = getattr(order, "legs", None)
            if order_legs:
                leg_short_puts: list[float] = []
                leg_long_puts: list[float] = []
                leg_short_calls: list[float] = []
                leg_long_calls: list[float] = []
                leg_underlying: Optional[str] = None
                for leg in order_legs:
                    lsym = str(getattr(leg, "symbol", "")).strip().upper()
                    parsed_leg = parse_occ_option_symbol(lsym)
                    if parsed_leg is None:
                        continue
                    l_und, _l_exp, l_right, l_strike = parsed_leg
                    leg_underlying = leg_underlying or l_und
                    l_side_raw = getattr(leg, "side", "")
                    l_side = str(getattr(l_side_raw, "value", l_side_raw)).lower()
                    l_intent_raw = getattr(leg, "position_intent", "")
                    l_intent = str(getattr(l_intent_raw, "value", l_intent_raw)).lower()
                    if "sell_to_open" in l_intent or (not l_intent and l_side == "sell"):
                        if l_right == OptionRight.PUT:
                            leg_short_puts.append(l_strike)
                        else:
                            leg_short_calls.append(l_strike)
                    elif "buy_to_open" in l_intent or (not l_intent and l_side == "buy"):
                        if l_right == OptionRight.PUT:
                            leg_long_puts.append(l_strike)
                        else:
                            leg_long_calls.append(l_strike)
                if leg_underlying:
                    p_risk = (
                        max(0.0, max(leg_short_puts) - min(leg_long_puts)) * 100.0 * remaining
                        if (leg_short_puts and leg_long_puts)
                        else sum(100.0 * s * remaining for s in leg_short_puts)
                    )
                    c_risk = (
                        max(0.0, max(leg_long_calls) - min(leg_short_calls)) * 100.0 * remaining
                        if (leg_short_calls and leg_long_calls)
                        else sum(100.0 * s * remaining for s in leg_short_calls)
                    )
                    wing_risk = max(p_risk, c_risk)
                    exposure_by_ticker[leg_underlying] = exposure_by_ticker.get(leg_underlying, 0.0) + wing_risk
                    broker_reserved_csp_collateral += wing_risk
                continue

            sym = str(getattr(order, "symbol", "") or "").strip().upper()
            side_raw = getattr(order, "side", "")
            side_str = str(getattr(side_raw, "value", side_raw)).lower()

            parsed_occ = parse_occ_option_symbol(sym)
            if parsed_occ is not None:
                underlying, _exp, right, strike = parsed_occ
                if side_str == "sell" and right == OptionRight.PUT:
                    notional = 100.0 * strike * remaining
                    exposure_by_ticker[underlying] = exposure_by_ticker.get(underlying, 0.0) + notional
                    broker_reserved_csp_collateral += notional
            elif sym and side_str == "buy":
                underlying = self.extract_underlying_ticker(sym)
                limit_p = getattr(order, "limit_price", None)
                notional_p = getattr(order, "notional", None)
                if limit_p is not None:
                    est_notional = remaining * float(limit_p)
                elif notional_p is not None:
                    est_notional = float(notional_p)
                else:
                    est_notional = 0.0
                exposure_by_ticker[underlying] = exposure_by_ticker.get(underlying, 0.0) + est_notional

        start_eq = daily_starting_equity if daily_starting_equity is not None else last_equity
        peak_eq = max(start_eq, total_equity)

        return PortfolioSnapshot(
            as_of_ny=clock.timestamp_ny,
            total_equity=total_equity,
            non_marginable_buying_power=non_marginable_bp,
            daily_starting_equity=start_eq,
            intraday_peak_equity=peak_eq,
            options_buying_power=options_bp,
            broker_reserved_csp_collateral=broker_reserved_csp_collateral,
            ticker_notional_exposure=exposure_by_ticker,
        )

    @staticmethod
    def extract_underlying_ticker(symbol: str) -> str:
        """Extract root equity ticker from either an equity symbol or an OCC option symbol."""
        parsed = parse_occ_option_symbol(symbol)
        if parsed is not None:
            return parsed[0]
        clean = symbol.strip().upper()
        if len(clean) <= 6 and all(ch.isalpha() or ch == "." for ch in clean):
            return clean
        prefix = []
        for ch in clean:
            if ch.isalpha() or ch == ".":
                prefix.append(ch)
            else:
                break
        return "".join(prefix) or clean

    def cancel_all_open_orders(self) -> Any:
        """Global kill-switch action required by TRD-POL-006 (`TradingClient.cancel_orders()`)."""
        logger.warning("Executing global cancel_orders() on Alpaca Paper account.")
        return self._with_retry("cancel_orders", lambda: self.trading_client.cancel_orders())

    def submit_order_intent(
        self,
        intent: OrderIntent,
        client_order_id: str,
        require_market_open: bool = True,
    ) -> BrokerOrderResult:
        """
        Translate a RiskGateway-approved `OrderIntent` into an Alpaca order request and submit it.
        Note: For `BUY_EQUITY`, submits the entry order cleanly first; once filled, the orchestrator
        submits a true broker-native `TrailingStopOrderRequest` (`trail_percent`) so Alpaca handles
        real trailing stops rather than fixed stops.
        """
        self._verify_trading_client_is_paper(self.trading_client)

        if require_market_open:
            clock = self.get_market_clock()
            if not clock.is_open:
                raise RuntimeError(
                    f"Order submission blocked: US market is closed (as_of_ny={clock.timestamp_ny.isoformat()})."
                )

        order_request = self.build_order_request(intent=intent, client_order_id=client_order_id)
        raw_order = self._with_retry(
            f"submit_order:{intent.ticker}:{intent.action.value}",
            lambda: self.trading_client.submit_order(order_data=order_request),
        )
        return self._normalize_order(raw_order, fallback_client_id=client_order_id)

    def submit_trailing_stop(
        self,
        ticker: str,
        qty: int,
        trailing_stop_pct: float,
        client_order_id: str,
    ) -> BrokerOrderResult:
        """Submit a real GTC broker-native trailing-stop sell order for a filled equity position."""
        self._verify_trading_client_is_paper(self.trading_client)
        trail_percent = round(trailing_stop_pct * 100.0, 2)
        req = TrailingStopOrderRequest(
            symbol=ticker.strip().upper(),
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            trail_percent=trail_percent,
            client_order_id=client_order_id,
        )
        raw_order = self._with_retry(
            f"submit_trailing_stop:{ticker}",
            lambda: self.trading_client.submit_order(order_data=req),
        )
        return self._normalize_order(raw_order, fallback_client_id=client_order_id)

    @staticmethod
    def build_order_request(intent: OrderIntent, client_order_id: str) -> Any:
        """Construct the exact alpaca-py OrderRequest object for `intent`."""
        if intent.action in (TradeAction.BUY_EQUITY, TradeAction.ADD_DIP_TRANCHE):
            if intent.limit_price is not None:
                return LimitOrderRequest(
                    symbol=intent.ticker,
                    qty=intent.qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    limit_price=round(intent.limit_price, 2),
                    client_order_id=client_order_id,
                )
            if intent.action == TradeAction.ADD_DIP_TRANCHE:
                raise ValueError("ADD_DIP_TRANCHE requires an explicit limit_price.")
            return MarketOrderRequest(
                symbol=intent.ticker,
                qty=intent.qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                client_order_id=client_order_id,
            )

        if intent.action in (TradeAction.SELL_CSP, TradeAction.SELL_CC):
            if not intent.option_symbol or intent.limit_price is None:
                raise ValueError(f"{intent.action.value} requires option_symbol and limit_price.")
            return LimitOrderRequest(
                symbol=intent.option_symbol,
                qty=intent.qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                limit_price=round(intent.limit_price, 2),
                client_order_id=client_order_id,
            )

        if intent.action in (TradeAction.BUY_TO_CLOSE_CSP, TradeAction.BUY_TO_CLOSE_CC):
            if not intent.option_symbol or intent.limit_price is None:
                raise ValueError(f"{intent.action.value} requires option_symbol and limit_price.")
            return LimitOrderRequest(
                symbol=intent.option_symbol,
                qty=intent.qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=round(intent.limit_price, 2),
                client_order_id=client_order_id,
            )

        if intent.action in (
            TradeAction.OPEN_BULL_PUT_SPREAD,
            TradeAction.OPEN_IRON_CONDOR,
            TradeAction.CLOSE_BULL_PUT_SPREAD,
            TradeAction.CLOSE_IRON_CONDOR,
        ):
            if not (2 <= len(intent.legs) <= 4) or intent.limit_price is None or intent.limit_price <= 0:
                raise ValueError(f"{intent.action.value} requires 2-4 option legs and positive limit_price.")
            is_open_credit = intent.action in (TradeAction.OPEN_BULL_PUT_SPREAD, TradeAction.OPEN_IRON_CONDOR)
            # Alpaca MLEG API convention: negative limit_price = net credit received; positive = net debit paid.
            signed_limit = -abs(round(intent.limit_price, 2)) if is_open_credit else abs(round(intent.limit_price, 2))
            alpaca_legs = [
                OptionLegRequest(
                    symbol=leg.option_symbol,
                    ratio_qty=leg.ratio_qty,
                    side=OrderSide.SELL if leg.side == "SELL" else OrderSide.BUY,
                    position_intent=PositionIntent(leg.position_intent),
                )
                for leg in intent.legs
            ]
            return LimitOrderRequest(
                order_class=OrderClass.MLEG,
                qty=intent.qty,
                time_in_force=TimeInForce.DAY,
                limit_price=signed_limit,
                legs=alpaca_legs,
                client_order_id=client_order_id,
            )

        raise ValueError(f"Unsupported TradeAction for broker order submission: {intent.action}")

    # =========================================================================
    # Market Data Adapters (Bars, Quotes, News, Options Chains)
    # =========================================================================

    def get_stock_daily_bars(self, ticker: str, days: int = 60) -> list[Any]:
        """Fetch historical daily bars (with 16-minute delay on SIP feed for free plans)."""
        now_utc = datetime.now(timezone.utc)
        feed = DataFeed(self.settings.ALPACA_BARS_FEED)
        end_dt = now_utc - timedelta(minutes=16) if feed == DataFeed.SIP else now_utc
        req = StockBarsRequest(
            symbol_or_symbols=[ticker.strip().upper()],
            timeframe=TimeFrame.Day,
            start=now_utc - timedelta(days=days),
            end=end_dt,
            feed=feed,
        )
        barset = self._with_retry(
            f"get_stock_bars:{ticker}",
            lambda: self.stock_data_client.get_stock_bars(req),
        )
        data = getattr(barset, "data", {}) or {}
        return list(data.get(ticker.strip().upper(), []))

    def get_stock_quote_snapshot(self, ticker: str) -> tuple[float, float, Optional[float]]:
        """Return (bid, ask, last_trade_price) for `ticker` using configured quote feed."""
        sym = ticker.strip().upper()
        feed = DataFeed(self.settings.ALPACA_QUOTE_FEED)
        bid = 0.0
        ask = 0.0
        last_trade: Optional[float] = None

        quotes = self._with_retry(
            f"get_stock_latest_quote:{sym}",
            lambda: self.stock_data_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=[sym], feed=feed)
            ),
        )
        q = quotes.get(sym) if isinstance(quotes, dict) else None
        if q is not None:
            bid = float(getattr(q, "bid_price", 0.0) or 0.0)
            ask = float(getattr(q, "ask_price", 0.0) or 0.0)

        if hasattr(self.stock_data_client, "get_stock_latest_trade"):
            try:
                trades = self._with_retry(
                    f"get_stock_latest_trade:{sym}",
                    lambda: self.stock_data_client.get_stock_latest_trade(
                        StockLatestTradeRequest(symbol_or_symbols=[sym], feed=feed)
                    ),
                )
                tr = trades.get(sym) if isinstance(trades, dict) else None
                if tr is not None:
                    last_trade = float(getattr(tr, "price", 0.0) or 0.0) or None
            except Exception:
                pass

        return bid, ask, last_trade

    def get_recent_news_summary(self, ticker: str, limit: int = 5) -> str:
        """Fetch recent headlines/summaries from Alpaca News API for untrusted catalyst synthesis."""
        if not hasattr(self.news_client, "get_news"):
            return ""
        sym = ticker.strip().upper()
        try:
            news_set = self._with_retry(
                f"get_news:{sym}",
                lambda: self.news_client.get_news(NewsRequest(symbols=sym, limit=limit)),
            )
            items = (getattr(news_set, "data", {}) or {}).get("news", [])
            snippets: list[str] = []
            for item in items[:limit]:
                headline = str(getattr(item, "headline", "") or "").strip()
                summary = str(getattr(item, "summary", "") or "").strip()
                if headline:
                    snippets.append(f"{headline}: {summary}" if summary else headline)
            return " | ".join(snippets)
        except Exception as exc:
            logger.warning(f"News lookup unavailable for {sym}: {exc}", extra={"ticker": sym})
            return ""

    def get_option_chain_quotes(
        self,
        ticker: str,
        right: OptionRight,
        min_dte: int,
        max_dte: int,
        as_of_date: date,
    ) -> list[OptionContractQuote]:
        """Fetch option chain snapshots from Alpaca and normalize into `OptionContractQuote` objects."""
        sym = ticker.strip().upper()
        contract_type = ContractType.PUT if right == OptionRight.PUT else ContractType.CALL
        req = OptionChainRequest(
            underlying_symbol=sym,
            type=contract_type,
            expiration_date_gte=as_of_date + timedelta(days=min_dte),
            expiration_date_lte=as_of_date + timedelta(days=max_dte),
        )
        chain = self._with_retry(
            f"get_option_chain:{sym}:{right.value}",
            lambda: self.option_data_client.get_option_chain(req),
        )
        results: list[OptionContractQuote] = []
        for occ_sym, snap in (chain or {}).items():
            parsed = parse_occ_option_symbol(str(occ_sym))
            if parsed is None:
                continue
            underlying, exp_date, parsed_right, strike = parsed
            lq = getattr(snap, "latest_quote", None)
            if lq is None:
                continue
            bid = float(getattr(lq, "bid_price", 0.0) or 0.0)
            ask = float(getattr(lq, "ask_price", 0.0) or 0.0)
            if bid <= 0.0 or ask <= 0.0 or ask < bid:
                continue
            greeks = getattr(snap, "greeks", None)
            delta = float(greeks.delta) if getattr(greeks, "delta", None) is not None else None
            theta = float(greeks.theta) if getattr(greeks, "theta", None) is not None else None
            iv_val = getattr(snap, "implied_volatility", None)
            iv = float(iv_val) if iv_val is not None else None
            results.append(
                OptionContractQuote(
                    option_symbol=str(occ_sym),
                    underlying_ticker=underlying,
                    right=parsed_right,
                    strike_price=strike,
                    expiration_date=exp_date,
                    bid=bid,
                    ask=ask,
                    delta=delta,
                    theta=theta,
                    implied_volatility=iv,
                )
            )
        return results

    def get_option_quote(self, option_symbol: str) -> Optional[OptionContractQuote]:
        """Fetch latest quote for a single OCC option symbol."""
        parsed = parse_occ_option_symbol(option_symbol)
        if parsed is None:
            return None
        underlying, exp_date, right, strike = parsed
        sym = option_symbol.strip().upper()
        quotes = self._with_retry(
            f"get_option_latest_quote:{sym}",
            lambda: self.option_data_client.get_option_latest_quote(
                OptionLatestQuoteRequest(symbol_or_symbols=[sym])
            ),
        )
        lq = quotes.get(sym) if isinstance(quotes, dict) else None
        if lq is None:
            return None
        bid = float(getattr(lq, "bid_price", 0.0) or 0.0)
        ask = float(getattr(lq, "ask_price", 0.0) or 0.0)
        if bid <= 0.0 or ask <= 0.0 or ask < bid:
            return None
        return OptionContractQuote(
            option_symbol=sym,
            underlying_ticker=underlying,
            right=right,
            strike_price=strike,
            expiration_date=exp_date,
            bid=bid,
            ask=ask,
        )
