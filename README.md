# `alpaca-gemini-trader`

A modular, fail-closed automated equities and options **Paper Trading** prototype in Python (`3.11+`) that cleanly separates **probabilistic AI analysis** (`google-genai`) from **deterministic risk enforcement and order execution** (`alpaca-py` + SQLite).

---

## Core Security & Runtime Invariants

1. **Zero-Secret-Leakage Policy:** Credentials (`APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, `GEMINI_API_KEY`) are loaded exclusively from a local `.env` file (`0600`) via `pydantic-settings` with `repr=False`. Never paste API keys into chat logs or commit `.env`.
2. **Fail-Closed Paper Trading Lock:** `AlpacaBroker` hardcodes `paper=True` on `TradingClient` and validates `TRADING_MODE == "PAPER"`, `ALPACA_PAPER_BASE_URL == "https://paper-api.alpaca.markets"`, and underlying client `BaseURL.TRADING_PAPER` / `_sandbox`. Any attempt to set `TRADING_MODE != "PAPER"` or target `api.alpaca.markets` raises a fatal `SecurityConfigurationError`.
3. **Strict Execution Boundary:** `GeminiAnalyst` (`src/intelligence/analyst.py`) holds zero brokerage credentials and outputs a strict `TradeProposal` Pydantic schema (`extra="forbid"`) containing no share quantities, option strikes, or stop prices. Deterministic Python strategies compute sizing and submit `OrderIntent` objects into `RiskGateway`. Missing `GEMINI_API_KEY` or API failures fail closed (`TradeAction.REJECT`) unless `ANALYST_MODE=RULES` is explicitly selected for deterministic no-key operation.
4. **Prompt-Injection Quarantine:** External text (congressional disclosures, 13F filings, news headlines) is stripped of instruction-override patterns and wrapped in `<UNTRUSTED_EXTERNAL_DATA>...</UNTRUSTED_EXTERNAL_DATA>` delimiters; detected injections short-circuit deterministically before calling the LLM.
5. **Timezone & Market Clock Determinism:** All timestamps, date buckets, entry blackouts, and DTE calculations are anchored to `zoneinfo.ZoneInfo("America/New_York")` and verified against Alpaca's `get_clock()` API.

---

## Architecture Highlights

- **Two-Step Trailing Stop Execution:** Equity entries (`BUY_EQUITY`) submit a standalone market order, track it in `tracked_orders`, and once filled on reconciliation submit a genuine broker-native GTC `TrailingStopOrderRequest(trail_percent=...)`.
- **Wheel Strategy State Machine & Reconciliation:** Sells ~10% OTM Cash-Secured Puts (`14–30` DTE) with a 25% max bid-ask spread haircut check (`Decimal` penny rounding), monitors open contracts for 50% early profit-taking (`BUY_TO_CLOSE_CSP` / `BUY_TO_CLOSE_CC` exempt from rate cooldown), reconciles option fills, and automatically transitions `CSP_OPEN -> ASSIGNED_SHARES` / `IDLE` and `CC_OPEN -> IDLE` / `ASSIGNED_SHARES` from live Alpaca positions.
- **Bounded Analyst Caching & Offline `RULES` Mode:** `ANALYST_MODE=GEMINI` (default `gemini-3.8-flash`) caches proposals per ticker/strategy/date for `ANALYST_CACHE_MINUTES` (default `15`). `ANALYST_MODE=RULES` provides a deterministic rules-based analyst using ATR, RSI, 20d return/volatility, IV rank, and liquidity without external LLM calls.

---

## Mandatory Risk Policies (`src/execution/risk_gateway.py`)

Every `OrderIntent` is synchronously inspected by `RiskGateway` prior to `TradingClient.submit_order()`, and state (`order_cooldowns`, `dip_ladders`, `tracked_orders`) is committed **only after** confirmed broker acceptance:

| Policy ID | Name | Deterministic Rule |
|---|---|---|
| `TRD-POL-001` | **Single-Ticker Concentration Cap** | Rejects any entry pushing total notional exposure (long equity + `100 * strike * |qty|` short put assignment notional + pending open orders) above `MAX_POSITION_PCT` (default `5%` of total equity). |
| `TRD-POL-002` | **Cash-Secured Put Collateral Lock** | Verifies effective non-marginable / options buying power covers `100 * strike_price * qty` before selling any CSP, avoiding double-counting collateral already reserved by Alpaca. |
| `TRD-POL-003` | **Covered Call Cost-Basis Floor** | Blocks any Covered Call whose strike price is below the recorded net cost basis (`gross_cost_basis - cumulative_premiums_collected`) of the assigned 100-share lot. |
| `TRD-POL-004` | **Bounded Dip-Laddering Cap** | Limits scaled dip buys to at most `2` add-on tranches per position, subject to `TRD-POL-001`, and pauses dip-buying if 20-day realized volatility exceeds `MAX_REALIZED_VOL_20D`. |
| `TRD-POL-005` | **Idempotency & Rate Cooldown** | Computes deterministic `client_order_id` hashes (`strategy_id + ticker + action + date_bucket`) and enforces a 15-minute per-ticker entry cooldown window (closing exits exempt from cooldown while preserving idempotency). |
| `TRD-POL-006` | **Portfolio Intraday Circuit Breaker** | Measures daily realized + unrealized loss against `daily_starting_equity` (while also persisting `intraday_peak_equity`). If drawdown exceeds `MAX_DAILY_DRAWDOWN_PCT` (default `2%`), immediately calls `cancel_orders()`, locks `system_state` to `HALTED`, and blocks all execution until manually unlocked via CLI. |

---

## Local Setup & Safe Credential Configuration

### 1. Create Virtual Environment & Install Dependencies

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

### 2. Interactive Credential Setup & Connection Verification

Use the built-in hidden prompt setup (`getpass`, writes `.env` with `0600` permissions) or copy `.env.example` to `.env`:

```bash
.venv/bin/python -m src.cli setup
.venv/bin/python -m src.cli verify
```

Notes on Market Data & Analyst Keys:
- **Alpaca Paper Trading:** Includes equities + level-3 options paper trading, SIP historical bars (`end <= now - 16 min` on free paper tier), IEX real-time quotes (`ALPACA_QUOTE_FEED=iex`), and Benzinga news.
- **Gemini Analyst:** Set `GEMINI_API_KEY` in `.env` for `ANALYST_MODE=GEMINI` (`gemini-3.8-flash`), or set `ANALYST_MODE=RULES` in `.env` to run deterministic strategy analysis without a Gemini key.

---

## Verification & Operations

### 1. Run Deterministic Test Suite (Zero Network Calls Required)

```bash
.venv/bin/pytest -v
```

### 2. Initialize Database & Check System Status

```bash
.venv/bin/python -m src.cli init-db
.venv/bin/python -m src.cli status
```

### 3. Dry-Run & Live Paper Orchestrator

```bash
# Single cycle with live Paper market data, zero live order submissions:
.venv/bin/python -m src.main --once --dry-run

# Single live paper cycle (respects market clock & entry blackout window):
.venv/bin/python -m src.main --once

# Continuous paper loop:
.venv/bin/python -m src.main --interval 60
```

### Multi-Leg Defined-Risk Credit Spreads (`OrderClass.MLEG`) & Zero-Cost `RULES` Mode

- **Credit Spreads (`StrategyType.CREDIT_SPREAD`)**: Targets `~$30/week` (`WEEKLY_CREDIT_TARGET_USD`) on `CREDIT_SPREAD_WATCHLIST` (default `GOOGL`) using defined-risk multi-leg `OrderClass.MLEG` orders:
  - **Iron Condor (`OPEN_IRON_CONDOR`)**: 4-leg defined-risk neutral spread (~8–12 delta short put/call wings) requiring positive net credit after 25% spread haircut and non-overlapping wing collateral caps.
  - **Deeper Bull Put Credit Spread (`OPEN_BULL_PUT_SPREAD`)**: 2-leg defined-risk deeper-OTM put credit spread (~4–8 delta, strictly below any open Iron Condor long put strike) for stacked conservative income.
  - **Automated Exits (`CLOSE_IRON_CONDOR` / `CLOSE_BULL_PUT_SPREAD`)**: Takes profit at 50% of entry credit (`SPREAD_PROFIT_TAKE_RATIO=0.50`), exits on 2.5x credit stop-loss (`SPREAD_STOP_LOSS_MULTIPLE=2.5`), or closes before expiry (< 1 DTE).
  - **Net Exposure Accounting (`TRD-POL-001` / `TRD-POL-002`)**: Pairs multi-leg option wings per `(underlying, expiration)` so defined-risk collateral is bounded by `max(put_width, call_width) * 100 * qty - net_credit`, enforcing the strict 5% single-ticker equity cap (`MAX_POSITION_PCT=0.05`) without naked short exposure.
- **Zero-Cost Testing (`ANALYST_MODE=RULES`)**: Set `ANALYST_MODE=RULES` in `.env` to run deterministic indicator & liquidity gating with **zero calls** (and `$0.00` charges) to `GEMINI_API_KEY`. When you add your own Google AI Studio Free-Tier `GEMINI_API_KEY`, switch back to `ANALYST_MODE=GEMINI`.

### 4. Manually Clear a `HALTED` Circuit Breaker (`TRD-POL-006`)

```bash
.venv/bin/python -m src.cli unlock
```

### 5. Launch Read-Only Streamlit Observability Dashboard

```bash
.venv/bin/streamlit run src/dashboard.py
```
