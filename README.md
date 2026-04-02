# Week 4 Capstone: BTC Volatility Trading Agent with MCP

**Student**: Thomas Jensen, Cohort 1  
**Course**: The AI Engineer — Core Track  
**Repository**: https://github.com/thomasplato/TheAIEngineer_Week4

---

## Quick Start

### Prerequisites

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Running the System

**Terminal 1 — Start the MCP server:**
```bash
python trading_server.py
```

**Terminal 2 — Run the orchestrator (50 cycles):**
```bash
python orchestrator.py --steps 50
```

**Terminal 2 — Or replay a previous run (no server needed):**
```bash
python orchestrator.py --replay memory/trace.jsonl
```

### Running Tests

```bash
# With server running in Terminal 1:
python test_integration.py
```

### Output Files

All output goes to the `memory/` directory:

| File | Description |
|------|-------------|
| `trace.jsonl` | Cycle-by-cycle log with correlation IDs (for `--replay`) |
| `summary.json` | Final performance metrics and run metadata |
| `state.json` | Current portfolio state |
| `trade_journal.json` | Complete trade history |
| `prediction_log.json` | All model predictions |

---

## Project Overview

This project implements an **AI-powered Bitcoin trading agent** that combines a transformer model for 24-hour forward volatility prediction with an MCP architecture for clean separation of concerns and a deterministic signal-based strategy for safe trade execution.

The system predicts Bitcoin volatility using hourly data and uses these predictions to generate trading signals. While trading Bitcoin options would have been ideal (leveraging volatility directly), Alpaca's API doesn't support BTC options, so the volatility predictions are instead used as trading signals for spot Bitcoin. In Week 3 I built a Transformer model to predict SPY volatility — but I like that Bitcoin is trading 24/7.

### Pipeline Architecture

The project follows a four-phase development pipeline where each phase builds upon the previous. The first two phases focus on model development: raw Bitcoin data is fetched from Alpaca, engineered into 16 features, and used to train a decoder-only transformer that predicts 24-hour forward volatility. The trained model achieves meaningful predictions that outperform naive baselines and is saved as `volatility_transformer.pt`.

In the third phase, the trained model is tested on held-out data to validate that volatility predictions can generate profitable trading signals when combined with momentum indicators and realistic transaction costs. This backtesting phase confirms the model's practical utility before moving to the MCP implementation.

The fourth phase serves as a critical bridge between the transformer model and the MCP trading agent. The notebook `btc_volatility_04_mcp_agent_data.ipynb` runs the trained transformer across the entire dataset (train, validation, and test) to generate a complete historical record. For each hourly observation, it computes the volatility surprise (the ratio of predicted to realized volatility), which indicates when the market is calmer or more volatile than expected. It also generates LONG/SHORT trading signals based on surprise thresholds, calculates 6-hour momentum indicators for trend confirmation, and estimates bid/ask spreads for realistic trade simulation. The output, `btc_agent_replay.csv`, becomes the data source for the MCP trading agent.

Finally, in Week 4, the MCP architecture brings everything together through two Python modules: `trading_server.py` and `orchestrator.py`. The MCP design separates data and tool provision (the server) from decision-making and execution (the orchestrator). The agent "replays" the historical data, with a deterministic finite-state machine selecting actions at each time step based on the transformer's volatility surprise, momentum signals, and current portfolio state. The orchestrator follows the Observe → Plan → Act → Learn loop prescribed by the MCP framework: observing market conditions via tool calls and resource reads, planning via signal-based rules, acting via the execute_trade tool, and learning by persisting structured cycle records with correlation IDs. This design ensures safety through bounded actions, budget enforcement, and a complete audit trail — while the deterministic planner could be replaced by an LLM-based strategy selector (as explored in an earlier version using GPT-4o Mini) without changing the MCP architecture.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│            MCP SERVER  (trading_server.py)            │
│            Transport: WebSocket JSON-RPC 2.0          │
│                                                       │
│  Tools (5):                                           │
│  ├─ get_market_data     → Price, bid/ask, vol        │
│  ├─ run_prediction      → Transformer predictions     │
│  ├─ execute_trade       → Paper trading at bid/ask    │
│  ├─ get_performance     → P&L, win rate, budgets     │
│  └─ get_replay_status   → Progress through data      │
│                                                       │
│  Resources (4):                                       │
│  ├─ memory://trades/latest    → Last trade            │
│  ├─ memory://performance      → Portfolio metrics     │
│  ├─ memory://state            → Cash + BTC position   │
│  └─ memory://replay/progress  → Replay progress      │
│                                                       │
│  Data: btc_agent_replay.csv (6,598 hourly rows)      │
│  Budgets: 500 trades, 50,000 tool calls per session   │
└────────────────────┬─────────────────────────────────┘
                     │  WebSocket (ws://127.0.0.1:8765)
                     │
┌────────────────────▼─────────────────────────────────┐
│          ORCHESTRATOR  (orchestrator.py)               │
│                                                       │
│  Cycle: Observe → Plan → Act → Learn                  │
│                                                       │
│  1. OBSERVE: call tools + read resources              │
│  2. PLAN:   deterministic FSM (signal-based)          │
│  3. ACT:    execute_trade via MCP tool call           │
│  4. LEARN:  record cycle with correlation ID          │
│                                                       │
│  Features:                                            │
│  ├─ UUID4 correlation IDs per cycle                   │
│  ├─ JSONL structured trace logging                    │
│  ├─ --replay flag for offline playback                │
│  └─ Budget tracking in every cycle record             │
└───────────────────────────────────────────────────────┘
```

---

## Sample Run: 500-Cycle Backtest Results

The following output was produced by running `python orchestrator.py --steps 500`:

```
======================================================================
ORCHESTRATOR: Observe → Plan → Act → Learn
======================================================================
[LOOP] Historical data: 6,598 rows
[LOOP] Max cycles: 500
[LOOP] Trace log: memory/trace.jsonl

  Cycle    50 | 2025-05-05 03:00:00 | $ 94,055.45 |   SHORT → HOLD | P&L: $    -95.62
  Cycle   100 | 2025-05-07 05:00:00 | $ 96,370.36 | NEUTRAL → HOLD | P&L: $    +40.23
  Cycle   150 | 2025-05-09 07:00:00 | $103,670.30 | NEUTRAL → HOLD | P&L: $   +770.22
  Cycle   200 | 2025-05-11 09:00:00 | $104,368.11 |    LONG → HOLD | P&L: $   +795.23
  Cycle   250 | 2025-05-13 11:00:00 | $103,704.32 | NEUTRAL → HOLD | P&L: $   +617.85
  Cycle   300 | 2025-05-15 13:00:00 | $102,138.32 |    LONG → HOLD | P&L: $   +432.57
  Cycle   350 | 2025-05-17 15:00:00 | $102,992.53 |    LONG →  BUY | P&L: $   +244.10
  Cycle   400 | 2025-05-19 17:00:00 | $105,427.80 | NEUTRAL → HOLD | P&L: $    +62.23
  Cycle   450 | 2025-05-21 19:00:00 | $108,680.30 | NEUTRAL → HOLD | P&L: $   +311.59

======================================================================
RUN COMPLETE — MAX_CYCLES
======================================================================

  Performance:
    Initial Capital:  $100,000.00
    Final Value:      $100,281.20
    P&L:              $+281.20 (+0.28%)

  Actions: 29 buys, 29 sells, 442 holds
  Peak P&L:  $+845.60
  Worst P&L: $-175.50

  Execution:
    Cycles: 500
    Time:   7.9s (0.1 min)
    Rate:   63 cycles/sec

  Date range: 2025-05-03 02:00:00 → 2025-05-23 21:00:00
======================================================================
```

### Summary (memory/summary.json)

```json
{
  "run_metadata": {
    "end_reason": "max_cycles",
    "total_cycles": 500,
    "elapsed_seconds": 7.9,
    "cycles_per_second": 63.4,
    "trace_file": "memory/trace.jsonl"
  },
  "performance": {
    "initial_capital": 100000.0,
    "final_value": 100281.2,
    "pnl_usd": 281.2,
    "pnl_pct": 0.2812,
    "final_position_btc": 0.0
  },
  "signal_distribution": {
    "LONG": 155,
    "SHORT": 128,
    "NEUTRAL": 217
  },
  "action_distribution": {
    "BUY": 29,
    "SELL": 29,
    "HOLD": 442
  },
  "date_range": {
    "start": "2025-05-03 02:00:00",
    "end": "2025-05-23 21:00:00"
  }
}
```

### Sample Trace Record (memory/trace.jsonl)

Each line in the trace file is one complete cycle with its correlation ID. Here is one example (formatted for readability):

```json
{
  "correlation_id": "bc9f83c1-23e9-4b1b-8f93-c058020d8603",
  "cycle_number": 1,
  "timestamp": "2026-04-02T08:47:54.389851+00:00",
  "datetime_market": "2025-05-03 02:00:00",
  "price": 96562.65,
  "signal": "NEUTRAL",
  "vol_surprise": 0.6336,
  "momentum": -0.0053,
  "planned_action": "HOLD",
  "plan_reason": "NEUTRAL signal — waiting",
  "executed_action": "HOLD",
  "quantity": 0.0,
  "position_after": 0.0,
  "total_value": 100000.0,
  "pnl_usd": 0.0,
  "cycle_latency_ms": 5.72,
  "budget_trades_remaining": 500,
  "budget_calls_remaining": 49995
}
```

---

## Integration Test Results

Running `python test_integration.py` with the server active produces:

```
======================================================================
INTEGRATION TESTS — MCP Trading Agent
======================================================================

--- Protocol Tests ---
  ✓ initialize handshake — version=2024-11-05
  ✓ tools/list — 5 tools present
  ✓ resources/list — 4+ resources

--- Tool Tests ---
  ✓ get_market_data — returns price/bid/ask — price=$107,842.63
  ✓ run_prediction — returns signal + vol_surprise — signal=NEUTRAL, vol=0.782
  ✓ execute_trade HOLD — succeeds
  ✓ execute_trade BUY — succeeds — position=0.01
  ✓ get_performance — returns P&L + budget — P&L=$+280.63

--- Resource Tests ---
  ✓ resource read memory://state — returned 1 content block(s)
  ✓ resource read memory://performance — returned 1 content block(s)
  ✓ resource read memory://trades/latest — returned 1 content block(s)
  ✓ resource read memory://replay/progress — returned 1 content block(s)

--- Integration Tests ---
  ✓ full cycle 1 — complete — action=HOLD, P&L=$+275.35
  ✓ full cycle 2 — complete — action=HOLD, P&L=$+276.55

--- Replay Tests ---
  ✓ replay mode — trace file exists — 500 cycle(s) recorded
  ✓ replay mode — trace records valid — each record has correlation_id + cycle_number

======================================================================
RESULTS: 16/16 tests passed
All tests passed!
======================================================================
```

---

## MCP Protocol Compliance

This project follows the MCP specification as described in HandoutWeek4:

### Protocol (Section 2.1)
- **Transport**: WebSocket at `ws://127.0.0.1:8765`
- **Format**: JSON-RPC 2.0 with `jsonrpc`, `method`, `params`, `id` fields
- **Handshake**: `initialize` → capabilities response with tools + resources
- **Tool calls**: `tools/call` with name + arguments → structured result
- **Resource reads**: `resources/read` with URI → data without side effects

### Tool Contracts (Section 2.2)
Each tool declares: purpose, JSON input schema, output format, and side effects.
See `tool_registry.json` for the complete registry.

### Telemetry (Section 2.3)
- **Correlation IDs**: Every cycle gets a UUID4, logged in trace.jsonl
- **Latency tracking**: Each tool call measures execution time in milliseconds
- **Budget counters**: Trades and tool calls tracked per session
- **Replay traces**: JSONL log enables deterministic replay via `--replay`

### Memory + Resources (Section 5.3)
- State persisted as JSON files in `memory/` directory
- Exposed via 4 resource URIs for the client's Observe phase
- Concise summaries (3–4 fields per resource) to stay within token budgets

### Orchestrator (Section 5.4)
- Deterministic FSM planner (signal → action mapping)
- Budget enforcement checked every cycle
- Clean exit on end-of-data, max-cycles, or budget exhaustion

---

## Development Pipeline (Weeks 1–4)

### Phase 1: Data Pipeline
**File**: `btc_volatility_01_data_pipeline.ipynb` (Colab)
- Alpaca API: hourly BTC/USD data (2021–2026)
- 16 engineered features: log returns, volatility measures, volume, cyclical time
- Output: train/val/test CSV splits (~44,000 samples)

### Phase 2: Model Training
**File**: `btc_volatility_02_model_training.ipynb` (Colab)
- Decoder-only Transformer: 128 dims, 4 heads, 4 layers, 795K parameters
- Trained with early stopping on validation loss
- Compared against EWMA, historical mean, naive baselines

### Phase 3: Trading Backtest
**File**: `btc_volatility_03_trading_backtest.ipynb` (Colab)
- Volatility predictions → trading signals with momentum confirmation
- Realistic simulation with bid/ask spread transaction costs

### Phase 4: MCP Data Preparation
**File**: `btc_volatility_04_mcp_agent_data.ipynb` (Colab)
- Full-dataset inference → volatility surprise ratios
- LONG/SHORT signals from 1.1x surprise thresholds
- Output: `btc_agent_replay.csv` for the MCP agent

---

## File Manifest

| File | Role | HandoutWeek4 Section |
|------|------|---------------------|
| `trading_server.py` | MCP server (5 tools, 4 resources) | 5.2, 5.3 |
| `orchestrator.py` | Agent loop + --replay CLI | 5.4, 5.5, 5.6 |
| `tool_registry.json` | Tool schemas, budgets, config | 4.2 |
| `test_integration.py` | End-to-end verification | 5.7 |
| `requirements.txt` | Python dependencies | 5.1 |
| `btc_agent_replay.csv` | Historical data (6,598 rows) | Data |
| `memory/` | Persistent state + logs | 5.3 |

---

## Optional Colab Path

The Colab notebooks (Phases 1–4) are self-contained and do not require MCP.
The MCP components (this repo) run locally in Python 3.10+.

If you want to run the MCP server inside Colab:
1. Install websockets: `!pip install websockets`
2. Use a tunnel (cloudflared or ngrok) to expose the WebSocket port
3. Run the orchestrator locally, connecting to the tunnel URL

**Limitation**: Colab restricts long-running processes and doesn't persist between sessions. Local execution is the canonical path.

---

## Verification Commands

```bash
# 1. Start server
python trading_server.py

# 2. Run integration tests (in a second terminal)
python test_integration.py

# 3. Run a 50-cycle live session
python orchestrator.py --steps 50

# 4. Replay the trace (no server needed)
python orchestrator.py --replay memory/trace.jsonl

# 5. Inspect outputs
cat memory/summary.json
head -5 memory/trace.jsonl
```

---

## Reflections

### 1. On the Rapid Pace of AI Development

Your Week 3 feedback included a striking observation: *"When reading 'Superintelligence' by Nick Bostrom in 2017, his 'fast takeoff' scenario seemed so science fiction. I personally feel like finding myself in such a scenario now."*

That comment resonated deeply with my reading: I recently spent considerable time with **'What is Intelligence: Lessons from AI About Evolution, Computing, and Minds'** by Blaise Agüera y Arcas (Google VP). This 500-page work synthesizes computer science, evolution, mathematics, psychology, neuroscience, and philosophy to argue that intelligence is better understood as a spectrum of evolving capabilities across systems, rather than a uniquely human essence that machines either have or lack.

Under this framework, LLMs today are in a way intelligent.

Implications for Finance: If we're approaching anything resembling Bostrom's "fast takeoff," being at the forefront of AI in specialized domains becomes crucial. Finance, with its complexity and data richness, will see significant disruption. I find it exciting to study an article like 'Artificial Intelligence vs. Efficient Markets: A critical Reassessment of Predictive Models in the Big Data Era' by Antonio Pagliaro (2025) (https://doi.org/10.3390/electronics14091721) or 'The Virtue of Complexity in Return Prediction' by Kelly, Malamud and Zhou (2023) (DOI: 10.1111/jofi.13298).

### 2. On Teaching and Learning Foundations

A paradox of modern AI tools: They make problem-solving easier while potentially eroding foundational understanding. I fought with this tension throughout this project.

I see this in my mathematics teaching: Students can now solve complex problems with AI assistance, but may skip the conceptual foundations that enable deeper insight and transfer to novel situations. Did I do the same here?

**The tension**:
- **Getting to the solution** is important for learning (concrete examples, immediate feedback)
- **Building foundations** is important for future capability (abstraction, generalization, creativity)

In this project, I extensively used Claude (and Gemini in Colab) for code generation. However, I insisted on "Teaching Moments" in every cell — detailed explanations of what the code does, why it's structured that way, and the underlying financial/technical concepts.

### 3. On the MCP Architecture

Building the MCP agent taught me that the protocol is really about **discipline**: every tool call has a contract, every action is logged, every budget is enforced. This is the difference between a demo and a production system. The Observe → Plan → Act → Learn loop with correlation IDs makes the agent's behavior fully traceable — you can replay any decision and understand exactly why it happened.

---

## Disclaimer

**I do not claim to have passed this course — I did not do independent coding. Far from it.** Claude and Gemini generated most of the implementation code. However, I:
- Designed the overall architecture and pipeline
- Specified all requirements and constraints
- Verified conceptual correctness at each stage
- Learned the underlying principles through extensive teaching moments
- Built understanding of transformer architectures, MCP protocols, and financial modeling

---

**Thomas Jensen**  
Cohort 1, The AI Engineer  
April 2026
