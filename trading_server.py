#!/usr/bin/env python3
"""
MCP Trading Server — BTC Volatility Agent (Historical Replay)
==============================================================

This file implements the SERVER side of the Model Context Protocol (MCP).

WHY MCP?
--------
MCP is a JSON-RPC 2.0 protocol over WebSocket that standardizes how an AI agent
(the "client" or "orchestrator") talks to tool providers (the "server").
Think of it like a universal plug: any MCP client can connect to any MCP server
without custom wiring, because they all speak the same protocol.

PROTOCOL FLOW (from HandoutWeek4, Section 2.1):
------------------------------------------------
1. Client opens a WebSocket connection to the server
2. Client sends "initialize" → Server replies with capabilities (tools + resources)
3. Client calls "tools/call" with a tool name and arguments → Server executes and returns
4. Client calls "resources/read" with a URI → Server returns stored data
5. Every request/response carries an "id" for correlation (linking request to response)

WHAT THIS SERVER PROVIDES:
--------------------------
- 5 Tools:  get_market_data, run_prediction, execute_trade, get_performance, get_replay_status
- 4 Resources: memory://trades/latest, memory://performance, memory://state, memory://replay/progress
- Budget enforcement: limits on trades and tool calls per session
- Memory persistence: all state saved as JSON files for auditability
- Historical replay: steps through btc_agent_replay.csv one row at a time

ARCHITECTURE NOTE (from HandoutWeek4, Section 2.2):
----------------------------------------------------
Every tool answers four questions:
  1. Purpose: WHY should the agent call this instead of prompting?
  2. Inputs: JSON schema with defaults, ranges, guardrail hints
  3. Outputs: typed payload + status + metrics (latency, cost)
  4. Side effects: what gets persisted, how to roll back

References:
  [1] Y. J. Hilpisch, "AI Agents & Automation", The AI Engineer Program, 2025
  [2] Y. J. Hilpisch, "Software, ML, and AI Engineering", The AI Engineer Program, 2025
"""

import asyncio
import json
import time
import uuid
import websockets
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

# =============================================================================
# CONFIGURATION
# =============================================================================
# These could also be loaded from tool_registry.json (see that file).
# The handout (Section 5.1) says: "Define environment variables for secrets
# and tool limits; never bake tokens into MCP payloads."

HOST = "127.0.0.1"
PORT = 8765
MEMORY_DIR = Path("memory")
DATA_FILE = Path("btc_agent_replay.csv")

# Create memory directory for persistent state
# The handout (Section 5.3) says: "Store observations and tool outcomes
# as short JSON documents. Index them by alert ID + timestamp."
MEMORY_DIR.mkdir(exist_ok=True)


# =============================================================================
# HISTORICAL DATA PROVIDER
# =============================================================================
# This class manages stepping through our CSV file one row at a time.
# In a live system, this would be replaced by a real-time data feed.
# The "replay" concept is key for the capstone: it lets us test and
# debug the agent without risking real money or needing live markets.

class HistoricalDataProvider:
    """
    Manages historical data replay — steps through btc_agent_replay.csv
    one observation at a time.
    
    The CSV was generated in Phase 4 (btc_volatility_04_mcp_agent_data.ipynb)
    by running our trained transformer model across the entire BTC dataset.
    Each row contains: price, bid, ask, realized_vol, predicted_vol,
    vol_surprise, momentum, signal, spread_bps.
    """
    
    def __init__(self, csv_path: Path):
        print(f"[DATA] Loading historical data from {csv_path}...")
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        
        # current_index tracks where we are in the replay
        self.current_index = 0
        self.max_index = len(self.df) - 1
        # is_complete prevents us from reading past the end
        self.is_complete = False
        
        print(f"[DATA] Loaded {len(self.df):,} rows")
        print(f"[DATA] Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        print(f"[DATA] Signal distribution: {self.df['signal'].value_counts().to_dict()}")
    
    def get_current_row(self) -> Optional[Dict]:
        """
        Get the current time step's data.
        Returns None if we've already exhausted all rows.
        """
        if self.is_complete or self.current_index > self.max_index:
            self.is_complete = True
            return None
        
        row = self.df.iloc[self.current_index]
        return row.to_dict()
    
    def advance(self) -> bool:
        """
        Move to the next time step.
        Returns True if there's more data, False if we just hit the end.
        
        This is called AFTER we've processed the current row, so the
        orchestrator sees the data before we move on.
        """
        if self.current_index < self.max_index:
            self.current_index += 1
            return True
        else:
            self.is_complete = True
            return False
    
    def get_progress(self) -> Dict:
        """Get current replay progress — used by get_replay_status tool."""
        return {
            "current_index": self.current_index,
            "total_rows": len(self.df),
            "progress_pct": round((self.current_index + 1) / len(self.df) * 100, 2),
            "is_complete": self.is_complete,
            "current_datetime": str(self.df.iloc[self.current_index]['datetime'])
                if not self.is_complete else None,
        }


# Global instance — created in main()
data_provider: Optional[HistoricalDataProvider] = None


# =============================================================================
# BUDGET ENFORCEMENT
# =============================================================================
# The handout (Section 5.4, point 3) says:
# "Track budgets (tokens, runtime, dollars) per loop.
#  Abort if any limit is exceeded."
#
# In our case we track:
#  - Number of trades (to prevent runaway trading)
#  - Number of tool calls (to prevent infinite loops)
#
# A production system would also track:
#  - LLM token usage and dollar cost
#  - Wall-clock time per cycle
#  - Network latency budgets (see Figure 1 in the handout)

@dataclass
class Budgets:
    """
    Session-level budget enforcement.
    
    Each field has a maximum and a counter. Before any action,
    the orchestrator (via the server) checks can_trade() or
    can_call_tool(). If the budget is exhausted, the server
    returns an error instead of executing.
    """
    max_trades_per_session: int = 500
    max_tool_calls_per_session: int = 50000
    trades_this_session: int = 0
    tool_calls_this_session: int = 0
    
    def can_trade(self) -> bool:
        return self.trades_this_session < self.max_trades_per_session
    
    def can_call_tool(self) -> bool:
        return self.tool_calls_this_session < self.max_tool_calls_per_session
    
    def record_trade(self):
        self.trades_this_session += 1
    
    def record_tool_call(self):
        self.tool_calls_this_session += 1
    
    def summary(self) -> Dict:
        """Return budget status for telemetry."""
        return {
            "trades_used": self.trades_this_session,
            "trades_max": self.max_trades_per_session,
            "trades_remaining": self.max_trades_per_session - self.trades_this_session,
            "tool_calls_used": self.tool_calls_this_session,
            "tool_calls_max": self.max_tool_calls_per_session,
        }


budgets = Budgets()


# =============================================================================
# MEMORY MANAGEMENT
# =============================================================================
# The handout (Section 5.3) explains:
# "Persistent state keeps loops coherent. Store observations and tool
#  outcomes as short JSON documents. Index them by alert ID + timestamp."
#
# We use simple JSON files in the memory/ directory. Each file serves
# a specific purpose:
#   - state.json:          Current portfolio (position, cash)
#   - trade_journal.json:  Complete history of all trades
#   - prediction_log.json: Every prediction the model made
#   - replay_log.json:     System-level replay events

def load_json(filename: str, default: Any = None) -> Any:
    """Load a JSON file from the memory directory, returning default if missing."""
    path = MEMORY_DIR / filename
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return default if default is not None else {}


def save_json(filename: str, data: Any) -> None:
    """Save data as JSON to the memory directory."""
    path = MEMORY_DIR / filename
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_state() -> Dict:
    """Load current portfolio state (position + cash)."""
    default_state = {
        "position": 0.0,
        "cash": 100000.0,
        "last_price": None,
        "last_update": None,
    }
    return load_json("state.json", default_state)


def save_state(state: Dict) -> None:
    save_state_to = state.copy()
    save_json("state.json", save_state_to)


def append_trade(trade: Dict) -> None:
    """Append a trade to the persistent journal."""
    journal = load_json("trade_journal.json", [])
    journal.append(trade)
    save_json("trade_journal.json", journal)


def append_prediction(prediction: Dict) -> None:
    """Append a prediction to the persistent log."""
    log = load_json("prediction_log.json", [])
    log.append(prediction)
    save_json("prediction_log.json", log)


def reset_memory():
    """
    Clear all memory files for a fresh start.
    Called at server startup so each session begins clean.
    """
    for filename in ["state.json", "trade_journal.json",
                     "prediction_log.json", "replay_log.json"]:
        path = MEMORY_DIR / filename
        if path.exists():
            path.unlink()
    print("[MEMORY] Cleared all memory files for fresh session")


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================
# These are the "contracts" advertised to the client during the
# initialize handshake. The handout (Section 2.2) says each tool
# must declare: name, description, and a JSON schema for inputs.
#
# The client uses this information to know WHAT it can call and
# WHAT arguments to provide — like an API specification.

TOOLS = [
    {
        "name": "get_market_data",
        "description": "Fetch current BTC market data from historical replay. "
                       "Returns price, bid, ask, volatility, and spread.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "default": "BTC/USD",
                    "description": "Trading pair symbol",
                },
            },
            "required": [],
        },
    },
    {
        "name": "run_prediction",
        "description": "Get pre-computed volatility prediction from the transformer model. "
                       "Returns vol_surprise, momentum, and trading signal. "
                       "Also advances the replay to the next time step.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "execute_trade",
        "description": "Execute a paper trade (BUY/SELL/HOLD) at historical bid/ask prices. "
                       "Enforces budget limits and persists to trade journal.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["BUY", "SELL", "HOLD"],
                    "description": "Trade direction",
                },
                "quantity": {
                    "type": "number",
                    "default": 0.01,
                    "description": "BTC quantity to trade",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this trade was chosen (for audit trail)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "get_performance",
        "description": "Get current portfolio performance metrics: P&L, position, win rate.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_replay_status",
        "description": "Get progress through the historical data replay.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

# =============================================================================
# RESOURCE DEFINITIONS
# =============================================================================
# Resources are READ-ONLY data the client can fetch via "resources/read".
# The handout (Section 5.3, point 2) says:
# "Expose the store through MCP's resource listing so the client
#  can 'Observe' before planning."
#
# The difference between tools and resources:
#   - Tools DO something (execute trades, run predictions)
#   - Resources EXPOSE data (read current state, read trade history)
#
# URIs use the "memory://" scheme — this is a convention, not a
# standard. It signals that the data comes from the agent's memory.

RESOURCES = [
    {
        "uri": "memory://trades/latest",
        "name": "Latest Trade",
        "description": "Most recent trade from the journal",
    },
    {
        "uri": "memory://performance",
        "name": "Performance Summary",
        "description": "Current P&L, position size, win rate",
    },
    {
        "uri": "memory://state",
        "name": "Agent State",
        "description": "Current portfolio state (cash + BTC position)",
    },
    {
        "uri": "memory://replay/progress",
        "name": "Replay Progress",
        "description": "How far through the historical data we are",
    },
]


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================
# Each function below is the actual logic behind a tool.
# When the client calls "tools/call" with a tool name, the server
# routes to the matching function here.

def tool_get_market_data(symbol: str = "BTC/USD") -> Dict:
    """
    OBSERVE phase data source.
    
    Returns the current row from btc_agent_replay.csv, which contains
    price, bid, ask, realized volatility, and spread. If the replay
    has finished, returns an end_of_data status so the orchestrator
    knows to stop gracefully.
    """
    global data_provider
    
    if data_provider.is_complete:
        return {
            "status": "end_of_data",
            "message": "Historical replay complete — no more data",
            "total_rows": len(data_provider.df),
        }
    
    row = data_provider.get_current_row()
    if row is None:
        return {"status": "end_of_data", "message": "No more data"}
    
    market_data = {
        "symbol": symbol,
        "datetime": str(row['datetime']),
        "price": float(row['price']),
        "bid": float(row['bid']),
        "ask": float(row['ask']),
        "realized_vol_24h": float(row['realized_vol']),
        "spread_bps": float(row['spread_bps']),
        "replay_index": data_provider.current_index,
        "replay_total": len(data_provider.df),
    }
    
    # Side effect: update last known price in memory
    state = load_state()
    state["last_price"] = market_data["price"]
    state["last_update"] = market_data["datetime"]
    save_state(state)
    
    return {"status": "success", "data": market_data}


def tool_run_prediction() -> Dict:
    """
    Serves the transformer model's pre-computed prediction.
    
    In Phase 4, we ran the trained transformer across all data and
    stored the results in btc_agent_replay.csv. Each row has:
      - predicted_vol: what the model thinks 24h volatility will be
      - realized_vol:  what actually happened
      - vol_surprise:  predicted / realized (>1 means expecting MORE vol)
      - momentum:      6-hour price trend
      - signal:        LONG / SHORT based on surprise thresholds
    
    IMPORTANT: This tool advances the replay index AFTER returning
    the current row's data, so the orchestrator sees the prediction
    before we move to the next time step.
    """
    global data_provider
    
    if data_provider.is_complete:
        return {"status": "end_of_data", "message": "Replay complete"}
    
    row = data_provider.get_current_row()
    if row is None:
        return {"status": "end_of_data", "message": "Replay complete"}
    
    prediction = {
        "datetime": str(row['datetime']),
        "predicted_vol": float(row['predicted_vol']),
        "realized_vol": float(row['realized_vol']),
        "vol_surprise": float(row['vol_surprise']),
        "momentum": float(row['momentum']),
        "signal": row['signal'],
        "replay_index": data_provider.current_index,
    }
    
    # Side effect: persist prediction for audit trail
    append_prediction(prediction)
    
    # Advance to next time step (returns False if no more data)
    has_more = data_provider.advance()
    
    return {
        "status": "success",
        "data": prediction,
        "has_more_data": has_more,
    }


def tool_execute_trade(action: str, quantity: float = 0.01, reason: str = "") -> Dict:
    """
    ACT phase — execute a paper trade at historical bid/ask prices.
    
    Safety checks:
      1. Budget: are we within our trade limit?
      2. Capital: do we have enough cash to buy?
      3. Position: do we have enough BTC to sell?
    
    Trades execute at the ASK price (buying) or BID price (selling),
    which simulates realistic transaction costs from the spread.
    """
    if not budgets.can_trade():
        return {"status": "error", "error": "Trade budget exceeded"}
    
    state = load_state()
    
    # Get prices from the current row (before advance)
    idx = max(0, data_provider.current_index - 1)
    row = data_provider.df.iloc[idx]
    bid = float(row['bid'])
    ask = float(row['ask'])
    price = float(row['price'])
    timestamp = str(row['datetime'])
    
    # HOLD requires no execution
    if action == "HOLD":
        return {
            "status": "success",
            "data": {
                "action": "HOLD",
                "reason": reason,
                "position": state["position"],
                "cash": state["cash"],
            },
        }
    
    # BUY: deduct cash, add BTC
    if action == "BUY":
        cost = quantity * ask  # We pay the ask price (higher)
        if cost > state["cash"]:
            return {
                "status": "error",
                "error": f"Insufficient cash: need ${cost:.2f}, have ${state['cash']:.2f}",
            }
        state["cash"] -= cost
        state["position"] += quantity
        fill_price = ask
        
    # SELL: add cash, deduct BTC
    elif action == "SELL":
        if quantity > state["position"]:
            return {
                "status": "error",
                "error": f"Insufficient BTC: want {quantity}, have {state['position']}",
            }
        proceeds = quantity * bid  # We receive the bid price (lower)
        state["cash"] += proceeds
        state["position"] -= quantity
        fill_price = bid
    else:
        return {"status": "error", "error": f"Unknown action: {action}"}
    
    # Persist updated state
    save_state(state)
    
    # Record trade in journal (side effect for audit trail)
    trade = {
        "timestamp": timestamp,
        "replay_index": idx,
        "action": action,
        "quantity": quantity,
        "fill_price": fill_price,
        "mid_price": price,
        "spread_bps": float(row['spread_bps']),
        "reason": reason,
        "position_after": state["position"],
        "cash_after": state["cash"],
    }
    append_trade(trade)
    budgets.record_trade()
    
    return {
        "status": "success",
        "data": {
            "trade": trade,
            "budget_remaining": budgets.max_trades_per_session - budgets.trades_this_session,
        },
    }


def tool_get_performance() -> Dict:
    """
    LEARN phase data — returns portfolio performance metrics.
    
    This is what the orchestrator reads after each cycle to understand
    how well (or poorly) the agent is doing, enabling the "Learn" step.
    """
    state = load_state()
    journal = load_json("trade_journal.json", [])
    
    current_price = state.get("last_price") or 95000
    position_value = state["position"] * current_price
    total_value = state["cash"] + position_value
    initial_value = 100000.0
    pnl_usd = total_value - initial_value
    pnl_pct = (pnl_usd / initial_value) * 100
    
    buys = [t for t in journal if t["action"] == "BUY"]
    sells = [t for t in journal if t["action"] == "SELL"]
    
    # Simple win/loss counting: compare each sell to its corresponding buy
    wins = sum(1 for i, s in enumerate(sells) if i < len(buys) and s["fill_price"] > buys[i]["fill_price"])
    losses = sum(1 for i, s in enumerate(sells) if i < len(buys) and s["fill_price"] <= buys[i]["fill_price"])
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    
    return {
        "status": "success",
        "data": {
            "total_value_usd": round(total_value, 2),
            "cash_usd": round(state["cash"], 2),
            "position_btc": round(state["position"], 6),
            "position_value_usd": round(position_value, 2),
            "pnl_usd": round(pnl_usd, 2),
            "pnl_pct": round(pnl_pct, 4),
            "num_buys": len(buys),
            "num_sells": len(sells),
            "num_trades": len(journal),
            "win_rate_pct": round(win_rate, 1),
            "budget": budgets.summary(),
        },
    }


def tool_get_replay_status() -> Dict:
    """Returns how far through the historical data we've progressed."""
    if data_provider is None:
        return {"status": "error", "error": "Data provider not initialized"}
    return {"status": "success", "data": data_provider.get_progress()}


# Map tool names to their handler functions
TOOL_HANDLERS = {
    "get_market_data": tool_get_market_data,
    "run_prediction": tool_run_prediction,
    "execute_trade": tool_execute_trade,
    "get_performance": tool_get_performance,
    "get_replay_status": tool_get_replay_status,
}


# =============================================================================
# RESOURCE HANDLERS
# =============================================================================
# Resources are the "read" side of memory. The orchestrator calls
# resources/read with a URI to fetch current state without side effects.

def get_resource(uri: str) -> Dict:
    """
    Route a resource URI to the appropriate data.
    
    Unlike tools, resources are read-only — they don't change state.
    The handout calls this the "Observe" capability: the agent reads
    its own memory before planning the next action.
    """
    if uri == "memory://trades/latest":
        journal = load_json("trade_journal.json", [])
        return {"data": journal[-1] if journal else None}
    elif uri == "memory://performance":
        return tool_get_performance()
    elif uri == "memory://state":
        return {"data": load_state()}
    elif uri == "memory://replay/progress":
        return tool_get_replay_status()
    return {"error": f"Unknown resource: {uri}"}


# =============================================================================
# MCP PROTOCOL HANDLERS
# =============================================================================
# These functions implement the JSON-RPC 2.0 methods that MCP defines.
# The handout (Section 2.1) describes the three-phase flow:
#   1. initialize   — capabilities handshake
#   2. tools/call   — invoke a tool
#   3. resources/read — fetch stored data
#
# Additional standard methods: tools/list, resources/list

def handle_initialize(params: Dict) -> Dict:
    """
    CAPABILITIES HANDSHAKE (Step 1 of MCP).
    
    The client sends "initialize" with its name/version.
    The server responds with:
      - protocolVersion: which MCP version we speak
      - serverInfo: our identity
      - capabilities: what features we support (tools, resources)
    
    Until this succeeds, the client cannot call any tools.
    (HandoutWeek4, Section 2.1, point 1)
    """
    return {
        "protocolVersion": "2024-11-05",
        "serverInfo": {
            "name": "btc-volatility-trading-server",
            "version": "1.0.0",
        },
        "capabilities": {
            "tools": {},       # We support the tools/call method
            "resources": {},   # We support the resources/read method
        },
    }


def handle_tools_list(params: Dict) -> Dict:
    """Return the catalog of available tools with their JSON schemas."""
    return {"tools": TOOLS}


def handle_tools_call(params: Dict) -> Dict:
    """
    TOOL INVOCATION (Step 3 of MCP).
    
    The client sends a tool name + arguments. We:
      1. Check the tool call budget
      2. Look up the handler function
      3. Execute it with the provided arguments
      4. Return structured output with status, data, and metrics
    
    The response format uses "content" with type "text" — this is
    the MCP standard for returning tool results as JSON strings.
    (HandoutWeek4, Section 2.1, point 3)
    """
    # Budget check BEFORE executing
    if not budgets.can_call_tool():
        return {
            "isError": True,
            "content": [{"type": "text", "text": json.dumps({
                "error": "Tool call budget exceeded",
                "budget": budgets.summary(),
            })}],
        }
    budgets.record_tool_call()
    
    tool_name = params.get("name")
    arguments = params.get("arguments", {})
    
    if tool_name not in TOOL_HANDLERS:
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
        }
    
    # Measure latency for telemetry (HandoutWeek4, Section 2.3)
    start = time.time()
    try:
        result = TOOL_HANDLERS[tool_name](**arguments)
        latency_ms = round((time.time() - start) * 1000, 2)
        # Attach metrics to every tool response
        result["_metrics"] = {"latency_ms": latency_ms, "tool": tool_name}
        return {"content": [{"type": "text", "text": json.dumps(result)}]}
    except Exception as e:
        return {
            "isError": True,
            "content": [{"type": "text", "text": json.dumps({
                "error": str(e), "tool": tool_name,
            })}],
        }


def handle_resources_list(params: Dict) -> Dict:
    """Return the catalog of available resources."""
    return {"resources": RESOURCES}


def handle_resources_read(params: Dict) -> Dict:
    """
    RESOURCE ACCESS (Step 2 of MCP).
    
    The client requests data by URI. Unlike tools, this is read-only
    and has no side effects — perfect for the "Observe" phase.
    (HandoutWeek4, Section 2.1, point 2)
    """
    uri = params.get("uri", "")
    result = get_resource(uri)
    return {"contents": [{"uri": uri, "text": json.dumps(result)}]}


def route_request(method: str, params: Dict) -> Dict:
    """
    Route an incoming JSON-RPC method to the right handler.
    
    This is the server's "dispatcher" — every MCP message arrives here,
    and we look up which function should handle it.
    """
    routes = {
        "initialize": handle_initialize,
        "tools/list": handle_tools_list,
        "tools/call": handle_tools_call,
        "resources/list": handle_resources_list,
        "resources/read": handle_resources_read,
    }
    handler = routes.get(method)
    if handler:
        return handler(params)
    return {"error": {"code": -32601, "message": f"Unknown method: {method}"}}


# =============================================================================
# WEBSOCKET SERVER
# =============================================================================
# The transport layer. MCP runs over WebSocket so messages can flow
# in both directions. Each message is a JSON-RPC 2.0 frame with:
#   {"jsonrpc": "2.0", "id": <correlation_id>, "method": "...", "params": {...}}
#
# The server replies with:
#   {"jsonrpc": "2.0", "id": <same_id>, "result": {...}}
#
# The matching "id" field is how the client correlates requests with
# responses — this is the "correlation ID" the handout emphasizes.

async def handle_connection(websocket):
    """Handle a single WebSocket client connection."""
    client_id = id(websocket)
    print(f"[SERVER] Client {client_id} connected")
    
    try:
        async for message in websocket:
            try:
                request = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }))
                continue
            
            method = request.get("method", "")
            params = request.get("params", {})
            req_id = request.get("id")
            
            # Route to the appropriate handler
            result = route_request(method, params)
            
            # Build JSON-RPC 2.0 response with matching correlation ID
            response = {"jsonrpc": "2.0", "id": req_id, "result": result}
            await websocket.send(json.dumps(response))
            
    except websockets.exceptions.ConnectionClosed:
        print(f"[SERVER] Client {client_id} disconnected")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    global data_provider
    
    print("=" * 70)
    print("BTC VOLATILITY TRADING — MCP SERVER")
    print("Week 4 Capstone: Model Context Protocol Agent")
    print("=" * 70)
    
    # Load historical data
    try:
        data_provider = HistoricalDataProvider(DATA_FILE)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Place btc_agent_replay.csv in the same directory as this script.")
        return
    
    # Clear memory for a fresh session
    reset_memory()
    
    print(f"\n[SERVER] Starting on ws://{HOST}:{PORT}")
    print(f"[SERVER] Memory directory: {MEMORY_DIR.absolute()}")
    print(f"[SERVER] Tools: {[t['name'] for t in TOOLS]}")
    print(f"[SERVER] Resources: {[r['uri'] for r in RESOURCES]}")
    print(f"[SERVER] Budgets: {budgets.max_trades_per_session} trades, "
          f"{budgets.max_tool_calls_per_session} tool calls")
    print(f"\n[SERVER] Waiting for client connection... (Ctrl+C to stop)\n")
    
    async with websockets.serve(handle_connection, HOST, PORT):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
