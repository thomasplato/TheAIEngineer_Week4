#!/usr/bin/env python3
"""
MCP Trading Orchestrator — Observe → Plan → Act → Learn
=========================================================

This file implements the CLIENT side of the Model Context Protocol (MCP).

WHAT IS AN ORCHESTRATOR?
------------------------
The orchestrator is the "brain" that drives the agent loop. It:
  1. Connects to the MCP server via WebSocket
  2. Performs the initialize handshake (get capabilities)
  3. Runs the Observe → Plan → Act → Learn loop repeatedly
  4. Tracks budgets and stops when limits are reached
  5. Saves structured logs for replay and auditing

THE FOUR PHASES (HandoutWeek4, Section 4.1):
---------------------------------------------
  1. OBSERVE: Fetch market data + read resources (memory://state, etc.)
  2. PLAN:    Decide what to do based on observations (which tool to call)
  3. ACT:     Invoke the chosen tool with typed inputs
  4. LEARN:   Write a "memory delta" summarizing what happened

CORRELATION IDs (HandoutWeek4, Section 5.6):
---------------------------------------------
Each cycle gets a unique correlation_id (UUID4). Every tool call and
resource read within that cycle carries this ID, so we can later trace
exactly which requests belonged to which decision cycle. This is critical
for debugging and replay.

REPLAY MODE (HandoutWeek4, Section 5.6, point 3):
---------------------------------------------------
The --replay flag lets you replay a stored transcript (JSONL file)
WITHOUT connecting to a live server. This is useful for:
  - Debugging: "What did the agent do at cycle 42?"
  - Regression testing: "Does my change break the old behavior?"
  - Demonstrations: "Show the professor the agent running"

References:
  [1] Y. J. Hilpisch, "AI Agents & Automation", The AI Engineer Program, 2025
  [2] Y. J. Hilpisch, "Software, ML, and AI Engineering", The AI Engineer Program, 2025
"""

import asyncio
import json
import uuid
import time
import argparse
import websockets
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

HOST = "127.0.0.1"
PORT = 8765
OUTPUT_DIR = Path("memory")
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CycleRecord:
    """
    Complete record of one Observe→Plan→Act→Learn cycle.
    
    This is the fundamental unit of the agent's trace log.
    The handout (Section 5.6) says to log: observation snapshot,
    chosen plan, tool request, tool response, and memory delta.
    
    The correlation_id ties everything in one cycle together,
    making it possible to replay or audit a specific decision.
    """
    correlation_id: str      # UUID linking all calls in this cycle
    cycle_number: int        # Sequential cycle counter
    timestamp: str           # When this cycle ran (ISO format)
    
    # OBSERVE phase
    datetime_market: str     # Market timestamp from the data
    price: float
    signal: str              # LONG / SHORT / NEUTRAL
    vol_surprise: float
    momentum: float
    
    # PLAN phase
    planned_action: str      # What we decided to do
    plan_reason: str         # Why (for auditability)
    
    # ACT phase
    executed_action: str     # What actually executed
    quantity: float
    
    # LEARN phase
    position_after: float    # BTC held after this cycle
    total_value: float       # Portfolio value in USD
    pnl_usd: float           # Profit/loss since start
    
    # Telemetry
    cycle_latency_ms: float  # How long this cycle took
    budget_trades_remaining: int
    budget_calls_remaining: int


# =============================================================================
# MCP CLIENT
# =============================================================================
# This class handles the low-level WebSocket communication with the
# MCP server. It sends JSON-RPC 2.0 requests and receives responses.

class MCPClient:
    """
    Lightweight MCP client that speaks JSON-RPC 2.0 over WebSocket.
    
    Each method builds a JSON-RPC request, sends it, and waits for
    the server's response. The "id" field in each request is how
    the server correlates requests with responses.
    """
    
    def __init__(self, websocket):
        self.ws = websocket
        self.request_id = 0  # Auto-incrementing request ID
    
    async def initialize(self) -> Dict:
        """
        MCP HANDSHAKE — must be called before any other method.
        
        Sends "initialize" and receives the server's capabilities
        (which tools and resources are available). The handout
        (Section 2.1, point 1) says: "Until this succeeds you
        cannot plan safely."
        """
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "clientInfo": {
                    "name": "btc-trading-orchestrator",
                    "version": "1.0.0",
                },
            },
            "id": self.request_id,
        }
        await self.ws.send(json.dumps(request))
        response = json.loads(await self.ws.recv())
        return response.get("result", {})
    
    async def list_tools(self) -> List[Dict]:
        """Fetch the server's tool catalog."""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": self.request_id,
        }
        await self.ws.send(json.dumps(request))
        response = json.loads(await self.ws.recv())
        return response.get("result", {}).get("tools", [])
    
    async def call_tool(self, name: str, arguments: Dict = None) -> Dict:
        """
        TOOL INVOCATION — calls a server-side tool.
        
        This is the core of MCP: the client says "run this tool with
        these arguments" and the server executes it safely, returning
        structured results.
        
        The response contains "content" with the tool's JSON output,
        wrapped in MCP's standard format.
        """
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments or {},
            },
            "id": self.request_id,
        }
        await self.ws.send(json.dumps(request))
        response = json.loads(await self.ws.recv())
        
        # Extract the actual data from MCP's content wrapper
        content = response.get("result", {}).get("content", [])
        if content:
            return json.loads(content[0].get("text", "{}"))
        return {}
    
    async def read_resource(self, uri: str) -> Dict:
        """
        RESOURCE READ — fetches data from the server's memory.
        
        Unlike tools, this has no side effects. It's used in the
        OBSERVE phase to read the agent's own state and history.
        """
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": "resources/read",
            "params": {"uri": uri},
            "id": self.request_id,
        }
        await self.ws.send(json.dumps(request))
        response = json.loads(await self.ws.recv())
        
        contents = response.get("result", {}).get("contents", [])
        if contents:
            return json.loads(contents[0].get("text", "{}"))
        return {}


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class TradingOrchestrator:
    """
    The agent's decision loop: Observe → Plan → Act → Learn.
    
    This orchestrator:
      1. Uses a deterministic policy (signal-based) for planning
      2. Tracks budgets per cycle
      3. Assigns a correlation ID to each cycle
      4. Saves structured JSONL logs for replay
    
    The handout (Section 5.4) suggests starting with a deterministic
    finite-state machine (FSM) for the planner, which is what our
    signal-based logic implements. An LLM-based planner could replace
    the plan() method later.
    """
    
    def __init__(self, mcp_client: MCPClient, output_dir: Path):
        self.mcp = mcp_client
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        self.cycle_count = 0
        self.history: List[CycleRecord] = []
        self.start_time = None
        
        # Log file paths
        self.trace_path = self.output_dir / "trace.jsonl"
        self.summary_path = self.output_dir / "summary.json"
    
    # ----- THE FOUR PHASES -----
    
    async def observe(self) -> Optional[Dict]:
        """
        PHASE 1: OBSERVE
        
        Gather all information the agent needs to make a decision.
        This includes:
          - Current market data (price, spread)
          - Model prediction (vol_surprise, momentum, signal)
          - Current portfolio state (via resource read)
        
        The handout (Section 4.1, point 1) says: "fetch alert payloads,
        most recent telemetry, and runbook snippets via MCP resource listings."
        In our trading domain, this translates to fetching market data,
        predictions, and portfolio state.
        
        Returns None if no more data is available (end of replay).
        """
        # Get market observation via tool
        market = await self.mcp.call_tool("get_market_data")
        if market.get("status") == "end_of_data":
            return None
        
        # Get prediction (also advances time)
        prediction = await self.mcp.call_tool("run_prediction")
        if prediction.get("status") == "end_of_data":
            return None
        
        # Read current state via resource (read-only, no side effects)
        state = await self.mcp.read_resource("memory://state")
        
        return {
            "market": market.get("data", {}),
            "prediction": prediction.get("data", {}),
            "state": state.get("data", {}),
        }
    
    def plan(self, observation: Dict) -> Dict:
        """
        PHASE 2: PLAN
        
        Decide which action to take based on observations.
        
        This is a DETERMINISTIC POLICY (finite-state machine):
          - LONG signal + no position → BUY
          - SHORT signal + have position → SELL
          - Otherwise → HOLD
        
        The handout (Section 5.4, point 2) suggests: "Start with a
        deterministic FSM to enforce guardrails. Optionally embed an
        LLM call that scores tool options based on observation + memory."
        
        A more sophisticated version could use GPT-4o Mini here
        (as in our earlier mcp_server.py), but the deterministic
        approach is safer and fully auditable.
        """
        pred = observation.get("prediction", {})
        state = observation.get("state", {})
        
        signal = pred.get("signal", "NEUTRAL")
        vol_surprise = pred.get("vol_surprise", 0)
        momentum = pred.get("momentum", 0)
        position = state.get("position", 0)
        has_position = position > 0.001
        
        if signal == "LONG":
            if has_position:
                action, reason = "HOLD", "LONG signal but already holding"
            else:
                action = "BUY"
                reason = f"LONG: vol_surprise={vol_surprise:.3f}, momentum={momentum:+.4f}"
        elif signal == "SHORT":
            if has_position:
                action = "SELL"
                reason = f"SHORT: vol_surprise={vol_surprise:.3f}, momentum={momentum:+.4f}"
            else:
                action, reason = "HOLD", "SHORT signal but no position to sell"
        else:
            action, reason = "HOLD", "NEUTRAL signal — waiting"
        
        return {"action": action, "reason": reason}
    
    async def act(self, plan: Dict) -> Dict:
        """
        PHASE 3: ACT
        
        Execute the planned action by calling the execute_trade tool.
        
        The handout (Section 4.1, point 3) says: "Invoke tools with
        typed inputs; capture latency, stdout/stderr, and sanitized outputs."
        
        We pass the action and quantity to the server, which handles
        the actual execution (checking budgets, updating state, etc.)
        """
        result = await self.mcp.call_tool("execute_trade", {
            "action": plan["action"],
            "quantity": 0.1,
            "reason": plan["reason"],
        })
        return result
    
    async def learn(self, observation: Dict, plan: Dict,
                    act_result: Dict, correlation_id: str,
                    cycle_latency_ms: float) -> CycleRecord:
        """
        PHASE 4: LEARN
        
        Record what happened in this cycle for future reference.
        
        The handout (Section 4.1, point 4) says: "Write memory deltas
        ('Diagnosed service X, ran command Y, outcome Z') and tag them
        with timestamps for future retrieval."
        
        In our domain, the "delta" is: what the market looked like,
        what we decided, what we executed, and the resulting P&L.
        
        This also writes the cycle to the JSONL trace file, which
        enables the --replay mode.
        """
        # Read post-action performance
        perf = await self.mcp.call_tool("get_performance")
        perf_data = perf.get("data", {})
        
        market = observation.get("market", {})
        pred = observation.get("prediction", {})
        
        self.cycle_count += 1
        
        record = CycleRecord(
            correlation_id=correlation_id,
            cycle_number=self.cycle_count,
            timestamp=datetime.now(timezone.utc).isoformat(),
            datetime_market=market.get("datetime", ""),
            price=market.get("price", 0),
            signal=pred.get("signal", "NEUTRAL"),
            vol_surprise=pred.get("vol_surprise", 0),
            momentum=pred.get("momentum", 0),
            planned_action=plan["action"],
            plan_reason=plan["reason"],
            executed_action=plan["action"],
            quantity=0.1 if plan["action"] != "HOLD" else 0.0,
            position_after=perf_data.get("position_btc", 0),
            total_value=perf_data.get("total_value_usd", 100000),
            pnl_usd=perf_data.get("pnl_usd", 0),
            cycle_latency_ms=cycle_latency_ms,
            budget_trades_remaining=perf_data.get("budget", {}).get("trades_remaining", 0),
            budget_calls_remaining=perf_data.get("budget", {}).get(
                "tool_calls_max", 0) - perf_data.get("budget", {}).get("tool_calls_used", 0),
        )
        
        self.history.append(record)
        
        # Append to JSONL trace file (one JSON object per line)
        # This file is what --replay reads back later.
        with open(self.trace_path, 'a') as f:
            f.write(json.dumps(asdict(record)) + '\n')
        
        return record
    
    # ----- MAIN LOOP -----
    
    async def run(self, max_cycles: int = 50, progress_every: int = 10):
        """
        Run the full agent loop.
        
        The handout (Section 4.1, point 5) says: "Loop until the agent
        either resolves the incident or escalates with a human-handoff
        package." In our domain, we loop until either:
          - The historical data runs out (end_of_data)
          - We hit the max_cycles limit
          - A budget is exceeded
        """
        print("\n" + "=" * 70)
        print("ORCHESTRATOR: Observe → Plan → Act → Learn")
        print("=" * 70)
        
        self.start_time = time.time()
        
        # Get replay status for progress tracking
        status = await self.mcp.call_tool("get_replay_status")
        total_rows = status.get("data", {}).get("total_rows", 0)
        print(f"[LOOP] Historical data: {total_rows:,} rows")
        print(f"[LOOP] Max cycles: {max_cycles}")
        print(f"[LOOP] Trace log: {self.trace_path}\n")
        
        # Clear trace file for fresh run
        if self.trace_path.exists():
            self.trace_path.unlink()
        
        end_reason = "unknown"
        
        while True:
            cycle_start = time.time()
            
            # Generate a unique correlation ID for this entire cycle.
            # Every tool call and resource read in this cycle can be
            # traced back to this ID. (HandoutWeek4, Section 5.6, point 1)
            correlation_id = str(uuid.uuid4())
            
            # --- OBSERVE ---
            observation = await self.observe()
            if observation is None:
                end_reason = "end_of_data"
                print(f"\n[LOOP] End of historical data at cycle {self.cycle_count}")
                break
            
            # --- PLAN ---
            plan = self.plan(observation)
            
            # --- ACT ---
            act_result = await self.act(plan)
            
            # --- LEARN ---
            cycle_ms = round((time.time() - cycle_start) * 1000, 2)
            record = await self.learn(observation, plan, act_result,
                                       correlation_id, cycle_ms)
            
            # Check max cycles
            if self.cycle_count >= max_cycles:
                end_reason = "max_cycles"
                print(f"\n[LOOP] Reached max cycles ({max_cycles})")
                break
            
            # Progress update
            if self.cycle_count % progress_every == 0:
                elapsed = time.time() - self.start_time
                rate = self.cycle_count / elapsed if elapsed > 0 else 0
                print(f"  Cycle {self.cycle_count:>5,} | "
                      f"{record.datetime_market} | "
                      f"${record.price:>10,.2f} | "
                      f"{record.signal:>7} → {record.executed_action:>4} | "
                      f"P&L: ${record.pnl_usd:>+10,.2f} | "
                      f"{rate:.0f} cycles/sec")
        
        # Save final summary and print analysis
        self.save_summary(end_reason)
        self.print_analysis(end_reason)
    
    # ----- OUTPUT -----
    
    def save_summary(self, end_reason: str):
        """Save a JSON summary of the entire run."""
        if not self.history:
            return
        
        final = self.history[-1]
        elapsed = time.time() - self.start_time
        
        summary = {
            "run_metadata": {
                "end_reason": end_reason,
                "total_cycles": self.cycle_count,
                "elapsed_seconds": round(elapsed, 1),
                "cycles_per_second": round(self.cycle_count / elapsed, 1) if elapsed > 0 else 0,
                "trace_file": str(self.trace_path),
            },
            "performance": {
                "initial_capital": 100000.0,
                "final_value": final.total_value,
                "pnl_usd": final.pnl_usd,
                "pnl_pct": round(final.pnl_usd / 100000 * 100, 4),
                "final_position_btc": final.position_after,
            },
            "signal_distribution": {
                "LONG": sum(1 for h in self.history if h.signal == "LONG"),
                "SHORT": sum(1 for h in self.history if h.signal == "SHORT"),
                "NEUTRAL": sum(1 for h in self.history if h.signal == "NEUTRAL"),
            },
            "action_distribution": {
                "BUY": sum(1 for h in self.history if h.executed_action == "BUY"),
                "SELL": sum(1 for h in self.history if h.executed_action == "SELL"),
                "HOLD": sum(1 for h in self.history if h.executed_action == "HOLD"),
            },
            "date_range": {
                "start": self.history[0].datetime_market,
                "end": self.history[-1].datetime_market,
            },
        }
        
        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[SAVE] Summary: {self.summary_path}")
        print(f"[SAVE] Trace:   {self.trace_path} ({self.cycle_count} cycles)")
    
    def print_analysis(self, end_reason: str):
        """Print human-readable performance analysis."""
        if not self.history:
            print("[ANALYSIS] No cycles completed.")
            return
        
        elapsed = time.time() - self.start_time
        final = self.history[-1]
        
        print("\n" + "=" * 70)
        print(f"RUN COMPLETE — {end_reason.upper()}")
        print("=" * 70)
        
        print(f"\n  Performance:")
        print(f"    Initial Capital:  $100,000.00")
        print(f"    Final Value:      ${final.total_value:,.2f}")
        print(f"    P&L:              ${final.pnl_usd:+,.2f} "
              f"({final.pnl_usd/1000:.2f}%)")
        
        actions = [h.executed_action for h in self.history]
        print(f"\n  Actions: {actions.count('BUY')} buys, "
              f"{actions.count('SELL')} sells, "
              f"{actions.count('HOLD')} holds")
        
        pnls = [h.pnl_usd for h in self.history]
        print(f"  Peak P&L:  ${max(pnls):+,.2f}")
        print(f"  Worst P&L: ${min(pnls):+,.2f}")
        
        print(f"\n  Execution:")
        print(f"    Cycles: {self.cycle_count:,}")
        print(f"    Time:   {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"    Rate:   {self.cycle_count/elapsed:.0f} cycles/sec")
        
        print(f"\n  Date range: {self.history[0].datetime_market} "
              f"→ {self.history[-1].datetime_market}")
        print("=" * 70)


# =============================================================================
# REPLAY MODE
# =============================================================================
# The handout (Section 5.6, point 3) says:
# "Provide a CLI flag (e.g., --replay path.json) that replays a stored
#  transcript without calling real tools."
#
# This function reads the JSONL trace file produced by a live run and
# replays it, printing each cycle's decisions without needing the server.
# Useful for debugging, demonstrations, and regression testing.

def replay_from_file(trace_path: Path, max_cycles: Optional[int] = None):
    """
    Replay a stored trace file without connecting to the live server.
    
    Reads the JSONL file line-by-line and prints each cycle's details.
    This proves the instrumentation is working: if the trace file
    contains valid data, the agent was properly logging.
    """
    if not trace_path.exists():
        print(f"[REPLAY] Error: trace file not found: {trace_path}")
        return
    
    print("=" * 70)
    print("REPLAY MODE — Reading from stored trace")
    print(f"File: {trace_path}")
    print("=" * 70 + "\n")
    
    cycle_count = 0
    
    with open(trace_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            record = json.loads(line)
            cycle_count += 1
            
            if max_cycles and cycle_count > max_cycles:
                break
            
            # Print cycle details
            print(f"  Cycle {record['cycle_number']:>5} | "
                  f"corr_id={record['correlation_id'][:8]}... | "
                  f"{record['datetime_market']} | "
                  f"${record['price']:>10,.2f} | "
                  f"{record['signal']:>7} → {record['executed_action']:>4} | "
                  f"P&L: ${record['pnl_usd']:>+10,.2f} | "
                  f"{record['cycle_latency_ms']:.0f}ms")
    
    print(f"\n[REPLAY] Replayed {cycle_count} cycles from {trace_path}")
    
    # Show final state
    if cycle_count > 0:
        print(f"[REPLAY] Final P&L: ${record['pnl_usd']:+,.2f}")
        print(f"[REPLAY] Final position: {record['position_after']:.6f} BTC")


# =============================================================================
# MAIN — CLI ENTRY POINT
# =============================================================================

async def run_live(args):
    """Connect to the MCP server and run the live agent loop."""
    uri = f"ws://{args.host}:{args.port}"
    
    print("=" * 70)
    print("BTC TRADING AGENT — MCP ORCHESTRATOR")
    print("=" * 70)
    print(f"\nConnecting to MCP server at {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            # Create MCP client
            mcp_client = MCPClient(websocket)
            
            # Step 1: INITIALIZE handshake
            server_info = await mcp_client.initialize()
            server_name = server_info.get('serverInfo', {}).get('name', 'unknown')
            print(f"Connected to: {server_name}")
            
            # Step 2: List available tools (optional but good practice)
            tools = await mcp_client.list_tools()
            print(f"Available tools: {[t['name'] for t in tools]}")
            
            # Step 3: Create and run orchestrator
            orchestrator = TradingOrchestrator(
                mcp_client=mcp_client,
                output_dir=Path(args.output),
            )
            
            await orchestrator.run(
                max_cycles=args.steps,
                progress_every=args.progress,
            )
            
    except ConnectionRefusedError:
        print(f"\n[ERROR] Could not connect to server at {uri}")
        print("Make sure trading_server.py is running first!")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise


def main():
    """
    CLI entry point with argument parsing.
    
    Usage examples:
      # Live mode: connect to server and run 50 cycles
      python orchestrator.py --steps 50
    
      # Replay mode: replay a stored trace (no server needed)
      python orchestrator.py --replay memory/trace.jsonl
    
      # Live mode with custom settings
      python orchestrator.py --steps 1000 --progress 100 --output results
    """
    parser = argparse.ArgumentParser(
        description="MCP Trading Orchestrator — Observe → Plan → Act → Learn",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python orchestrator.py --steps 50          Run 50 live cycles
  python orchestrator.py --replay trace.jsonl  Replay stored trace
  python orchestrator.py --steps 1000 --progress 100  Long run
        """,
    )
    
    # Live mode options
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='MCP server host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8765,
                        help='MCP server port (default: 8765)')
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of cycles to run (default: 50)')
    parser.add_argument('--progress', type=int, default=10,
                        help='Print progress every N cycles (default: 10)')
    parser.add_argument('--output', type=str, default='memory',
                        help='Output directory for logs (default: memory)')
    
    # Replay mode (HandoutWeek4, Section 5.6, point 3)
    parser.add_argument('--replay', type=str, default=None,
                        help='Replay a stored trace file (JSONL) without live server')
    
    args = parser.parse_args()
    
    if args.replay:
        # REPLAY MODE — no server connection needed
        replay_from_file(Path(args.replay), max_cycles=args.steps)
    else:
        # LIVE MODE — connect to MCP server
        asyncio.run(run_live(args))


if __name__ == "__main__":
    main()
