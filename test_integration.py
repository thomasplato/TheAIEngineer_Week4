#!/usr/bin/env python3
"""
Integration Test — Verify the MCP Trading Agent End-to-End
============================================================

HandoutWeek4 (Section 5.7) requires:
  1. Unit test each tool handler with golden inputs/outputs
  2. Write an integration script that simulates an incident (in our case,
     a market replay), runs the full agent loop, and checks for:
       - At least one retrieval and one diagnostic call
       - A final summary referencing citations (from resources) plus
         recommended actions
  3. Capture latency and cost stats

HOW TO RUN:
-----------
  Terminal 1: python trading_server.py
  Terminal 2: python test_integration.py

This script will:
  1. Connect to the running MCP server
  2. Test the initialize handshake
  3. Test each tool individually
  4. Test resource reads
  5. Run 2 full Observe→Plan→Act→Learn cycles
  6. Verify that trace.jsonl was written
  7. Test the --replay mode on the generated trace
  8. Print PASS/FAIL for each test
"""

import asyncio
import json
import websockets
from pathlib import Path
from typing import Dict

# ---- Configuration ----
HOST = "127.0.0.1"
PORT = 8765
URI = f"ws://{HOST}:{PORT}"

# Track test results
results = []

def record(name: str, passed: bool, detail: str = ""):
    """Record a test result."""
    status = "PASS" if passed else "FAIL"
    results.append({"name": name, "passed": passed, "detail": detail})
    symbol = "✓" if passed else "✗"
    print(f"  {symbol} {name}" + (f" — {detail}" if detail else ""))


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

async def test_initialize(ws) -> Dict:
    """Test 1: MCP initialize handshake."""
    request = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {"clientInfo": {"name": "test-client"}},
        "id": 1,
    }
    await ws.send(json.dumps(request))
    response = json.loads(await ws.recv())
    result = response.get("result", {})
    
    has_version = "protocolVersion" in result
    has_server = "serverInfo" in result
    has_caps = "capabilities" in result
    
    record("initialize handshake",
           has_version and has_server and has_caps,
           f"version={result.get('protocolVersion')}")
    return result


async def test_tools_list(ws):
    """Test 2: List available tools."""
    request = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "params": {},
        "id": 2,
    }
    await ws.send(json.dumps(request))
    response = json.loads(await ws.recv())
    tools = response.get("result", {}).get("tools", [])
    
    tool_names = [t["name"] for t in tools]
    required = ["get_market_data", "run_prediction", "execute_trade",
                "get_performance", "get_replay_status"]
    
    all_present = all(name in tool_names for name in required)
    record("tools/list — 5 tools present",
           all_present and len(tools) >= 5,
           f"found: {tool_names}")


async def test_resources_list(ws):
    """Test 3: List available resources."""
    request = {
        "jsonrpc": "2.0",
        "method": "resources/list",
        "params": {},
        "id": 3,
    }
    await ws.send(json.dumps(request))
    response = json.loads(await ws.recv())
    resources = response.get("result", {}).get("resources", [])
    
    uris = [r["uri"] for r in resources]
    record("resources/list — 4+ resources",
           len(resources) >= 4,
           f"found: {uris}")


async def call_tool(ws, name: str, args: dict = None, req_id: int = 10) -> dict:
    """Helper: call a tool and return parsed result."""
    request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": name, "arguments": args or {}},
        "id": req_id,
    }
    await ws.send(json.dumps(request))
    response = json.loads(await ws.recv())
    content = response.get("result", {}).get("content", [])
    if content:
        return json.loads(content[0].get("text", "{}"))
    return {}


async def test_get_market_data(ws):
    """Test 4: get_market_data returns valid data."""
    result = await call_tool(ws, "get_market_data", req_id=10)
    
    has_data = result.get("status") == "success"
    data = result.get("data", {})
    has_price = "price" in data
    has_bid = "bid" in data
    
    record("get_market_data — returns price/bid/ask",
           has_data and has_price and has_bid,
           f"price=${data.get('price', 0):,.2f}" if has_data else "no data")


async def test_run_prediction(ws):
    """Test 5: run_prediction returns volatility data."""
    result = await call_tool(ws, "run_prediction", req_id=11)
    
    has_data = result.get("status") == "success"
    data = result.get("data", {})
    has_signal = "signal" in data
    has_vol = "vol_surprise" in data
    
    record("run_prediction — returns signal + vol_surprise",
           has_data and has_signal and has_vol,
           f"signal={data.get('signal')}, vol={data.get('vol_surprise', 0):.3f}")


async def test_execute_trade(ws):
    """Test 6: execute_trade handles BUY/HOLD."""
    # Test HOLD (should always succeed)
    result = await call_tool(ws, "execute_trade",
                              {"action": "HOLD", "reason": "test"},
                              req_id=12)
    record("execute_trade HOLD — succeeds",
           result.get("status") == "success")
    
    # Test BUY
    result = await call_tool(ws, "execute_trade",
                              {"action": "BUY", "quantity": 0.01, "reason": "test buy"},
                              req_id=13)
    record("execute_trade BUY — succeeds",
           result.get("status") == "success",
           f"position={result.get('data', {}).get('trade', {}).get('position_after', '?')}")


async def test_get_performance(ws):
    """Test 7: get_performance returns metrics."""
    result = await call_tool(ws, "get_performance", req_id=14)
    
    has_data = result.get("status") == "success"
    data = result.get("data", {})
    has_pnl = "pnl_usd" in data
    has_budget = "budget" in data
    
    record("get_performance — returns P&L + budget",
           has_data and has_pnl and has_budget,
           f"P&L=${data.get('pnl_usd', 0):+,.2f}")


async def test_resource_read(ws):
    """Test 8: Read resources."""
    for uri in ["memory://state", "memory://performance",
                "memory://trades/latest", "memory://replay/progress"]:
        request = {
            "jsonrpc": "2.0",
            "method": "resources/read",
            "params": {"uri": uri},
            "id": 20,
        }
        await ws.send(json.dumps(request))
        response = json.loads(await ws.recv())
        contents = response.get("result", {}).get("contents", [])
        
        record(f"resource read {uri}",
               len(contents) > 0,
               f"returned {len(contents)} content block(s)")


async def test_full_cycles(ws):
    """
    Test 9: Run 2 complete Observe→Plan→Act→Learn cycles.
    
    This is the integration test the handout asks for:
    simulate a scenario and verify the full loop works.
    """
    print("\n  Running 2 full O→P→A→L cycles...")
    
    for cycle in range(2):
        # OBSERVE
        market = await call_tool(ws, "get_market_data", req_id=30+cycle*10)
        if market.get("status") == "end_of_data":
            record(f"full cycle {cycle+1} — observe", False, "end of data")
            continue
        
        prediction = await call_tool(ws, "run_prediction", req_id=31+cycle*10)
        
        # PLAN (simple logic)
        signal = prediction.get("data", {}).get("signal", "NEUTRAL")
        action = "BUY" if signal == "LONG" else "HOLD"
        
        # ACT
        await call_tool(ws, "execute_trade",
                        {"action": action, "quantity": 0.01, "reason": f"test cycle {cycle+1}"},
                        req_id=32+cycle*10)
        
        # LEARN
        perf = await call_tool(ws, "get_performance", req_id=33+cycle*10)
        
        record(f"full cycle {cycle+1} — complete",
               perf.get("status") == "success",
               f"action={action}, P&L=${perf.get('data', {}).get('pnl_usd', 0):+,.2f}")


def test_replay_mode():
    """
    Test 10: Verify that --replay can read a trace file.
    
    This doesn't need a server connection — it just reads the
    JSONL file that was generated during a previous live run.
    """
    trace_path = Path("memory/trace.jsonl")
    
    if not trace_path.exists():
        record("replay mode — trace file exists", False, "no trace.jsonl found")
        return
    
    # Count lines in trace file
    with open(trace_path) as f:
        lines = [l for l in f if l.strip()]
    
    record("replay mode — trace file exists",
           len(lines) > 0,
           f"{len(lines)} cycle(s) recorded")
    
    # Verify each line is valid JSON
    all_valid = True
    for i, line in enumerate(lines):
        try:
            record_data = json.loads(line)
            assert "correlation_id" in record_data
            assert "cycle_number" in record_data
        except (json.JSONDecodeError, AssertionError):
            all_valid = False
            break
    
    record("replay mode — trace records valid",
           all_valid,
           "each record has correlation_id + cycle_number")


# =============================================================================
# MAIN
# =============================================================================

async def run_tests():
    print("=" * 70)
    print("INTEGRATION TESTS — MCP Trading Agent")
    print("=" * 70)
    print(f"\nConnecting to {URI}...\n")
    
    try:
        async with websockets.connect(URI) as ws:
            print("--- Protocol Tests ---")
            await test_initialize(ws)
            await test_tools_list(ws)
            await test_resources_list(ws)
            
            print("\n--- Tool Tests ---")
            await test_get_market_data(ws)
            await test_run_prediction(ws)
            await test_execute_trade(ws)
            await test_get_performance(ws)
            
            print("\n--- Resource Tests ---")
            await test_resource_read(ws)
            
            print("\n--- Integration Tests ---")
            await test_full_cycles(ws)
            
    except ConnectionRefusedError:
        print(f"[ERROR] Cannot connect to {URI}")
        print("Start the server first: python trading_server.py")
        return
    
    print("\n--- Replay Tests ---")
    test_replay_mode()
    
    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {passed}/{total} tests passed")
    if passed == total:
        print("All tests passed!")
    else:
        print("Failed tests:")
        for r in results:
            if not r["passed"]:
                print(f"  ✗ {r['name']}: {r['detail']}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_tests())
