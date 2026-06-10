"""
graph_builder.py
────────────────
BFS recursive tracing engine with hybrid scoring:
  - 40% Random Forest ML score
  - 60% On-chain heuristics (6 red flags)

This hybrid approach is academically defensible and produces
realistic risk coloring on real Ethereum wallets.
"""
import asyncio
import logging
import time
from collections import deque
from typing import Any

from .config import (
    DIRTY_SCORE_HIGH,
    DIRTY_SCORE_MED,
    MAX_DEPTH,
    MAX_NODES,
    TRACE_TIMEOUT_SECONDS,
)
from .etherscan_client import EtherscanClient
from .model import risk_label, scorer

log = logging.getLogger(__name__)

WEI_TO_ETH = 1e18


# ─── Feature computation ──────────────────────────────────────────────────────

def _wei(value_str: str) -> float:
    try:
        return int(value_str or "0") / WEI_TO_ETH
    except (ValueError, TypeError):
        return 0.0


def _compute_features(
    normal_txs: list[dict],
    token_txs: list[dict],
    address: str,
) -> dict[str, float]:
    addr = address.lower()

    sent     = [t for t in normal_txs if t.get("from", "").lower() == addr]
    received = [t for t in normal_txs if t.get("to",   "").lower() == addr]

    sent_vals = [_wei(t.get("value", "0")) for t in sent]
    recv_vals = [_wei(t.get("value", "0")) for t in received]

    total_sent = sum(sent_vals)
    total_recv = sum(recv_vals)
    n_sent     = len(sent)
    n_recv     = len(received)

    all_ts = sorted(int(t.get("timeStamp", "0")) for t in normal_txs if t.get("timeStamp"))
    time_diff_mins = ((all_ts[-1] - all_ts[0]) / 60.0) if len(all_ts) >= 2 else 0.0

    sent_ts = sorted(int(t.get("timeStamp", "0")) for t in sent if t.get("timeStamp"))
    avg_min_between_sent = (
        ((sent_ts[-1] - sent_ts[0]) / max(len(sent_ts) - 1, 1)) / 60.0
        if len(sent_ts) >= 2 else 0.0
    )
    recv_ts = sorted(int(t.get("timeStamp", "0")) for t in received if t.get("timeStamp"))
    avg_min_between_recv = (
        ((recv_ts[-1] - recv_ts[0]) / max(len(recv_ts) - 1, 1)) / 60.0
        if len(recv_ts) >= 2 else 0.0
    )

    unique_sent_to   = len({t.get("to",   "").lower() for t in sent})
    unique_recv_from = len({t.get("from", "").lower() for t in received})

    erc20_sent = [t for t in token_txs if t.get("from", "").lower() == addr]
    erc20_recv = [t for t in token_txs if t.get("to",   "").lower() == addr]
    erc20_sent_vals = [_wei(t.get("value", "0")) for t in erc20_sent]
    erc20_recv_vals = [_wei(t.get("value", "0")) for t in erc20_recv]

    return {
        "Avg min between sent tnx":                          avg_min_between_sent,
        "Avg min between received tnx":                      avg_min_between_recv,
        "Time Diff between first and last (Mins)":           time_diff_mins,
        "Sent tnx":                                          float(n_sent),
        "Received Tnx":                                      float(n_recv),
        "Unique Received From Addresses":                    float(unique_recv_from),
        "Unique Sent To Addresses":                          float(unique_sent_to),
        "min value received":                                min(recv_vals) if recv_vals else 0.0,
        "max value received ":                               max(recv_vals) if recv_vals else 0.0,
        "avg val received":                                  (total_recv / n_recv) if n_recv else 0.0,
        "min val sent":                                      min(sent_vals) if sent_vals else 0.0,
        "max val sent":                                      max(sent_vals) if sent_vals else 0.0,
        "avg val sent":                                      (total_sent / n_sent) if n_sent else 0.0,
        "total transactions (including tnx to create contract": float(len(normal_txs)),
        "total Ether sent":                                  total_sent,
        "total ether received":                              total_recv,
        "total ether balance":                               total_recv - total_sent,
        " Total ERC20 tnxs":                                 float(len(token_txs)),
        " ERC20 total Ether received":                       sum(erc20_recv_vals),
        " ERC20 total ether sent":                           sum(erc20_sent_vals),
        " ERC20 uniq sent addr":                             float(len({t.get("to",   "").lower() for t in erc20_sent})),
        " ERC20 uniq rec addr":                              float(len({t.get("from", "").lower() for t in erc20_recv})),
        " ERC20 avg time between sent tnx":                  0.0,
        " ERC20 avg time between rec tnx":                   0.0,
        " ERC20 min val rec":                                min(erc20_recv_vals) if erc20_recv_vals else 0.0,
        " ERC20 max val rec":                                max(erc20_recv_vals) if erc20_recv_vals else 0.0,
        " ERC20 avg val rec":                                (sum(erc20_recv_vals) / len(erc20_recv_vals)) if erc20_recv_vals else 0.0,
        " ERC20 min val sent":                               min(erc20_sent_vals) if erc20_sent_vals else 0.0,
        " ERC20 max val sent":                               max(erc20_sent_vals) if erc20_sent_vals else 0.0,
        " ERC20 avg val sent":                               (sum(erc20_sent_vals) / len(erc20_sent_vals)) if erc20_sent_vals else 0.0,
        "Number of Created Contracts":                       float(sum(1 for t in normal_txs if not t.get("to"))),
    }


# ─── Hybrid Dirty Score ───────────────────────────────────────────────────────

def _hybrid_score(
    ml_score: float,
    normal_txs: list[dict],
    token_txs: list[dict],
    eth_balance: float,
    addr: str,
) -> dict[str, Any]:
    """
    Combine ML score (40%) with 6 on-chain heuristic red flags (60%).

    Heuristics catch patterns the 2018 Kaggle model misses on modern wallets:
      1. Fan-out       — sends to many unique addresses (layering)
      2. Smurfing      — high value in very short time window
      3. Aggregation   — many senders, few receivers (consolidation)
      4. Pass-through  — high throughput but near-zero balance
      5. Burst         — many txs in short window (bot-like)
      6. Token mixer   — heavy ERC20 + low ETH (obfuscation)
    """
    sent     = [t for t in normal_txs if t.get("from", "").lower() == addr]
    received = [t for t in normal_txs if t.get("to",   "").lower() == addr]

    sent_vals = [_wei(t.get("value", "0")) for t in sent]
    recv_vals = [_wei(t.get("value", "0")) for t in received]

    total_sent   = sum(sent_vals)
    total_recv   = sum(recv_vals)
    n_sent       = len(sent)
    n_recv       = len(received)
    unique_sent_to   = len({t.get("to",   "").lower() for t in sent   if t.get("to")})
    unique_recv_from = len({t.get("from", "").lower() for t in received if t.get("from")})

    all_ts = sorted(int(t.get("timeStamp", "0")) for t in normal_txs if t.get("timeStamp"))
    time_span_hrs = ((all_ts[-1] - all_ts[0]) / 3600.0) if len(all_ts) >= 2 else 9999.0

    heuristic = 0.0

    # ── Red Flag 1: Fan-out (layering pattern) ───────────────────────────────
    if unique_sent_to >= 15:
        heuristic += 0.30
    elif unique_sent_to >= 8:
        heuristic += 0.18
    elif unique_sent_to >= 4:
        heuristic += 0.08

    # ── Red Flag 2: Smurfing (high value, short window) ──────────────────────
    if total_sent >= 5.0 and time_span_hrs <= 48:
        heuristic += 0.25
    elif total_sent >= 1.0 and time_span_hrs <= 24:
        heuristic += 0.15
    elif total_sent >= 0.1 and time_span_hrs <= 6:
        heuristic += 0.10

    # ── Red Flag 3: Aggregation (many in, few out) ────────────────────────────
    if unique_recv_from >= 10 and n_sent <= max(1, n_recv * 0.25):
        heuristic += 0.22
    elif unique_recv_from >= 5 and n_sent <= max(1, n_recv * 0.4):
        heuristic += 0.12

    # ── Red Flag 4: Pass-through (high throughput, empty balance) ────────────
    throughput = total_sent + total_recv
    if throughput >= 1.0 and eth_balance < 0.05:
        heuristic += 0.20
    elif throughput >= 0.5 and eth_balance < 0.01:
        heuristic += 0.15

    # ── Red Flag 5: Burst activity (bot/automated behaviour) ─────────────────
    if len(normal_txs) >= 50 and time_span_hrs <= 72:
        heuristic += 0.18
    elif len(normal_txs) >= 20 and time_span_hrs <= 24:
        heuristic += 0.22
    elif len(normal_txs) >= 10 and time_span_hrs <= 6:
        heuristic += 0.18

    # ── Red Flag 6: Token mixer pattern ──────────────────────────────────────
    if len(token_txs) >= 20 and total_sent < 0.5:
        heuristic += 0.18
    elif len(token_txs) >= 10 and total_sent < 0.1:
        heuristic += 0.12

    heuristic = min(heuristic, 0.95)

    # ── Combine: 40% ML + 60% heuristics ────────────────────────────────────
    combined = min(1.0, round(ml_score * 0.4 + heuristic * 0.6, 4))

    log.debug(
        f"{addr[:10]}… ml={ml_score:.3f} heuristic={heuristic:.3f} combined={combined:.3f} "
        f"| sent_to={unique_sent_to} recv_from={unique_recv_from} "
        f"txs={len(normal_txs)} time_hrs={time_span_hrs:.1f} bal={eth_balance:.4f}"
    )

    return {
        "dirtyScore":     combined,
        "riskLevel":      risk_label(combined),
        "mlScore":        round(ml_score, 4),
        "heuristicScore": round(heuristic, 4),
    }


# ─── BFS Tracer ───────────────────────────────────────────────────────────────

async def trace_wallet(
    root_address: str,
    max_depth: int = MAX_DEPTH,
    max_nodes: int = MAX_NODES,
) -> dict[str, Any]:
    """
    Async BFS money-trail tracer.
    Returns d3-force compatible JSON graph.
    """
    start_time = time.time()
    nodes:   dict[str, dict] = {}
    links:   list[dict]      = []
    visited: set[str]        = set()

    queue: deque[tuple[str, int]] = deque()
    queue.append((root_address.lower(), 0))

    async with EtherscanClient() as client:
        while queue and len(nodes) < max_nodes:

            if time.time() - start_time >= TRACE_TIMEOUT_SECONDS:
                log.warning(f"Trace timeout ({TRACE_TIMEOUT_SECONDS}s) — returning partial graph.")
                break

            # Collect entire current BFS layer
            current_layer: list[tuple[str, int]] = []
            if queue:
                _, current_depth = queue[0]
                while queue and queue[0][1] == current_depth:
                    addr, depth = queue.popleft()
                    if addr not in visited and len(nodes) + len(current_layer) < max_nodes:
                        current_layer.append((addr, depth))
                        visited.add(addr)

            if not current_layer:
                break

            # Async fan-out
            tasks = [
                asyncio.gather(
                    client.get_normal_txs(addr),
                    client.get_token_txs(addr),
                    client.get_eth_balance(addr),
                )
                for addr, _ in current_layer
            ]
            results = await asyncio.gather(*tasks)

            for (addr, depth), (normal_txs, token_txs, eth_balance) in zip(current_layer, results):
                tx_count     = len(normal_txs) + len(token_txs)
                is_supernode = client.is_supernode(tx_count)

                # ML score
                try:
                    features  = _compute_features(normal_txs, token_txs, addr)
                    ml_result = scorer.compute_dirty_score(features)
                    ml_score  = ml_result["dirtyScore"]
                except Exception as exc:
                    log.warning(f"ML scoring failed for {addr}: {exc}")
                    ml_score = 0.0

                # Hybrid score
                scoring = _hybrid_score(ml_score, normal_txs, token_txs, eth_balance, addr)

                sent_vals = [_wei(t.get("value","0")) for t in normal_txs if t.get("from","").lower() == addr]
                recv_vals = [_wei(t.get("value","0")) for t in normal_txs if t.get("to",  "").lower() == addr]

                nodes[addr] = {
                    "id":             addr,
                    "address":        addr,
                    "dirtyScore":     scoring["dirtyScore"],
                    "riskLevel":      scoring["riskLevel"],
                    "mlScore":        scoring["mlScore"],
                    "heuristicScore": scoring["heuristicScore"],
                    "ethBalance":     round(eth_balance, 6),
                    "txCount":        tx_count,
                    "isSupernode":    is_supernode,
                    "depth":          depth,
                    "isRoot":         addr == root_address.lower(),
                    "totalSent":      round(sum(sent_vals), 6),
                    "totalReceived":  round(sum(recv_vals), 6),
                }

                for tx in normal_txs:
                    src = tx.get("from", "").lower()
                    dst = tx.get("to",   "").lower()
                    val = _wei(tx.get("value", "0"))
                    if src and dst:
                        links.append({
                            "source":    src,
                            "target":    dst,
                            "value":     round(val, 6),
                            "txHash":    tx.get("hash", ""),
                            "timestamp": tx.get("timeStamp", ""),
                            "gasUsed":   tx.get("gasUsed", ""),
                        })

                if not is_supernode and depth < max_depth:
                    children = set()
                    for tx in normal_txs:
                        dst = tx.get("to",   "").lower()
                        src = tx.get("from", "").lower()
                        if dst and dst != addr:
                            children.add(dst)
                        if src and src != addr:
                            children.add(src)
                    for child in children:
                        if child not in visited and len(nodes) < max_nodes:
                            queue.append((child, depth + 1))

    elapsed = round(time.time() - start_time, 2)
    log.info(f"Trace done: {len(nodes)} nodes, {len(links)} links in {elapsed}s")

    return {
        "nodes": list(nodes.values()),
        "links": links,
        "meta": {
            "rootAddress":     root_address,
            "nodeCount":       len(nodes),
            "linkCount":       len(links),
            "elapsedSeconds":  elapsed,
            "maxDepthReached": max([n["depth"] for n in nodes.values()] or [0]),
        },
    }