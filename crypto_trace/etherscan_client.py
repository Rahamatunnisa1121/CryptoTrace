"""
etherscan_client.py
────────────────────
Async wrapper around the Etherscan API.

Key design decisions:
  • asyncio.Semaphore(MAX_CALLS_PER_SEC) prevents 429 errors on free tier (5 req/s)
  • In-memory LRU-style cache keyed by (address, action) avoids duplicate API hits
    within a single trace session
  • "Rolling window" cap: fetch only the last MAX_TX_PER_WALLET transactions per
    address so the BFS stays within the 30-second budget
"""
import asyncio
import logging
import time
from typing import Any

import aiohttp

from .config import (
    ETHERSCAN_API_KEY,
    ETHERSCAN_BASE_URL,
    ETHERSCAN_MAX_CALLS_PER_SEC,
    ETHERSCAN_MAX_TX_PER_WALLET,
    SUPERNODE_TX_THRESHOLD,
)

log = logging.getLogger(__name__)

# Global semaphore: shared across all concurrent callers
_semaphore: asyncio.Semaphore | None = None


def get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(ETHERSCAN_MAX_CALLS_PER_SEC)
    return _semaphore


class EtherscanClient:
    """
    Async Etherscan client.

    Usage (within an async context):
        async with EtherscanClient() as client:
            txs = await client.get_normal_txs("0xABCD...")
    """

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        # Simple in-memory request cache for a single trace session
        self._cache: dict[str, Any] = {}

    async def __aenter__(self) -> "EtherscanClient":
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._session:
            await self._session.close()

    # ─── Core request ────────────────────────────────────────────────────────

    async def _get(self, params: dict[str, str]) -> dict[str, Any]:
        """Rate-limited GET with retries (up to 3)."""
        cache_key = str(sorted(params.items()))
        if cache_key in self._cache:
            return self._cache[cache_key]

        params["apikey"] = ETHERSCAN_API_KEY
        params["chainid"] = "1"   # Ethereum mainnet
        sem = get_semaphore()

        for attempt in range(3):
            async with sem:
                try:
                    assert self._session is not None
                    async with self._session.get(
                        ETHERSCAN_BASE_URL, params=params, ssl=True
                    ) as resp:
                        resp.raise_for_status()
                        data: dict = await resp.json()

                    if data.get("status") == "1":
                        self._cache[cache_key] = data
                        return data

                    # Etherscan returns status=0 for "no transactions"
                    if data.get("message") in ("No transactions found", "No records found"):
                        empty: dict = {"status": "0", "result": [], "message": "No tx"}
                        self._cache[cache_key] = empty
                        return empty

                    log.warning(f"Etherscan API error: {data.get('message')} | params={params}")
                    return {"status": "0", "result": [], "message": data.get("message", "")}

                except aiohttp.ClientError as exc:
                    log.warning(f"HTTP error (attempt {attempt + 1}): {exc}")
                    if attempt < 2:
                        await asyncio.sleep(1.0 * (attempt + 1))

        return {"status": "0", "result": [], "message": "Max retries exceeded"}

    # ─── Public helpers ──────────────────────────────────────────────────────

    async def get_normal_txs(self, address: str) -> list[dict]:
        """
        Fetch the last MAX_TX_PER_WALLET normal (ETH transfer) transactions
        for an address. Sorted newest-first for the rolling window.
        """
        params = {
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": "0",
            "endblock": "99999999",
            "page": "1",
            "offset": str(ETHERSCAN_MAX_TX_PER_WALLET),
            "sort": "desc",
        }
        data = await self._get(params)
        return data.get("result", []) or []

    async def get_internal_txs(self, address: str) -> list[dict]:
        """Fetch internal (contract-call) transactions."""
        params = {
            "module": "account",
            "action": "txlistinternal",
            "address": address,
            "startblock": "0",
            "endblock": "99999999",
            "page": "1",
            "offset": str(ETHERSCAN_MAX_TX_PER_WALLET),
            "sort": "desc",
        }
        data = await self._get(params)
        return data.get("result", []) or []

    async def get_token_txs(self, address: str) -> list[dict]:
        """Fetch ERC-20 token transfer transactions."""
        params = {
            "module": "account",
            "action": "tokentx",
            "address": address,
            "startblock": "0",
            "endblock": "99999999",
            "page": "1",
            "offset": str(ETHERSCAN_MAX_TX_PER_WALLET),
            "sort": "desc",
        }
        data = await self._get(params)
        return data.get("result", []) or []

    async def get_eth_balance(self, address: str) -> float:
        """Get ETH balance in ether."""
        params = {
            "module": "account",
            "action": "balance",
            "address": address,
            "tag": "latest",
        }
        data = await self._get(params)
        wei = int(data.get("result", "0") or "0")
        return wei / 1e18

    def is_supernode(self, tx_count: int) -> bool:
        """
        Detect high-volume exchange wallets that would explode the graph.
        These are NOT recursed into but ARE added as leaf nodes.
        """
        return tx_count >= SUPERNODE_TX_THRESHOLD
