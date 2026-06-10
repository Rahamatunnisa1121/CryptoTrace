# ⛓ CryptoTrace — Blockchain Forensic Dashboard

> Recursive BFS wallet tracer + Random Forest dirty scoring over Ethereum transactions.  
> B.Tech Major Project | Blockchain Forensics

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────┐
│                       ML Pipeline                         │
│  Kaggle Dataset → data_prep → feature_selection → model   │
│  (51 cols)         clean+scale   RF Gini top-15   .pkl   │
└────────────────────────────┬──────────────────────────────┘
                             │ model.pkl + features_15.json
┌────────────────────────────▼──────────────────────────────┐
│                     FastAPI Backend                        │
│  POST /trace → graph_builder (async BFS)                  │
│              → etherscan_client (rate-limited aiohttp)    │
│              → scorer.compute_dirty_score()               │
│              → JSON graph (nodes + links)                 │
└────────────────────────────┬──────────────────────────────┘
                             │ react-force-graph JSON
┌────────────────────────────▼──────────────────────────────┐
│                   React Frontend                           │
│  ForceGraph2D  ·  Node risk coloring  ·  Sidebar panel   │
│  Min-score filter  ·  Depth/node controls  ·  Retrace    │
└───────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
cryptotrace/
├── crypto_trace/
│   ├── config.py              # All env-backed constants
│   ├── data_prep.py           # Step 1: clean + scale Kaggle dataset
│   ├── feature_selection.py   # Step 2: RF Gini importance → top 15 features
│   ├── model.py               # Step 3: train final RF + DirtyScorer
│   ├── etherscan_client.py    # Async Etherscan wrapper (rate-limited)
│   ├── graph_builder.py       # BFS tracer + feature computation
│   └── api.py                 # FastAPI endpoints
├── data/
│   ├── raw/                   # Place Kaggle CSV here
│   └── processed/             # Auto-generated outputs
├── frontend/
│   ├── src/
│   │   ├── App.js             # Main dashboard component
│   │   ├── App.css            # Dark forensic theme
│   │   └── index.js
│   └── package.json
├── tests/
│   ├── test_data_prep.py
│   ├── test_model.py
│   ├── test_graph_builder.py
│   └── test_api.py
├── notebooks/                 # EDA experiments
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## Setup

### 1. Clone & environment

```bash
git clone <your-repo>
cd cryptotrace

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment variables

```bash
cp .env.example .env
# Edit .env — add your Etherscan API key
ETHERSCAN_API_KEY=your_key_here
```

Get a free Etherscan API key at: https://etherscan.io/register

### 3. Dataset

Download the **Ethereum Fraud Detection Dataset** from Kaggle:  
https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset

Place `transaction_dataset.csv` in `data/raw/`.

---

## ML Pipeline (run once)

```bash
# Step 1 — Clean, scale, save preprocessing pipeline
python -m crypto_trace.data_prep

# Step 2 — RF Gini importance → select top 15 features
python -m crypto_trace.feature_selection

# Step 3 — Train final model on 15 features, save model.pkl
python -m crypto_trace.model
```

Outputs in `data/processed/`:
- `optimized_transaction_dataset.csv` — scaled 51-feature dataset
- `optimized_15_features.csv`         — reduced 15-feature training set
- `features_15.json`                  — selected feature names (used at runtime)
- `model.pkl`                         — final RF pipeline
- `feature_importance.png`            — Gini importance bar chart

---

## Run the Backend

```bash
uvicorn crypto_trace.api:app --reload --host 0.0.0.0 --port 8000
```

API docs (Swagger UI): http://localhost:8000/docs

### Endpoints

| Method | Path      | Description                              |
|--------|-----------|------------------------------------------|
| GET    | `/health` | Liveness check + model loaded status     |
| POST   | `/trace`  | BFS trace from address → JSON graph      |
| GET    | `/score`  | Quick dirty score for a single address   |

**POST /trace request body:**
```json
{
  "address": "0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe",
  "maxDepth": 5,
  "maxNodes": 200
}
```

---

## Run the Frontend

```bash
cd frontend
npm install
npm start
```

Opens at http://localhost:3000

---

## Run Tests

```bash
# Unit tests (no API key or model needed)
pytest tests/ -v

# With integration tests (requires model.pkl + Etherscan key)
pytest tests/ -v -m integration
```

---

## Key Technical Decisions

### Why Random Forest for feature selection?
Random Forest computes **Gini Importance** (Mean Decrease in Impurity) for every feature across all decision trees. Features that do not meaningfully separate fraud from normal transactions score near zero and are dropped.

This is called **RF-RFE (Recursive Feature Elimination)** in academic literature:
- Zhou et al. (2025) — *Enhancing Fraud Detection in the Ethereum Blockchain Using Ensemble Learning*, PeerJ Computer Science Vol. 11
- Liu et al. — *Big Data Cleaning Based on Improved CLOF and Random Forest*, IEEE Xplore

Reducing from 51 → 15 features cuts inference time ~70%, which is critical for the ≤30s BFS trace requirement.

### Why asyncio.Semaphore for Etherscan?
Etherscan free tier = **5 req/s**. Without a semaphore, `asyncio.gather` fires all concurrent requests simultaneously, causing `429 Too Many Requests` errors. The semaphore caps concurrency at exactly 5 without blocking the event loop.

### Why cap "Supernode" wallets?
Exchange wallets (Binance, Coinbase, etc.) have 100,000+ outgoing transactions. Recursing into them would explode the graph to millions of nodes. Any wallet with `tx_count ≥ 1000` is flagged as a supernode, added as a leaf, and not recursed further.

### Feature schema alignment
The 15 features fed to the live model must **exactly match** the Kaggle training features — same names, same order, same scale. `graph_builder._compute_features()` computes these from the last `N` Etherscan transactions (rolling window). This is documented as a conscious performance–accuracy trade-off.

---

## Performance Targets

| Metric        | Target         | Strategy                                    |
|---------------|----------------|---------------------------------------------|
| Trace latency | ≤ 30 seconds   | Async fan-out + node/depth caps             |
| Node count    | ≤ 500          | BFS node limit enforced in graph_builder    |
| Graph render  | ≥ 30 FPS       | WebGL via react-force-graph (Three.js)      |
| API security  | TLS 1.2+       | Etherscan uses HTTPS; deploy with nginx/TLS |

---

## Known Limitations & Future Work

- **Rolling window**: Features are computed from the last 100 txs only (not full history) to stay within the time budget.
- **ERC-20 features**: Some ERC-20 time-delta features are zeroed out in the live path (they require significantly more API calls).
- **No persistent caching**: The in-memory cache is per-request. A Redis cache layer would prevent redundant Etherscan calls across sessions.
- **Frontend**: Currently single-page; adding a history panel and graph export (PNG/JSON) would be valuable.
