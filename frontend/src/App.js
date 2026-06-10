import React, { useState, useRef, useCallback, useEffect } from "react";
import * as d3 from "d3";
import axios from "axios";
import "./App.css";

const API = "http://localhost:8000";

// ─── Helpers ─────────────────────────────────────────────────────────────────
const riskColor = (score, isRoot) => {
  if (isRoot) return "#f4d03f";
  if (score >= 0.65) return "#e74c3c";
  if (score >= 0.35) return "#e67e22";
  return "#27ae60";
};

const riskBadgeClass = (level) =>
  ({ HIGH: "badge-high", MEDIUM: "badge-med", LOW: "badge-low" })[level] ||
  "badge-unknown";

const shortenAddr = (addr) =>
  addr ? `${addr.slice(0, 6)}…${addr.slice(-4)}` : "—";

const formatEth = (val) =>
  val !== undefined && val !== null ? `${Number(val).toFixed(4)} ETH` : "—";

function nodeRadius(n) {
  return Math.max(
    5,
    Math.min(20, (n.txCount || 1) / 15 + (n.dirtyScore || 0) * 8),
  );
}

// ─── D3 Force Graph Canvas ────────────────────────────────────────────────────
function ForceGraph({ nodes, links, onNodeClick }) {
  const canvasRef = useRef(null);
  const simRef = useRef(null);
  const transformRef = useRef({ x: 0, y: 0, k: 1 });

  useEffect(() => {
    if (!nodes.length) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    const simNodes = nodes.map((n) => ({ ...n }));
    const nodeById = Object.fromEntries(simNodes.map((n) => [n.id, n]));
    const simLinks = links
      .map((l) => ({
        ...l,
        source: nodeById[l.source] || l.source,
        target: nodeById[l.target] || l.target,
      }))
      .filter((l) => l.source && l.target);

    if (simRef.current) simRef.current.stop();

    const sim = d3
      .forceSimulation(simNodes)
      .force(
        "link",
        d3
          .forceLink(simLinks)
          .id((d) => d.id)
          .distance(60)
          .strength(0.5),
      )
      .force("charge", d3.forceManyBody().strength(-120))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force("collision", d3.forceCollide().radius(12));

    simRef.current = sim;

    const draw = () => {
      ctx.save();
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = "#0a0e1a";
      ctx.fillRect(0, 0, W, H);

      const { x, y, k } = transformRef.current;
      ctx.translate(x, y);
      ctx.scale(k, k);

      // Links
      simLinks.forEach((l) => {
        if (!l.source.x || !l.target.x) return;
        ctx.beginPath();
        ctx.moveTo(l.source.x, l.source.y);
        ctx.lineTo(l.target.x, l.target.y);
        ctx.strokeStyle = "rgba(100,140,200,0.25)";
        ctx.lineWidth = Math.max(0.5, Math.log10((l.value || 0.001) + 1));
        ctx.stroke();

        // Arrowhead
        const angle = Math.atan2(
          l.target.y - l.source.y,
          l.target.x - l.source.x,
        );
        const r = nodeRadius(l.target);
        const ax = l.target.x - Math.cos(angle) * (r + 4);
        const ay = l.target.y - Math.sin(angle) * (r + 4);
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(
          ax - 6 * Math.cos(angle - 0.4),
          ay - 6 * Math.sin(angle - 0.4),
        );
        ctx.lineTo(
          ax - 6 * Math.cos(angle + 0.4),
          ay - 6 * Math.sin(angle + 0.4),
        );
        ctx.closePath();
        ctx.fillStyle = "rgba(100,140,200,0.4)";
        ctx.fill();
      });

      // Nodes
      simNodes.forEach((n) => {
        if (!n.x) return;
        const r = nodeRadius(n);
        const color = riskColor(n.dirtyScore, n.isRoot);

        if (n.dirtyScore >= 0.65) {
          ctx.beginPath();
          ctx.arc(n.x, n.y, r + 6, 0, 2 * Math.PI);
          ctx.fillStyle = "rgba(231,76,60,0.15)";
          ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(n.x, n.y, r, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();

        if (n.isRoot) {
          ctx.beginPath();
          ctx.arc(n.x, n.y, r + 3, 0, 2 * Math.PI);
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2;
          ctx.stroke();
        }

        if (n.isSupernode) {
          ctx.beginPath();
          ctx.arc(n.x, n.y, r + 3, 0, 2 * Math.PI);
          ctx.strokeStyle = "#9b59b6";
          ctx.lineWidth = 2;
          ctx.setLineDash([4, 3]);
          ctx.stroke();
          ctx.setLineDash([]);
        }

        if (k > 0.8) {
          ctx.font = `${Math.max(8, 10 / k)}px monospace`;
          ctx.fillStyle = "rgba(200,216,240,0.85)";
          ctx.textAlign = "center";
          ctx.fillText(shortenAddr(n.address), n.x, n.y + r + 10 / k);
        }
      });

      ctx.restore();
    };

    sim.on("tick", draw);

    // Zoom & pan
    const zoom = d3
      .zoom()
      .scaleExtent([0.1, 8])
      .on("zoom", (e) => {
        transformRef.current = {
          x: e.transform.x,
          y: e.transform.y,
          k: e.transform.k,
        };
        draw();
      });
    d3.select(canvas).call(zoom);

    // Click to select node
    const handleClick = (e) => {
      const rect = canvas.getBoundingClientRect();
      const mx =
        (e.clientX - rect.left - transformRef.current.x) /
        transformRef.current.k;
      const my =
        (e.clientY - rect.top - transformRef.current.y) /
        transformRef.current.k;
      const hit = simNodes.find((n) => {
        const dx = n.x - mx;
        const dy = n.y - my;
        return Math.sqrt(dx * dx + dy * dy) <= nodeRadius(n) + 4;
      });
      if (hit) onNodeClick(hit);
    };
    canvas.addEventListener("click", handleClick);

    return () => {
      sim.stop();
      canvas.removeEventListener("click", handleClick);
      d3.select(canvas).on(".zoom", null);
    };
  }, [nodes, links, onNodeClick]);

  return (
    <canvas
      ref={canvasRef}
      width={window.innerWidth - 320}
      height={window.innerHeight - 48}
      style={{ display: "block", background: "#0a0e1a" }}
    />
  );
}

// ─── NodePanel ───────────────────────────────────────────────────────────────
function NodePanel({ node, onRetrace }) {
  if (!node) return null;
  return (
    <div className="node-panel">
      <div className="panel-title">
        <span className="panel-icon">🔍</span> Wallet Details
        {node.isSupernode && <span className="supernode-badge">EXCHANGE</span>}
      </div>
      <div className="panel-addr" title={node.address}>
        {node.address}
      </div>
      <div className="panel-row">
        <span>Dirty Score</span>
        <span>
          <span className={`badge ${riskBadgeClass(node.riskLevel)}`}>
            {node.riskLevel}
          </span>{" "}
          {(node.dirtyScore * 100).toFixed(1)}%
        </span>
      </div>
      <div className="panel-score-bar">
        <div
          className="panel-score-fill"
          style={{
            width: `${node.dirtyScore * 100}%`,
            background: riskColor(node.dirtyScore, false),
          }}
        />
      </div>
      <div className="panel-row">
        <span>ETH Balance</span>
        <span>{formatEth(node.ethBalance)}</span>
      </div>
      <div className="panel-row">
        <span>Total Sent</span>
        <span>{formatEth(node.totalSent)}</span>
      </div>
      <div className="panel-row">
        <span>Total Received</span>
        <span>{formatEth(node.totalReceived)}</span>
      </div>
      <div className="panel-row">
        <span>Tx Count</span>
        <span>{node.txCount?.toLocaleString()}</span>
      </div>
      <div className="panel-row">
        <span>Hop Depth</span>
        <span>{node.depth}</span>
      </div>
      <button
        className="retrace-btn"
        onClick={() => onRetrace(node.address)}
        disabled={node.isSupernode}
      >
        Trace from this wallet →
      </button>
    </div>
  );
}

// ─── Legend ──────────────────────────────────────────────────────────────────
function Legend() {
  return (
    <div className="legend">
      <div className="legend-title">Risk Legend</div>
      {[
        { color: "#f4d03f", label: "Root Wallet" },
        { color: "#27ae60", label: "Low Risk (<35%)" },
        { color: "#e67e22", label: "Medium Risk (35–65%)" },
        { color: "#e74c3c", label: "High Risk (>65%)" },
        { color: "#9b59b6", label: "Exchange / Supernode" },
      ].map(({ color, label }) => (
        <div key={label} className="legend-row">
          <span className="dot" style={{ background: color }} /> {label}
        </div>
      ))}
    </div>
  );
}

// ─── MetaBar ─────────────────────────────────────────────────────────────────
function MetaBar({ meta, loading }) {
  if (loading)
    return <div className="meta-bar pulsing">Tracing… please wait</div>;
  if (!meta) return null;
  return (
    <div className="meta-bar">
      <span>🔗 {meta.nodeCount} wallets</span>
      <span>↔ {meta.linkCount} txs</span>
      <span>⏱ {meta.elapsedSeconds}s</span>
      <span>📏 Max depth {meta.maxDepthReached}</span>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [address, setAddress] = useState("");
  const [maxDepth, setMaxDepth] = useState(5);
  const [maxNodes, setMaxNodes] = useState(200);
  const [minScore, setMinScore] = useState(0);
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [filteredGraph, setFilteredGraph] = useState({ nodes: [], links: [] });
  const [meta, setMeta] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedNode, setSelectedNode] = useState(null);

  useEffect(() => {
    const nodeIds = new Set(
      graphData.nodes
        .filter((n) => n.dirtyScore >= minScore || n.isRoot)
        .map((n) => n.id),
    );
    setFilteredGraph({
      nodes: graphData.nodes.filter((n) => nodeIds.has(n.id)),
      links: graphData.links.filter(
        (l) =>
          nodeIds.has(l.source?.id || l.source) &&
          nodeIds.has(l.target?.id || l.target),
      ),
    });
  }, [graphData, minScore]);

  const doTrace = useCallback(
    async (addr) => {
      const a = (addr || address).trim().toLowerCase();
      if (!a.startsWith("0x") || a.length !== 42) {
        setError(
          "Invalid Ethereum address. Must start with 0x and be 42 chars.",
        );
        return;
      }
      setError("");
      setLoading(true);
      setSelectedNode(null);
      setGraphData({ nodes: [], links: [] });
      setMeta(null);
      try {
        const { data } = await axios.post(`${API}/trace`, {
          address: a,
          maxDepth,
          maxNodes,
        });
        setGraphData({ nodes: data.nodes, links: data.links });
        setMeta(data.meta);
      } catch (err) {
        setError(err?.response?.data?.detail || err.message || "Trace failed.");
      } finally {
        setLoading(false);
      }
    },
    [address, maxDepth, maxNodes],
  );

  const handleNodeClick = useCallback((n) => setSelectedNode(n), []);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="logo">
          <span className="logo-icon">⛓</span>
          <span className="logo-text">CryptoTrace</span>
        </div>

        <div className="sidebar-label">Wallet Address</div>
        <input
          className="addr-input"
          value={address}
          onChange={(e) => setAddress(e.target.value)}
          placeholder="0x…"
          onKeyDown={(e) => e.key === "Enter" && doTrace()}
        />

        <div className="controls-row">
          <label>
            <span>Depth</span>
            <input
              type="number"
              min={1}
              max={10}
              value={maxDepth}
              onChange={(e) => setMaxDepth(Number(e.target.value))}
            />
          </label>
          <label>
            <span>Max Nodes</span>
            <input
              type="number"
              min={10}
              max={500}
              step={10}
              value={maxNodes}
              onChange={(e) => setMaxNodes(Number(e.target.value))}
            />
          </label>
        </div>

        <button
          className="trace-btn"
          onClick={() => doTrace()}
          disabled={loading}
        >
          {loading ? "Tracing…" : "▶ Trace Wallet"}
        </button>

        {error && <div className="error-msg">{error}</div>}

        <div className="divider" />
        <div className="sidebar-label">Filter by Min Risk Score</div>
        <input
          type="range"
          min={0}
          max={0.9}
          step={0.05}
          value={minScore}
          onChange={(e) => setMinScore(Number(e.target.value))}
        />
        <div className="filter-val">{(minScore * 100).toFixed(0)}%</div>

        <div className="divider" />
        <Legend />
        <div className="divider" />
        <NodePanel node={selectedNode} onRetrace={doTrace} />
      </aside>

      <main className="canvas-area">
        <MetaBar meta={meta} loading={loading} />

        {filteredGraph.nodes.length === 0 && !loading && (
          <div className="empty-state">
            <div className="empty-icon">🔗</div>
            <div>
              Enter an Ethereum wallet address and press{" "}
              <strong>Trace Wallet</strong>
            </div>
            <div className="empty-sub">
              The forensic engine will map up to {maxNodes} connected wallets
              across {maxDepth} hops and score each one for suspicious activity.
            </div>
          </div>
        )}

        <ForceGraph
          nodes={filteredGraph.nodes}
          links={filteredGraph.links}
          onNodeClick={handleNodeClick}
        />
      </main>
    </div>
  );
}
