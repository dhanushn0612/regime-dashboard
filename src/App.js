import { useState, useEffect } from "react";
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";

// ── REGIME CONFIG ─────────────────────────────────────────────────────
const REGIME_CONFIG = {
  "Strong Bull":    { color: "#00ff87", bg: "rgba(0,255,135,0.12)" },
  "Mild Bull":      { color: "#7dffb3", bg: "rgba(125,255,179,0.10)" },
  "Neutral/Choppy": { color: "#ffd166", bg: "rgba(255,209,102,0.10)" },
  "Mild Bear":      { color: "#ff6b6b", bg: "rgba(255,107,107,0.10)" },
  "Strong Bear":    { color: "#ff2d55", bg: "rgba(255,45,85,0.12)" },
};

const scoreColor = (s) =>
  s > 75 ? "#00ff87" : s > 55 ? "#7dffb3" : s > 40 ? "#ffd166" : s > 20 ? "#ff6b6b" : "#ff2d55";

// ── SUBCOMPONENTS ─────────────────────────────────────────────────────
const GaugeArc = ({ score, label, size = 110 }) => {
  const r = 46, cx = size / 2, cy = size / 2 + 8;
  const toXY = (deg, rad) => ({
    x: cx + rad * Math.cos((deg * Math.PI) / 180),
    y: cy + rad * Math.sin((deg * Math.PI) / 180),
  });
  const arcPath = (from, to, ri, ro) => {
    const s1 = toXY(from, ro), e1 = toXY(to, ro);
    const s2 = toXY(to, ri), e2 = toXY(from, ri);
    const lg = to - from > 180 ? 1 : 0;
    return `M${s1.x} ${s1.y} A${ro} ${ro} 0 ${lg} 1 ${e1.x} ${e1.y} L${s2.x} ${s2.y} A${ri} ${ri} 0 ${lg} 0 ${e2.x} ${e2.y}Z`;
  };
  const angle = -210 + (score / 100) * 240;
  const needle = toXY(angle, r - 8);
  const color = scoreColor(score);
  return (
    <svg width={size} height={size * 0.85} viewBox={`0 0 ${size} ${size * 0.85}`}>
      <path d={arcPath(-210, 30, 36, 50)} fill="#1a1a2e" />
      <path d={arcPath(-210, angle, 36, 50)} fill={color} opacity={0.85} />
      <line x1={cx} y1={cy} x2={needle.x} y2={needle.y} stroke={color} strokeWidth="2.5" strokeLinecap="round" />
      <circle cx={cx} cy={cy} r="4" fill={color} />
      <text x={cx} y={cy + 16} textAnchor="middle" fill={color} fontSize="13" fontWeight="700" fontFamily="monospace">{score}</text>
      <text x={cx} y={cy + 28} textAnchor="middle" fill="#888" fontSize="7" fontFamily="monospace" letterSpacing="1">{label}</text>
    </svg>
  );
};

const RegimeBadge = ({ label }) => {
  const cfg = REGIME_CONFIG[label] || { color: "#aaa", bg: "transparent" };
  return (
    <span style={{
      background: cfg.bg, border: `1px solid ${cfg.color}40`,
      color: cfg.color, padding: "3px 12px", borderRadius: "4px",
      fontSize: "11px", letterSpacing: "1.5px", textTransform: "uppercase", fontWeight: "600",
    }}>{label}</span>
  );
};

const DimBar = ({ label, score, weight }) => {
  const color = scoreColor(score);
  return (
    <div style={{ marginBottom: "12px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
        <span style={{ color: "#aaa", fontSize: "11px", letterSpacing: "1px" }}>
          {label} <span style={{ color: "#555", fontSize: "10px" }}>({(weight * 100).toFixed(0)}%)</span>
        </span>
        <span style={{ color, fontSize: "13px", fontWeight: "700" }}>{score}</span>
      </div>
      <div style={{ height: "6px", background: "#1a1a2e", borderRadius: "3px", overflow: "hidden" }}>
        <div style={{ width: `${score}%`, height: "100%", background: color, borderRadius: "3px", boxShadow: `0 0 8px ${color}80` }} />
      </div>
    </div>
  );
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div style={{ background: "#0d0d1a", border: "1px solid #2a2a4a", padding: "10px 14px", borderRadius: "6px", fontSize: "11px" }}>
      <div style={{ color: "#666", marginBottom: "6px" }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, marginBottom: "2px" }}>
          {p.name}: <span style={{ color: "#fff" }}>{typeof p.value === 'number' ? p.value.toFixed(1) : p.value}</span>
        </div>
      ))}
      {d?.regime_label && <div style={{ color: REGIME_CONFIG[d.regime_label]?.color || "#aaa", marginTop: "4px" }}>{d.regime_label}</div>}
    </div>
  );
};

// ── MAIN APP ──────────────────────────────────────────────────────────
export default function App() {
  const [history, setHistory]   = useState([]);
  const [current, setCurrent]   = useState(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState(null);
  const [activeTab, setActiveTab] = useState("overview");

  useEffect(() => {
    Promise.all([
      fetch('/regime_current.json').then(r => r.json()),
      fetch('/regime_history.json').then(r => r.json()),
    ])
      .then(([curr, hist]) => {
        setCurrent(curr);
        setHistory(hist);
        setLoading(false);
      })
      .catch(e => {
        setError("Could not load regime data. Run the classifier first.");
        setLoading(false);
      });
  }, []);

  if (loading) return (
    <div style={{ background: "#07071a", minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", color: "#555", fontFamily: "monospace" }}>
      Loading regime data...
    </div>
  );

  if (error || !current) return (
    <div style={{ background: "#07071a", minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", color: "#ff6b6b", fontFamily: "monospace", padding: "32px", textAlign: "center" }}>
      {error || "No data available."}<br /><br />
      <span style={{ color: "#555", fontSize: "12px" }}>Run: python data_pipeline/run_classifier.py</span>
    </div>
  );

  const cfg = REGIME_CONFIG[current.regime_label] || { color: "#aaa", bg: "transparent" };
  const tabs = ["overview", "dimensions", "history", "signals"];
  const regimeDist = history.reduce((acc, d) => { acc[d.regime_label] = (acc[d.regime_label] || 0) + 1; return acc; }, {});

  return (
    <div style={{ background: "#07071a", minHeight: "100vh", color: "#e0e0e0", fontFamily: "monospace" }}>

      {/* Header */}
      <div style={{ borderBottom: "1px solid #1a1a2e", padding: "18px 32px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div>
          <div style={{ fontSize: "18px", fontWeight: "800", letterSpacing: "2px", color: "#fff" }}>
            REGIME<span style={{ color: cfg.color }}>.</span>CLASSIFIER
          </div>
          <div style={{ fontSize: "10px", color: "#555", letterSpacing: "2px", marginTop: "2px" }}>INDIAN EQUITY · MULTI-DIMENSIONAL · NSE/BSE</div>
        </div>
        <div style={{ textAlign: "right" }}>
          <RegimeBadge label={current.regime_label} />
          <div style={{ fontSize: "10px", color: "#555", marginTop: "4px" }}>AS OF {current.date}</div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ borderBottom: "1px solid #1a1a2e", padding: "0 32px", display: "flex" }}>
        {tabs.map(tab => (
          <button key={tab} onClick={() => setActiveTab(tab)} style={{
            background: "none", border: "none", cursor: "pointer", padding: "12px 18px",
            fontSize: "10px", letterSpacing: "1.5px", textTransform: "uppercase",
            color: activeTab === tab ? cfg.color : "#555",
            borderBottom: activeTab === tab ? `2px solid ${cfg.color}` : "2px solid transparent",
            marginBottom: "-1px",
          }}>{tab}</button>
        ))}
      </div>

      <div style={{ padding: "24px 32px" }}>

        {activeTab === "overview" && (
          <div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr 1.4fr", gap: "12px", marginBottom: "24px" }}>
              {[
                { label: "TREND",      score: current.trend_score,      weight: 0.30 },
                { label: "VOLATILITY", score: current.volatility_score, weight: 0.25 },
                { label: "BREADTH",    score: current.breadth_score,    weight: 0.25 },
                { label: "FLOW",       score: current.flow_score,       weight: 0.20 },
              ].map(d => (
                <div key={d.label} style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "16px", textAlign: "center" }}>
                  <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "8px" }}>{d.label}</div>
                  <GaugeArc score={d.score} label={d.label} />
                  <div style={{ fontSize: "9px", color: "#555", marginTop: "4px" }}>weight {(d.weight * 100).toFixed(0)}%</div>
                </div>
              ))}
              <div style={{ background: cfg.bg, border: `1px solid ${cfg.color}30`, borderRadius: "8px", padding: "16px", textAlign: "center", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
                <div style={{ fontSize: "9px", color: "#888", letterSpacing: "2px", marginBottom: "8px" }}>COMPOSITE</div>
                <div style={{ fontSize: "52px", fontWeight: "700", color: cfg.color, lineHeight: 1 }}>{current.composite_score}</div>
                <div style={{ fontSize: "9px", color: cfg.color, marginTop: "6px", opacity: 0.7 }}>/100</div>
                <div style={{ marginTop: "12px" }}><RegimeBadge label={current.regime_label} /></div>
                <div style={{ fontSize: "10px", color: "#777", marginTop: "12px" }}>Nifty {current.nifty_price?.toLocaleString()} · VIX {current.india_vix}</div>
              </div>
            </div>

            <div style={{ background: cfg.bg, border: `1px solid ${cfg.color}25`, borderRadius: "8px", padding: "14px 20px", marginBottom: "24px", display: "flex", alignItems: "center", gap: "12px" }}>
              <div style={{ fontSize: "9px", color: cfg.color, letterSpacing: "2px", whiteSpace: "nowrap" }}>RECOMMENDED ACTION</div>
              <div style={{ width: "1px", height: "20px", background: cfg.color, opacity: 0.3 }} />
              <div style={{ fontSize: "12px", color: "#ddd" }}>{current.recommended_action}</div>
            </div>

            {history.length > 0 && (
              <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "16px" }}>
                <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "14px" }}>COMPOSITE SCORE HISTORY</div>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={history}>
                    <defs>
                      <linearGradient id="cg" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={cfg.color} stopOpacity={0.3} />
                        <stop offset="95%" stopColor={cfg.color} stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="#1a1a2e" strokeDasharray="3 3" />
                    <XAxis dataKey="date" tick={{ fill: "#555", fontSize: 9 }} tickFormatter={d => d?.slice(2, 7)} />
                    <YAxis domain={[0, 100]} tick={{ fill: "#555", fontSize: 9 }} />
                    <Tooltip content={<CustomTooltip />} />
                    {[75, 55, 40, 20].map(v => <ReferenceLine key={v} y={v} stroke="#ffffff15" strokeDasharray="4 4" />)}
                    <Area type="monotone" dataKey="composite_score" stroke={cfg.color} fill="url(#cg)" strokeWidth={2} dot={false} name="Composite" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}

        {activeTab === "dimensions" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
            {[
              { label: "TREND",             score: current.trend_score,      dataKey: "trend_score",      color: "#00d4ff", weight: 0.30, desc: "SMA relationships, rate of change, 52-week proximity" },
              { label: "VOLATILITY",        score: current.volatility_score, dataKey: "volatility_score", color: "#ff6b6b", weight: 0.25, desc: "India VIX level, VIX change, realized vol vs historical" },
              { label: "BREADTH",           score: current.breadth_score,    dataKey: "breadth_score",    color: "#ffd166", weight: 0.25, desc: "% stocks above 50/200 DMA, advance/decline ratio" },
              { label: "FLOW & SENTIMENT",  score: current.flow_score,       dataKey: "flow_score",       color: "#bf5af2", weight: 0.20, desc: "FII/DII flows, VIX slope sentiment proxy" },
            ].map(dim => (
              <div key={dim.label} style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "18px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
                  <div>
                    <div style={{ fontSize: "11px", color: dim.color, letterSpacing: "2px", fontWeight: "600" }}>{dim.label}</div>
                    <div style={{ fontSize: "9px", color: "#555", marginTop: "3px" }}>{dim.desc}</div>
                  </div>
                  <div style={{ fontSize: "28px", color: dim.color, fontWeight: "700" }}>{dim.score}</div>
                </div>
                <div style={{ height: "8px", background: "#1a1a2e", borderRadius: "4px", overflow: "hidden", marginBottom: "6px" }}>
                  <div style={{ width: `${dim.score}%`, height: "100%", background: dim.color, boxShadow: `0 0 10px ${dim.color}60` }} />
                </div>
                <div style={{ fontSize: "9px", color: "#444", marginBottom: "14px" }}>weight {(dim.weight * 100).toFixed(0)}%</div>
                {history.length > 0 && (
                  <ResponsiveContainer width="100%" height={90}>
                    <LineChart data={history.slice(-52)}>
                      <XAxis hide /><YAxis hide domain={[0, 100]} />
                      <Tooltip content={<CustomTooltip />} />
                      <Line type="monotone" dataKey={dim.dataKey} stroke={dim.color} strokeWidth={1.5} dot={false} name={dim.label} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
            ))}
          </div>
        )}

        {activeTab === "history" && history.length > 0 && (
          <div>
            <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "16px", marginBottom: "16px" }}>
              <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "14px" }}>NIFTY 50 vs COMPOSITE REGIME</div>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={history}>
                  <CartesianGrid stroke="#1a1a2e" strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fill: "#555", fontSize: 9 }} tickFormatter={d => d?.slice(2, 7)} />
                  <YAxis yAxisId="p" tick={{ fill: "#555", fontSize: 9 }} tickFormatter={v => (v / 1000).toFixed(0) + "k"} />
                  <YAxis yAxisId="s" orientation="right" domain={[0, 100]} tick={{ fill: "#555", fontSize: 9 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Line yAxisId="p" type="monotone" dataKey="nifty_price" stroke="#ffffff40" strokeWidth={1.5} dot={false} name="Nifty" />
                  <Line yAxisId="s" type="monotone" dataKey="composite_score" stroke={cfg.color} strokeWidth={2} dot={false} name="Composite" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "16px" }}>
              <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "14px" }}>REGIME TIMELINE</div>
              <div style={{ display: "flex", height: "32px", borderRadius: "4px", overflow: "hidden", gap: "1px" }}>
                {history.map((d, i) => (
                  <div key={i} title={`${d.date}: ${d.regime_label} (${d.composite_score})`}
                    style={{ flex: 1, background: REGIME_CONFIG[d.regime_label]?.color || "#555", opacity: 0.75 }} />
                ))}
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "9px", color: "#555", marginTop: "6px" }}>
                <span>{history[0]?.date}</span><span>{history[history.length - 1]?.date}</span>
              </div>
              <div style={{ marginTop: "20px", display: "flex", gap: "8px", flexWrap: "wrap" }}>
                {Object.entries(regimeDist).sort((a, b) => b[1] - a[1]).map(([label, count]) => {
                  const pct = (count / history.length * 100).toFixed(1);
                  const color = REGIME_CONFIG[label]?.color || "#aaa";
                  return (
                    <div key={label} style={{ background: REGIME_CONFIG[label]?.bg, border: `1px solid ${color}30`, borderRadius: "6px", padding: "8px 12px" }}>
                      <div style={{ fontSize: "9px", color, letterSpacing: "1px" }}>{label}</div>
                      <div style={{ fontSize: "18px", color, fontWeight: "700" }}>{pct}%</div>
                      <div style={{ fontSize: "9px", color: "#555" }}>{count} weeks</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {activeTab === "signals" && (
          <div>
            <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "18px", marginBottom: "16px" }}>
              <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "16px" }}>DIMENSION BREAKDOWN — CURRENT</div>
              <DimBar label="TREND"      score={current.trend_score}      weight={0.30} />
              <DimBar label="VOLATILITY" score={current.volatility_score} weight={0.25} />
              <DimBar label="BREADTH"    score={current.breadth_score}    weight={0.25} />
              <DimBar label="FLOW"       score={current.flow_score}       weight={0.20} />
            </div>

            <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "18px", marginBottom: "16px" }}>
              <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "16px" }}>ALLOCATION FRAMEWORK</div>
              {[
                { range: "75–100", label: "Strong Bull",    color: "#00ff87", equity: "90–100%", style: "Momentum + Small/Midcap",  cash: "0–10%" },
                { range: "55–75",  label: "Mild Bull",      color: "#7dffb3", equity: "65–85%",  style: "Quality + Large Cap",      cash: "15–35%" },
                { range: "40–55",  label: "Neutral/Choppy", color: "#ffd166", equity: "40–60%",  style: "Defensive + Low Vol",      cash: "40–60%" },
                { range: "20–40",  label: "Mild Bear",      color: "#ff6b6b", equity: "15–35%",  style: "Debt + Gold",              cash: "65–85%" },
                { range: "0–20",   label: "Strong Bear",    color: "#ff2d55", equity: "0–15%",   style: "Capital Preservation",     cash: "85–100%" },
              ].map(r => (
                <div key={r.range} style={{
                  display: "flex", alignItems: "center", gap: "12px", padding: "10px 12px",
                  borderRadius: "6px", marginBottom: "6px",
                  background: current.regime_label === r.label ? REGIME_CONFIG[r.label]?.bg : "transparent",
                  border: current.regime_label === r.label ? `1px solid ${r.color}30` : "1px solid transparent",
                }}>
                  <div style={{ width: "6px", height: "6px", borderRadius: "50%", background: r.color }} />
                  <div style={{ width: "50px", fontSize: "9px", color: "#555" }}>{r.range}</div>
                  <div style={{ flex: 1, fontSize: "11px", color: r.color }}>{r.label}</div>
                  <div style={{ fontSize: "10px", color: "#aaa", width: "80px" }}>Eq: {r.equity}</div>
                  <div style={{ fontSize: "10px", color: "#666", flex: 1 }}>{r.style}</div>
                  <div style={{ fontSize: "10px", color: "#888", width: "60px", textAlign: "right" }}>Cash: {r.cash}</div>
                </div>
              ))}
            </div>

            <div style={{ background: "rgba(255,209,102,0.05)", border: "1px solid rgba(255,209,102,0.2)", borderRadius: "8px", padding: "14px 18px" }}>
              <div style={{ fontSize: "10px", color: "#ffd166", letterSpacing: "1px", marginBottom: "6px" }}>⚡ TO UNLOCK REAL FII/DII FLOW DATA</div>
              <div style={{ fontSize: "10px", color: "#888", lineHeight: 1.6 }}>
                Download from <span style={{ color: "#ffd166" }}>nseindia.com → Market Data → FII/DII Activity</span><br />
                Pass as DataFrame: <span style={{ color: "#ffd166" }}>columns = ['date', 'FII_Net', 'DII_Net']</span><br />
                Call: <span style={{ color: "#ffd166" }}>classifier.load_data(fii_dii_df=df)</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
