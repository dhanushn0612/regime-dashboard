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

        {activeTab === "sectors" && (
          <div>
            {!sector ? (
              <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "32px", textAlign: "center" }}>
                <div style={{ fontSize: "11px", color: "#555", letterSpacing: "2px", marginBottom: "8px" }}>NO SECTOR DATA</div>
                <div style={{ fontSize: "12px", color: "#888" }}>Run python data_pipeline/sector_rotation.py to generate sector allocations.</div>
              </div>
            ) : (
              <div>
                {/* Header */}
                <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "18px", marginBottom: "16px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
                    <div>
                      <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "4px" }}>SECTOR ROTATION ENGINE</div>
                      <div style={{ fontSize: "11px", color: "#888" }}>{sector.date} · {sector.model_used}</div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div style={{ fontSize: "28px", fontWeight: "700", color: sector.sectors_held === 0 ? "#ff2d55" : "#00ff87", fontFamily: "monospace" }}>
                        {sector.sectors_held === 0 ? "CASH" : `${sector.sectors_held} SECTORS`}
                      </div>
                      <div style={{ fontSize: "10px", color: "#888", marginTop: "2px" }}>
                        {Math.round(sector.cash_weight * 100)}% cash
                      </div>
                    </div>
                  </div>

                  {/* Action box */}
                  <div style={{
                    background: sector.sectors_held === 0 ? "rgba(255,45,85,0.08)" : "rgba(0,255,135,0.08)",
                    border: `1px solid ${sector.sectors_held === 0 ? "#ff2d5530" : "#00ff8730"}`,
                    borderRadius: "6px", padding: "10px 14px",
                  }}>
                    <div style={{ fontSize: "10px", color: sector.sectors_held === 0 ? "#ff2d55" : "#00ff87", letterSpacing: "1px" }}>
                      {sector.sectors_held === 0
                        ? "STRONG BEAR — Full cash. No sector deployment until regime recovers above 20."
                        : `DEPLOYING INTO ${sector.sectors_held} SECTOR${sector.sectors_held > 1 ? "S" : ""}`}
                    </div>
                  </div>
                </div>

                {/* Allocations table */}
                {sector.allocations && sector.allocations.length > 0 ? (
                  <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "18px", marginBottom: "16px" }}>
                    <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "14px" }}>CURRENT ALLOCATIONS</div>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "monospace", fontSize: "12px" }}>
                      <thead>
                        <tr style={{ borderBottom: "1px solid #1a1a2e" }}>
                          {["SECTOR", "WEIGHT", "PRED RET", "1M", "3M", "RATIONALE"].map(h => (
                            <td key={h} style={{ padding: "6px 8px", color: "#555", fontSize: "9px", letterSpacing: "1px" }}>{h}</td>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {sector.allocations.map((a, i) => (
                          <tr key={i} style={{ borderBottom: "1px solid #0d0d2a" }}>
                            <td style={{ padding: "10px 8px", color: "#00ff87", fontWeight: "700" }}>{a.sector}</td>
                            <td style={{ padding: "10px 8px", color: "#fff" }}>{(a.weight * 100).toFixed(1)}%</td>
                            <td style={{ padding: "10px 8px", color: a.predicted_ret >= 0 ? "#00ff87" : "#ff2d55" }}>
                              {a.predicted_ret >= 0 ? "+" : ""}{a.predicted_ret?.toFixed(1)}%
                            </td>
                            <td style={{ padding: "10px 8px", color: a.ret_1m >= 0 ? "#7dffb3" : "#ff6b6b" }}>
                              {a.ret_1m >= 0 ? "+" : ""}{a.ret_1m?.toFixed(1)}%
                            </td>
                            <td style={{ padding: "10px 8px", color: a.ret_3m >= 0 ? "#7dffb3" : "#ff6b6b" }}>
                              {a.ret_3m >= 0 ? "+" : ""}{a.ret_3m?.toFixed(1)}%
                            </td>
                            <td style={{ padding: "10px 8px", color: "#666", fontSize: "11px" }}>{a.rationale}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "18px", marginBottom: "16px" }}>
                    <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "8px" }}>NO ACTIVE ALLOCATIONS</div>
                    <div style={{ fontSize: "12px", color: "#888" }}>
                      Regime too weak to deploy. All {sector.excluded_sectors?.length || 0} sectors failed rule filter.
                    </div>
                  </div>
                )}

                {/* Excluded sectors */}
                {sector.excluded_sectors && sector.excluded_sectors.length > 0 && (
                  <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "18px", marginBottom: "16px" }}>
                    <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "14px" }}>
                      EXCLUDED BY RULES ({sector.excluded_sectors.length} sectors)
                    </div>
                    {sector.excluded_sectors.map((e, i) => (
                      <div key={i} style={{
                        display: "flex", justifyContent: "space-between", alignItems: "center",
                        padding: "8px 0", borderBottom: i < sector.excluded_sectors.length - 1 ? "1px solid #0d0d2a" : "none",
                      }}>
                        <span style={{ color: "#ff6b6b", fontFamily: "monospace", fontSize: "12px", fontWeight: "700" }}>
                          {e.sector}
                        </span>
                        <span style={{ color: "#555", fontSize: "11px", fontFamily: "monospace" }}>
                          {e.reasons?.join("  ·  ")}
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Regime context */}
                <div style={{
                  background: "rgba(255,209,102,0.05)", border: "1px solid rgba(255,209,102,0.2)",
                  borderRadius: "8px", padding: "14px 18px",
                }}>
                  <div style={{ fontSize: "10px", color: "#ffd166", letterSpacing: "1px", marginBottom: "6px" }}>
                    HOW SECTOR COUNT IS DETERMINED
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: "8px", marginTop: "8px" }}>
                    {[
                      { score: "75-100", label: "Strong Bull", n: 4 },
                      { score: "55-75",  label: "Mild Bull",   n: 3 },
                      { score: "40-55",  label: "Neutral",     n: 2 },
                      { score: "20-40",  label: "Mild Bear",   n: 1 },
                      { score: "0-20",   label: "Strong Bear", n: 0 },
                    ].map(r => (
                      <div key={r.label} style={{
                        background: sector.regime_label === r.label ? "rgba(255,209,102,0.1)" : "transparent",
                        border: `1px solid ${sector.regime_label === r.label ? "#ffd16640" : "#1a1a2e"}`,
                        borderRadius: "6px", padding: "8px", textAlign: "center",
                      }}>
                        <div style={{ fontSize: "18px", fontWeight: "700", color: "#ffd166", fontFamily: "monospace" }}>{r.n}</div>
                        <div style={{ fontSize: "9px", color: "#888", marginTop: "2px" }}>{r.label}</div>
                        <div style={{ fontSize: "9px", color: "#555" }}>{r.score}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
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
  const [sector, setSector] = useState(null);
  const [screener, setScreener] = useState(null);
  const [portfolio, setPortfolio] = useState(null);
  const [risk, setRisk] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch('/regime_current.json').then(r => r.json()),
      fetch('/regime_history.json').then(r => r.json()),
      fetch('/sector_current.json').then(r => r.json()).catch(() => null),
      fetch('/screener_current.json').then(r => r.json()).catch(() => null),
      fetch('/portfolio_current.json').then(r => r.json()).catch(() => null),
      fetch('/risk_current.json').then(r => r.json()).catch(() => null),
    ])
      .then(([curr, hist, sec, scr, port, rsk]) => {
        setCurrent(curr);
        setHistory(hist);
        setSector(sec);
        setScreener(scr);
        setPortfolio(port);
        setRisk(rsk);
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
  const tabs = ["overview", "dimensions", "history", "sectors", "screener", "portfolio", "risk", "signals"];
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


        {activeTab === "screener" && (
          <div>
            {!screener ? (
              <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "32px", textAlign: "center" }}>
                <div style={{ fontSize: "11px", color: "#555", letterSpacing: "2px", marginBottom: "8px" }}>NO SCREENER DATA</div>
                <div style={{ fontSize: "12px", color: "#888" }}>Run python data_pipeline/stock_screener.py to generate stock rankings.</div>
              </div>
            ) : (
              <div>
                {/* Header */}
                <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "18px", marginBottom: "16px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
                    <div>
                      <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "4px" }}>STOCK SCREENER — LAYER 3</div>
                      <div style={{ fontSize: "11px", color: "#888" }}>{screener.date} · {screener.stocks_screened} stocks screened · {screener.stocks_passed} passed filters</div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div style={{ fontSize: "28px", fontWeight: "700", color: screener.stocks_selected === 0 ? "#ff6b6b" : "#00ff87", fontFamily: "monospace" }}>
                        {screener.stocks_selected === 0 ? "WATCH" : screener.stocks_selected}
                      </div>
                      <div style={{ fontSize: "10px", color: "#888", marginTop: "2px" }}>
                        {screener.stocks_selected === 0 ? "list only" : "selected stocks"}
                      </div>
                    </div>
                  </div>
                  <div style={{
                    background: screener.stocks_selected === 0 ? "rgba(255,107,107,0.08)" : "rgba(0,255,135,0.08)",
                    border: `1px solid ${screener.stocks_selected === 0 ? "#ff6b6b30" : "#00ff8730"}`,
                    borderRadius: "6px", padding: "10px 14px", fontSize: "10px",
                    color: screener.stocks_selected === 0 ? "#ff6b6b" : "#00ff87", letterSpacing: "1px",
                  }}>
                    {screener.note}
                  </div>
                </div>

                {/* Factor legend */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "8px", marginBottom: "16px" }}>
                  {[
                    { label: "MOMENTUM", key: "f_momentum", color: "#00d4ff", desc: "1M/3M/6M/12M price" },
                    { label: "QUALITY",  key: "f_quality",  color: "#bf5af2", desc: "Trend + vol confirm" },
                    { label: "LOW VOL",  key: "f_lowvol",   color: "#ffd166", desc: "Vol + drawdown" },
                    { label: "EARNINGS", key: "f_earnings", color: "#00ff87", desc: "Growth consistency" },
                  ].map(f => (
                    <div key={f.key} style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "6px", padding: "10px 12px" }}>
                      <div style={{ fontSize: "9px", color: f.color, letterSpacing: "1px", marginBottom: "2px" }}>{f.label}</div>
                      <div style={{ fontSize: "9px", color: "#555" }}>{f.desc}</div>
                    </div>
                  ))}
                </div>

                {/* Stock table */}
                {screener.stocks && screener.stocks.length > 0 ? (
                  <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "18px", marginBottom: "16px", overflowX: "auto" }}>
                    <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "14px" }}>
                      SELECTED STOCKS — RANKED BY ML SCORE
                    </div>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "monospace", fontSize: "11px", minWidth: "700px" }}>
                      <thead>
                        <tr style={{ borderBottom: "1px solid #1a1a2e" }}>
                          {["#", "TICKER", "SCORE", "MOM", "QUAL", "VOL", "EARN", "3M", "1M", "VOL%", "MAX DD"].map(h => (
                            <td key={h} style={{ padding: "6px 8px", color: "#555", fontSize: "9px", letterSpacing: "1px" }}>{h}</td>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {screener.stocks.map((st, i) => (
                          <tr key={i} style={{ borderBottom: "1px solid #0d0d2a" }}>
                            <td style={{ padding: "10px 8px", color: "#555", fontSize: "10px" }}>{i + 1}</td>
                            <td style={{ padding: "10px 8px", color: "#fff", fontWeight: "700" }}>{st.name}</td>
                            <td style={{ padding: "10px 8px", color: "#00ff87" }}>{st.ml_score?.toFixed(1)}</td>
                            <td style={{ padding: "10px 8px", color: "#00d4ff" }}>{st.f_momentum?.toFixed(0)}</td>
                            <td style={{ padding: "10px 8px", color: "#bf5af2" }}>{st.f_quality?.toFixed(0)}</td>
                            <td style={{ padding: "10px 8px", color: "#ffd166" }}>{st.f_lowvol?.toFixed(0)}</td>
                            <td style={{ padding: "10px 8px", color: "#00ff87" }}>{st.f_earnings?.toFixed(0)}</td>
                            <td style={{ padding: "10px 8px", color: st.ret_3m >= 0 ? "#7dffb3" : "#ff6b6b" }}>
                              {st.ret_3m >= 0 ? "+" : ""}{st.ret_3m?.toFixed(1)}%
                            </td>
                            <td style={{ padding: "10px 8px", color: st.ret_1m >= 0 ? "#7dffb3" : "#ff6b6b" }}>
                              {st.ret_1m >= 0 ? "+" : ""}{st.ret_1m?.toFixed(1)}%
                            </td>
                            <td style={{ padding: "10px 8px", color: "#888" }}>{st.ann_vol_pct?.toFixed(0)}%</td>
                            <td style={{ padding: "10px 8px", color: "#ff6b6b" }}>{st.max_dd_pct?.toFixed(0)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "32px", marginBottom: "16px", textAlign: "center" }}>
                    <div style={{ fontSize: "11px", color: "#555", letterSpacing: "2px", marginBottom: "8px" }}>WATCHLIST MODE</div>
                    <div style={{ fontSize: "12px", color: "#888" }}>
                      {screener.stocks_passed} stocks passed quality filters but regime is too weak for deployment.
                      These will become buy candidates when regime recovers above 40.
                    </div>
                  </div>
                )}

                {/* Regime stock count guide */}
                <div style={{ background: "rgba(0,212,255,0.05)", border: "1px solid rgba(0,212,255,0.2)", borderRadius: "8px", padding: "14px 18px" }}>
                  <div style={{ fontSize: "10px", color: "#00d4ff", letterSpacing: "1px", marginBottom: "8px" }}>REGIME-DEPENDENT STOCK COUNT</div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: "8px" }}>
                    {[
                      { label: "Strong Bull", n: 15, score: "75-100" },
                      { label: "Mild Bull",   n: 10, score: "55-75" },
                      { label: "Neutral",     n: 5,  score: "40-55" },
                      { label: "Mild Bear",   n: 0,  score: "20-40" },
                      { label: "Strong Bear", n: 0,  score: "0-20" },
                    ].map(r => (
                      <div key={r.label} style={{
                        background: screener.regime_label === r.label ? "rgba(0,212,255,0.1)" : "transparent",
                        border: `1px solid ${screener.regime_label === r.label ? "#00d4ff40" : "#1a1a2e"}`,
                        borderRadius: "6px", padding: "8px", textAlign: "center",
                      }}>
                        <div style={{ fontSize: "18px", fontWeight: "700", color: "#00d4ff", fontFamily: "monospace" }}>{r.n}</div>
                        <div style={{ fontSize: "9px", color: "#888", marginTop: "2px" }}>{r.label}</div>
                        <div style={{ fontSize: "9px", color: "#555" }}>{r.score}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}


        {activeTab === "portfolio" && (
          <div>
            {!portfolio ? (
              <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "32px", textAlign: "center" }}>
                <div style={{ fontSize: "11px", color: "#555", letterSpacing: "2px", marginBottom: "8px" }}>NO PORTFOLIO DATA</div>
                <div style={{ fontSize: "12px", color: "#888" }}>Run python data_pipeline/portfolio_construction.py to generate portfolio.</div>
              </div>
            ) : (
              <div>
                {/* Header */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: "12px", marginBottom: "16px" }}>
                  {[
                    { label: "EQUITY", value: Math.round(portfolio.equity_allocation * 100) + "%", color: portfolio.equity_allocation > 0.5 ? "#00ff87" : "#ffd166" },
                    { label: "CASH", value: Math.round(portfolio.cash_allocation * 100) + "%", color: "#888" },
                    { label: "POSITIONS", value: portfolio.positions?.length || 0, color: "#00d4ff" },
                    { label: "STATUS", value: portfolio.status === "cash_mode" ? "CASH" : "ACTIVE", color: portfolio.status === "cash_mode" ? "#ff6b6b" : "#00ff87" },
                  ].map(m => (
                    <div key={m.label} style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "16px", textAlign: "center" }}>
                      <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "8px" }}>{m.label}</div>
                      <div style={{ fontSize: "28px", fontWeight: "700", color: m.color, fontFamily: "monospace" }}>{m.value}</div>
                    </div>
                  ))}
                </div>

                {/* Analytics */}
                {portfolio.analytics && Object.keys(portfolio.analytics).length > 0 && (
                  <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "18px", marginBottom: "16px" }}>
                    <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "14px" }}>PORTFOLIO ANALYTICS</div>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px" }}>
                      {[
                        { label: "Exp. Return", value: portfolio.analytics.expected_return_pct?.toFixed(1) + "%", color: "#00ff87" },
                        { label: "Exp. Volatility", value: portfolio.analytics.expected_vol_pct?.toFixed(1) + "%", color: "#ffd166" },
                        { label: "Sharpe Ratio", value: portfolio.analytics.sharpe_ratio?.toFixed(2), color: "#00d4ff" },
                        { label: "Eff. Stocks", value: portfolio.analytics.effective_n_stocks?.toFixed(1), color: "#bf5af2" },
                      ].map(a => (
                        <div key={a.label} style={{ textAlign: "center" }}>
                          <div style={{ fontSize: "22px", fontWeight: "700", color: a.color, fontFamily: "monospace" }}>{a.value}</div>
                          <div style={{ fontSize: "9px", color: "#555", marginTop: "4px" }}>{a.label}</div>
                        </div>
                      ))}
                    </div>
                    {portfolio.transaction_costs?.total_cost_bps > 0 && (
                      <div style={{ marginTop: "12px", padding: "8px 12px", background: "rgba(255,107,107,0.08)", borderRadius: "6px", fontSize: "10px", color: "#ff6b6b" }}>
                        Transaction cost: {portfolio.transaction_costs.total_cost_bps?.toFixed(1)} bps  ·  Rs {portfolio.transaction_costs.total_cost?.toLocaleString()} on Rs 10L portfolio
                      </div>
                    )}
                  </div>
                )}

                {/* Rebalance status */}
                <div style={{
                  background: portfolio.rebalance_needed ? "rgba(255,209,102,0.08)" : "rgba(0,200,150,0.08)",
                  border: `1px solid ${portfolio.rebalance_needed ? "#ffd16630" : "#00c89630"}`,
                  borderRadius: "8px", padding: "12px 16px", marginBottom: "16px",
                  fontSize: "10px", color: portfolio.rebalance_needed ? "#ffd166" : "#00c896", letterSpacing: "1px",
                }}>
                  {portfolio.rebalance_needed ? "REBALANCE REQUIRED" : "NO REBALANCE NEEDED"}
                  <span style={{ color: "#888", marginLeft: "12px", fontWeight: "400" }}>{portfolio.rebalance_reason}</span>
                </div>

                {/* Positions table */}
                {portfolio.positions && portfolio.positions.length > 0 ? (
                  <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "18px", marginBottom: "16px", overflowX: "auto" }}>
                    <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "14px" }}>POSITIONS — MEAN-VARIANCE OPTIMISED</div>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "monospace", fontSize: "11px", minWidth: "650px" }}>
                      <thead>
                        <tr style={{ borderBottom: "1px solid #1a1a2e" }}>
                          {["TICKER", "PORTFOLIO WT", "EQUITY WT", "EXP RET", "SECTOR", "ML SCORE"].map(h => (
                            <td key={h} style={{ padding: "6px 8px", color: "#555", fontSize: "9px", letterSpacing: "1px" }}>{h}</td>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {portfolio.positions.map((p, i) => (
                          <tr key={i} style={{ borderBottom: "1px solid #0d0d2a" }}>
                            <td style={{ padding: "10px 8px", color: "#fff", fontWeight: "700" }}>{p.name}</td>
                            <td style={{ padding: "10px 8px", color: "#00ff87" }}>{(p.target_weight * 100).toFixed(1)}%</td>
                            <td style={{ padding: "10px 8px", color: "#7dffb3" }}>{(p.equity_weight * 100).toFixed(1)}%</td>
                            <td style={{ padding: "10px 8px", color: p.expected_ret >= 0 ? "#7dffb3" : "#ff6b6b" }}>
                              {p.expected_ret >= 0 ? "+" : ""}{p.expected_ret?.toFixed(1)}%
                            </td>
                            <td style={{ padding: "10px 8px", color: "#888" }}>{p.sector}</td>
                            <td style={{ padding: "10px 8px", color: "#00d4ff" }}>{p.ml_score?.toFixed(1)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "32px", marginBottom: "16px", textAlign: "center" }}>
                    <div style={{ fontSize: "11px", color: "#555", letterSpacing: "2px", marginBottom: "8px" }}>CASH MODE</div>
                    <div style={{ fontSize: "12px", color: "#888" }}>
                      {portfolio.note || "Regime too weak for equity deployment."}
                    </div>
                    <div style={{ fontSize: "11px", color: "#ffd166", marginTop: "8px" }}>
                      {Math.round(portfolio.equity_allocation * 100)}% equity  ·  {Math.round(portfolio.cash_allocation * 100)}% cash
                    </div>
                  </div>
                )}

                {/* Sector breakdown */}
                {portfolio.analytics?.sector_weights && Object.keys(portfolio.analytics.sector_weights).length > 0 && (
                  <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "18px", marginBottom: "16px" }}>
                    <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "14px" }}>SECTOR BREAKDOWN</div>
                    {Object.entries(portfolio.analytics.sector_weights).sort((a,b) => b[1]-a[1]).map(([sec, wt]) => (
                      <div key={sec} style={{ marginBottom: "8px" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "3px" }}>
                          <span style={{ fontSize: "11px", color: "#aaa" }}>{sec}</span>
                          <span style={{ fontSize: "11px", color: "#00ff87", fontFamily: "monospace" }}>{(wt*100).toFixed(1)}%</span>
                        </div>
                        <div style={{ height: "4px", background: "#1a1a2e", borderRadius: "2px" }}>
                          <div style={{ width: `${wt*100}%`, height: "100%", background: "#00ff87", borderRadius: "2px" }} />
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Methodology note */}
                <div style={{ background: "rgba(26,107,255,0.05)", border: "1px solid rgba(26,107,255,0.2)", borderRadius: "8px", padding: "14px 18px" }}>
                  <div style={{ fontSize: "10px", color: "#1a6bff", letterSpacing: "1px", marginBottom: "6px" }}>METHODOLOGY</div>
                  <div style={{ fontSize: "10px", color: "#888", lineHeight: 1.6 }}>
                    Weights computed via Maximum Sharpe Ratio optimisation using Ledoit-Wolf shrinkage covariance estimator.
                    Expected returns blend 3M momentum (40%), ML score premium (40%), and mean-reversion adjustment (20%).
                    Total equity exposure scaled by regime: {Math.round(portfolio.equity_allocation*100)}% in current {portfolio.regime_label} regime.
                    Transaction costs: 20bps large cap, 40bps mid cap round-trip.
                  </div>
                </div>
              </div>
            )}
          </div>
        )}


        {activeTab === "risk" && (
          <div>
            {!risk ? (
              <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "32px", textAlign: "center" }}>
                <div style={{ fontSize: "11px", color: "#555", letterSpacing: "2px", marginBottom: "8px" }}>NO RISK DATA</div>
                <div style={{ fontSize: "12px", color: "#888" }}>Run python data_pipeline/risk_management.py</div>
              </div>
            ) : (
              <div>
                {/* Aggregate risk banner */}
                <div style={{
                  background: `${risk.aggregate?.color}15`,
                  border: `1px solid ${risk.aggregate?.color}40`,
                  borderRadius: "8px", padding: "18px 20px", marginBottom: "16px",
                  display: "flex", justifyContent: "space-between", alignItems: "center",
                }}>
                  <div>
                    <div style={{ fontSize: "9px", color: risk.aggregate?.color, letterSpacing: "2px", marginBottom: "4px" }}>RISK STATUS — {risk.date}</div>
                    <div style={{ fontSize: "14px", color: "#fff", fontWeight: "600" }}>{risk.aggregate?.overall_message}</div>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <div style={{ fontSize: "40px", fontWeight: "700", color: risk.aggregate?.color, fontFamily: "monospace", lineHeight: 1 }}>{risk.aggregate?.risk_score}</div>
                    <div style={{ fontSize: "9px", color: "#555", marginTop: "4px" }}>/100 risk score</div>
                  </div>
                </div>

                {/* Position size impact */}
                {risk.aggregate?.position_size_mult < 1.0 && (
                  <div style={{ background: "rgba(255,107,107,0.08)", border: "1px solid #ff6b6b30", borderRadius: "8px", padding: "14px 18px", marginBottom: "16px" }}>
                    <div style={{ fontSize: "10px", color: "#ff6b6b", letterSpacing: "1px", marginBottom: "6px" }}>AUTO POSITION REDUCTION TRIGGERED</div>
                    <div style={{ display: "flex", gap: "24px", fontSize: "12px" }}>
                      <span style={{ color: "#888" }}>Base equity: <span style={{ color: "#fff" }}>{risk.aggregate?.base_equity_pct}%</span></span>
                      <span style={{ color: "#555" }}>→</span>
                      <span style={{ color: "#888" }}>Adjusted equity: <span style={{ color: "#ff6b6b", fontWeight: "700" }}>{risk.aggregate?.adjusted_equity_pct}%</span></span>
                      <span style={{ color: "#888" }}>Multiplier: <span style={{ color: "#ff6b6b" }}>{risk.aggregate?.position_size_mult}x</span></span>
                    </div>
                  </div>
                )}

                {/* Individual rules */}
                <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "18px", marginBottom: "16px" }}>
                  <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "14px" }}>
                    RISK RULES — {risk.aggregate?.triggered_count}/6 TRIGGERED
                  </div>
                  {risk.rules?.map((rule, i) => {
                    const colors = { INFO: "#00c896", WARN: "#ffd166", ALERT: "#ff6b6b", CRITICAL: "#ff2d55" };
                    const col = colors[rule.severity] || "#888";
                    const ruleNames = {
                      'portfolio_drawdown': 'Portfolio Drawdown',
                      'stock_stop_loss': 'Stock Stop Loss',
                      'factor_concentration': 'Factor Concentration',
                      'correlation_spike': 'Correlation Spike',
                      'regime_deterioration': 'Regime Deterioration',
                      'market_anomaly': 'ML Anomaly Detection',
                    };
                    return (
                      <div key={i} style={{
                        display: "flex", alignItems: "center", gap: "12px",
                        padding: "10px 0",
                        borderBottom: i < risk.rules.length - 1 ? "1px solid #0d0d2a" : "none",
                      }}>
                        <div style={{
                          width: "8px", height: "8px", borderRadius: "50%",
                          background: col, flexShrink: 0,
                          boxShadow: rule.triggered ? `0 0 6px ${col}` : "none",
                        }} />
                        <div style={{ width: "180px", flexShrink: 0 }}>
                          <div style={{ fontSize: "11px", color: "#fff", fontWeight: "600" }}>{ruleNames[rule.rule] || rule.rule}</div>
                          <div style={{ fontSize: "9px", color: col, letterSpacing: "1px", marginTop: "2px" }}>{rule.severity}</div>
                        </div>
                        <div style={{ flex: 1, fontSize: "11px", color: "#888" }}>{rule.message}</div>
                        {rule.triggered && (
                          <div style={{ fontSize: "10px", color: col, whiteSpace: "nowrap" }}>{rule.action}</div>
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* Thresholds reference */}
                <div style={{ background: "rgba(26,107,255,0.05)", border: "1px solid rgba(26,107,255,0.2)", borderRadius: "8px", padding: "14px 18px" }}>
                  <div style={{ fontSize: "10px", color: "#1a6bff", letterSpacing: "1px", marginBottom: "10px" }}>RISK THRESHOLDS (INDUSTRY STANDARD)</div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "8px" }}>
                    {[
                      { label: "Max Drawdown", value: "15%" },
                      { label: "Stock Stop Loss", value: "12%" },
                      { label: "Factor Concentration", value: "60%" },
                      { label: "Correlation Spike", value: "0.75" },
                      { label: "Regime Drop (5d)", value: "20pts" },
                      { label: "Critical Regime", value: "< 15" },
                    ].map(t => (
                      <div key={t.label} style={{ padding: "8px 10px", background: "#0d0d1a", borderRadius: "6px" }}>
                        <div style={{ fontSize: "9px", color: "#555", marginBottom: "2px" }}>{t.label}</div>
                        <div style={{ fontSize: "13px", color: "#fff", fontFamily: "monospace", fontWeight: "700" }}>{t.value}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
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

        {activeTab === "sectors" && (
          <div>
            {!sector ? (
              <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "32px", textAlign: "center" }}>
                <div style={{ fontSize: "11px", color: "#555", letterSpacing: "2px", marginBottom: "8px" }}>NO SECTOR DATA</div>
                <div style={{ fontSize: "12px", color: "#888" }}>Run python data_pipeline/sector_rotation.py to generate sector allocations.</div>
              </div>
            ) : (
              <div>
                {/* Header */}
                <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "18px", marginBottom: "16px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
                    <div>
                      <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "4px" }}>SECTOR ROTATION ENGINE</div>
                      <div style={{ fontSize: "11px", color: "#888" }}>{sector.date} · {sector.model_used}</div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div style={{ fontSize: "28px", fontWeight: "700", color: sector.sectors_held === 0 ? "#ff2d55" : "#00ff87", fontFamily: "monospace" }}>
                        {sector.sectors_held === 0 ? "CASH" : `${sector.sectors_held} SECTORS`}
                      </div>
                      <div style={{ fontSize: "10px", color: "#888", marginTop: "2px" }}>
                        {Math.round(sector.cash_weight * 100)}% cash
                      </div>
                    </div>
                  </div>

                  {/* Action box */}
                  <div style={{
                    background: sector.sectors_held === 0 ? "rgba(255,45,85,0.08)" : "rgba(0,255,135,0.08)",
                    border: `1px solid ${sector.sectors_held === 0 ? "#ff2d5530" : "#00ff8730"}`,
                    borderRadius: "6px", padding: "10px 14px",
                  }}>
                    <div style={{ fontSize: "10px", color: sector.sectors_held === 0 ? "#ff2d55" : "#00ff87", letterSpacing: "1px" }}>
                      {sector.sectors_held === 0
                        ? "STRONG BEAR — Full cash. No sector deployment until regime recovers above 20."
                        : `DEPLOYING INTO ${sector.sectors_held} SECTOR${sector.sectors_held > 1 ? "S" : ""}`}
                    </div>
                  </div>
                </div>

                {/* Allocations table */}
                {sector.allocations && sector.allocations.length > 0 ? (
                  <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "18px", marginBottom: "16px" }}>
                    <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "14px" }}>CURRENT ALLOCATIONS</div>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "monospace", fontSize: "12px" }}>
                      <thead>
                        <tr style={{ borderBottom: "1px solid #1a1a2e" }}>
                          {["SECTOR", "WEIGHT", "PRED RET", "1M", "3M", "RATIONALE"].map(h => (
                            <td key={h} style={{ padding: "6px 8px", color: "#555", fontSize: "9px", letterSpacing: "1px" }}>{h}</td>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {sector.allocations.map((a, i) => (
                          <tr key={i} style={{ borderBottom: "1px solid #0d0d2a" }}>
                            <td style={{ padding: "10px 8px", color: "#00ff87", fontWeight: "700" }}>{a.sector}</td>
                            <td style={{ padding: "10px 8px", color: "#fff" }}>{(a.weight * 100).toFixed(1)}%</td>
                            <td style={{ padding: "10px 8px", color: a.predicted_ret >= 0 ? "#00ff87" : "#ff2d55" }}>
                              {a.predicted_ret >= 0 ? "+" : ""}{a.predicted_ret?.toFixed(1)}%
                            </td>
                            <td style={{ padding: "10px 8px", color: a.ret_1m >= 0 ? "#7dffb3" : "#ff6b6b" }}>
                              {a.ret_1m >= 0 ? "+" : ""}{a.ret_1m?.toFixed(1)}%
                            </td>
                            <td style={{ padding: "10px 8px", color: a.ret_3m >= 0 ? "#7dffb3" : "#ff6b6b" }}>
                              {a.ret_3m >= 0 ? "+" : ""}{a.ret_3m?.toFixed(1)}%
                            </td>
                            <td style={{ padding: "10px 8px", color: "#666", fontSize: "11px" }}>{a.rationale}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "18px", marginBottom: "16px" }}>
                    <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "8px" }}>NO ACTIVE ALLOCATIONS</div>
                    <div style={{ fontSize: "12px", color: "#888" }}>
                      Regime too weak to deploy. All {sector.excluded_sectors?.length || 0} sectors failed rule filter.
                    </div>
                  </div>
                )}

                {/* Excluded sectors */}
                {sector.excluded_sectors && sector.excluded_sectors.length > 0 && (
                  <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "18px", marginBottom: "16px" }}>
                    <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "14px" }}>
                      EXCLUDED BY RULES ({sector.excluded_sectors.length} sectors)
                    </div>
                    {sector.excluded_sectors.map((e, i) => (
                      <div key={i} style={{
                        display: "flex", justifyContent: "space-between", alignItems: "center",
                        padding: "8px 0", borderBottom: i < sector.excluded_sectors.length - 1 ? "1px solid #0d0d2a" : "none",
                      }}>
                        <span style={{ color: "#ff6b6b", fontFamily: "monospace", fontSize: "12px", fontWeight: "700" }}>
                          {e.sector}
                        </span>
                        <span style={{ color: "#555", fontSize: "11px", fontFamily: "monospace" }}>
                          {e.reasons?.join("  ·  ")}
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Regime context */}
                <div style={{
                  background: "rgba(255,209,102,0.05)", border: "1px solid rgba(255,209,102,0.2)",
                  borderRadius: "8px", padding: "14px 18px",
                }}>
                  <div style={{ fontSize: "10px", color: "#ffd166", letterSpacing: "1px", marginBottom: "6px" }}>
                    HOW SECTOR COUNT IS DETERMINED
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: "8px", marginTop: "8px" }}>
                    {[
                      { score: "75-100", label: "Strong Bull", n: 4 },
                      { score: "55-75",  label: "Mild Bull",   n: 3 },
                      { score: "40-55",  label: "Neutral",     n: 2 },
                      { score: "20-40",  label: "Mild Bear",   n: 1 },
                      { score: "0-20",   label: "Strong Bear", n: 0 },
                    ].map(r => (
                      <div key={r.label} style={{
                        background: sector.regime_label === r.label ? "rgba(255,209,102,0.1)" : "transparent",
                        border: `1px solid ${sector.regime_label === r.label ? "#ffd16640" : "#1a1a2e"}`,
                        borderRadius: "6px", padding: "8px", textAlign: "center",
                      }}>
                        <div style={{ fontSize: "18px", fontWeight: "700", color: "#ffd166", fontFamily: "monospace" }}>{r.n}</div>
                        <div style={{ fontSize: "9px", color: "#888", marginTop: "2px" }}>{r.label}</div>
                        <div style={{ fontSize: "9px", color: "#555" }}>{r.score}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
