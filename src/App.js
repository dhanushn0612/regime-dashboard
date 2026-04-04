import { useState, useEffect } from "react";
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, RadarChart, Radar, PolarGrid, PolarAngleAxis, Cell } from "recharts";

// ── REGIME CONFIG ──────────────────────────────────────────────────────
const REGIME_CONFIG = {
  "Strong Bull":    { color: "#00ff87", bg: "rgba(0,255,135,0.12)" },
  "Mild Bull":      { color: "#7dffb3", bg: "rgba(125,255,179,0.10)" },
  "Neutral/Choppy": { color: "#ffd166", bg: "rgba(255,209,102,0.10)" },
  "Mild Bear":      { color: "#ff6b6b", bg: "rgba(255,107,107,0.10)" },
  "Strong Bear":    { color: "#ff2d55", bg: "rgba(255,45,85,0.12)" },
};

const scoreColor = (s) =>
  s > 75 ? "#00ff87" : s > 55 ? "#7dffb3" : s > 40 ? "#ffd166" : s > 20 ? "#ff6b6b" : "#ff2d55";

const pnlColor = (v) => (v >= 0 ? "#00ff87" : "#ff6b6b");
const fmt = (v, dec = 1) => (v === null || v === undefined ? "—" : Number(v).toFixed(dec));
const fmtPct = (v, dec = 1) => (v >= 0 ? "+" : "") + fmt(v, dec) + "%";
const fmtCr = (v) => {
  if (!v) return "—";
  if (Math.abs(v) >= 10000000) return "₹" + (v / 10000000).toFixed(2) + "Cr";
  if (Math.abs(v) >= 100000)  return "₹" + (v / 100000).toFixed(1) + "L";
  return "₹" + v.toLocaleString("en-IN");
};

// ── SHARED COMPONENTS ──────────────────────────────────────────────────
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
    <span style={{ background: cfg.bg, border: `1px solid ${cfg.color}40`, color: cfg.color, padding: "3px 12px", borderRadius: "4px", fontSize: "11px", letterSpacing: "1.5px", textTransform: "uppercase", fontWeight: "600" }}>{label}</span>
  );
};

const Card = ({ title, badge, children, accent }) => (
  <div style={{ background: "#0d0d1a", border: `1px solid ${accent || "#1a1a2e"}`, borderRadius: "8px", padding: "18px", marginBottom: "16px" }}>
    {title && (
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px" }}>
        <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px" }}>{title}</div>
        {badge}
      </div>
    )}
    {children}
  </div>
);

const MetricCard = ({ label, value, sub, color = "#fff", highlight }) => (
  <div style={{ background: highlight ? `${highlight}10` : "#131326", border: `1px solid ${highlight || "#2a2a4a"}40`, borderRadius: "6px", padding: "16px", textAlign: "center", position: "relative" }}>
    {highlight && <div style={{ position: "absolute", top: "-8px", left: "50%", transform: "translateX(-50%)", background: highlight, color: "#000", fontSize: "8px", fontWeight: "800", padding: "2px 8px", borderRadius: "10px", letterSpacing: "1px", whiteSpace: "nowrap" }}>{highlight === "#00d4ff" ? "CORE VALUE PROP" : "STOCK PICKING EDGE"}</div>}
    <div style={{ fontSize: "9px", color: highlight || "#888", letterSpacing: "1px", marginBottom: "6px", marginTop: highlight ? "4px" : "0" }}>{label}</div>
    <div style={{ fontSize: "24px", fontWeight: "700", color, fontFamily: "monospace" }}>{value}</div>
    {sub && <div style={{ fontSize: "9px", color: "#555", marginTop: "4px" }}>{sub}</div>}
  </div>
);

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div style={{ background: "#0d0d1a", border: "1px solid #2a2a4a", padding: "10px 14px", borderRadius: "6px", fontSize: "11px" }}>
      <div style={{ color: "#666", marginBottom: "6px" }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, marginBottom: "2px" }}>
          {p.name}: <span style={{ color: "#fff" }}>{typeof p.value === "number" ? p.value.toFixed(1) : p.value}</span>
        </div>
      ))}
      {d?.regime_label && <div style={{ color: REGIME_CONFIG[d.regime_label]?.color || "#aaa", marginTop: "4px" }}>{d.regime_label}</div>}
    </div>
  );
};

const NoData = ({ msg, cmd }) => (
  <Card>
    <div style={{ textAlign: "center", padding: "24px" }}>
      <div style={{ fontSize: "11px", color: "#555", letterSpacing: "2px", marginBottom: "8px" }}>NO DATA</div>
      <div style={{ fontSize: "12px", color: "#888" }}>{msg}</div>
      {cmd && <div style={{ fontSize: "11px", color: "#555", marginTop: "8px", fontFamily: "monospace" }}>{cmd}</div>}
    </div>
  </Card>
);

// ── YIELD CURVE TAB ────────────────────────────────────────────────────
const YieldCurveTab = ({ yc }) => {
  if (!yc) return <NoData msg="Yield curve data not available." cmd="python data_pipeline/fixed_income.py" />;

  const tenors = Object.entries(yc.yields || {}).map(([k, v]) => ({ tenor: k.toUpperCase(), yield: v }));
  const spreads = yc.spreads || {};
  const sectors = yc.sector_signals || {};
  const ycColor = scoreColor(yc.yc_score || 50);
  const regimeColors = { Expansion: "#00ff87", Recovery: "#7dffb3", Neutral: "#ffd166", Slowdown: "#ff6b6b", Contraction: "#ff2d55" };
  const ycRegColor = regimeColors[yc.yc_regime] || "#aaa";

  return (
    <div>
      {/* Header metrics */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "12px", marginBottom: "16px" }}>
        <MetricCard label="10Y G-SEC YIELD" value={`${fmt(yc.yields?.["10y"] || yc.yields?.y10y, 2)}%`} sub={`RBI Repo: ${fmt(yc.repo_rate, 2)}%`} color="#00d4ff" />
        <MetricCard label="10Y — 2Y SPREAD" value={`${(spreads["10y_2y"] >= 0 ? "+" : "")}${fmt(spreads["10y_2y"], 2)}%`}
          sub={spreads["10y_2y"] > 0.5 ? "Steep — bullish" : spreads["10y_2y"] > 0 ? "Flat — neutral" : "Inverted — caution"} color={spreads["10y_2y"] > 0.3 ? "#00ff87" : spreads["10y_2y"] > 0 ? "#ffd166" : "#ff6b6b"} />
        <MetricCard label="POLICY GAP" value={`${spreads.policy_gap >= 0 ? "+" : ""}${fmt(spreads.policy_gap, 2)}%`}
          sub={spreads.policy_gap < 0 ? "Market pricing cuts" : "Market pricing hikes"} color={spreads.policy_gap < 0 ? "#00ff87" : "#ff6b6b"} />
        <div style={{ background: `${ycColor}12`, border: `1px solid ${ycColor}40`, borderRadius: "6px", padding: "16px", textAlign: "center" }}>
          <div style={{ fontSize: "9px", color: ycColor, letterSpacing: "1px", marginBottom: "6px" }}>YC SCORE</div>
          <div style={{ fontSize: "36px", fontWeight: "700", color: ycColor, fontFamily: "monospace" }}>{yc.yc_score}</div>
          <div style={{ marginTop: "6px" }}><span style={{ background: `${ycRegColor}20`, border: `1px solid ${ycRegColor}40`, color: ycRegColor, padding: "2px 8px", borderRadius: "4px", fontSize: "10px", fontWeight: "600" }}>{yc.yc_regime}</span></div>
        </div>
      </div>

      {/* Yield curve chart */}
      {tenors.length > 0 && (
        <Card title="INDIA G-SEC YIELD CURVE">
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={tenors} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <defs>
                <linearGradient id="ycg" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#00d4ff" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#1a1a2e" strokeDasharray="3 3" />
              <XAxis dataKey="tenor" tick={{ fill: "#555", fontSize: 10 }} />
              <YAxis domain={["auto", "auto"]} tick={{ fill: "#555", fontSize: 10 }} tickFormatter={v => v + "%"} />
              <Tooltip formatter={(v) => [fmt(v, 2) + "%", "Yield"]} contentStyle={{ background: "#0d0d1a", border: "1px solid #2a2a4a", borderRadius: "6px", fontSize: "11px" }} />
              {yc.repo_rate && <ReferenceLine y={yc.repo_rate} stroke="#ffffff20" strokeDasharray="4 4" label={{ value: "Repo", fill: "#555", fontSize: 9 }} />}
              <Area type="monotone" dataKey="yield" stroke="#00d4ff" fill="url(#ycg)" strokeWidth={2} dot={{ fill: "#00d4ff", r: 4 }} name="Yield" />
            </AreaChart>
          </ResponsiveContainer>
        </Card>
      )}

      {/* Interpretation + sectors */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
        <Card title="MACRO INTERPRETATION">
          <div style={{ fontSize: "12px", color: "#bbb", lineHeight: "1.6" }}>{yc.interpretation || "—"}</div>
          <div style={{ marginTop: "12px", fontSize: "10px", color: "#555" }}>Source: {yc.source} · {yc.date}</div>
        </Card>
        <Card title="SECTOR SIGNALS">
          {Object.entries(sectors).map(([sector, signal]) => {
            const col = signal === "bullish" ? "#00ff87" : signal === "bearish" ? "#ff6b6b" : "#555";
            const dot = signal === "bullish" ? "▲" : signal === "bearish" ? "▼" : "●";
            return (
              <div key={sector} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "7px 0", borderBottom: "1px solid #1a1a2e" }}>
                <span style={{ fontSize: "12px", color: "#bbb" }}>{sector}</span>
                <span style={{ fontSize: "11px", color: col, fontWeight: "700" }}>{dot} {signal}</span>
              </div>
            );
          })}
        </Card>
      </div>
    </div>
  );
};

// ── REGIME MONITOR TAB ─────────────────────────────────────────────────
const RegimeMonitorTab = ({ monitor }) => {
  if (!monitor) return <NoData msg="Regime monitor data not available." cmd="python data_pipeline/regime_monitor.py" />;

  const triggerColors = { CRITICAL: "#ff2d55", ALERT: "#ff6b6b", WARN: "#ffd166", OK: "#00ff87" };
  const triggers = monitor.triggers || [];
  const actionRequired = monitor.action_required;

  return (
    <div>
      {/* Action banner */}
      {actionRequired && (
        <div style={{ background: "rgba(255,45,85,0.15)", border: "1px solid #ff2d5560", borderRadius: "8px", padding: "16px 20px", marginBottom: "16px" }}>
          <div style={{ fontSize: "11px", color: "#ff2d55", letterSpacing: "2px", fontWeight: "700", marginBottom: "6px" }}>⚠ ACTION REQUIRED</div>
          <div style={{ fontSize: "13px", color: "#fff" }}>{monitor.primary_message}</div>
          {monitor.target_equity !== undefined && (
            <div style={{ fontSize: "11px", color: "#ff6b6b", marginTop: "6px" }}>
              Reduce equity: {fmt(monitor.current_equity, 0)}% → {fmt(monitor.target_equity, 0)}%
            </div>
          )}
        </div>
      )}

      {/* Score velocity chart */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "12px", marginBottom: "16px" }}>
        <MetricCard label="CURRENT SCORE" value={fmt(monitor.composite_score, 1)} color={scoreColor(monitor.composite_score || 50)} />
        <MetricCard label="SCORE CHANGE (5D)" value={`${(monitor.score_change >= 0 ? "+" : "")}${fmt(monitor.score_change, 1)}`} color={monitor.score_change >= 0 ? "#00ff87" : "#ff6b6b"} sub="vs 5 days ago" />
        <MetricCard label="REGIME" value={monitor.regime_label || "—"} color={REGIME_CONFIG[monitor.regime_label]?.color || "#aaa"} />
        <MetricCard label="BREADTH" value={fmt(monitor.breadth, 0)} color={scoreColor(monitor.breadth || 50)} sub="score /100" />
      </div>

      {/* Triggers */}
      <Card title="INTRA-MONTH TRIGGERS">
        {triggers.length === 0 ? (
          <div style={{ fontSize: "12px", color: "#555", textAlign: "center", padding: "12px" }}>No trigger data available.</div>
        ) : triggers.map((t, i) => {
          const col = triggerColors[t.severity] || "#888";
          return (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: "12px", padding: "12px 14px", background: t.fired ? `${col}10` : "transparent", border: `1px solid ${t.fired ? col + "30" : "#1a1a2e"}`, borderRadius: "6px", marginBottom: "8px" }}>
              <div style={{ width: "10px", height: "10px", borderRadius: "50%", background: t.fired ? col : "#333", flexShrink: 0, boxShadow: t.fired ? `0 0 8px ${col}` : "none" }} />
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: "11px", color: t.fired ? "#fff" : "#666", fontWeight: "600" }}>{t.name}</div>
                <div style={{ fontSize: "10px", color: "#555", marginTop: "2px" }}>{t.description}</div>
              </div>
              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: "10px", color: col, fontWeight: "700", letterSpacing: "1px" }}>{t.fired ? t.severity : "OK"}</div>
                {t.value !== undefined && <div style={{ fontSize: "10px", color: "#555", marginTop: "2px" }}>{fmt(t.value, 1)}</div>}
              </div>
            </div>
          );
        })}
      </Card>

      <div style={{ fontSize: "10px", color: "#555", textAlign: "right" }}>Last checked: {monitor.date || "—"}</div>
    </div>
  );
};

// ── PAPER PORTFOLIO TAB ────────────────────────────────────────────────
const PaperPortfolioTab = ({ paper }) => {
  if (!paper) return <NoData msg="Paper portfolio not initialised." cmd="python data_pipeline/paper_portfolio.py --init --capital 2500000" />;

  const port = paper.portfolio || {};
  const pnl  = paper.pnl || {};
  const perf = paper.performance || {};
  const positions = paper.positions || [];
  const history = paper.history_30d || [];
  const action = paper.action_required;

  return (
    <div>
      {/* Action alert */}
      {action && paper.action && (
        <div style={{ background: "rgba(255,45,85,0.15)", border: "1px solid #ff2d5560", borderRadius: "8px", padding: "14px 18px", marginBottom: "16px" }}>
          <div style={{ fontSize: "11px", color: "#ff2d55", fontWeight: "700", letterSpacing: "2px", marginBottom: "4px" }}>⚠ ACTION REQUIRED — {paper.action.worst_severity}</div>
          <div style={{ fontSize: "12px", color: "#fff" }}>{paper.action.primary_message}</div>
        </div>
      )}

      {/* Key metrics */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "12px", marginBottom: "16px" }}>
        <MetricCard label="PORTFOLIO VALUE" value={fmtCr(port.value)} color="#fff" sub={`Capital: ${fmtCr(port.capital)}`} />
        <MetricCard label="TODAY P&L" value={fmtPct(pnl.today_pct)} color={pnlColor(pnl.today_pct)} sub={fmtCr(pnl.today)} />
        <MetricCard label="TOTAL P&L" value={fmtPct(pnl.total_pct)} color={pnlColor(pnl.total_pct)} sub={fmtCr(pnl.total)} />
        <MetricCard label="EQUITY" value={`${fmt(port.equity_pct, 0)}%`} color={scoreColor((port.equity_pct || 0) * 1.1)} sub={`Cash: ${fmt(100 - (port.equity_pct || 0), 0)}%`} />
        <MetricCard label="DRAWDOWN" value={`${fmt(pnl.drawdown, 1)}%`} color={pnl.drawdown < -10 ? "#ff2d55" : pnl.drawdown < -5 ? "#ff6b6b" : "#00ff87"} sub="from high-water mark" />
        <MetricCard label="POSITIONS" value={port.positions || 0} color="#00d4ff" sub={`${paper.days_live || 0} days live`} />
      </div>

      {/* Equity curve */}
      {history.length > 1 && (
        <Card title="30-DAY EQUITY CURVE">
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={history}>
              <defs>
                <linearGradient id="ppg" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00ff87" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#00ff87" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#1a1a2e" strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fill: "#555", fontSize: 9 }} tickFormatter={d => d?.slice(5)} />
              <YAxis tick={{ fill: "#555", fontSize: 9 }} tickFormatter={v => "₹" + (v / 100000).toFixed(0) + "L"} />
              <Tooltip formatter={(v) => [fmtCr(v), "Value"]} contentStyle={{ background: "#0d0d1a", border: "1px solid #2a2a4a", borderRadius: "6px", fontSize: "11px" }} />
              <Area type="monotone" dataKey="portfolio_value" stroke="#00ff87" fill="url(#ppg)" strokeWidth={2} dot={false} name="Portfolio" />
            </AreaChart>
          </ResponsiveContainer>
        </Card>
      )}

      {/* Performance stats */}
      {Object.keys(perf).length > 0 && (
        <Card title="LIVE PERFORMANCE METRICS">
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px" }}>
            <MetricCard label="CAGR" value={`${fmtPct(perf.cagr, 1)}`} color="#00ff87" />
            <MetricCard label="SHARPE" value={fmt(perf.sharpe, 2)} color="#00d4ff" />
            <MetricCard label="MAX DRAWDOWN" value={`${fmt(perf.max_dd, 1)}%`} color="#ff6b6b" />
            <MetricCard label="WIN RATE" value={`${fmt(perf.win_rate, 0)}%`} color="#ffd166" />
          </div>
        </Card>
      )}

      {/* Positions table */}
      {positions.length > 0 ? (
        <Card title={`POSITIONS — ${positions.length} STOCKS`}>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "monospace", fontSize: "11px", minWidth: "600px" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid #1a1a2e" }}>
                  {["TICKER", "SHARES", "ENTRY", "CURRENT", "P&L %", "WEIGHT"].map(h => (
                    <td key={h} style={{ padding: "6px 8px", color: "#555", fontSize: "9px", letterSpacing: "1px" }}>{h}</td>
                  ))}
                </tr>
              </thead>
              <tbody>
                {positions.sort((a, b) => (b.weight || 0) - (a.weight || 0)).map((p, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid #0d0d2a" }}>
                    <td style={{ padding: "10px 8px", color: "#fff", fontWeight: "700" }}>{(p.ticker || "").replace(".NS", "")}</td>
                    <td style={{ padding: "10px 8px", color: "#888" }}>{p.shares}</td>
                    <td style={{ padding: "10px 8px", color: "#888" }}>₹{fmt(p.cost_price, 0)}</td>
                    <td style={{ padding: "10px 8px", color: "#fff" }}>₹{fmt(p.current_price, 0)}</td>
                    <td style={{ padding: "10px 8px", color: pnlColor(p.unrealised_pct) }}>{fmtPct(p.unrealised_pct, 1)}</td>
                    <td style={{ padding: "10px 8px", color: "#00d4ff" }}>{fmt(p.weight, 1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      ) : (
        <Card>
          <div style={{ textAlign: "center", padding: "24px", fontSize: "12px", color: "#888" }}>
            No positions held. Regime in defensive mode — capital in fixed income.
          </div>
        </Card>
      )}

      {/* Recent trades */}
      {(paper.recent_trades || []).length > 0 && (
        <Card title="RECENT TRADES">
          {paper.recent_trades.slice(-5).reverse().map((t, i) => (
            <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "8px 0", borderBottom: "1px solid #1a1a2e", fontSize: "11px" }}>
              <span style={{ color: t.type === "BUY" ? "#00ff87" : "#ff6b6b", fontWeight: "700", width: "40px" }}>{t.type}</span>
              <span style={{ color: "#fff" }}>{(t.ticker || "").replace(".NS", "")}</span>
              <span style={{ color: "#888" }}>{t.shares} @ ₹{fmt(t.price, 0)}</span>
              <span style={{ color: "#555" }}>{t.date}</span>
            </div>
          ))}
        </Card>
      )}
    </div>
  );
};

// ── FIXED INCOME TAB ───────────────────────────────────────────────────
const FixedIncomeTab = ({ fi }) => {
  if (!fi) return <NoData msg="Fixed income backtest data not available." cmd="python data_pipeline/fixed_income.py" />;

  const orig = fi.original || {};
  const adj  = fi.fi_adjusted || {};
  const nif  = fi.nifty || {};
  const cont = fi.fi_contribution || {};
  const byReg = fi.by_regime || {};

  const compareData = [
    { name: "Total Return", strategy: orig.total, fi: adj.total, nifty: nif.total },
    { name: "Ann. Return", strategy: orig.ann, fi: adj.ann, nifty: nif.ann },
  ];

  return (
    <div>
      {/* FI contribution hero */}
      <div style={{ background: "rgba(0,255,135,0.08)", border: "1px solid #00ff8730", borderRadius: "8px", padding: "16px 20px", marginBottom: "16px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div style={{ fontSize: "9px", color: "#00ff87", letterSpacing: "2px", marginBottom: "4px" }}>FIXED INCOME CONTRIBUTION</div>
          <div style={{ fontSize: "14px", color: "#fff" }}>Deploying idle cash into liquid/overnight funds adds real return</div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: "40px", fontWeight: "700", color: "#00ff87", fontFamily: "monospace" }}>+{fmt(cont.ann_return_added, 2)}%</div>
          <div style={{ fontSize: "10px", color: "#888" }}>p.a. additional return</div>
        </div>
      </div>

      {/* Comparison metrics */}
      <Card title="18-YEAR BACKTEST — CASH=0% vs WITH FI DEPLOYMENT">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: "10px" }}>
          {[
            { label: "TOTAL RETURN",   a: orig.total + "%", b: adj.total + "%",   n: nif.total + "%" },
            { label: "ANN. RETURN",    a: orig.ann + "%",   b: adj.ann + "%",     n: nif.ann + "%" },
            { label: "SHARPE RATIO",   a: fmt(orig.sharpe, 2), b: fmt(adj.sharpe, 2), n: fmt(nif.sharpe, 2) },
            { label: "MAX DRAWDOWN",   a: orig.max_dd + "%", b: adj.max_dd + "%", n: nif.max_dd + "%" },
            { label: "CALMAR RATIO",   a: fmt(orig.calmar, 2), b: fmt(adj.calmar, 2), n: fmt(nif.calmar, 2) },
          ].map(m => (
            <div key={m.label} style={{ background: "#131326", border: "1px solid #2a2a4a", borderRadius: "6px", padding: "12px", textAlign: "center" }}>
              <div style={{ fontSize: "8px", color: "#555", letterSpacing: "1px", marginBottom: "8px" }}>{m.label}</div>
              <div style={{ fontSize: "11px", color: "#888", marginBottom: "2px" }}>Cash=0%: <span style={{ color: "#aaa" }}>{m.a}</span></div>
              <div style={{ fontSize: "14px", color: "#00ff87", fontWeight: "700", marginBottom: "2px" }}>+FI: {m.b}</div>
              <div style={{ fontSize: "11px", color: "#555" }}>Nifty: {m.n}</div>
            </div>
          ))}
        </div>
      </Card>

      {/* By regime */}
      <Card title="FI DEPLOYMENT BY REGIME">
        {Object.entries(byReg).map(([regime, data]) => {
          const col = REGIME_CONFIG[regime]?.color || "#aaa";
          return (
            <div key={regime} style={{ display: "flex", alignItems: "center", gap: "12px", padding: "10px 0", borderBottom: "1px solid #1a1a2e" }}>
              <div style={{ width: "6px", height: "6px", borderRadius: "50%", background: col, flexShrink: 0 }} />
              <div style={{ width: "120px", fontSize: "11px", color: col }}>{regime}</div>
              <div style={{ fontSize: "10px", color: "#888", width: "60px" }}>{data.months}M</div>
              <div style={{ fontSize: "10px", color: "#555", width: "100px" }}>Cash: {fmt(data.avg_cash_pct, 0)}%</div>
              <div style={{ flex: 1, fontSize: "11px", color: "#fff" }}>{data.instrument}</div>
              <div style={{ fontSize: "12px", color: "#00ff87", fontWeight: "700" }}>{fmt(data.avg_fi_annual, 2)}% p.a.</div>
            </div>
          );
        })}
      </Card>

      <div style={{ fontSize: "10px", color: "#555", textAlign: "center", marginTop: "8px" }}>
        FI rates based on historical RBI repo rate. Overnight: repo−10bps · Liquid: repo · Short duration: repo+25bps
      </div>
    </div>
  );
};

// ── MAIN APP ───────────────────────────────────────────────────────────
export default function App() {
  const [history,   setHistory]   = useState([]);
  const [current,   setCurrent]   = useState(null);
  const [loading,   setLoading]   = useState(true);
  const [error,     setError]     = useState(null);
  const [activeTab, setActiveTab] = useState("overview");

  const [sector,   setSector]   = useState(null);
  const [screener, setScreener] = useState(null);
  const [portfolio,setPortfolio]= useState(null);
  const [risk,     setRisk]     = useState(null);
  const [backtest, setBacktest] = useState(null);
  const [yc,       setYc]       = useState(null);
  const [monitor,  setMonitor]  = useState(null);
  const [paper,    setPaper]    = useState(null);
  const [fi,       setFi]       = useState(null);

  useEffect(() => {
    Promise.all([
      fetch("/regime_current.json").then(r => r.json()),
      fetch("/regime_history.json").then(r => r.json()),
      fetch("/sector_current.json").then(r => r.json()).catch(() => null),
      fetch("/screener_current.json").then(r => r.json()).catch(() => null),
      fetch("/portfolio_current.json").then(r => r.json()).catch(() => null),
      fetch("/risk_current.json").then(r => r.json()).catch(() => null),
      fetch("/institutional_backtest.json").then(r => r.json()).catch(() => null),
      fetch("/yield_curve.json").then(r => r.json()).catch(() => null),
      fetch("/regime_monitor.json").then(r => r.json()).catch(() => null),
      fetch("/paper_portfolio.json").then(r => r.json()).catch(() => null),
      fetch("/fi_backtest.json").then(r => r.json()).catch(() => null),
    ])
      .then(([curr, hist, sec, scr, port, rsk, bt, ycData, mon, pp, fiData]) => {
        setCurrent(curr); setHistory(hist); setSector(sec); setScreener(scr);
        setPortfolio(port); setRisk(rsk); setBacktest(bt); setYc(ycData);
        setMonitor(mon); setPaper(pp); setFi(fiData);
        setLoading(false);
      })
      .catch(() => { setError("Could not load regime data. Run the classifier first."); setLoading(false); });
  }, []);

  if (loading) return (
    <div style={{ background: "#07071a", minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", color: "#555", fontFamily: "monospace" }}>
      Loading institutional framework...
    </div>
  );
  if (error || !current) return (
    <div style={{ background: "#07071a", minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", color: "#ff6b6b", fontFamily: "monospace", padding: "32px", textAlign: "center" }}>
      {error || "No data available."}<br /><br />
      <span style={{ color: "#555", fontSize: "12px" }}>Run: python data_pipeline/run_classifier.py</span>
    </div>
  );

  const cfg = REGIME_CONFIG[current.regime_label] || { color: "#aaa", bg: "transparent" };
  const regimeDist = history.reduce((acc, d) => { acc[d.regime_label] = (acc[d.regime_label] || 0) + 1; return acc; }, {});

  const TABS = [
    { id: "overview",    label: "Overview" },
    { id: "dimensions",  label: "Dimensions" },
    { id: "history",     label: "History" },
    { id: "sectors",     label: "Sectors" },
    { id: "screener",    label: "Screener" },
    { id: "portfolio",   label: "Portfolio" },
    { id: "risk",        label: "Risk" },
    { id: "signals",     label: "Signals" },
    { id: "performance", label: "Backtest" },
    { id: "yieldcurve",  label: "Yield Curve" },
    { id: "monitor",     label: "Monitor" },
    { id: "paper",       label: "Paper Port" },
    { id: "fixedincome", label: "Fixed Income" },
  ];

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
      <div style={{ borderBottom: "1px solid #1a1a2e", padding: "0 32px", display: "flex", overflowX: "auto" }}>
        {TABS.map(tab => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{
            background: "none", border: "none", cursor: "pointer", padding: "12px 14px",
            fontSize: "10px", letterSpacing: "1.5px", textTransform: "uppercase",
            color: activeTab === tab.id ? cfg.color : "#555",
            borderBottom: activeTab === tab.id ? `2px solid ${cfg.color}` : "2px solid transparent",
            marginBottom: "-1px", whiteSpace: "nowrap"
          }}>{tab.label}</button>
        ))}
      </div>

      <div style={{ padding: "24px 32px" }}>

        {/* OVERVIEW */}
        {activeTab === "overview" && (
          <div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "12px", marginBottom: "24px" }}>
              {[
                { label: "TREND",        score: current.trend_score,      weight: 0.28 },
                { label: "VOLATILITY",   score: current.volatility_score, weight: 0.23 },
                { label: "BREADTH",      score: current.breadth_score,    weight: 0.22 },
                { label: "FLOW",         score: current.flow_score,       weight: 0.17 },
                { label: "YIELD CURVE",  score: yc?.yc_score ?? current.yc_score ?? 50, weight: 0.10 },
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

        {/* DIMENSIONS */}
        {activeTab === "dimensions" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
            {[
              { label: "TREND",            score: current.trend_score,      dataKey: "trend_score",      color: "#00d4ff", weight: 0.28, desc: "SMA relationships, rate of change, 52-week proximity" },
              { label: "VOLATILITY",       score: current.volatility_score, dataKey: "volatility_score", color: "#ff6b6b", weight: 0.23, desc: "India VIX level, VIX change, realized vol vs historical" },
              { label: "BREADTH",          score: current.breadth_score,    dataKey: "breadth_score",    color: "#ffd166", weight: 0.22, desc: "% stocks above 50/200 DMA, advance/decline ratio" },
              { label: "FLOW & SENTIMENT", score: current.flow_score,       dataKey: "flow_score",       color: "#bf5af2", weight: 0.17, desc: "FII/DII flows normalised by market cap, DII trend" },
              { label: "YIELD CURVE",      score: yc?.yc_score ?? current.yc_score ?? 50, dataKey: "yc_score", color: "#00ff87", weight: 0.10, desc: "10Y-2Y G-Sec spread, policy gap, absolute level" },
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
                {history.length > 0 && dim.dataKey !== "yc_score" && (
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

        {/* HISTORY */}
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

        {/* SECTORS — unchanged */}
        {activeTab === "sectors" && (
          <div>
            {!sector ? (
              <NoData msg="Run python data_pipeline/sector_rotation.py to generate sector allocations." />
            ) : (
              <div>
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
                      <div style={{ fontSize: "10px", color: "#888", marginTop: "2px" }}>{Math.round(sector.cash_weight * 100)}% cash</div>
                    </div>
                  </div>
                  <div style={{ background: sector.sectors_held === 0 ? "rgba(255,45,85,0.08)" : "rgba(0,255,135,0.08)", border: `1px solid ${sector.sectors_held === 0 ? "#ff2d5530" : "#00ff8730"}`, borderRadius: "6px", padding: "10px 14px" }}>
                    <div style={{ fontSize: "10px", color: sector.sectors_held === 0 ? "#ff2d55" : "#00ff87", letterSpacing: "1px" }}>
                      {sector.sectors_held === 0 ? "STRONG BEAR — Full cash. No sector deployment." : `DEPLOYING INTO ${sector.sectors_held} SECTOR${sector.sectors_held > 1 ? "S" : ""}`}
                    </div>
                  </div>
                </div>
                {sector.allocations?.length > 0 && (
                  <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "18px", marginBottom: "16px", overflowX: "auto" }}>
                    <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "14px" }}>CURRENT ALLOCATIONS</div>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "monospace", fontSize: "12px", minWidth: "500px" }}>
                      <thead><tr style={{ borderBottom: "1px solid #1a1a2e" }}>{["SECTOR","WEIGHT","PRED RET","1M","3M","RATIONALE"].map(h => <td key={h} style={{ padding: "6px 8px", color: "#555", fontSize: "9px", letterSpacing: "1px" }}>{h}</td>)}</tr></thead>
                      <tbody>{sector.allocations.map((a, i) => (
                        <tr key={i} style={{ borderBottom: "1px solid #0d0d2a" }}>
                          <td style={{ padding: "10px 8px", color: "#00ff87", fontWeight: "700" }}>{a.sector}</td>
                          <td style={{ padding: "10px 8px", color: "#fff" }}>{(a.weight * 100).toFixed(1)}%</td>
                          <td style={{ padding: "10px 8px", color: a.predicted_ret >= 0 ? "#00ff87" : "#ff2d55" }}>{a.predicted_ret >= 0 ? "+" : ""}{a.predicted_ret?.toFixed(1)}%</td>
                          <td style={{ padding: "10px 8px", color: a.ret_1m >= 0 ? "#7dffb3" : "#ff6b6b" }}>{a.ret_1m >= 0 ? "+" : ""}{a.ret_1m?.toFixed(1)}%</td>
                          <td style={{ padding: "10px 8px", color: a.ret_3m >= 0 ? "#7dffb3" : "#ff6b6b" }}>{a.ret_3m >= 0 ? "+" : ""}{a.ret_3m?.toFixed(1)}%</td>
                          <td style={{ padding: "10px 8px", color: "#666", fontSize: "11px" }}>{a.rationale}</td>
                        </tr>
                      ))}</tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* SCREENER — unchanged */}
        {activeTab === "screener" && (
          <div>
            {!screener ? <NoData msg="Run python data_pipeline/stock_screener.py" /> : (
              <div>
                <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "18px", marginBottom: "16px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
                    <div>
                      <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "4px" }}>STOCK SCREENER — LAYER 3</div>
                      <div style={{ fontSize: "11px", color: "#888" }}>{screener.date} · {screener.stocks_screened} screened · {screener.stocks_passed} passed</div>
                    </div>
                    <div style={{ fontSize: "28px", fontWeight: "700", color: screener.stocks_selected === 0 ? "#ff6b6b" : "#00ff87", fontFamily: "monospace" }}>{screener.stocks_selected === 0 ? "WATCH" : screener.stocks_selected}</div>
                  </div>
                  <div style={{ background: screener.stocks_selected === 0 ? "rgba(255,107,107,0.08)" : "rgba(0,255,135,0.08)", border: `1px solid ${screener.stocks_selected === 0 ? "#ff6b6b30" : "#00ff8730"}`, borderRadius: "6px", padding: "10px 14px", fontSize: "10px", color: screener.stocks_selected === 0 ? "#ff6b6b" : "#00ff87", letterSpacing: "1px" }}>{screener.note}</div>
                </div>
                {screener.stocks?.length > 0 && (
                  <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "18px", overflowX: "auto" }}>
                    <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "14px" }}>SELECTED STOCKS — RANKED BY ML SCORE</div>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "monospace", fontSize: "11px", minWidth: "700px" }}>
                      <thead><tr style={{ borderBottom: "1px solid #1a1a2e" }}>{["#","TICKER","SCORE","MOM","QUAL","VOL","EARN","3M","1M","VOL%","MAX DD"].map(h => <td key={h} style={{ padding: "6px 8px", color: "#555", fontSize: "9px", letterSpacing: "1px" }}>{h}</td>)}</tr></thead>
                      <tbody>{screener.stocks.map((st, i) => (
                        <tr key={i} style={{ borderBottom: "1px solid #0d0d2a" }}>
                          <td style={{ padding: "10px 8px", color: "#555" }}>{i+1}</td>
                          <td style={{ padding: "10px 8px", color: "#fff", fontWeight: "700" }}>{st.name}</td>
                          <td style={{ padding: "10px 8px", color: "#00ff87" }}>{st.ml_score?.toFixed(1)}</td>
                          <td style={{ padding: "10px 8px", color: "#00d4ff" }}>{st.f_momentum?.toFixed(0)}</td>
                          <td style={{ padding: "10px 8px", color: "#bf5af2" }}>{st.f_quality?.toFixed(0)}</td>
                          <td style={{ padding: "10px 8px", color: "#ffd166" }}>{st.f_lowvol?.toFixed(0)}</td>
                          <td style={{ padding: "10px 8px", color: "#00ff87" }}>{st.f_earnings?.toFixed(0)}</td>
                          <td style={{ padding: "10px 8px", color: st.ret_3m >= 0 ? "#7dffb3" : "#ff6b6b" }}>{st.ret_3m >= 0 ? "+" : ""}{st.ret_3m?.toFixed(1)}%</td>
                          <td style={{ padding: "10px 8px", color: st.ret_1m >= 0 ? "#7dffb3" : "#ff6b6b" }}>{st.ret_1m >= 0 ? "+" : ""}{st.ret_1m?.toFixed(1)}%</td>
                          <td style={{ padding: "10px 8px", color: "#888" }}>{st.ann_vol_pct?.toFixed(0)}%</td>
                          <td style={{ padding: "10px 8px", color: "#ff6b6b" }}>{st.max_dd_pct?.toFixed(0)}%</td>
                        </tr>
                      ))}</tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* PORTFOLIO — unchanged */}
        {activeTab === "portfolio" && (
          <div>
            {!portfolio ? <NoData msg="Run python data_pipeline/portfolio_construction.py" /> : (
              <div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "12px", marginBottom: "16px" }}>
                  {[
                    { label: "EQUITY",    value: Math.round(portfolio.equity_allocation*100)+"%", color: portfolio.equity_allocation > 0.5 ? "#00ff87" : "#ffd166" },
                    { label: "CASH",      value: Math.round(portfolio.cash_allocation*100)+"%",   color: "#888" },
                    { label: "POSITIONS", value: portfolio.positions?.length || 0,                color: "#00d4ff" },
                    { label: "STATUS",    value: portfolio.status === "cash_mode" ? "CASH" : "ACTIVE", color: portfolio.status === "cash_mode" ? "#ff6b6b" : "#00ff87" },
                  ].map(m => (
                    <div key={m.label} style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "16px", textAlign: "center" }}>
                      <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "8px" }}>{m.label}</div>
                      <div style={{ fontSize: "28px", fontWeight: "700", color: m.color, fontFamily: "monospace" }}>{m.value}</div>
                    </div>
                  ))}
                </div>
                {portfolio.positions?.length > 0 && (
                  <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "18px", overflowX: "auto" }}>
                    <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "14px" }}>POSITIONS — MEAN-VARIANCE OPTIMISED</div>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "monospace", fontSize: "11px", minWidth: "500px" }}>
                      <thead><tr style={{ borderBottom: "1px solid #1a1a2e" }}>{["TICKER","PORTFOLIO WT","EQUITY WT","EXP RET","SECTOR","ML SCORE"].map(h => <td key={h} style={{ padding: "6px 8px", color: "#555", fontSize: "9px", letterSpacing: "1px" }}>{h}</td>)}</tr></thead>
                      <tbody>{portfolio.positions.map((p, i) => (
                        <tr key={i} style={{ borderBottom: "1px solid #0d0d2a" }}>
                          <td style={{ padding: "10px 8px", color: "#fff", fontWeight: "700" }}>{p.name}</td>
                          <td style={{ padding: "10px 8px", color: "#00ff87" }}>{(p.target_weight*100).toFixed(1)}%</td>
                          <td style={{ padding: "10px 8px", color: "#7dffb3" }}>{(p.equity_weight*100).toFixed(1)}%</td>
                          <td style={{ padding: "10px 8px", color: p.expected_ret >= 0 ? "#7dffb3" : "#ff6b6b" }}>{p.expected_ret >= 0 ? "+" : ""}{p.expected_ret?.toFixed(1)}%</td>
                          <td style={{ padding: "10px 8px", color: "#888" }}>{p.sector}</td>
                          <td style={{ padding: "10px 8px", color: "#00d4ff" }}>{p.ml_score?.toFixed(1)}</td>
                        </tr>
                      ))}</tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* RISK — unchanged */}
        {activeTab === "risk" && (
          <div>
            {!risk ? <NoData msg="Run python data_pipeline/risk_management.py" /> : (
              <div>
                <div style={{ background: `${risk.aggregate?.color}15`, border: `1px solid ${risk.aggregate?.color}40`, borderRadius: "8px", padding: "18px 20px", marginBottom: "16px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div>
                    <div style={{ fontSize: "9px", color: risk.aggregate?.color, letterSpacing: "2px", marginBottom: "4px" }}>RISK STATUS — {risk.date}</div>
                    <div style={{ fontSize: "14px", color: "#fff", fontWeight: "600" }}>{risk.aggregate?.overall_message}</div>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <div style={{ fontSize: "40px", fontWeight: "700", color: risk.aggregate?.color, fontFamily: "monospace", lineHeight: 1 }}>{risk.aggregate?.risk_score}</div>
                    <div style={{ fontSize: "9px", color: "#555", marginTop: "4px" }}>/100 risk score</div>
                  </div>
                </div>
                {risk.rules?.map((rule, i) => {
                  const colors = { INFO: "#00c896", WARN: "#ffd166", ALERT: "#ff6b6b", CRITICAL: "#ff2d55" };
                  const col = colors[rule.severity] || "#888";
                  return (
                    <div key={i} style={{ display: "flex", alignItems: "center", gap: "12px", padding: "10px 14px", background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "6px", marginBottom: "8px" }}>
                      <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: col, flexShrink: 0, boxShadow: rule.triggered ? `0 0 6px ${col}` : "none" }} />
                      <div style={{ width: "180px", flexShrink: 0 }}>
                        <div style={{ fontSize: "11px", color: "#fff", fontWeight: "600" }}>{rule.rule}</div>
                        <div style={{ fontSize: "9px", color: col, letterSpacing: "1px", marginTop: "2px" }}>{rule.severity}</div>
                      </div>
                      <div style={{ flex: 1, fontSize: "11px", color: "#888" }}>{rule.message}</div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* SIGNALS — unchanged */}
        {activeTab === "signals" && (
          <div>
            <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "18px", marginBottom: "16px" }}>
              <div style={{ fontSize: "9px", color: "#555", letterSpacing: "2px", marginBottom: "16px" }}>ALLOCATION FRAMEWORK</div>
              {[
                { range: "75–100", label: "Strong Bull",    color: "#00ff87", equity: "90–100%", style: "Momentum + Small/Midcap",       cash: "0–10%" },
                { range: "55–75",  label: "Mild Bull",      color: "#7dffb3", equity: "65–88%",  style: "Quality + Momentum mix",        cash: "12–35%" },
                { range: "40–55",  label: "Neutral/Choppy", color: "#ffd166", equity: "40–60%",  style: "Defensive + Low Vol + FI",      cash: "40–60%" },
                { range: "20–40",  label: "Mild Bear",      color: "#ff6b6b", equity: "15–35%",  style: "15% Index + 85% Liquid Fund",   cash: "65–85%" },
                { range: "0–20",   label: "Strong Bear",    color: "#ff2d55", equity: "0–15%",   style: "Capital Preservation / Liquid", cash: "85–100%" },
              ].map(r => (
                <div key={r.range} style={{ display: "flex", alignItems: "center", gap: "12px", padding: "10px 12px", borderRadius: "6px", marginBottom: "6px", background: current.regime_label === r.label ? REGIME_CONFIG[r.label]?.bg : "transparent", border: current.regime_label === r.label ? `1px solid ${r.color}30` : "1px solid transparent" }}>
                  <div style={{ width: "6px", height: "6px", borderRadius: "50%", background: r.color }} />
                  <div style={{ width: "50px", fontSize: "9px", color: "#555" }}>{r.range}</div>
                  <div style={{ flex: 1, fontSize: "11px", color: r.color }}>{r.label}</div>
                  <div style={{ fontSize: "10px", color: "#aaa", width: "80px" }}>Eq: {r.equity}</div>
                  <div style={{ fontSize: "10px", color: "#666", flex: 1 }}>{r.style}</div>
                  <div style={{ fontSize: "10px", color: "#888", width: "70px", textAlign: "right" }}>Cash: {r.cash}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* PERFORMANCE / BACKTEST */}
        {activeTab === "performance" && (
          <div>
            {!backtest ? <NoData msg="Run python data_pipeline/backtest_institutional.py" /> : (
              <div>
                <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "20px", marginBottom: "20px" }}>
                  <div style={{ fontSize: "12px", color: "#00d4ff", letterSpacing: "2px", fontWeight: "700", marginBottom: "4px" }}>PART 1: ASSET ALLOCATION STRESS TEST (18 YEARS)</div>
                  <div style={{ fontSize: "10px", color: "#888", marginBottom: "16px" }}>Nifty 50 Index + Cash only. Survivorship bias: 0.0%</div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px" }}>
                    <MetricCard label="STRATEGY CAGR" value={fmt(backtest.part1_metrics?.port_ann * 100, 1) + "%"} color="#00ff87" sub={"vs Nifty " + fmt(backtest.part1_metrics?.nifty_ann * 100, 1) + "%"} />
                    <MetricCard label="STRATEGY MAX DD" value={fmt(backtest.part1_metrics?.max_dd * 100, 1) + "%"} color="#00d4ff" sub={"vs Nifty " + fmt(backtest.part1_metrics?.nifty_dd * 100, 1) + "%"} highlight="#00d4ff" />
                    <MetricCard label="ALPHA (ANNUAL)" value={fmtPct(backtest.part1_metrics?.alpha_ann * 100, 1)} color="#ffd166" sub="drawdown insurance" />
                    <MetricCard label="SHARPE RATIO" value={fmt(backtest.part1_metrics?.sharpe, 2)} color="#bf5af2" sub="risk-adjusted" />
                  </div>
                </div>
                <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "8px", padding: "20px" }}>
                  <div style={{ fontSize: "12px", color: "#bf5af2", letterSpacing: "2px", fontWeight: "700", marginBottom: "4px" }}>PART 2: SECURITY SELECTION ALPHA TEST (5 YEARS)</div>
                  <div style={{ fontSize: "10px", color: "#888", marginBottom: "16px" }}>Nifty 500 universe. 1.5% p.a. synthetic delisting drag applied.</div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px" }}>
                    <MetricCard label="STRATEGY CAGR" value={fmt(backtest.part2_metrics?.port_ann * 100, 1) + "%"} color="#00ff87" sub={"vs Nifty " + fmt(backtest.part2_metrics?.nifty_ann * 100, 1) + "%"} />
                    <MetricCard label="ANNUAL ALPHA" value={"+" + fmt(Math.max(0, backtest.part2_metrics?.alpha_ann * 100), 1) + "%"} color="#bf5af2" sub="stock picking edge" highlight="#bf5af2" />
                    <MetricCard label="STRATEGY MAX DD" value={fmt(backtest.part2_metrics?.max_dd * 100, 1) + "%"} color="#00d4ff" sub="downside capped" />
                    <MetricCard label="SHARPE RATIO" value={fmt(backtest.part2_metrics?.sharpe, 2)} color="#ffd166" sub="risk-adjusted" />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* NEW TABS */}
        {activeTab === "yieldcurve"  && <YieldCurveTab yc={yc} />}
        {activeTab === "monitor"     && <RegimeMonitorTab monitor={monitor} />}
        {activeTab === "paper"       && <PaperPortfolioTab paper={paper} />}
        {activeTab === "fixedincome" && <FixedIncomeTab fi={fi} />}

      </div>
    </div>
  );
}