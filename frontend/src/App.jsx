import React, { useState, useEffect } from 'react';
import {
  Activity, CloudRain, Droplets, Mountain, Navigation, ShieldAlert,
  Wind, Sun, Cloud, Thermometer, Image as ImageIcon, UploadCloud,
  Bot, Satellite, AlertTriangle, CheckCircle, Radio, TrendingUp,
  Layers, Zap, BarChart2, BookOpen, X, ChevronLeft, ChevronRight,
  Moon, Sun as SunIcon, Brain
} from 'lucide-react';
import axios from 'axios';

const API = 'http://localhost:8000/api/v1';

const NOTEBOOKS = [
  { name: 'EDA & Data Analysis',       file: 'data_analysis.ipynb',         color: '#60a5fa', desc: 'Heatmaps, distributions, outlier detection' },
  { name: 'KMeans Clustering',          file: 'clustering.ipynb',            color: '#34d399', desc: 'Spatial risk zone segmentation' },
  { name: 'CNN Satellite Vision',       file: 'cnn_satellite.ipynb',         color: '#38bdf8', desc: 'U-Net CNN flood segmentation' },
  { name: 'LSTM Time-Series',           file: 'lstm_model.ipynb',            color: '#a78bfa', desc: 'Temporal flood prediction' },
  { name: 'LSTM Classifier',            file: 'lstm_classifier.ipynb',       color: '#f472b6', desc: 'Binary flood classification' },
  { name: 'Kaggle S5E3 XGBoost',        file: 'kaggle_s5e3_predictor.ipynb', color: '#fbbf24', desc: 'Rainfall prediction with XGBoost' },
];

const Slider = ({ icon: Icon, color, label, unit, min, max, step, value, onChange }) => {
  const [local, setLocal] = useState(value);
  const pct = ((local - min) / (max - min)) * 100;
  return (
    <div className="slider-row">
      <div className="slider-meta">
        <span className="slider-label"><Icon size={13} color={color} />{label}</span>
        <span className="slider-val" style={{ color }}>{local}<span className="slider-unit">{unit}</span></span>
      </div>
      <div className="slider-track">
        <div className="slider-fill" style={{ width: `${pct}%`, background: color }} />
        <input type="range" min={min} max={max} step={step} value={local}
          onChange={e => setLocal(parseFloat(e.target.value))}
          onMouseUp={() => onChange(local)} onTouchEnd={() => onChange(local)} />
      </div>
    </div>
  );
};

const Chip = ({ label, value, color = '#38bdf8' }) => (
  <div className="chip">
    <span className="chip-label">{label}</span>
    <span className="chip-value" style={{ color }}>{value}</span>
  </div>
);

const SectionHeader = ({ icon: Icon, color, title, badge }) => (
  <div className="section-header">
    <div className="section-icon" style={{ background: `${color}18`, border: `1px solid ${color}40` }}>
      <Icon size={16} color={color} />
    </div>
    <span className="section-title">{title}</span>
    {badge && <span className="section-badge" style={{ color, borderColor: `${color}50`, background: `${color}12` }}>{badge}</span>}
  </div>
);

const ShapChart = ({ values, floodColor }) => {
  if (!values || values.length === 0) return null;
  const top = values.slice(0, 6);
  return (
    <div className="shap-section">
      <div className="shap-title"><Brain size={12} />XAI — Feature Impact (SHAP)</div>
      {top.map(s => (
        <div key={s.feature} className="shap-row">
          <span className="shap-label">{s.feature}</span>
          <div className="shap-bar-wrap">
            <div className="shap-bar" style={{
              width: `${s.pct}%`,
              background: s.direction === 'positive' ? floodColor : '#64748b'
            }} />
          </div>
          <span className="shap-pct">{s.pct}%</span>
        </div>
      ))}
    </div>
  );
};

export default function App() {
  const [theme, setTheme] = useState('light');
  const [tab, setTab] = useState('dashboard');
  const [plots, setPlots] = useState([]);
  const [lightbox, setLightbox] = useState(null);

  const [params, setParams] = useState({ rainfall: 145, temperature: 28.4, humidity: 85, river_discharge: 600, water_level: 5.2, elevation: 45, risk_zone: 1 });
  const [kaggleParams, setKaggleParams] = useState({ pressure: 1015, maxtemp: 28, temparature: 24, mintemp: 20, dewpoint: 18, humidity: 85, cloud: 90, sunshine: 1.5, winddirection: 180, windspeed: 20 });

  const [prediction, setPrediction] = useState(null);
  const [kagglePrediction, setKagglePrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [visionFile, setVisionFile] = useState(null);
  const [visionUrl, setVisionUrl] = useState('');
  const [visionResult, setVisionResult] = useState(null);
  const [visionLoading, setVisionLoading] = useState(false);
  const [visionPreview, setVisionPreview] = useState(null);
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  useEffect(() => { const t = setInterval(() => setTime(new Date()), 1000); return () => clearInterval(t); }, []);
  useEffect(() => { axios.get(`${API}/plots`).then(r => setPlots(r.data)).catch(() => {}); }, []);

  useEffect(() => {
    if (!lightbox) return;
    const h = e => {
      if (e.key === 'ArrowRight') setLightbox(l => ({ ...l, index: Math.min(l.index + 1, l.list.length - 1) }));
      if (e.key === 'ArrowLeft')  setLightbox(l => ({ ...l, index: Math.max(l.index - 1, 0) }));
      if (e.key === 'Escape')     setLightbox(null);
    };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [lightbox]);

  useEffect(() => {
    setLoading(true);
    const attempt = (retries) => {
      axios.post(`${API}/predict`, params)
        .then(r => { setPrediction(r.data); setLoading(false); })
        .catch(() => {
          if (retries > 0) setTimeout(() => attempt(retries - 1), 2000);
          else { setPrediction({ prediction_status: 'API OFFLINE', flood_probability: 0 }); setLoading(false); }
        });
    };
    attempt(2);
  }, [params]);

  useEffect(() => {
    axios.post(`${API}/kaggle/predict`, kaggleParams)
      .then(r => setKagglePrediction(r.data))
      .catch(() => setKagglePrediction({ error: 'Model offline' }));
  }, [kaggleParams]);

  const runVision = async () => {
    setVisionLoading(true); setVisionResult(null);
    try {
      let payload = {};
      let previewUrl = visionUrl || null;
      if (visionFile) {
        const b64 = await new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(r.result.split(',')[1]); r.onerror = rej; r.readAsDataURL(visionFile); });
        previewUrl = URL.createObjectURL(visionFile);
        payload = { image_base64: b64 };
      } else { payload = { image_url: visionUrl }; }
      setVisionPreview(previewUrl);
      const r = await axios.post(`${API}/vision`, payload);
      setVisionResult(r.data);
    } catch (e) {
      setVisionResult({ status: 'ERROR', message: e?.response?.data?.detail || e?.message || 'Request failed' });
    }
    setVisionLoading(false);
  };

  const statusColor = s => {
    if (!s) return '#64748b';
    if (s.includes('DANGER') || s.includes('OFFLINE')) return '#ef4444';
    if (s.includes('SAFE')) return '#10b981';
    return '#f59e0b';
  };

  const floodColor = statusColor(prediction?.prediction_status);
  const rainColor = kagglePrediction?.status === 'RAINING' ? '#3b82f6' : kagglePrediction?.status === 'CLEAR' ? '#10b981' : '#64748b';
  const groups = [...new Set(plots.map(p => p.group))];

  return (
    <div className="app">
      {/* ── Topbar ── */}
      <header className="topbar">
        <div className="topbar-left">
          <div className="logo-mark"><Zap size={18} color="#fff" /></div>
          <div>
            <div className="logo-title">Aqua<span>Alert</span> AI</div>
            <div className="logo-sub">Flood Detection &amp; Early Warning Intelligence</div>
          </div>
        </div>
        <div className="topbar-right">
          <div className="status-pill online"><span className="pulse-dot" />API ONLINE</div>
          <div className="clock">{time.toLocaleTimeString('en-GB')}</div>
          <div className="date-chip">{time.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })}</div>
          <button className="theme-toggle" onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')} title="Toggle theme">
            {theme === 'dark' ? <SunIcon size={16} /> : <Moon size={16} />}
          </button>
        </div>
      </header>

      {/* ── Tab Bar ── */}
      <nav className="tab-bar">
        <button className={`tab-btn ${tab === 'dashboard' ? 'active' : ''}`} onClick={() => setTab('dashboard')}>
          <Activity size={15} />Dashboard
        </button>
        <button className={`tab-btn ${tab === 'analytics' ? 'active' : ''}`} onClick={() => setTab('analytics')}>
          <BarChart2 size={15} />Analytics &amp; Notebooks
        </button>
      </nav>

      <main className="main">

        {/* ══════════ DASHBOARD ══════════ */}
        {tab === 'dashboard' && <>

          <div className="row-2col">
            {/* Telemetry */}
            <div className="card">
              <SectionHeader icon={Navigation} color="#60a5fa" title="Topographic Telemetry" badge="LIVE" />
              <div className="card-body">
                <Slider icon={CloudRain} color="#60a5fa" label="Rainfall" unit=" mm" min={0} max={800} step={1} value={params.rainfall} onChange={v => setParams(p => ({ ...p, rainfall: v }))} />
                <Slider icon={Droplets} color="#34d399" label="River Discharge" unit=" m³/s" min={0} max={3000} step={10} value={params.river_discharge} onChange={v => setParams(p => ({ ...p, river_discharge: v }))} />
                <Slider icon={Activity} color="#fbbf24" label="Water Level" unit=" m" min={0} max={25} step={0.1} value={params.water_level} onChange={v => setParams(p => ({ ...p, water_level: v }))} />
                <Slider icon={Mountain} color="#a78bfa" label="Elevation" unit=" m" min={-5} max={300} step={1} value={params.elevation} onChange={v => setParams(p => ({ ...p, elevation: v }))} />
                <div className="zone-row">
                  <span className="zone-label">Risk Zone</span>
                  <div className="zone-btns">
                    {[['Zone 1', 0, '#10b981'], ['Zone 2', 1, '#f59e0b'], ['Zone 3', 2, '#ef4444']].map(([lbl, val, col]) => (
                      <button key={val} className="zone-btn"
                        style={params.risk_zone === val ? { borderColor: col, color: col, background: `${col}20`, boxShadow: `0 0 12px ${col}30` } : {}}
                        onClick={() => setParams(p => ({ ...p, risk_zone: val }))}>{lbl}</button>
                    ))}
                  </div>
                </div>
                <p style={{ fontSize: '0.68rem', color: 'var(--muted2)', lineHeight: 1.4 }}>
                  Zone = KMeans cluster (0=safe · 1=moderate · 2=danger) — weighted feature in XGBoost.
                </p>
              </div>
            </div>

            {/* Flood Risk + SHAP */}
            <div className="card flood-card">
              <SectionHeader icon={ShieldAlert} color={floodColor} title="Flood Risk Assessment" badge="XGBoost + SHAP" />
              <div className="flood-body">
                <div className="flood-ring">
                  <svg viewBox="0 0 120 120" className="ring-svg">
                    <circle cx="60" cy="60" r="50" className="ring-bg" />
                    <circle cx="60" cy="60" r="50" className="ring-fg"
                      style={{ stroke: floodColor, strokeDasharray: `${(prediction?.flood_probability ?? 0) * 3.14} 314` }} />
                  </svg>
                  <div className="ring-inner">
                    <span className="ring-pct" style={{ color: floodColor }}>{loading ? '—' : `${(prediction?.flood_probability ?? 0).toFixed(1)}%`}</span>
                    <span className="ring-sub">Flood Risk</span>
                  </div>
                </div>
                <div className="flood-status" style={{ color: floodColor }}>{loading ? 'COMPUTING...' : (prediction?.prediction_status ?? 'AWAITING')}</div>
                {prediction?.recommended_action && (
                  <div className="action-banner" style={{ borderColor: `${floodColor}50`, background: `${floodColor}10` }}>
                    <AlertTriangle size={14} color={floodColor} /><span>{prediction.recommended_action}</span>
                  </div>
                )}
                <ShapChart values={prediction?.shap_values} floodColor={floodColor} />
                <div className="flood-chips">
                  <Chip label="Rainfall" value={`${params.rainfall} mm`} color="#60a5fa" />
                  <Chip label="Discharge" value={`${params.river_discharge} m³/s`} color="#34d399" />
                  <Chip label="Water Lvl" value={`${params.water_level} m`} color="#fbbf24" />
                </div>
              </div>
            </div>
          </div>

          <div className="row-2col">
            {/* Atmospheric */}
            <div className="card">
              <SectionHeader icon={Cloud} color="#f472b6" title="Atmospheric Engine" badge="Kaggle S5E3" />
              <div className="card-body">
                <Slider icon={Thermometer} color="#f87171" label="Temperature" unit="°C" min={-10} max={50} step={0.1} value={kaggleParams.temparature} onChange={v => setKaggleParams(p => ({ ...p, temparature: v }))} />
                <Slider icon={Cloud} color="#60a5fa" label="Cloud Cover" unit="%" min={0} max={100} step={1} value={kaggleParams.cloud} onChange={v => setKaggleParams(p => ({ ...p, cloud: v }))} />
                <Slider icon={Wind} color="#a78bfa" label="Wind Speed" unit=" km/h" min={0} max={80} step={0.5} value={kaggleParams.windspeed} onChange={v => setKaggleParams(p => ({ ...p, windspeed: v }))} />
                <Slider icon={Sun} color="#fbbf24" label="Sunshine" unit=" hrs" min={0} max={15} step={0.1} value={kaggleParams.sunshine} onChange={v => setKaggleParams(p => ({ ...p, sunshine: v }))} />
              </div>
            </div>

            {/* Rain */}
            <div className="card rain-card">
              <SectionHeader icon={TrendingUp} color={rainColor} title="Rainfall Prediction" badge="XGBoost" />
              <div className="rain-body">
                <div className="rain-icon-wrap" style={{ background: `${rainColor}15`, border: `1px solid ${rainColor}30` }}>
                  <CloudRain size={48} color={rainColor} />
                </div>
                <div className="rain-status" style={{ color: rainColor }}>{kagglePrediction?.status ?? 'LOADING'}</div>
                {kagglePrediction?.rainfall_probability !== undefined && (
                  <div className="rain-prob">
                    <div className="rain-bar-wrap"><div className="rain-bar-fill" style={{ width: `${kagglePrediction.rainfall_probability}%`, background: rainColor }} /></div>
                    <span style={{ color: rainColor, fontWeight: 700 }}>{kagglePrediction.rainfall_probability.toFixed(1)}%</span>
                  </div>
                )}
                {kagglePrediction?.recommended_action && <p className="rain-action">{kagglePrediction.recommended_action}</p>}
                {kagglePrediction?.error && <p className="rain-error">{kagglePrediction.error}</p>}
                <div className="flood-chips" style={{ marginTop: '1rem' }}>
                  <Chip label="Temp" value={`${kaggleParams.temparature}°C`} color="#f87171" />
                  <Chip label="Cloud" value={`${kaggleParams.cloud}%`} color="#60a5fa" />
                  <Chip label="Wind" value={`${kaggleParams.windspeed} km/h`} color="#a78bfa" />
                </div>
              </div>
            </div>
          </div>

          {/* CNN Vision */}
          <div className="card">
            <SectionHeader icon={Satellite} color="#38bdf8" title="Satellite Flood Vision — CNN Segmentation" badge="LIVE" />
            <div className="vision-grid">
              <div className="vision-upload">
                <label htmlFor="vf" className={`dropzone ${visionFile ? 'active' : ''}`}>
                  <UploadCloud size={28} color={visionFile ? '#38bdf8' : undefined} style={{ color: visionFile ? '#38bdf8' : 'var(--muted)' }} />
                  <span className="dz-title">{visionFile ? visionFile.name : 'Drop satellite image here'}</span>
                  <span className="dz-sub">PNG · JPG · TIFF · GeoTIFF</span>
                  <input id="vf" type="file" accept="image/*" style={{ display: 'none' }}
                    onChange={e => { setVisionFile(e.target.files?.[0] || null); setVisionUrl(''); }} />
                </label>
                <div className="or-divider"><span>or paste URL</span></div>
                <div className="url-input-wrap">
                  <ImageIcon size={14} color="var(--muted)" />
                  <input className="url-input" value={visionUrl}
                    onChange={e => { setVisionUrl(e.target.value); setVisionFile(null); }}
                    placeholder="https://... or data/raw/flood-area/Image/0.jpg" />
                </div>
                <button className="infer-btn" disabled={visionLoading || (!visionFile && !visionUrl)} onClick={runVision}>
                  {visionLoading ? <><span className="spinner" />Processing...</> : <><Bot size={15} />Run CNN Inference</>}
                </button>
                {(visionFile || visionUrl) && !visionLoading && (
                  <button className="clear-btn" onClick={() => { setVisionFile(null); setVisionUrl(''); setVisionResult(null); setVisionPreview(null); }}>Clear</button>
                )}
              </div>

              <div className="vision-result">
                {!visionResult && !visionLoading && (
                  <div className="vr-empty">
                    <Layers size={36} style={{ color: 'var(--muted)' }} />
                    <span>No analysis yet</span>
                    <span className="vr-hint">Upload an image or provide a URL to begin CNN inference</span>
                  </div>
                )}
                {visionLoading && (
                  <div className="vr-loading">
                    <div className="scan-bar"><div className="scan-fill" /></div>
                    <Radio size={20} color="#38bdf8" />
                    <span>PROCESSING SATELLITE FEED...</span>
                  </div>
                )}
                {visionResult && !visionLoading && (() => {
                  const err = visionResult.status === 'ERROR';
                  const flooded = visionResult.flood_state === 'FLOODED';
                  const score = visionResult.flood_percentage ?? 0;
                  const ac = err ? '#ef4444' : flooded ? '#f97316' : '#10b981';
                  const Ic = err ? AlertTriangle : flooded ? AlertTriangle : CheckCircle;
                  return (
                    <div className="vr-result" style={{ '--ac': ac }}>
                      <div className="vr-top">
                        <div className="vr-dot" style={{ background: ac, boxShadow: `0 0 10px ${ac}` }} />
                        <span className="vr-state" style={{ color: ac }}>{err ? 'ERROR' : visionResult.flood_state}</span>
                        {!err && <span className="vr-mode">{visionResult.mode}</span>}
                        <Ic size={18} color={ac} style={{ marginLeft: 'auto' }} />
                      </div>
                      {!err && <>
                        <div className="vr-score-row">
                          <span className="vr-score-label">Flood Confidence</span>
                          <span className="vr-score-val" style={{ color: ac }}>{score}%</span>
                        </div>
                        <div className="vr-bar"><div className="vr-bar-fill" style={{ width: `${score}%`, background: `linear-gradient(90deg,${ac}80,${ac})` }} /></div>
                        <div className="vr-metrics">
                          <div className="vr-metric"><span className="vr-metric-label">Risk Level</span><span className="vr-metric-val" style={{ color: ac }}>{score > 60 ? 'HIGH' : score > 30 ? 'MODERATE' : 'LOW'}</span></div>
                          <div className="vr-metric"><span className="vr-metric-label">Model</span><span className="vr-metric-val">CNN U-Net</span></div>
                          <div className="vr-metric"><span className="vr-metric-label">Resolution</span><span className="vr-metric-val">128×128</span></div>
                        </div>
                      </>}
                      <p className="vr-msg">{err ? visionResult.message : flooded ? `⚠ High flood probability. ${score}% of the scanned area exhibits flood signatures.` : `✓ Area appears safe. ${score}% flood signature — within acceptable threshold.`}</p>
                      {/* GradCAM */}
                      {!err && visionResult.gradcam && (
                        <div>
                          <div className="gradcam-wrap" style={{ position: 'relative' }}>
                            {visionPreview && (
                              <img className="gradcam-base" src={visionPreview} alt="original"
                                style={{ width: '100%', display: 'block', borderRadius: '8px' }} />
                            )}
                            <img
                              style={{
                                position: visionPreview ? 'absolute' : 'relative',
                                inset: 0, width: '100%', height: '100%',
                                borderRadius: '8px', mixBlendMode: visionPreview ? 'screen' : 'normal',
                                display: 'block'
                              }}
                              src={`data:image/png;base64,${visionResult.gradcam}`}
                              alt="gradcam heatmap"
                            />
                          </div>
                          <div className="gradcam-label">GradCAM — Flood Activation Heatmap (red = high risk)</div>
                        </div>
                      )}
                    </div>
                  );
                })()}
              </div>
            </div>
          </div>
        </>}

        {/* ══════════ ANALYTICS ══════════ */}
        {tab === 'analytics' && (
          <div className="analytics-page">
            <div className="analytics-section">
              <div className="analytics-heading">
                <BookOpen size={18} color="var(--blue)" />
                <span>Model Notebooks</span>
                <span className="analytics-count">{NOTEBOOKS.length} notebooks</span>
              </div>
              <div className="nb-grid">
                {NOTEBOOKS.map(nb => (
                  <div key={nb.file} className="nb-card">
                    <div className="nb-accent" style={{ background: nb.color }} />
                    <div className="nb-info">
                      <span className="nb-name" style={{ color: nb.color }}>{nb.name}</span>
                      <span className="nb-desc">{nb.desc}</span>
                      <span className="nb-file">{nb.file}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {groups.map(group => {
              const gp = plots.filter(p => p.group === group);
              return (
                <div key={group} className="analytics-section">
                  <div className="analytics-heading">
                    <BarChart2 size={18} color="var(--blue)" />
                    <span>{group} Visualizations</span>
                    <span className="analytics-count">{gp.length} charts</span>
                  </div>
                  <div className="plot-grid">
                    {gp.map((p, i) => (
                      <div key={p.file} className="plot-card" onClick={() => setLightbox({ index: i, list: gp })}>
                        <div className="plot-img-wrap">
                          <img src={`${API}/plots/image?file=${encodeURIComponent(p.file)}`} alt={p.title} loading="lazy" />
                          <div className="plot-overlay"><ImageIcon size={20} color="#fff" /></div>
                        </div>
                        <div className="plot-label">{p.title}</div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}

            {plots.length === 0 && (
              <div className="analytics-empty">
                <BarChart2 size={44} style={{ color: 'var(--muted)' }} />
                <span>No plots found</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--muted2)' }}>Run the notebooks to generate visualizations in outputs/plots/</span>
              </div>
            )}
          </div>
        )}
      </main>

      {/* ── Lightbox ── */}
      {lightbox && (
        <div className="lightbox-overlay" onClick={() => setLightbox(null)}>
          <div className="lightbox-box" onClick={e => e.stopPropagation()}>
            <button className="lb-close" onClick={() => setLightbox(null)}><X size={18} /></button>
            <button className="lb-nav lb-prev" disabled={lightbox.index === 0} onClick={() => setLightbox(l => ({ ...l, index: l.index - 1 }))}><ChevronLeft size={24} /></button>
            <img className="lb-img" src={`${API}/plots/image?file=${encodeURIComponent(lightbox.list[lightbox.index].file)}`} alt={lightbox.list[lightbox.index].title} />
            <button className="lb-nav lb-next" disabled={lightbox.index === lightbox.list.length - 1} onClick={() => setLightbox(l => ({ ...l, index: l.index + 1 }))}><ChevronRight size={24} /></button>
            <div className="lb-caption">
              <span>{lightbox.list[lightbox.index].title}</span>
              <span className="lb-counter">{lightbox.index + 1} / {lightbox.list.length}</span>
            </div>
          </div>
        </div>
      )}

      <footer className="footer">
        <span>AquaAlert AI — Flood Intelligence Platform</span>
        <span>XGBoost · LSTM · CNN U-Net · SHAP XAI</span>
        <span>FastAPI v1 · Port 8000</span>
      </footer>
    </div>
  );
}
