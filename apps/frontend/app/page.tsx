'use client';
import { useEffect, useState } from 'react';

type Patient = { patient_id: string; hours: number };
type Observation = { hour: number; HR: number | null; O2Sat: number | null; Temp: number | null; SBP: number | null; Resp: number | null; Lactate: number | null };
type Prediction = { risk_score: number | null; model_version: string | null; missing_fraction_6h: number; abstained: boolean; reason: string | null; trajectory: { hour: number; risk: number | null }[]; contributions: { feature: string; risk_delta: number }[] };
type Evidence = { id: string; title: string; url: string; text?: string; kind: string };
type Report = { final_report: string; trace: { node: string; status: string }[]; evidence?: Evidence[]; verification_passed: boolean };
type Model = { version: string; family: string; metrics: Record<string, unknown> };
type Extraction = { method: string; measurements: { field: string; value: number; unit: string }[]; limitations: string; rejected: { field: string; reason: string }[] };

async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api/${path}`, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : data.error?.message || `Request failed (${response.status})`);
  return data;
}

function Icon({ name, size = 20 }: { name: string; size?: number }) {
  const paths: Record<string, React.ReactNode> = {
    pulse: <path d="M2 12h5l3-8 4 16 3-8h5" />,
    grid: <><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></>,
    book: <><path d="M4 4h6l2 2 2-2h6v16h-6l-2 1-2-1H4zM12 6v15"/></>,
    file: <><path d="M6 3h8l4 4v14H6zM14 3v5h4M9 12h6M9 16h6"/></>,
    model: <><path d="M12 2l9 5v10l-9 5-9-5V7zM3 7l9 5 9-5M12 12v10"/></>,
    arrow: <path d="M5 12h14M13 6l6 6-6 6" />,
    check: <><path d="M12 2l8 3v6c0 5-8 11-8 11S4 16 4 11V5zM8 11l3 3 5-6"/></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name] || paths.pulse}</svg>;
}

function RiskChart({ data }: { data: Prediction['trajectory'] }) {
  const usable = data.filter(d => d.risk !== null);
  if (!usable.length) return <div className="empty">No model predictions yet. Train the demo model to see its trajectory.</div>;
  const maxHour = Math.max(...data.map(d => d.hour), 1);
  const point = (d: { hour: number; risk: number | null }) => `${46 + d.hour / maxHour * 664},${200 - (d.risk || 0) * 170}`;
  let drawing = '';
  let connected = false;
  data.forEach(d => { if (d.risk === null) connected = false; else { drawing += `${connected ? 'L' : 'M'}${point(d)} `; connected = true; } });
  return <svg className="chart" viewBox="0 0 740 238" role="img" aria-label="Hourly synthetic model score, from zero to one hundred percent">
    {[0, .25, .5, .75, 1].map(value => <g key={value}><line x1="46" x2="716" y1={200 - value * 170} y2={200 - value * 170} stroke="#e8eeef" strokeDasharray="4 5"/><text x="3" y={204 - value * 170} fill="#829194" fontSize="10">{value * 100}%</text></g>)}
    <path d={drawing} fill="none" stroke="#008f85" strokeWidth="3" strokeLinejoin="round"/>
    {[0, .25, .5, .75, 1].map(f => <text key={f} x={46 + f * 664} y="226" textAnchor="middle" fill="#829194" fontSize="11">Hour {Math.round(f * maxHour)}</text>)}
    <circle cx={46 + usable.at(-1)!.hour / maxHour * 664} cy={200 - usable.at(-1)!.risk! * 170} r="5" fill="#008f85" stroke="white" strokeWidth="2"/>
  </svg>;
}

export default function Home() {
  const [tab, setTab] = useState('overview');
  const [patients, setPatients] = useState<Patient[]>([]);
  const [patient, setPatient] = useState('DEMO-001');
  const [hour, setHour] = useState(35);
  const [timeline, setTimeline] = useState<Observation[]>([]);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [question, setQuestion] = useState('How has the model score changed over the last 12 hours?');
  const [report, setReport] = useState<Report | null>(null);
  const [evidence, setEvidence] = useState<Evidence[]>([]);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [document, setDocument] = useState('SYNTHETIC LAB REPORT\nLactate: 2.4 mmol/L\nWBC: 12.1 10^9/L\nTemperature: 38.2 C');
  const [extraction, setExtraction] = useState<Extraction | null>(null);

  useEffect(() => {
    Promise.all([api<Patient[]>('api/v1/patients'), api<Model[]>('api/v1/models')])
      .then(([p, m]) => { setPatients(p); setModels(m); }).catch(e => setError(e.message));
  }, []);
  useEffect(() => {
    let active = true;
    setLoading(true); setError(''); setReport(null); setPrediction(null);
    Promise.all([api<Observation[]>(`api/v1/patients/${patient}/timeline`), api<Prediction>('api/v1/predict', { patient_id: patient, hour })])
      .then(([t, p]) => { if (active) { setTimeline(t); setPrediction(p); } })
      .catch(e => { if (active) setError(e.message); }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [patient, hour]);

  async function query() {
    setBusy(true); setError('');
    try { setReport(await api<Report>('api/v1/agent/query', { patient_id: patient, hour, question })); }
    catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }
  async function search(live = false) {
    setBusy(true); setError('');
    try { setEvidence(await api<Evidence[]>(`api/v1/evidence${live ? '/pubmed' : ''}?q=${encodeURIComponent(question)}`)); }
    catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }
  async function extract() {
    setBusy(true); setError('');
    try { setExtraction(await api<Extraction>('api/v1/documents/analyze', { text: document, synthetic: true })); }
    catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }
  const observation = timeline.filter(t => t.hour <= hour).at(-1);
  const navigation = [['overview', 'grid', 'Patient workspace'], ['evidence', 'book', 'Evidence library'], ['documents', 'file', 'Document lab'], ['models', 'model', 'Model registry']];
  const percent = prediction?.risk_score == null ? '—' : `${(prediction.risk_score * 100).toFixed(1)}%`;
  const currentPatient = patients.find(p => p.patient_id === patient);

  return <div className="shell">
    <aside className="sidebar">
      <a className="brand" href="/"><span className="brand-icon"><Icon name="pulse" size={25}/></span><span>Healthcare<span className="brand-sub">SepsisAI</span></span></a>
      <div className="workspace-label">RESEARCH WORKSPACE <span>v0.1</span></div>
      <nav>{navigation.map(([id, icon, label]) => <button key={id} className={tab === id ? 'nav-item active' : 'nav-item'} onClick={() => { setTab(id); setError(''); }}><Icon name={icon}/>{label}</button>)}</nav>
      <div className="sidebar-note"><Icon name="check"/><strong>Built for exploration</strong><p>Synthetic records. Traceable tools. Human review at every step.</p><span className="tiny-pill">RESEARCH ONLY</span></div>
      <div className="profile"><span className="avatar">MK</span><span>Healthcare research<small>Independent project</small></span></div>
    </aside>

    <main>
      <header className="topbar"><span>Workspace <span className="breadcrumb">/ {navigation.find(n => n[0] === tab)?.[2]}</span></span><span className="demo-badge"><i/> Synthetic demo environment</span></header>
      <div className="content">
        <div className="heading"><div><div className="eyebrow">HEALTHCARE SEPSISAI</div><h1>{tab === 'overview' ? 'A clearer view of every hour.' : tab === 'evidence' ? 'Evidence, with a source.' : tab === 'documents' ? 'From report to structured data.' : 'Know what powers the prediction.'}</h1><p>{tab === 'overview' ? 'Explore patient trajectories, model signals, and the evidence behind them.' : 'An inspectable research workflow, from input to output.'}</p></div><span className="outline-pill">Research prototype</span></div>
        <div className="notice"><Icon name="check" size={18}/><span><strong>For research, not clinical decisions.</strong> Scores come from a synthetic-data model and are not validated clinical probabilities.</span></div>
        {error && <div className="error" role="alert">{error}</div>}

        {tab === 'overview' && <>
          <section className="patient-toolbar"><div className="patient-symbol">P</div><div><span className="label">PATIENT RECORD</span><label className="sr-only" htmlFor="patient">Select synthetic patient</label><select id="patient" value={patient} onChange={e => setPatient(e.target.value)}>{(patients.length ? patients : [{ patient_id: 'DEMO-001', hours: 36 }]).map(p => <option key={p.patient_id}>{p.patient_id}</option>)}</select></div><span className="muted">Synthetic ICU trajectory</span><div className="hour-control"><label htmlFor="hour">Observation hour <strong>{hour}</strong></label><input id="hour" type="range" min="0" max={(currentPatient?.hours || 36) - 1} value={hour} onChange={e => setHour(Number(e.target.value))}/></div><span className="small-status">{loading ? 'Loading…' : `${hour + 1} hours in view`}</span></section>
          <div className="vitals">{([['HR', 'Heart rate', 'bpm'], ['Temp', 'Temperature', '°C'], ['SBP', 'Systolic BP', 'mmHg'], ['Lactate', 'Lactate', 'mmol/L']] as const).map(([key, title, unit]) => <article className="vital" key={key}><div><span>{title}</span><Icon name="pulse" size={17}/></div><strong>{loading ? '…' : observation?.[key] ?? '—'} <small>{unit}</small></strong><span className="vital-caption">{observation?.[key] == null ? 'Measurement missing' : `Observed at hour ${hour}`}</span></article>)}</div>
          <div className="analysis-grid"><section className="panel trajectory"><div className="panel-title"><div><h2>Model score over time</h2><p>Hourly trajectory · available history only</p></div><span className="legend"><i/> Model score</span></div><div className="score-line"><strong>{loading ? '…' : percent}</strong><span>Current research score<small>{prediction?.model_version || 'No model loaded'}</small></span></div><RiskChart data={prediction?.trajectory || []}/>{prediction?.abstained && <div className="abstention">{prediction.reason}</div>}</section>
          <section className="panel contributions"><div className="panel-title"><div><h2>What moved the model?</h2><p>Grouped feature occlusion</p></div></div><p className="contribution-note">Change in score when a feature group is replaced by its training median.</p><div className="bars">{prediction?.contributions.slice(0, 5).map(item => <div className="bar-row" key={item.feature}><div><span>{item.feature}</span><strong>{(item.risk_delta * 100).toFixed(1)} pp</strong></div><div className="bar-track"><div style={{ width: `${Math.max(2, Math.min(100, Math.abs(item.risk_delta) * 400))}%`, background: item.risk_delta >= 0 ? '#008f85' : '#7a97ae' }}/></div></div>)}</div><div className="quality"><span>Missing measurements · last 6h</span><strong>{prediction ? `${(prediction.missing_fraction_6h * 100).toFixed(0)}%` : '—'}</strong></div><p className="fine-print">Model sensitivity does not establish clinical causation.</p></section></div>
          <section className="panel agent-panel"><div className="panel-title"><div className="agent-heading"><span className="agent-icon"><Icon name="model"/></span><div><h2>Ask the research workflow</h2><p>Patient data → prediction → explanation → evidence → verification</p></div></div><span className="tiny-pill teal">LANGGRAPH</span></div><div className="query-row"><label className="sr-only" htmlFor="query">Research question</label><input id="query" value={question} onChange={e => setQuestion(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !busy) query(); }}/><button className="primary" disabled={busy || loading || question.trim().length < 3} onClick={query}>{busy ? 'Running…' : 'Run analysis'}<Icon name="arrow" size={17}/></button></div><div className="chips">{['Explain model sensitivity and missingness', 'What are the limitations of sepsis prediction?'].map(q => <button key={q} onClick={() => setQuestion(q)}>{q}</button>)}</div>{report && <div className="report" aria-live="polite"><div className="trace">{report.trace.map(t => <span key={t.node}>{t.node.replaceAll('_', ' ')} <b>· {t.status.replaceAll('_', ' ')}</b></span>)}</div><p>{report.final_report}</p><div className="evidence-links">{report.evidence?.map(e => <a key={e.id} href={e.url.startsWith('https://') ? e.url : undefined} target="_blank" rel="noreferrer">{e.title} ↗</a>)}</div></div>}</section>
        </>}

        {tab === 'evidence' && <section className="panel"><div className="panel-title"><div><h2>Research reference search</h2><p>Local lexical index or live PubMed metadata. No generated medical claims.</p></div></div><div className="query-row"><label className="sr-only" htmlFor="search">Search evidence</label><input id="search" value={question} onChange={e => setQuestion(e.target.value)}/><button className="primary" disabled={busy} onClick={() => search()}>Search notes</button><button className="secondary" disabled={busy} onClick={() => search(true)}>Search PubMed ↗</button></div><div className="evidence-list">{evidence.length === 0 && <div className="empty">Search the research notes or retrieve current article metadata from PubMed.</div>}{evidence.map(e => <article key={e.id}><span className="label">{e.kind.replaceAll('_', ' ')} · {e.id}</span><h3>{e.title}</h3>{e.text && <p>{e.text}</p>}{e.url.startsWith('https://') ? <a href={e.url} target="_blank" rel="noreferrer">Open original source ↗</a> : <span className="muted">Repository reference: {e.url}</span>}</article>)}</div></section>}
        {tab === 'documents' && <section className="panel"><div className="panel-title"><div><h2>Synthetic document lab</h2><p>Text extraction with explicit units and plausibility validation. Image/VLM analysis is not implemented.</p></div></div><label htmlFor="document" className="label">SYNTHETIC REPORT TEXT</label><textarea id="document" rows={7} value={document} onChange={e => setDocument(e.target.value)}/><button className="primary" disabled={busy || !document.trim()} onClick={extract}>Extract measurements<Icon name="arrow" size={17}/></button>{extraction && <div className="report"><h3>Extracted fields</h3><table><thead><tr><th>Field</th><th>Value</th><th>Unit</th></tr></thead><tbody>{extraction.measurements.map((m, i) => <tr key={i}><td>{m.field}</td><td>{m.value}</td><td>{m.unit}</td></tr>)}</tbody></table>{!extraction.measurements.length && <p>No supported fields with explicit units were found.</p>}{extraction.rejected.map((r, i) => <p key={i}>Rejected {r.field}: {r.reason}</p>)}<p className="fine-print">{extraction.limitations}</p></div>}</section>}
        {tab === 'models' && <section className="panel"><div className="panel-title"><div><h2>Local model registry</h2><p>Metrics below describe held-out synthetic patients only.</p></div></div>{!models.length && <div className="empty">No trained artifact found. Run the training command in the README and restart the API.</div>}{models.map(m => <article className="model-card" key={m.version}><span className="tiny-pill teal">SYNTHETIC TRAINING</span><h3>{m.version}</h3><p>Family: {m.family} · Clinical validation: none</p><div className="metric-grid">{['auprc', 'auroc', 'brier', 'false_alert_episodes_per_nonseptic_patient_day'].map(k => <div key={k}><span>{k.replaceAll('_', ' ')}</span><strong>{typeof m.metrics[k] === 'number' ? (m.metrics[k] as number).toFixed(4) : '—'}</strong></div>)}</div><details><summary>Inspect evaluation metadata</summary><pre>{JSON.stringify(m.metrics, null, 2)}</pre></details></article>)}</section>}
        <footer><span><i/> HEALTHCARE SEPSISAI</span><span>Synthetic data · Reproducible research · Human oversight</span></footer>
      </div>
    </main>
  </div>;
}
