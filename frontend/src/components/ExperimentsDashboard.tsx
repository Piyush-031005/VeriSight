import React, { useEffect, useState } from 'react';
import { CheckCircle2, Clock, Layers, ShieldAlert, BarChart3 } from 'lucide-react';

interface MetricsPayload {
  experiments_summary?: any;
  text_eval?: any;
  audio_eval_l2?: any;
  exp3_robustness?: any;
  checkpoints?: Record<string, { exists: boolean; size_mb: number }>;
}

const CKPT_LABELS: Record<string, { title: string; owner: string }> = {
  'v2_visual_head.pt': { title: 'Visual Adapter (CLIP + EffNet-B0)', owner: 'Laptop 1' },
  'v2_text_head.pt': { title: 'Text Adapter (XLM-RoBERTa EN/HI)', owner: 'Laptop 1' },
  'v2_audio_xlsr_head.pt': { title: 'Audio Adapter (wav2vec2 XLS-R)', owner: 'Laptop 2' },
  'v2_fusion_consistency_gated.pt': { title: 'Consistency-Gated Fusion MLP', owner: 'Phase 3' },
};

export const ExperimentsDashboard: React.FC = () => {
  const [data, setData] = useState<MetricsPayload | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/v2/metrics')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { setData(d); setLoading(false); })
      .catch(() => { setData(null); setLoading(false); });
  }, []);

  if (loading) {
    return (
      <div className="exp-shell">
        <p style={{ color: 'var(--ink-3)', fontSize: 13 }}>Loading experiment benchmarks…</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="exp-shell">
        <div className="findings-error">
          <div className="findings-error-label">Backend offline</div>
          <p style={{ fontSize: 13, color: 'var(--ink-2)' }}>
            Start FastAPI (`python -m uvicorn src.api.main:app --port 8000`) to load live checkpoint and experiment metrics.
          </p>
        </div>
      </div>
    );
  }

  const exp = data.experiments_summary;
  const txt = data.text_eval;
  const aud = data.audio_eval_l2;
  const exp3 = data.exp3_robustness;
  const ckpts = data.checkpoints || {};
  const exp2 = exp?.exp2_fusion_comparison || {};
  const exp4 = exp?.exp4_modality_ablation || {};

  return (
    <div className="exp-shell">
      <div className="exp-header">
        <div>
          <div className="workspace-eyebrow">Research Evaluation &amp; Model Registry</div>
          <h1 className="workspace-title" style={{ fontSize: 30 }}>VeriSight Benchmarks</h1>
          <p className="workspace-subtitle">
            Live metrics from trained RTX 4050 adapter heads, 3-seed fusion comparison, cross-lingual Hindi/Hinglish evaluation, and modality ablation.
          </p>
        </div>
      </div>

      {/* 1. Model Checkpoints Grid */}
      <div className="findings-section-label" style={{ marginTop: 24 }}>Trained Adapter Checkpoints (&lt; 25 MB Git-Sync Ready)</div>
      <div className="exp-ckpt-grid">
        {Object.entries(CKPT_LABELS).map(([key, meta]) => {
          const info = ckpts[key] || { exists: false, size_mb: 0 };
          return (
            <div key={key} className={`exp-card ${info.exists ? 'exp-card--ready' : ''}`}>
              <div className="exp-card-top">
                <span className="exp-owner-pill">{meta.owner}</span>
                {info.exists ? (
                  <span className="exp-status exp-status--ok">
                    <CheckCircle2 size={12} /> {info.size_mb.toFixed(2)} MB
                  </span>
                ) : (
                  <span className="exp-status exp-status--wait">
                    <Clock size={12} /> Awaiting merge
                  </span>
                )}
              </div>
              <div className="exp-card-title">{meta.title}</div>
              <div className="exp-card-mono">{key}</div>
            </div>
          );
        })}
      </div>

      {/* 2. Exp 1 & Exp 2 Table */}
      <div className="exp-grid-two">
        <div className="exp-panel">
          <div className="exp-panel-header">
            <BarChart3 size={15} />
            <span>Exp 1 &amp; 2 · Unimodal vs. Fusion (3 Seeds)</span>
          </div>
          <table className="exp-table">
            <thead>
              <tr>
                <th>Model / Strategy</th>
                <th>F1 Score</th>
                <th>ROC-AUC</th>
                <th>ECE ↓</th>
              </tr>
            </thead>
            <tbody>
              {exp?.exp1_visual_only && (
                <tr>
                  <td>Visual Only (CLIP + EffNet-B0)</td>
                  <td className="mono">{exp.exp1_visual_only.f1.toFixed(4)}</td>
                  <td className="mono">{exp.exp1_visual_only.roc_auc.toFixed(4)}</td>
                  <td className="mono">{exp.exp1_visual_only.ece.toFixed(4)}</td>
                </tr>
              )}
              {txt && (
                <>
                  <tr>
                    <td>Text Only (XLM-R · English)</td>
                    <td className="mono">{txt.en_val_f1?.toFixed(4)}</td>
                    <td className="mono">—</td>
                    <td className="mono">—</td>
                  </tr>
                  <tr>
                    <td>Text Cross-Lingual (EN → Hindi/Hinglish)</td>
                    <td className="mono">{txt.hindi_hinglish_f1?.toFixed(4)}</td>
                    <td className="mono">Acc {txt.hindi_hinglish_acc?.toFixed(3)}</td>
                    <td className="mono">—</td>
                  </tr>
                </>
              )}
              {aud?.in_domain && (
                <>
                  <tr>
                    <td>Audio Only (XLS-R · ASVspoof 71k)</td>
                    <td className="mono">{aud.in_domain.f1?.toFixed(4)}</td>
                    <td className="mono">{aud.in_domain.auc?.toFixed(4)}</td>
                    <td className="mono">—</td>
                  </tr>
                  <tr>
                    <td>Audio OOD (Deepfake-Audio · 600)</td>
                    <td className="mono">{aud.out_of_domain?.f1?.toFixed(4)}</td>
                    <td className="mono">{aud.out_of_domain?.auc?.toFixed(4)}</td>
                    <td className="mono">—</td>
                  </tr>
                </>
              )}
              {['concat', 'cross_attention', 'consistency_gated'].map((k) => {
                const row = exp2[k];
                if (!row) return null;
                const isBest = k === 'consistency_gated';
                return (
                  <tr key={k} className={isBest ? 'exp-row-highlight' : ''}>
                    <td>
                      {k === 'concat' && 'Fusion A · Concat + MLP'}
                      {k === 'cross_attention' && 'Fusion B · Cross-Attention'}
                      {k === 'consistency_gated' && 'Fusion C · Consistency-Gated (Ours)'}
                    </td>
                    <td className="mono">{row.f1_mean.toFixed(4)} ± {row.f1_std.toFixed(3)}</td>
                    <td className="mono">{row.auc_mean.toFixed(4)}</td>
                    <td className="mono">{row.ece_mean.toFixed(4)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* 3. Exp 4 Modality Ablation & Calibration */}
        <div className="exp-panel">
          <div className="exp-panel-header">
            <Layers size={15} />
            <span>Exp 4 · Modality Ablation &amp; Calibration</span>
          </div>

          <div className="exp-ablation-list">
            {Object.entries(exp4).map(([k, val]: [string, any]) => (
              <div key={k} className="exp-ablation-row">
                <div className="exp-ablation-top">
                  <span>{k.replace('_', ' ').toUpperCase()}</span>
                  <span className="mono">F1 {val.f1.toFixed(4)} · AUC {val.roc_auc.toFixed(4)}</span>
                </div>
                <div className="signal-bar-track">
                  <div
                    className="signal-bar-fill signal-bar-fill--real"
                    style={{ width: `${Math.min(100, val.f1 * 100)}%`, background: 'var(--green)' }}
                  />
                </div>
              </div>
            ))}
          </div>

          {exp?.calibration && (
            <div className="exp-calib-strip">
              <div>
                <span className="exp-metric-k">Temperature T</span>
                <span className="exp-metric-v">{exp.calibration.temperature}</span>
              </div>
              <div>
                <span className="exp-metric-k">Calibrated ECE</span>
                <span className="exp-metric-v">{exp.calibration.calibrated_metrics?.ece}</span>
              </div>
              <div>
                <span className="exp-metric-k">UNCERTAIN Rate</span>
                <span className="exp-metric-v">{(exp.calibration.abstention_rate * 100).toFixed(1)}%</span>
              </div>
              <div>
                <span className="exp-metric-k">Deletion Drop</span>
                <span className="exp-metric-v">{exp.exp6_deletion_faithfulness_conf_drop}</span>
              </div>
            </div>
          )}

          {!data.exp3_robustness && (
            <div className="exp-pending-note">
              <ShieldAlert size={14} />
              <span>
                Audio XLS-R &amp; Exp 3 Compression Curves (H.264/JPEG/Noise) will appear here automatically once Laptop 2 merges <code>feat/laptop2-audio-asr-robustness</code>.
              </span>
            </div>
          )}
        </div>
      </div>

      {/* 4. Experiment 3 Generalization & Compression Stress Results (Laptop 2 Merged) */}
      {exp3 && (
        <div className="exp-panel" style={{ marginTop: 22 }}>
          <div className="exp-panel-header">
            <ShieldAlert size={15} />
            <span>Exp 3 · Cross-Dataset Generalization &amp; Social-Media Compression Stress</span>
          </div>
          <div className="exp-plots-grid">
            <div className="exp-plot-card">
              <div className="exp-plot-title">Compression &amp; Noise Degradation Curves (H.264 / JPEG / SNR)</div>
              <img src="/api/v2/plots/exp3_compression_curves.png" alt="Exp 3 compression robustness curves" className="exp-plot-img" />
            </div>
            <div className="exp-plot-card">
              <div className="exp-plot-title">Audio In-Domain vs Out-of-Domain Generalization</div>
              <img src="/api/v2/plots/exp3_audio_cross_dataset.png" alt="Exp 3 audio cross-dataset chart" className="exp-plot-img" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ExperimentsDashboard;
