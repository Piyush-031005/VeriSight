import React, { useState } from 'react';
import { ModalityType, PredictResponse, ImageCheckType } from '../types/api';

interface FindingsSheetProps {
  modality: ModalityType;
  file: File | null;
  textContent: string;
  result: PredictResponse | null;
  isLoading: boolean;
  error: string | null;
  onAnalyze: () => void;
  imageCheckType?: ImageCheckType;
}

const PATH_LABELS: Record<string, string> = {
  fusion:              'Multimodal Fusion (Visual + Audio)',
  video_only:          'Visual Face-Swap Stream',
  audio_only:          'Acoustic Speech Stream',
  image_only:          'Photographic Analysis',
  text_only:           'Linguistic Style Analysis',
  video_text_fusion:   'Audio + Visual + Caption',
  no_modalities:       'No Usable Signals',
};

export const FindingsSheet: React.FC<FindingsSheetProps> = ({
  modality,
  file,
  textContent,
  result,
  isLoading,
  error,
  onAnalyze,
  imageCheckType = 'face_swap',
}) => {
  const [detailsOpen, setDetailsOpen] = useState(false);

  const getAssessmentTitle = (res: PredictResponse) => {
    if (res.prediction === 'UNCERTAIN') {
      if (modality === 'image' && res.image?.score === null && res.image?.reason) return res.image.reason;
      if (modality === 'audio' && res.audio?.score === null && res.audio?.reason) return res.audio.reason;
      if (modality === 'text'  && res.text?.score  === null && res.text?.reason)  return 'Linguistic estimation omitted';
      if (modality === 'video' && res.video?.score === null && res.audio?.score === null) return 'No usable video or audio signals';
      return 'Inconclusive';
    }
    switch (res.prediction) {
      case 'REAL': return modality === 'text' ? 'Lower AI-writing style probability' : 'No manipulation detected';
      case 'FAKE': return modality === 'text' ? 'Elevated AI-writing style probability' : 'Manipulation evidence flagged';
      default:     return 'Inconclusive';
    }
  };

  const getAssessmentClass = (res: PredictResponse) => {
    switch (res.prediction) {
      case 'REAL':      return 'assessment-heading--real';
      case 'FAKE':      return 'assessment-heading--fake';
      default:          return 'assessment-heading--uncertain';
    }
  };

  const getCardClass = (res: PredictResponse) => {
    switch (res.prediction) {
      case 'REAL':      return 'assessment-card--real';
      case 'FAKE':      return 'assessment-card--fake';
      default:          return 'assessment-card--uncertain';
    }
  };

  const getBadgeClass = (res: PredictResponse) => {
    switch (res.prediction) {
      case 'REAL':      return 'assessment-verdict-badge--real';
      case 'FAKE':      return 'assessment-verdict-badge--fake';
      default:          return 'assessment-verdict-badge--uncertain';
    }
  };

  const getAssessmentMeaning = (res: PredictResponse) => {
    if (modality === 'text') return res.text?.reason || 'Experimental AI-writing style estimate. Independent verification required.';
    if (res.disagreement)   return 'The overall decision incorporates both stream findings below.';
    const branch = modality === 'image' ? res.image : res.audio;
    return branch?.reason || (res.prediction === 'FAKE'
      ? 'The selected detector flagged this sample.'
      : 'The selected detector found a low manipulation score.');
  };

  const getSignalBarFillClass = (score: number, band?: [number, number] | null) => {
    if (band && score >= band[0] && score <= band[1]) return 'signal-bar-fill--uncertain';
    return score >= 0.50 ? 'signal-bar-fill--fake' : 'signal-bar-fill--real';
  };

  const hasInput = modality === 'text' ? textContent.trim().length > 0 : file !== null;

  // ── State 1: Awaiting Input ────────────────────────────────────────
  if (!hasInput) {
    return (
      <div className="findings">
        <div className="findings-section-label">Analysis</div>
        <h2 className="findings-empty-heading">
          {modality === 'text' ? 'Awaiting textual input' : 'Awaiting media intake'}
        </h2>
        <p className="findings-empty-desc">
          {modality === 'text'
            ? 'Type or paste content into the inspection desk to begin analysis.'
            : 'Select or drop a file onto the monitor to begin inspection.'}
        </p>

        <hr className="findings-rule" />

        <div className="findings-section-label">Expected Findings</div>
        <div className="findings-index">
          <div className="findings-index-row">
            <span className="findings-index-label">Overall assessment</span>
            <span className="findings-index-value">Awaiting analysis</span>
          </div>
          {modality === 'video' && (
            <>
              <div className="findings-index-row">
                <span className="findings-index-label">Visual face-swap signal (8 frames)</span>
                <span className="findings-index-value">—</span>
              </div>
              <div className="findings-index-row">
                <span className="findings-index-label">Acoustic speech signal (soundtrack)</span>
                <span className="findings-index-value">—</span>
              </div>
            </>
          )}
          {modality === 'audio' && (
            <div className="findings-index-row">
              <span className="findings-index-label">Acoustic speech signal (16 kHz mono)</span>
              <span className="findings-index-value">—</span>
            </div>
          )}
          {modality === 'image' && (
            <div className="findings-index-row">
              <span className="findings-index-label">
                {imageCheckType === 'general_ai' ? 'General AI-image check' : 'Face-swap check (18% margin)'}
              </span>
              <span className="findings-index-value">—</span>
            </div>
          )}
          {modality === 'text' && (
            <div className="findings-index-row">
              <span className="findings-index-label">AI-writing style estimate</span>
              <span className="findings-index-value">—</span>
            </div>
          )}
        </div>
      </div>
    );
  }

  // ── State 2: Error ─────────────────────────────────────────────────
  if (error) {
    return (
      <div className="findings">
        <div className="findings-section-label">Analysis Status</div>
        <div className="findings-error">
          <div className="findings-error-label">Analysis could not be completed</div>
          <div style={{ fontSize: 13, color: 'var(--ink-2)', marginTop: 4 }}>{error}</div>
          <button className="retry-btn" type="button" onClick={onAnalyze}>Retry analysis</button>
        </div>
        <div className="findings-section-label">Sample Status</div>
        <div className="findings-index">
          <div className="findings-index-row">
            <span className="findings-index-label">Target modality</span>
            <span className="findings-index-value">{modality.toUpperCase()}</span>
          </div>
          <div className="findings-index-row">
            <span className="findings-index-label">Pipeline state</span>
            <span className="findings-index-value" style={{ color: 'var(--fake)' }}>Failed</span>
          </div>
        </div>
      </div>
    );
  }

  // ── State 3: Ready / Analyzing ─────────────────────────────────────
  if (!result) {
    return (
      <div className="findings">
        <button
          className={`run-analysis-btn${isLoading ? ' run-analysis-btn--loading' : ''}`}
          type="button"
          onClick={onAnalyze}
          disabled={isLoading}
        >
          {isLoading && <span className="run-analysis-spinner" aria-hidden="true" />}
          <span>{isLoading ? 'Running inference…' : `Run ${modality} analysis`}</span>
        </button>

        <hr className="findings-rule" />

        <div className="findings-section-label">Inspection Checklist</div>
        <div className="findings-index">
          <div className="findings-index-row">
            <span className="findings-index-label">Overall assessment</span>
            <span className="findings-index-value">{isLoading ? 'Running models…' : 'Ready to inspect'}</span>
          </div>
          {modality === 'video' && (
            <>
              <div className="findings-index-row">
                <span className="findings-index-label">Visual frames (CLIP + EffNet-B0 + Face)</span>
                <span className="findings-index-value">{isLoading ? 'Sampling 8 frames & Grad-CAM…' : 'Pending'}</span>
              </div>
              <div className="findings-index-row">
                <span className="findings-index-label">Audio soundtrack (wav2vec2 XLS-R)</span>
                <span className="findings-index-value">{isLoading ? 'Extracting speech to 16 kHz…' : 'Pending'}</span>
              </div>
            </>
          )}
          {modality === 'audio' && (
            <div className="findings-index-row">
              <span className="findings-index-label">Acoustic classifier (wav2vec2 XLS-R)</span>
              <span className="findings-index-value">{isLoading ? 'Windowing & evaluating speech…' : 'Pending'}</span>
            </div>
          )}
          {modality === 'image' && (
            <div className="findings-index-row">
              <span className="findings-index-label">
                {imageCheckType === 'general_ai' ? 'AI-image hybrid (Diffusion + Visual v2)' : 'Face-swap forensic (Visual v2 + Face)'}
              </span>
              <span className="findings-index-value">{isLoading ? 'Evaluating image pixels & Grad-CAM…' : 'Pending'}</span>
            </div>
          )}
          {modality === 'text' && (
            <div className="findings-index-row">
              <span className="findings-index-label">Multilingual style & claim check (XLM-RoBERTa)</span>
              <span className="findings-index-value">{isLoading ? 'Tokenizing EN/HI/Hinglish text…' : 'Pending'}</span>
            </div>
          )}
        </div>
      </div>
    );
  }

  // ── State 4: Completed Result ──────────────────────────────────────
  const { video, audio, timing, warnings, image, text, cross_modal, disagreement, v2_multi_verify } = result;
  const filteredWarnings = (warnings || []).filter((w) => !w.startsWith('[UNVALIDATED]'));
  const consistency = v2_multi_verify?.consistency_scores;
  const hasConsistency =
    consistency &&
    (consistency.visual_text != null || consistency.audio_text != null || consistency.av_sync != null);

  return (
    <div className="findings">
      {/* 1. Assessment Card */}
      <div className={`assessment-card ${getCardClass(result)}`}>
        <div className="assessment-verdict-row">
          <span className={`assessment-verdict-badge ${getBadgeClass(result)}`}>
            {result.prediction}
          </span>
          {result.confidence != null && (
            <span style={{ fontSize: 12, color: 'var(--ink-3)' }}>
              {(result.confidence * 100).toFixed(1)}% confidence
            </span>
          )}
        </div>
        <h2 className={`assessment-heading ${getAssessmentClass(result)}`}>
          {getAssessmentTitle(result)}
        </h2>
        <p className="assessment-meaning">{getAssessmentMeaning(result)}</p>

        {disagreement && (
          <div className="disagreement-callout">
            <div className="disagreement-callout-label">Cross-Modal Finding</div>
            <div className="disagreement-callout-text">{disagreement}</div>
          </div>
        )}
      </div>

      <hr className="findings-rule" />

      {/* 2. Modality Signals */}
      <div className="findings-section-label">Modality Signals</div>
      <div className="findings-index">

        {/* VIDEO: Visual */}
        {modality === 'video' && video && (
          <div className="signal-row">
            <div className="signal-row-header">
              <span className="signal-row-label">Visual face-swap ({video.model})</span>
              <span className="signal-row-value">
                {video.score !== null
                  ? `${(video.score * 100).toFixed(1)}%`
                  : <span style={{ color: 'var(--ink-4)' }}>{video.reason || 'No usable face'}</span>}
              </span>
            </div>
            {video.score !== null ? (
              <>
                <div className="signal-bar-track">
                  <div
                    className={`signal-bar-fill ${getSignalBarFillClass(video.score, video.band)}`}
                    style={{ width: `${Math.min(100, Math.max(0, video.score * 100))}%` }}
                  />
                </div>
                <div className="signal-note">
                  Median across {video.faces_detected}/{video.frames_analyzed} frames · 18% crop margin
                </div>
              </>
            ) : (
              <div className="signal-note">{video.reason || 'No human faces detected across sampled frames.'}</div>
            )}
          </div>
        )}

        {/* VIDEO: Audio */}
        {modality === 'video' && audio && (
          <div className="signal-row">
            <div className="signal-row-header">
              <span className="signal-row-label">Acoustic speech ({audio.model})</span>
              <span className="signal-row-value">
                {audio.score !== null
                  ? `${(audio.score * 100).toFixed(1)}%`
                  : <span style={{ color: 'var(--ink-4)' }}>{audio.reason || 'Silence or no speech'}</span>}
              </span>
            </div>
            {audio.score !== null ? (
              <>
                <div className="signal-bar-track">
                  <div
                    className={`signal-bar-fill ${getSignalBarFillClass(audio.score, audio.band)}`}
                    style={{ width: `${Math.min(100, Math.max(0, audio.score * 100))}%` }}
                  />
                </div>
                <div className="signal-note">
                  Evaluated over {audio.duration_seconds.toFixed(1)}s · 16 kHz mono
                </div>
              </>
            ) : (
              <div className="signal-note">{audio.reason || 'Audio stream missing, silent, or too short.'}</div>
            )}
          </div>
        )}

        {/* AUDIO only */}
        {modality === 'audio' && audio && (
          <div className="signal-row">
            <div className="signal-row-header">
              <span className="signal-row-label">Acoustic classifier ({audio.model})</span>
              <span className="signal-row-value">
                {audio.score !== null
                  ? `${(audio.score * 100).toFixed(1)}%`
                  : <span style={{ color: 'var(--ink-4)' }}>{audio.reason || 'Silence or insufficient speech'}</span>}
              </span>
            </div>
            {audio.score !== null ? (
              <div className="signal-bar-track">
                <div
                  className={`signal-bar-fill ${getSignalBarFillClass(audio.score, audio.band)}`}
                  style={{ width: `${Math.min(100, Math.max(0, audio.score * 100))}%` }}
                />
              </div>
            ) : (
              <div className="signal-note">{audio.reason}</div>
            )}
          </div>
        )}

        {/* IMAGE */}
        {modality === 'image' && image && (
          <div className="signal-row">
            <div className="signal-row-header">
              <span className="signal-row-label">
                {image.check_type === 'general_ai' ? 'General AI-image' : 'Face-swap check'} ({image.model})
              </span>
              <span className="signal-row-value">
                {image.score !== null
                  ? `${(image.score * 100).toFixed(1)}%`
                  : <span style={{ color: 'var(--ink-4)' }}>{image.reason || 'No usable face'}</span>}
              </span>
            </div>
            {image.score !== null ? (
              <div className="signal-bar-track">
                <div
                  className={`signal-bar-fill ${getSignalBarFillClass(image.score)}`}
                  style={{ width: `${Math.min(100, Math.max(0, image.score * 100))}%` }}
                />
              </div>
            ) : (
              <div className="signal-note">{image.reason}</div>
            )}
          </div>
        )}

        {/* TEXT */}
        {modality === 'text' && text && (
          <div className="signal-row">
            <div className="signal-row-header">
              <span className="signal-row-label">
                {text.label || 'AI-writing style estimate'} ({text.model})
              </span>
              <span className="signal-row-value">
                {text.score !== null
                  ? `${(text.score * 100).toFixed(1)}% AI probability`
                  : <span style={{ color: 'var(--ink-4)' }}>{text.reason || 'Omitted'}</span>}
              </span>
            </div>
            {text.score !== null ? (
              <div className="signal-bar-track">
                <div
                  className={`signal-bar-fill ${getSignalBarFillClass(text.score, text.band)}`}
                  style={{ width: `${Math.min(100, Math.max(0, text.score * 100))}%` }}
                />
              </div>
            ) : (
              <div className="signal-note">{text.reason}</div>
            )}
          </div>
        )}

        {/* Uncertainty band notes */}
        {[['Visual', video?.band], ['Acoustic', audio?.band], ['Writing style', text?.band], ['Fusion', result.fusion?.band]]
          .filter(([, b]) => Array.isArray(b))
          .map(([label, band]) => (
            <p className="evidence-note" key={String(label)}>
              {label} uncertainty band {(band as number[])[0].toFixed(2)}–{(band as number[])[1].toFixed(2)} (uncalibrated)
            </p>
          ))}

        {/* Cross-modal transcript */}
        {cross_modal && (
          <section className="signal-row">
            <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--ink-2)', marginBottom: 4 }}>
              Speech transcript
            </h3>
            <p style={{ fontSize: 13, color: 'var(--ink-2)' }}>
              {cross_modal.transcript || cross_modal.transcript_status}
            </p>
            {cross_modal.reason && <p style={{ fontSize: 12, color: 'var(--ink-3)', marginTop: 4 }}>{cross_modal.reason}</p>}
            {cross_modal.wer != null && (
              <p style={{ fontSize: 12, color: 'var(--ink-3)', marginTop: 4 }}>
                Caption WER {cross_modal.wer.toFixed(3)} · CER {cross_modal.cer?.toFixed(3)}{' '}
                · {cross_modal.similarity_method} {cross_modal.cross_modal_consistency?.toFixed(3)}
              </p>
            )}
          </section>
        )}

        {/* Transcript text analysis (video/audio mode) */}
        {text && modality !== 'text' && (
          <p className="evidence-note">
            Transcript AI-writing estimate: {text.score?.toFixed(4) ?? text.reason}
          </p>
        )}

        {/* Sentence-level breakdown */}
        {!!text?.sentences?.length && (
          <details className="signal-row" open>
            <summary>Sentence estimates ({text.sentences.length} sampled)</summary>
            <p style={{ fontSize: 12, color: 'var(--ink-3)', marginBottom: 6 }}>
              Each sentence is scored independently. Passage estimate uses overlapping token windows.
            </p>
            {text.sentences.map((s) => (
              <p key={s.start} style={{ fontSize: 12, margin: '4px 0' }}>
                <mark>{s.text}</mark> <strong>{s.score.toFixed(4)}</strong>
              </p>
            ))}
          </details>
        )}

        {!!text?.spans?.length && (
          <details className="signal-row">
            <summary>Writing-style evidence by span ({text.spans.length})</summary>
            {text.spans.map((s) => (
              <blockquote key={s.start}>
                <strong>Chars {s.start + 1}–{s.end} · {s.score.toFixed(4)}</strong>
                <p><mark>{s.text}</mark></p>
              </blockquote>
            ))}
          </details>
        )}

        {result.fusion?.score != null && (
          <p className="evidence-note">Uncalibrated fusion score {result.fusion.score.toFixed(4)}</p>
        )}
      </div>

      {/* 2B. VeriSight v2 Multi-Verification & Explainability */}
      {(hasConsistency ||
        v2_multi_verify?.gradcam_overlay_b64 ||
        (v2_multi_verify?.token_attributions && v2_multi_verify.token_attributions.length > 0) ||
        v2_multi_verify?.grounded_summary) && (
        <>
          <hr className="findings-rule" />
          <div className="findings-section-label">Multi-Verification &amp; Explainability</div>

          {hasConsistency && (
            <div className="v2-verify-card">
              <div className="v2-verify-header">
                <span>Layer 2 · Cross-Modal Double Check</span>
                {consistency?.is_contradiction && (
                  <span className="v2-badge-contradiction">Contradiction Detected</span>
                )}
              </div>
              <div className="v2-consistency-grid">
                {consistency?.av_sync != null && (
                  <div className="v2-metric-box">
                    <span className="v2-metric-label">Voice ↔ Video Sync</span>
                    <span className="v2-metric-val">{(consistency.av_sync * 100).toFixed(0)}%</span>
                  </div>
                )}
                {consistency?.audio_text != null && (
                  <div className="v2-metric-box">
                    <span className="v2-metric-label">Voice ↔ Caption</span>
                    <span className="v2-metric-val">{(consistency.audio_text * 100).toFixed(0)}%</span>
                  </div>
                )}
                {consistency?.visual_text != null && (
                  <div className="v2-metric-box">
                    <span className="v2-metric-label">Visual ↔ Caption</span>
                    <span className="v2-metric-val">{(consistency.visual_text * 100).toFixed(0)}%</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {v2_multi_verify?.gradcam_overlay_b64 && (
            <div className="v2-gradcam-card">
              <div className="v2-verify-header">
                <span>Visual Grad-CAM Saliency Map</span>
                <span style={{ fontSize: 11, color: 'var(--ink-4)' }}>224×224 overlay</span>
              </div>
              <img
                src={v2_multi_verify.gradcam_overlay_b64}
                alt="Grad-CAM facial manipulation heatmap"
                className="v2-gradcam-img"
              />
            </div>
          )}

          {v2_multi_verify?.token_attributions && v2_multi_verify.token_attributions.length > 0 && (
            <div className="v2-gradcam-card">
              <div className="v2-verify-header">
                <span>Top Influential Tokens (XLM-R Attention)</span>
                <span style={{ fontSize: 11, color: 'var(--ink-3)' }}>EN / HI / Hinglish</span>
              </div>
              <div className="v2-token-chips">
                {v2_multi_verify.token_attributions.map((item, idx) => (
                  <span key={`${item.token}-${idx}`} className="v2-token-chip">
                    <span>{item.token}</span>
                    <span className="v2-token-w">{(item.weight * 100).toFixed(1)}%</span>
                  </span>
                ))}
              </div>
            </div>
          )}

          {v2_multi_verify?.grounded_summary && (
            <div className="v2-grounded-box">
              <div className="v2-verify-header">
                <span>Evidence-Grounded Summary</span>
                {v2_multi_verify.uncertainty?.entropy != null && (
                  <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--ink-3)' }}>
                    H = {v2_multi_verify.uncertainty.entropy.toFixed(3)}
                  </span>
                )}
              </div>
              <p style={{ fontSize: 12, color: 'var(--ink-2)', lineHeight: 1.55, marginTop: 4 }}>
                {v2_multi_verify.grounded_summary}
              </p>
            </div>
          )}
        </>
      )}

      <hr className="findings-rule" />

      {/* 3. Technical Details */}
      <div className="tech-details">
        <button
          className="tech-details-toggle"
          type="button"
          onClick={() => setDetailsOpen(!detailsOpen)}
          aria-expanded={detailsOpen}
        >
          <span className={`tech-details-toggle-chevron${detailsOpen ? ' tech-details-toggle-chevron--open' : ''}`} />
          <span>Technical details &amp; forensics breakdown</span>
        </button>

        {detailsOpen && (
          <div className="tech-details-content">
            <div className="tech-detail-row">
              <span className="tech-detail-label">Decision path</span>
              <span className="tech-detail-value">{PATH_LABELS[result.prediction_path] || result.prediction_path}</span>
            </div>
            <div className="tech-detail-row">
              <span className="tech-detail-label">Confidence</span>
              <span className="tech-detail-value">
                {result.confidence == null ? 'Unavailable' : `${(result.confidence * 100).toFixed(1)}%`}
              </span>
            </div>
            {text && (
              <div className="tech-detail-row">
                <span className="tech-detail-label">Word count</span>
                <span className="tech-detail-value">{text.word_count} words</span>
              </div>
            )}
            {image && (
              <>
                <div className="tech-detail-row">
                  <span className="tech-detail-label">Detection model</span>
                  <span className="tech-detail-value">{image.model}</span>
                </div>
                {image.resolution && (
                  <div className="tech-detail-row">
                    <span className="tech-detail-label">Resolution</span>
                    <span className="tech-detail-value">{image.resolution}</span>
                  </div>
                )}
              </>
            )}
            {video && (
              <>
                <div className="tech-detail-row">
                  <span className="tech-detail-label">Frames analyzed</span>
                  <span className="tech-detail-value">{video.frames_analyzed} (evenly spaced)</span>
                </div>
                <div className="tech-detail-row">
                  <span className="tech-detail-label">Faces tracked</span>
                  <span className="tech-detail-value">{video.faces_detected} usable crops</span>
                </div>
              </>
            )}
            {audio && (
              <div className="tech-detail-row">
                <span className="tech-detail-label">Audio duration</span>
                <span className="tech-detail-value">{audio.duration_seconds.toFixed(2)}s</span>
              </div>
            )}
            <div className="tech-detail-row">
              <span className="tech-detail-label">Pipeline wall-clock</span>
              <span className="tech-detail-value">{timing.total_pipeline_sec.toFixed(2)}s</span>
            </div>
          </div>
        )}
      </div>

      {/* 4. Diagnostic warnings */}
      {filteredWarnings.length > 0 && (
        <>
          <hr className="findings-rule" />
          <div className="findings-section-label">Diagnostic Notes</div>
          <ul className="warnings-list">
            {filteredWarnings.map((warn, i) => <li key={i}>{warn}</li>)}
          </ul>
        </>
      )}
    </div>
  );
};

export default FindingsSheet;
