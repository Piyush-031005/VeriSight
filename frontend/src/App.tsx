import React, { useEffect, useState } from 'react';
import { Film, Music, Image as ImageIcon, FileText, BarChart2, ScanSearch } from 'lucide-react';
import { SourceMonitor } from './components/SourceMonitor';
import { FindingsSheet } from './components/FindingsSheet';
import { ExperimentsDashboard } from './components/ExperimentsDashboard';
import { usePredict } from './hooks/usePredict';
import { HealthResponse, ModalityType, ImageCheckType } from './types/api';
import './styles/global.css';

type ActiveView = 'inspector' | 'benchmarks';

const HEADINGS: Record<ModalityType, { eyebrow: string; title: string; subtitle: string }> = {
  video: {
    eyebrow: 'Multimodal Video Verification',
    title: 'Inspect video recording',
    subtitle:
      'Runs 3-Layer Multi-Verification: Visual face-swap + Grad-CAM saliency, acoustic speech analysis, and cross-modal consistency checks.',
  },
  audio: {
    eyebrow: 'Acoustic Forensics',
    title: 'Inspect acoustic recording',
    subtitle:
      'Evaluate 16 kHz speech waveforms for synthetic TTS or voice cloning markers and compare against optional captions.',
  },
  image: {
    eyebrow: 'Photographic Forensics',
    title: 'Inspect photographic media',
    subtitle:
      'Detect face-swap tampering (CLIP + EffNet-B0 + Grad-CAM) or diffusion/AI synthesis with visual forensics.',
  },
  text: {
    eyebrow: 'Multilingual Linguistic Check',
    title: 'Inspect textual content',
    subtitle:
      'Evaluate English, Hindi, or Hinglish passages for AI-generated writing style and misleading claims.',
  },
};

export const App: React.FC = () => {
  const [activeView, setActiveView] = useState<ActiveView>('inspector');
  const [modality, setModality] = useState<ModalityType>('video');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [textContent, setTextContent] = useState<string>('');
  const [fileError, setFileError] = useState<string | null>(null);
  const [imageCheckType, setImageCheckType] = useState<ImageCheckType>('face_swap');

  const { result, isLoading, error, analyzeMedia, reset } = usePredict();

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthChecked, setHealthChecked] = useState(false);
  const [showSystemDetails, setShowSystemDetails] = useState(false);

  useEffect(() => {
    fetch('/api/health')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => { setHealth(data); setHealthChecked(true); })
      .catch(() => { setHealth(null); setHealthChecked(true); });
  }, []);

  const handleModalityChange = (newModality: ModalityType) => {
    if (newModality === modality) return;
    setModality(newModality);
    setSelectedFile(null);
    setTextContent('');
    setFileError(null);
    reset();
  };

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setFileError(null);
    reset();
  };

  const handleTextChange = (text: string) => {
    setTextContent(text);
    if (result) reset();
  };

  const handleAnalyze = () => {
    if (!isLoading) analyzeMedia(modality, selectedFile, textContent, imageCheckType);
  };

  const activeHeading = HEADINGS[modality];

  return (
    <div>
      {/* ── Top Bar ─────────────────────────────────────────── */}
      <header className="topbar">
        <div className="topbar-brand">
          <div className="topbar-logo" aria-hidden="true">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.1" strokeLinecap="round" strokeLinejoin="round">
              <path d="M2.5 12s3.5-6.5 9.5-6.5S21.5 12 21.5 12s-3.5 6.5-9.5 6.5S2.5 12 2.5 12Z" />
              <circle cx="12" cy="12" r="3" />
              <path d="M12 2.5v2.2M12 19.3v2.2" />
            </svg>
          </div>
          <span className="topbar-wordmark">
            Veri<span>Sight</span>
          </span>
          <span className="topbar-tagline">Multimodal Verification</span>

          <div className="topbar-nav-switch" role="tablist" aria-label="Workspace view">
            <button
              type="button"
              role="tab"
              aria-selected={activeView === 'inspector'}
              className={`topbar-nav-btn${activeView === 'inspector' ? ' topbar-nav-btn--active' : ''}`}
              onClick={() => setActiveView('inspector')}
            >
              <ScanSearch size={13} />
              <span>Inspector</span>
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={activeView === 'benchmarks'}
              className={`topbar-nav-btn${activeView === 'benchmarks' ? ' topbar-nav-btn--active' : ''}`}
              onClick={() => setActiveView('benchmarks')}
            >
              <BarChart2 size={13} />
              <span>Benchmarks &amp; Exp</span>
            </button>
          </div>
        </div>

        <div className="topbar-right">
          {healthChecked && (
            <div className={`backend-pill${health ? ' backend-pill--ok' : ''}`}>
              <span className="backend-pill-dot" />
              <span>{health ? `API v${health.version} · ${health.device.toUpperCase()}` : 'Backend offline'}</span>
            </div>
          )}
          <button
            className="topbar-btn"
            type="button"
            onClick={() => setShowSystemDetails(!showSystemDetails)}
            aria-expanded={showSystemDetails}
          >
            {showSystemDetails ? 'Hide details' : 'System details'}
          </button>
        </div>
      </header>

      {/* ── System Details Drawer ───────────────────────────── */}
      {showSystemDetails && (
        <div className="system-details" role="region" aria-label="System diagnostic details">
          <div className="system-detail-item">
            <span className={`system-detail-dot ${health ? 'system-detail-dot--ok' : 'system-detail-dot--error'}`} />
            <span className="system-detail-label">Backend</span>
            <span className="system-detail-value">
              {health ? `Online (v${health.version})` : 'Unreachable'}
            </span>
          </div>
          <div className="system-detail-item">
            <span className="system-detail-label">Device</span>
            <span className="system-detail-value">{health ? health.device.toUpperCase() : '—'}</span>
          </div>
          <div className="system-detail-item">
            <span className="system-detail-label">FFmpeg</span>
            <span className="system-detail-value">
              {health ? (health.ffmpeg_available ? 'Available' : 'Missing') : '—'}
            </span>
          </div>
          <div className="system-detail-item">
            <span className="system-detail-label">Visual</span>
            <span className="system-detail-value">EfficientNet-B0 + Face-Crop + Grad-CAM</span>
          </div>
          <div className="system-detail-item">
            <span className="system-detail-label">Audio</span>
            <span className="system-detail-value">
              {health?.audio_model?.model_name && !health.audio_model.model_name.toLowerCase().includes('pranjal')
                ? health.audio_model.model_name
                : 'wav2vec2-large-xlsr (VeriSight Acoustic)'}
            </span>
          </div>
          <div className="system-detail-item">
            <span className="system-detail-label">Text</span>
            <span className="system-detail-value">XLM-RoBERTa (EN / HI / Hinglish)</span>
          </div>
        </div>
      )}

      {activeView === 'benchmarks' ? (
        <ExperimentsDashboard />
      ) : (
        <>
          {/* ── Workspace Heading + 3-Layer Pipeline Strip ──── */}
          <div className="workspace-heading">
            <div className="workspace-heading-row">
              <div>
                <div className="workspace-eyebrow">{activeHeading.eyebrow}</div>
                <h1 className="workspace-title">{activeHeading.title}</h1>
                <p className="workspace-subtitle">{activeHeading.subtitle}</p>
              </div>

              <div className="pipeline-stepper" aria-label="3-Layer Verification Pipeline">
                <div className={`pipeline-step${result || isLoading ? ' pipeline-step--active' : ''}`}>
                  <span className="pipeline-step-num">1</span>
                  <span>Unimodal Forensics</span>
                </div>
                <span className="pipeline-step-arrow">→</span>
                <div className={`pipeline-step${result?.v2_multi_verify ? ' pipeline-step--active' : ''}`}>
                  <span className="pipeline-step-num">2</span>
                  <span>Cross-Modal Check</span>
                </div>
                <span className="pipeline-step-arrow">→</span>
                <div className={`pipeline-step${result ? ' pipeline-step--done' : ''}`}>
                  <span className="pipeline-step-num">3</span>
                  <span>Calibrated Verdict</span>
                </div>
              </div>
            </div>

            {/* Modality Tabs */}
            <nav className="modality-nav" aria-label="Inspection modality">
              {(
                [
                  { key: 'video' as ModalityType, Icon: Film,      label: 'Video + Audio' },
                  { key: 'audio' as ModalityType, Icon: Music,     label: 'Audio' },
                  { key: 'image' as ModalityType, Icon: ImageIcon, label: 'Image' },
                  { key: 'text'  as ModalityType, Icon: FileText,  label: 'Text (EN / HI)' },
                ] as const
              ).map(({ key, Icon, label }) => (
                <button
                  key={key}
                  type="button"
                  className={`modality-tab${modality === key ? ' modality-tab--active' : ''}`}
                  onClick={() => handleModalityChange(key)}
                >
                  <Icon size={14} strokeWidth={2} />
                  <span>{label}</span>
                </button>
              ))}
            </nav>
          </div>

          {/* ── Main Two-Column Workspace ───────────────────── */}
          <main className="workspace">
            <SourceMonitor
              modality={modality}
              file={selectedFile}
              onFileSelect={handleFileSelect}
              textContent={textContent}
              onTextChange={handleTextChange}
              fileError={fileError}
              onFileError={setFileError}
              imageCheckType={imageCheckType}
              onImageCheckTypeChange={setImageCheckType}
            />

            <FindingsSheet
              modality={modality}
              file={selectedFile}
              textContent={textContent}
              result={result}
              isLoading={isLoading}
              error={error}
              onAnalyze={handleAnalyze}
              imageCheckType={imageCheckType}
            />
          </main>
        </>
      )}
    </div>
  );
};

export default App;
