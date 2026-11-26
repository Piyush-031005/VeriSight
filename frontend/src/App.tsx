import React, { useEffect, useState } from 'react';
import { Film, Music, Image as ImageIcon, FileText, Shield } from 'lucide-react';
import { SourceMonitor } from './components/SourceMonitor';
import { FindingsSheet } from './components/FindingsSheet';
import { usePredict } from './hooks/usePredict';
import { HealthResponse, ModalityType, ImageCheckType } from './types/api';
import './styles/global.css';

const HEADINGS: Record<ModalityType, { eyebrow: string; title: string; subtitle: string }> = {
  video: {
    eyebrow: 'Video Analysis',
    title: 'Inspect video recording',
    subtitle:
      'Sample 8 frames for face-swap manipulation, analyse the audio soundtrack, and evaluate cross-modal consistency signals.',
  },
  audio: {
    eyebrow: 'Audio Analysis',
    title: 'Inspect acoustic recording',
    subtitle:
      'Evaluate raw waveforms for synthetic speech markers using wav2vec2-large-xlsr deepfake classification.',
  },
  image: {
    eyebrow: 'Image Analysis',
    title: 'Inspect photographic media',
    subtitle:
      'Detect face-swap tampering via deepfakeguard-v2 or general AI-generated imagery via sdxl-detector.',
  },
  text: {
    eyebrow: 'Text Analysis',
    title: 'Inspect textual content',
    subtitle:
      'Evaluate writing style with RoBERTa-based AI-text detection to estimate synthetic authorship probability.',
  },
};

export const App: React.FC = () => {
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
            <Shield size={16} strokeWidth={2} />
          </div>
          <span className="topbar-wordmark">
            Veri<span>Sight</span>
          </span>
          <span className="topbar-tagline">Multimodal Deepfake Detection</span>
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
            <span className="system-detail-label">Visual Model</span>
            <span className="system-detail-value">Samson5827/deepfakeguard-v2</span>
          </div>
          <div className="system-detail-item">
            <span className="system-detail-label">Audio Model</span>
            <span className="system-detail-value">
              {health?.audio_model?.model_name ?? 'pranjal-pravesh/wav2vec2-large-xlsr'}
            </span>
          </div>
          <div className="system-detail-item">
            <span className="system-detail-label">Text Model</span>
            <span className="system-detail-value">fakespot-ai/roberta-base-ai-text-detection-v1</span>
          </div>
        </div>
      )}

      {/* ── Workspace Heading ───────────────────────────────── */}
      <div className="workspace-heading">
        <div className="workspace-eyebrow">{activeHeading.eyebrow}</div>
        <h1 className="workspace-title">{activeHeading.title}</h1>
        <p className="workspace-subtitle">{activeHeading.subtitle}</p>

        {/* Modality Tabs */}
        <nav className="modality-nav" aria-label="Inspection modality">
          {(
            [
              { key: 'video' as ModalityType, Icon: Film,      label: 'Video' },
              { key: 'audio' as ModalityType, Icon: Music,     label: 'Audio' },
              { key: 'image' as ModalityType, Icon: ImageIcon, label: 'Image' },
              { key: 'text'  as ModalityType, Icon: FileText,  label: 'Text'  },
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

      {/* ── Main Two-Column Workspace ───────────────────────── */}
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
    </div>
  );
};

export default App;
