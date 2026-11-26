import React, { useRef, useState, useCallback, useEffect } from 'react';
import { UploadCloud, FileAudio, FileImage } from 'lucide-react';
import { ModalityType, ImageCheckType } from '../types/api';

const VIDEO_EXTS = ['.mp4', '.avi', '.mov', '.webm', '.mkv'];
const AUDIO_EXTS = ['.wav', '.mp3', '.aac', '.flac', '.m4a', '.ogg'];
const IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.webp'];

const MAX_SIZE_BYTES = 100 * 1024 * 1024;

interface SourceMonitorProps {
  modality: ModalityType;
  file: File | null;
  onFileSelect: (file: File) => void;
  textContent: string;
  onTextChange: (text: string) => void;
  fileError: string | null;
  onFileError: (err: string | null) => void;
  imageCheckType?: ImageCheckType;
  onImageCheckTypeChange?: (val: ImageCheckType) => void;
}

interface MediaMeta {
  duration: number | null;
  width: number | null;
  height: number | null;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.floor((sec * 100) % 100);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(2, '0')}`;
}

function getFileExt(name: string): string {
  const idx = name.lastIndexOf('.');
  return idx >= 0 ? name.substring(idx).toLowerCase() : '';
}

export const SourceMonitor: React.FC<SourceMonitorProps> = ({
  modality,
  file,
  onFileSelect,
  textContent,
  onTextChange,
  fileError,
  onFileError,
  imageCheckType = 'face_swap',
  onImageCheckTypeChange,
}) => {
  const [dragOver, setDragOver] = useState(false);
  const [mediaMeta, setMediaMeta] = useState<MediaMeta>({ duration: null, width: null, height: null });
  const [mediaUrl, setMediaUrl] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    if (file && modality !== 'text') {
      const url = URL.createObjectURL(file);
      setMediaUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setMediaUrl(null);
    }
    setMediaMeta({ duration: null, width: null, height: null });
  }, [file, modality]);

  const getAllowedExtensions = () => {
    if (modality === 'audio') return AUDIO_EXTS;
    if (modality === 'image') return IMAGE_EXTS;
    return VIDEO_EXTS;
  };

  const validateAndSelect = useCallback(
    (fileList: FileList | null) => {
      onFileError(null);
      if (!fileList || fileList.length === 0) return;
      const f = fileList[0];
      const allowed = getAllowedExtensions();
      const ext = getFileExt(f.name);
      if (!allowed.includes(ext)) {
        onFileError(`Unsupported format '${ext}'. Accepted: ${allowed.join(', ')}`);
        return;
      }
      if (f.size > MAX_SIZE_BYTES) {
        onFileError('File exceeds the 100 MB limit.');
        return;
      }
      onFileSelect(f);
    },
    [modality, onFileSelect, onFileError]
  );

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    if (modality !== 'text') setDragOver(true);
  };
  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); e.stopPropagation(); };
  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    const rect = e.currentTarget.getBoundingClientRect();
    const { clientX, clientY } = e;
    if (clientX <= rect.left || clientX >= rect.right || clientY <= rect.top || clientY >= rect.bottom) {
      setDragOver(false);
    }
  };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    setDragOver(false);
    if (modality !== 'text' && e.dataTransfer.files?.length > 0) {
      validateAndSelect(e.dataTransfer.files);
    }
  };

  const handleVideoLoaded = () => {
    const v = videoRef.current;
    if (v) setMediaMeta({ duration: isFinite(v.duration) ? v.duration : null, width: v.videoWidth || null, height: v.videoHeight || null });
  };
  const handleAudioLoaded = () => {
    const a = audioRef.current;
    if (a) setMediaMeta({ duration: isFinite(a.duration) ? a.duration : null, width: null, height: null });
  };
  const handleImageLoaded = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    setMediaMeta({ duration: null, width: img.naturalWidth || null, height: img.naturalHeight || null });
  };

  const openFilePicker = () => fileInputRef.current?.click();

  const getMonitorLabel = () => {
    switch (modality) {
      case 'audio': return 'ACOUSTIC MONITOR';
      case 'image': return 'IMAGE VIEWPORT';
      case 'text':  return 'TEXT INSPECTION DESK';
      default:      return 'SOURCE MONITOR';
    }
  };

  const getDropPrompt = () => {
    switch (modality) {
      case 'audio': return 'Drop an audio recording here';
      case 'image': return 'Drop a photograph or face crop here';
      default:      return 'Drop a video recording here';
    }
  };

  const getAcceptedFormats = () =>
    getAllowedExtensions().map((e) => e.toUpperCase().replace('.', '')).join(' · ');

  const wordCount = textContent.trim() ? textContent.trim().split(/\s+/).length : 0;
  const charCount = textContent.length;

  return (
    <div
      className={`source-monitor${dragOver ? ' drag-over' : ''}`}
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Hidden file input */}
      {modality !== 'text' && (
        <input
          ref={fileInputRef}
          type="file"
          accept={getAllowedExtensions().join(',')}
          style={{ display: 'none' }}
          onChange={(e) => { validateAndSelect(e.target.files); e.target.value = ''; }}
          aria-label="Choose media file"
        />
      )}

      {/* Monitor title bar */}
      <div className="monitor-header">
        <div className="monitor-header-dots">
          <span className="monitor-dot monitor-dot--red" />
          <span className="monitor-dot monitor-dot--yellow" />
          <span className="monitor-dot monitor-dot--green" />
        </div>
        <span className="monitor-label">{getMonitorLabel()}</span>
        <div style={{ width: 48 }} />
      </div>

      {/* Main viewport */}
      <div className="monitor-viewport">
        {/* Corner marks */}
        <div className="monitor-corners"><span /></div>

        {/* TEXT MODE */}
        {modality === 'text' && (
          <div className="monitor-text-editor">
            <textarea
              className="monitor-textarea"
              value={textContent}
              onChange={(e) => onTextChange(e.target.value)}
              placeholder="Paste or type article text, transcription, or statement to evaluate for synthetic writing patterns…"
              aria-label="Text to inspect"
            />
          </div>
        )}

        {/* EMPTY STATE (file modes) */}
        {modality !== 'text' && !file && !dragOver && (
          <div className="monitor-empty">
            <div className="monitor-empty-icon">
              {modality === 'audio' ? <FileAudio size={24} /> :
               modality === 'image' ? <FileImage size={24} /> :
               <UploadCloud size={24} />}
            </div>
            <div>
              <p className="monitor-empty-text">{getDropPrompt()}</p>
              <p className="monitor-empty-subtext">{getAcceptedFormats()} · Max 100 MB</p>
            </div>
            <button className="monitor-choose-btn" onClick={openFilePicker} type="button">
              Choose file
            </button>
          </div>
        )}

        {/* DRAG OVER PROMPT */}
        {modality !== 'text' && !file && dragOver && (
          <p className="monitor-drag-text">Release to load media</p>
        )}

        {/* VIDEO */}
        {modality === 'video' && file && mediaUrl && (
          <video
            ref={videoRef}
            className="monitor-video"
            src={mediaUrl}
            controls
            onLoadedMetadata={handleVideoLoaded}
            preload="metadata"
          />
        )}

        {/* AUDIO */}
        {modality === 'audio' && file && mediaUrl && (
          <div className="monitor-audio-box">
            <div className="monitor-audio-waveform">
              {[0.1, 0.4, 0.2, 0.6, 0.3, 0.5, 0.15, 0.45, 0.25, 0.55, 0.35, 0.7].map((delay, i) => (
                <span key={i} className="monitor-audio-bar" style={{ animationDelay: `${delay}s` }} />
              ))}
            </div>
            <audio
              ref={audioRef}
              className="monitor-audio-player"
              src={mediaUrl}
              controls
              onLoadedMetadata={handleAudioLoaded}
            />
          </div>
        )}

        {/* IMAGE */}
        {modality === 'image' && file && mediaUrl && (
          <img
            className="monitor-image"
            src={mediaUrl}
            alt="Source preview"
            onLoad={handleImageLoaded}
          />
        )}

        {/* DRAG OVER REPLACE OVERLAY */}
        {file && dragOver && (
          <div style={{
            position: 'absolute', inset: 0,
            background: 'rgba(17,19,24,0.88)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            zIndex: 10, borderRadius: 'var(--radius-lg)',
          }}>
            <p className="monitor-drag-text">Release to replace media</p>
          </div>
        )}
      </div>

      {/* Image check type selector */}
      {modality === 'image' && onImageCheckTypeChange && (
        <div className="monitor-check-type">
          <div className="monitor-check-type-label">Inspection mode</div>
          <div className="check-type-grid">
            <button
              type="button"
              className={`check-type-btn${imageCheckType === 'face_swap' ? ' check-type-btn--active' : ''}`}
              onClick={() => onImageCheckTypeChange('face_swap')}
            >
              <div className="check-type-btn-title">Face-Swap Detection</div>
              <div className="check-type-btn-sub">deepfakeguard-v2 (ONNX)</div>
            </button>
            <button
              type="button"
              className={`check-type-btn${imageCheckType === 'general_ai' ? ' check-type-btn--active' : ''}`}
              onClick={() => onImageCheckTypeChange('general_ai')}
            >
              <div className="check-type-btn-title">General AI-Image</div>
              <div className="check-type-btn-sub">sdxl-detector</div>
            </button>
          </div>
        </div>
      )}

      {/* Reference caption for video/audio */}
      {(modality === 'video' || modality === 'audio') && (
        <div className="monitor-caption-row">
          <label htmlFor="caption-input">Caption</label>
          <input
            type="file"
            accept=".txt,.srt,.vtt"
            aria-label="Load caption file"
            onChange={async (event) => {
              const caption = event.target.files?.[0];
              if (!caption) return;
              if (caption.size > 200000) { onFileError('Caption exceeds the text limit.'); return; }
              const value = await caption.text();
              if (value.length > 50000) { onFileError('Caption exceeds 50,000 characters.'); return; }
              onTextChange(value);
            }}
          />
          <input
            id="caption-input"
            type="text"
            value={textContent}
            onChange={(e) => onTextChange(e.target.value)}
            placeholder="Paste spoken words or transcript for cross-modal analysis…"
          />
          {textContent.length > 0 && (
            <button
              type="button"
              onClick={() => onTextChange('')}
              style={{ background: 'none', border: 'none', color: '#5A6070', fontSize: 11, cursor: 'pointer', textDecoration: 'underline', flexShrink: 0 }}
            >
              Clear
            </button>
          )}
        </div>
      )}

      {/* Footer */}
      <div className="monitor-footer">
        {modality === 'text' ? (
          <span className="monitor-meta" style={{ width: '100%', display: 'flex', alignItems: 'center' }}>
            <span>{wordCount} words</span>
            <span>{charCount} chars</span>
            <span>~{Math.max(1, Math.ceil(wordCount / 200))} min read</span>
            {textContent.length > 0 && (
              <button className="monitor-replace-btn" onClick={() => onTextChange('')} type="button" style={{ marginLeft: 'auto' }}>
                Clear
              </button>
            )}
          </span>
        ) : file ? (
          <>
            <span className="monitor-filename" title={file.name}>{file.name}</span>
            <span className="monitor-meta">
              <span>{formatBytes(file.size)}</span>
              {mediaMeta.duration !== null && <span>{formatDuration(mediaMeta.duration)}</span>}
              {mediaMeta.width !== null && mediaMeta.height !== null && (
                <span>{mediaMeta.width}×{mediaMeta.height}</span>
              )}
            </span>
            <button className="monitor-replace-btn" onClick={openFilePicker} type="button">
              Replace
            </button>
          </>
        ) : (
          <span className="monitor-meta">
            <span>{getAcceptedFormats()}</span>
            <span>Max 100 MB</span>
          </span>
        )}
      </div>

      {/* Error */}
      {fileError && <div className="monitor-error">{fileError}</div>}
    </div>
  );
};

export default SourceMonitor;
