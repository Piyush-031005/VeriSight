export interface ModelReadiness {
  loaded: boolean;
  checkpoint_loaded: boolean;
  checkpoint_path?: string | null;
  is_mock: boolean;
  model_name: string;
  error?: string | null;
}

export interface VideoEvidence {
  band?: [number, number] | null;
  score: number | null;
  frames_analyzed: number;
  faces_detected: number;
  available: boolean;
  model: string;
  checkpoint_loaded?: boolean;
  is_mock?: boolean;
  reason?: string | null;
}

export interface AudioEvidence {
  band?: [number, number] | null;
  score: number | null;
  duration_seconds: number;
  available: boolean;
  model: string;
  checkpoint_loaded?: boolean;
  is_mock?: boolean;
  reason?: string | null;
}

export interface FusionEvidence {
  band?: [number, number] | null;
  score: number | null;
  method: string;
  threshold_used: number | null;
  threshold_name: string | null;
  checkpoint_loaded?: boolean;
  is_calibrated?: boolean;
}

export interface ImageEvidence {
  score: number | null;
  faces_detected: number;
  available: boolean;
  model: string;
  resolution?: string | null;
  reason?: string | null;
  check_type?: string;
  sdxl_score?: number | null;
}

export interface TextEvidence {
  sentences?: {start: number; end: number; text: string; score: number}[];
  spans?: {start: number; end: number; text: string; score: number; new_tokens: number}[];
  band?: [number, number] | null;
  score: number | null;
  word_count: number;
  burstiness_score?: number | null;
  perplexity_score?: number | null;
  lexical_diversity?: number | null;
  available: boolean;
  model: string;
  label?: string;
  reason?: string | null;
}

export interface CrossModalTextEvidence {
  transcript?: string | null;
  wer?: number | null;
  cer?: number | null;
  mismatch?: number | null;
  similarity_method?: string | null;
  cross_modal_consistency?: number | null;
  extracted_text_detected: boolean;
  ocr_word_count: number;
  transcript_status?: string | null;
  reason?: string | null;
}

export interface TimingEvidence {
  frame_sampling_sec?: number;
  face_detection_sec?: number;
  audio_extraction_sec?: number;
  video_inference_sec?: number;
  audio_inference_sec?: number;
  fusion_inference_sec?: number;
  total_pipeline_sec: number;
}

export interface PredictResponse {
  prediction: 'REAL' | 'FAKE' | 'UNCERTAIN';
  confidence: number | null;
  prediction_path: string;
  is_mock?: boolean;
  models_ready?: boolean;
  video: VideoEvidence;
  audio: AudioEvidence;
  fusion: FusionEvidence;
  timing: TimingEvidence;
  warnings: string[];
  image?: ImageEvidence;
  text?: TextEvidence;
  cross_modal?: CrossModalTextEvidence;
  disagreement?: string | null;
  v2_multi_verify?: {
    gradcam_overlay_b64?: string | null;
    token_attributions?: { token: string; weight: number }[];
    consistency_scores?: {
      visual_text?: number | null;
      audio_text?: number | null;
      av_sync?: number | null;
      is_contradiction?: boolean;
    };
    uncertainty?: {
      entropy?: number | null;
      verdict_note?: string | null;
    };
    grounded_summary?: string | null;
    v2_enrichment_sec?: number;
  } | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  device: string;
  ffmpeg_available: boolean;
  models_loaded: boolean;
  models_ready?: boolean;
  thresholds_calibrated?: boolean;
  video_model?: ModelReadiness | null;
  audio_model?: ModelReadiness | null;
  fusion_model?: ModelReadiness | null;
}

export type ModalityType = 'video' | 'audio' | 'image' | 'text';
export type ImageCheckType = 'face_swap' | 'general_ai';
