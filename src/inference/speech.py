"""Bounded mono PCM decoding, energy filtering and sequential acoustic inference."""
import io
import json
from pathlib import Path
import math
import shutil
import subprocess
import wave

import numpy as np

from .runtime import positive_index


class NoAudioStreamError(ValueError):
    """Raised when media has no decodable audio stream."""

MELODY = "MelodyMachine/Deepfake-audio-detection-V2"
GARY = "garystafford/wav2vec2-deepfake-voice-detector"
ONNX_AUDIO = "pranjal-pravesh/wav2vec2-large-xlsr-deepfake-audio-classification"
ONNX_ORIGIN = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
ONNX_REV = "4b1c4a294ab68ab444dac038fec8e8da2cc1982f"
ORIGIN_REV = "f7050b586236dc910d1157f430def2d0647b02b4"
AUDIO = ONNX_AUDIO
AUDIO_MODELS = {MELODY, "Hemgg/Deepfake-audio-detection", GARY, ONNX_AUDIO}
ASR = "Systran/faster-whisper-tiny"
RATE = 16000


def _decode_with_pyav(path):
    import av

    cap = 30 * RATE
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise NoAudioStreamError("Video contains no audio track")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=RATE)
        chunks = []
        total = 0
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                arr = resampled.to_ndarray().reshape(-1)
                if len(arr):
                    chunks.append(arr)
                    total += len(arr)
            if total >= cap:
                break
        for resampled in resampler.resample(None):
            arr = resampled.to_ndarray().reshape(-1)
            if len(arr):
                chunks.append(arr)
                total += len(arr)
    if not chunks:
        raise NoAudioStreamError("Empty audio stream")
    return np.concatenate(chunks)[:cap].astype(np.float32) / 32768.0


def decode(path):
    try:
        return _decode_with_pyav(path)
    except NoAudioStreamError:
        raise
    except Exception as pyav_exc:
        if shutil.which("ffmpeg") is None:
            raise ValueError(f"Audio decoding failed: {pyav_exc}") from pyav_exc
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
               "-t", "30", "-ac", "1", "-ar", str(RATE), "-c:a", "pcm_s16le",
               "-f", "wav", "pipe:1"]
    proc = subprocess.run(command, capture_output=True, timeout=60)
    if proc.returncode:
        err = proc.stderr.decode(errors="replace").strip()[-300:] or "Audio decoding failed"
        if "matches no streams" in err or "does not contain any stream" in err:
            raise NoAudioStreamError("Video contains no audio track")
        raise ValueError(err)
    with wave.open(io.BytesIO(proc.stdout)) as pcm:
        data = pcm.readframes(30 * RATE)
    if not data:
        raise NoAudioStreamError("Empty audio stream")
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.


def active_windows(wav, window_seconds=5):
    """20 ms energy frames. Keep internal pauses, weight windows by active time."""
    if len(wav) < RATE // 2:
        return
    frame = 320
    padded = np.pad(wav, (0, (-len(wav)) % frame)).reshape(-1, frame)
    active = (np.mean(padded**2, axis=1) >= .002**2) & (np.max(np.abs(padded), axis=1) >= .008)
    indices = np.flatnonzero(active)
    if not len(indices):
        return
    step = int(window_seconds * RATE / frame)
    for start in range(int(indices[0]), int(indices[-1])+1, step):
        end = min(start+step, int(indices[-1])+1)
        count = int(active[start:end].sum()) * frame
        if end == len(padded) and active[-1]:
            count -= len(padded) * frame - len(wav)
        duration = count / RATE
        if duration >= .5:
            yield wav[start*frame:end*frame], float(duration)


def _load_streamed_int8(model_dir):
    """Copy and quantize one pretrained linear layer at a time on CPU."""
    import torch
    from safetensors import safe_open
    from torch.ao.quantization import default_dynamic_qconfig
    from torch.ao.nn.quantized.dynamic import Linear as Int8Linear
    from transformers import AutoConfig, AutoModelForAudioClassification

    config = AutoConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        model = AutoModelForAudioClassification.from_config(config)
    file = str(Path(model_dir) / "model.safetensors")
    with safe_open(file, framework="pt") as archive:
        if set(archive.keys()) != set(model.state_dict()):
            raise ValueError("Checkpoint keys do not match every model parameter")
    paths = [name for name, _ in model.named_modules()]
    for name in paths:
        module = model.get_submodule(name)
        params = dict(module.named_parameters(recurse=False))
        buffers = dict(module.named_buffers(recurse=False))
        for attr, placeholder in (params | buffers).items():
            key = ".".join(filter(None, (name, attr)))
            with safe_open(file, framework="pt") as archive:
                tensor = archive.get_tensor(key).clone()
            if tensor.shape != placeholder.shape:
                raise ValueError(f"Checkpoint shape mismatch for {key}")
            if attr in params:
                tensor = torch.nn.Parameter(tensor, requires_grad=False)
            setattr(module, attr, tensor)
            del tensor
        if type(module) is torch.nn.Linear:
            parent, _, attr = name.rpartition(".")
            module.qconfig = default_dynamic_qconfig
            setattr(model.get_submodule(parent), attr, Int8Linear.from_float(module))
        del params, buffers, module
    return model.eval()


def _hf_cached_or_fetch(repo_id, filename, revision):
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo_id, filename, revision=revision, local_files_only=True)
    except Exception:
        return hf_hub_download(repo_id, filename, revision=revision)


def onnx_audio_files():
    """The export publishes the graph; its parent publishes labels and preprocessing."""
    model_path = _hf_cached_or_fetch(ONNX_AUDIO, "model_int8.onnx", ONNX_REV)
    config_path = _hf_cached_or_fetch(ONNX_ORIGIN, "config.json", ORIGIN_REV)
    _hf_cached_or_fetch(ONNX_ORIGIN, "preprocessor_config.json", ORIGIN_REV)
    return model_path, Path(config_path)


def _load_onnx_audio():
    import onnxruntime as ort
    from transformers import AutoFeatureExtractor
    model_path, config_path = onnx_audio_files()
    labels = json.loads(config_path.read_text(encoding="utf-8"))["id2label"]
    index = positive_index(labels, {"fake"}, {"real"})
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_prepacking", "1")
    model = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
    try:
        extractor = AutoFeatureExtractor.from_pretrained(config_path.parent, local_files_only=True)
    except Exception:
        extractor = AutoFeatureExtractor.from_pretrained(config_path.parent)
    return extractor, model, index


def _onnx_probability(samples, extractor, model, index):
    inputs = extractor(samples, sampling_rate=RATE, return_tensors="np", return_attention_mask=True)
    feed = {"input_values": inputs["input_values"].astype(np.float32, copy=False),
            "attention_mask": inputs["attention_mask"].astype(np.int32, copy=False)}
    logits = model.run(None, feed)[0].reshape(2)
    probs = np.exp(logits - logits.max())
    return float(probs[index] / probs.sum())


def load_audio(name):
    if name == ONNX_AUDIO:
        return _load_onnx_audio()
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    if name == GARY:
        from huggingface_hub import snapshot_download
        directory = snapshot_download(name, revision="c66306024a7ede0be291e9c4558b37634782dc4e",
                                      allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"])
        extractor = AutoFeatureExtractor.from_pretrained(directory)
        model = _load_streamed_int8(directory)
    else:
        extractor = AutoFeatureExtractor.from_pretrained(name)
        model, loading = AutoModelForAudioClassification.from_pretrained(
            name, use_safetensors=True, output_loading_info=True)
        if loading["missing_keys"] or loading["mismatched_keys"]:
            raise ValueError("Acoustic checkpoint did not supply all model weights")
    model.eval()
    return extractor, model, positive_index(model.config.id2label, {"fake", "aivoice"}, {"real", "humanvoice"})


def acoustic(wav, slot, window_seconds=5, name=AUDIO):
    import torch
    windows = list(active_windows(wav, window_seconds))
    seconds = sum(duration for _, duration in windows)
    result = dict(model=name, available=False, score=None, duration_seconds=len(wav)/RATE,
                  usable_seconds=seconds, windows=len(windows), reason=None, checkpoint_loaded=False)
    if not windows:
        result["reason"] = "Silence or less than 0.5 seconds of usable signal per window."
        return result
    scores = []
    try:
        with slot.use(name, lambda: load_audio(name)) as (extractor, model, index), torch.inference_mode():
            result["checkpoint_loaded"] = True
            for samples, weight in windows:
                if name == ONNX_AUDIO:
                    score = _onnx_probability(samples, extractor, model, index)
                else:
                    inputs = extractor(samples, sampling_rate=RATE, return_tensors="pt")
                    inputs["input_values"] = inputs["input_values"].to(model.dtype)
                    score = float(model(**inputs).logits.float().softmax(-1)[0, index])
                if not math.isfinite(score):
                    raise ValueError("Nonfinite acoustic score")
                scores.append((score, weight))
        total = 0.
        for score, weight in sorted(scores):
            total += weight
            if total >= seconds/2:
                result.update(score=score, available=True)
                break
    except Exception as exc:
        result["reason"] = f"Acoustic inference unavailable: {exc}"
    return result


def transcribe(wav, slot):
    from faster_whisper import WhisperModel

    def load():
        return WhisperModel(ASR, device="cpu", compute_type="int8", cpu_threads=2)

    with slot.use(ASR, load) as model:
        segments, info = model.transcribe(wav, beam_size=1, vad_filter=True,
                                          condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip(), info.language
