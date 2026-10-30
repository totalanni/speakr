"""
Speakr /asr blueprint — NATIVE Nemotron 3.5 ASR (0.6B) + Nemotron 3 Diarization
via NeMo-Speech.cpp (CPU-only, in-container) + wespeaker voiceprints.

This is the CPU-only fork of the ASR engine. NeMo-Speech.cpp's `serve`
process runs INSIDE this container (127.0.0.1:8080), loaded with:
  - asr:  nemotron-3.5-asr-streaming-0.6b.q8_0.gguf  (RNNT, CPU)
  - diar: Nemotron-3-Diarization.q8_0.gguf

Nemotron gives us STT + who-spoke-when (word-level speaker tags). It does NOT
emit voiceprint embeddings, so we reuse the existing wespeaker ONNX extractor
(speaker_embeddings_from_spans) to turn Nemotron's speaker spans into the
256-d voiceprints Speakr's speaker-management chain (voice_profiles /
embedding matcher / isMe / cross-recording continuity) consumes.

The existing `asr_endpoint` connector POSTs to this endpoint (whisper-asr-
webservice contract), so Speakr's webgui AND fieldy-backend both consume the
same pipeline with no changes to them.

Contract:
  POST /asr
  - multipart field `audio_file`
  - query params: task, output, language, diarize, min_speakers, max_speakers,
                  initial_prompt, hotwords, model, return_speaker_embeddings
  Response JSON:
  {
    "text": "<full text>",
    "language": "en",
    "segments": [ {"speaker": "SPEAKER_00", "text": "...", "start": 0.7, "end": 15.5}, ... ],
    "speaker_embeddings": { "SPEAKER_00": [256 floats], ... }   # only when asked
  }
"""
import logging
import os
import sys
import time
import uuid
import wave
import array
import tempfile
import urllib.request
from pathlib import Path

from flask import Blueprint, request, jsonify

log = logging.getLogger("speakr.asr")

asr_bp = Blueprint("asr", __name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_NEMO_URL = os.environ.get("NEMO_ASR_URL", "http://127.0.0.1:8080")
_NEMO_TOKEN = os.environ.get("NEMO_ASR_TOKEN", "")
_PAO_DIR = Path(os.environ.get("PAO_DIR", "/srv/pao"))
_TRANSCRIBE_TIMEOUT = int(os.environ.get("NEMO_TRANSCRIBE_TIMEOUT", "1800"))

_stt_word_gap = float(os.environ.get("ASR_WORD_GAP", "0.5"))   # silence that splits segments
_stt_max_seg = float(os.environ.get("ASR_MAX_SEG", "15.0"))    # max seconds per segment


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_wespeaker_pipeline = None


def _get_wespeaker():
    """The existing wespeaker ONNX extractor (voiceprints only)."""
    global _wespeaker_pipeline
    if _wespeaker_pipeline is None:
        pao = Path(os.environ.get("PAO_DIR", "/srv/pao"))
        if str(pao) not in sys.path:
            sys.path.insert(0, str(pao))
        # Reuse the ffmpeg decode shim (PyAV kwarg bug workaround).
        import utils.audio as ua
        ua.decode_audio = _decode_audio_ffmpeg
        from onnx_pyannote import ONNXSpeakerDiarization
        _wespeaker_pipeline = ONNXSpeakerDiarization(
            model_name="speaker-diarization-3.1",
            providers=["CPUExecutionProvider"])
        log.info("wespeaker extractor loaded")
    return _wespeaker_pipeline


# ---------------------------------------------------------------------------
# Audio decode: ffmpeg -> 16kHz mono WAV
# ---------------------------------------------------------------------------
import subprocess


def _to_16k_wav(in_path: str) -> str:
    out = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        out,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {r.stderr[-500:]}")
    return out


def _wav_to_float32(path: str):
    import numpy as np
    w = wave.open(path, "rb")
    pcm = array.array("h")
    pcm.frombytes(w.readframes(w.getnframes()))
    w.close()
    return np.array(pcm, dtype="float32") / 32768.0


def _decode_audio_ffmpeg(path, target_sr=16000):
    """Drop-in for pyannote's decode_audio. Returns float32 numpy array @ 16k."""
    wav = _to_16k_wav(path)
    return _wav_to_float32(wav)


# ---------------------------------------------------------------------------
# NeMo-Speech.cpp client
# ---------------------------------------------------------------------------
def _nemo_transcribe(wav_path: str):
    """POST the 16k wav to NeMo serve with diarization. Returns parsed JSON."""
    boundary = uuid.uuid4().hex
    with open(wav_path, "rb") as f:
        data = f.read()

    parts = []
    parts.append(("--%s\r\n" % boundary).encode())
    parts.append(b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n')
    parts.append(b"Content-Type: audio/wav\r\n\r\n")
    parts.append(data)
    parts.append(b"\r\n")
    parts.append(("--%s\r\n" % boundary).encode())
    parts.append(b'Content-Disposition: form-data; name="diarization"\r\n\r\n')
    parts.append(b"true\r\n")
    parts.append(("--%s\r\n" % boundary).encode())
    parts.append(b'Content-Disposition: form-data; name="response_format"\r\n\r\n')
    parts.append(b"verbose_json\r\n")
    parts.append(("--%s--\r\n" % boundary).encode())
    body = b"".join(parts)

    headers = {"Content-Type": "multipart/form-data; boundary=%s" % boundary}
    if _NEMO_TOKEN:
        headers["Authorization"] = "Bearer %s" % _NEMO_TOKEN

    req = urllib.request.Request(
        f"{_NEMO_URL}/v1/audio/transcriptions",
        data=body,
        headers=headers,
        method="POST",
    )
    r = urllib.request.urlopen(req, timeout=_TRANSCRIBE_TIMEOUT)
    return r.read().decode()


# ---------------------------------------------------------------------------
# Group Nemotron's word-level speaker tags into segments
# ---------------------------------------------------------------------------
def _words_to_segments(words):
    """Group Nemotron words into speaker-boundary + gap aware segments.

    Nemotron returns words sorted by time, each with a numeric `speaker`.
    We relabel numeric speakers to stable SPEAKER_XX names in first-appearance
    order, and split a new segment on: speaker change, silence gap, or max len.
    """
    if not words:
        return []

    # stable relabel: 1 -> SPEAKER_00, 2 -> SPEAKER_01, ... in first-appearance order
    remap = {}
    counter = 0
    for w in words:
        s = w.get("speaker")
        if s is None:
            s = 0
        if s not in remap:
            remap[s] = f"SPEAKER_{counter:02d}"
            counter += 1

    segs = []
    cur = ""
    cs = ce = 0.0
    cur_spk = None
    for w in words:
        ws = float(w.get("start", 0.0))
        we = float(w.get("end", 0.0))
        ww = w.get("word", "")
        spk = remap[w.get("speaker", 0)]

        if not cur:
            cs = ce = ws
            cur_spk = spk
        else:
            break_seg = (
                spk != cur_spk
                or (ws - ce > _stt_word_gap)
                or (we - cs > _stt_max_seg)
            )
            if break_seg:
                segs.append({"speaker": cur_spk, "text": cur.strip(),
                             "start": round(cs, 3), "end": round(ce, 3)})
                cur = ""
                cs = ce = ws
                cur_spk = spk
        cur += (" " if cur else "") + ww
        ce = we
    if cur.strip():
        segs.append({"speaker": cur_spk, "text": cur.strip(),
                     "start": round(cs, 3), "end": round(ce, 3)})
    return segs


# ---------------------------------------------------------------------------
# Build per-speaker time spans from words (for the wespeaker voiceprints)
# ---------------------------------------------------------------------------
def _spans_from_words(words):
    remap = {}
    counter = 0
    for w in words:
        s = w.get("speaker", 0)
        if s not in remap:
            remap[s] = f"SPEAKER_{counter:02d}"
            counter += 1
    spans = {}
    for w in words:
        spk = remap[w.get("speaker", 0)]
        spans.setdefault(spk, []).append((float(w.get("start", 0.0)), float(w.get("end", 0.0))))
    # merge contiguous/overlapping spans per speaker into a few clean windows
    out = {}
    for spk, lst in spans.items():
        lst.sort()
        merged = []
        for s, e in lst:
            if merged and s - merged[-1][1] <= 0.3:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        out[spk] = [tuple(m) for m in merged]
    return out


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------
@asr_bp.route("/asr", methods=["POST"])
def asr():
    t0 = time.time()
    f = request.files.get("audio_file")
    if f is None:
        return jsonify({"error": "audio_file is required"}), 400

    in_path = tempfile.mktemp(suffix="_in.bin")
    f.save(in_path)
    language = (request.args.get("language") or "en").strip()
    diarize = request.args.get("diarize", "true")
    diarize = str(diarize).lower() not in ("false", "0", "no")
    # return_speaker_embeddings can arrive as a form field (Speakr connector)
    # or a query arg — check both.
    _emb_flag = request.form.get("return_speaker_embeddings")
    if _emb_flag is None:
        _emb_flag = request.args.get("return_speaker_embeddings", "false")
    want_embs = str(_emb_flag).lower() in ("true", "1", "yes")

    try:
        wav = _to_16k_wav(in_path)
    except Exception as e:
        return jsonify({"error": f"audio decode failed: {e}"}), 400

    speaker_embeddings = {}
    try:
        raw = _nemo_transcribe(wav)
        import json as _json
        data = _json.loads(raw)
        words = data.get("words") or []

        if not words:
            text = (data.get("text") or "").strip()
            segs = [{"speaker": "SPEAKER_00", "text": text, "start": 0.0, "end": 0.0}] if text else []
        else:
            segs = _words_to_segments(words)

        if want_embs:
            try:
                pipe = _get_wespeaker()
                spans = _spans_from_words(words)
                speaker_embeddings = pipe.speaker_embeddings_from_spans(wav, spans)
            except Exception as e:
                log.exception("voiceprint extraction failed (continuing without embeddings)")
                speaker_embeddings = {}

        segments = segs
        full_text = data.get("text") or " ".join(s["text"] for s in segments)

        log.info("/asr: %d segs in %.1fs (words=%d, embs=%d)",
                 len(segments), time.time() - t0, len(words), len(speaker_embeddings))

        resp = {"text": full_text, "language": language, "segments": segments}
        if speaker_embeddings:
            resp["speaker_embeddings"] = speaker_embeddings
        return jsonify(resp)
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:800]
        log.exception("/asr NeMo HTTP error")
        return jsonify({"error": f"NeMo transcribe failed: {e.code} {body}"}), 502
    except Exception as e:
        log.exception("/asr failed")
        return jsonify({"error": str(e)}), 500
    finally:
        for p in (in_path,):
            try:
                os.unlink(p)
            except OSError:
                pass


@asr_bp.route("/health", methods=["GET"])
def health():
    ok = True
    detail = ""
    try:
        r = urllib.request.urlopen(f"{_NEMO_URL}/health", timeout=5)
        detail = r.read().decode()[:200]
    except Exception as e:
        ok = False
        detail = str(e)
    return jsonify({"status": "ok" if ok else "degraded",
                    "backend": "nemotron-nemocpp",
                    "nemo": detail}), 200 if ok else 503
