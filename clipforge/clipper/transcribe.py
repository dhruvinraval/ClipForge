from __future__ import annotations

from . import Cancelled

_MODELS: dict = {}


def transcribe(video, model_size="small", language=None, task="transcribe", offset=0.0,
               on_progress=None, cancelled=lambda: False):
    """Word-level transcription with faster-whisper (runs fully offline after first model download)."""
    from faster_whisper import WhisperModel

    model = _MODELS.get(model_size)
    if model is None:
        model = WhisperModel(model_size, device="auto", compute_type="auto")
        _MODELS[model_size] = model

    segments, info = model.transcribe(
        str(video), language=language or None, task=task, word_timestamps=True,
        vad_filter=True, beam_size=1,
    )
    total = max(float(info.duration or 0), 1.0)
    out = []
    for seg in segments:
        if cancelled():
            raise Cancelled()
        words = [
            {"word": w.word.strip(), "start": float(w.start) + offset, "end": float(w.end) + offset}
            for w in (seg.words or []) if w.word.strip()
        ]
        if words:
            out.append({"start": float(seg.start) + offset, "end": float(seg.end) + offset,
                        "text": seg.text.strip(), "words": words})
        if on_progress:
            on_progress(min(seg.end / total, 1.0))
    return {"language": info.language, "duration": float(info.duration or 0), "segments": out}
