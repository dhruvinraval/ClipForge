"""Highlight detection.

Every window of consecutive sentences (min_len..max_len seconds) is scored on hook strength, trigger words,
emotion, numbers, speaking pace, audio energy, sentence completeness and filler words. With action scoring on,
on-screen motion is measured too and fixed-length windows are added, so scenes without dialogue can win.
Selection skips overlapping, back-to-back and near-duplicate moments. An optional AI helper (Claude or a local
Ollama model) re-ranks the shortlist."""
from __future__ import annotations

import re
import subprocess

import numpy as np

from . import Cancelled
from .ai import annotate

HOOK_PHRASES = [
    "secret", "mistake", "never", "stop", "truth", "nobody", "everyone", "biggest", "worst",
    "best", "crazy", "insane", "shocking", "warning", "how to", "why you", "what if",
    "the reason", "you need", "you won't", "don't", "free", "money", "million", "hack",
    "trick", "changed my life", "game changer", "here's the thing", "most people", "wrong",
    "lesson", "the problem", "no one", "i was wrong", "unbelievable", "obsessed",
]
QUESTION_STARTS = {"how", "why", "what", "when", "who", "which", "can", "do", "does", "did",
                   "is", "are", "would", "should", "have", "ever", "imagine"}
FILLERS = re.compile(r"\b(um+|uh+|erm|you know|i mean|kind of|sort of)\b", re.I)


def build_sentences(segments):
    words = [w for s in segments for w in s["words"]]
    sents, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        gap = words[i + 1]["start"] - w["end"] if i + 1 < len(words) else 99
        if re.search(r"[.!?…]$", w["word"]) or gap > 0.8 or len(cur) >= 40:
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    return [{"start": s[0]["start"], "end": s[-1]["end"],
             "text": " ".join(x["word"] for x in s), "words": s} for s in sents]


def load_energy(video, audio_track=0):
    """Per-second loudness (0..1.5), normalised to the 95th percentile."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-map", f"0:a:{audio_track}?", "-vn", "-ac", "1",
           "-ar", "8000", "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    n = len(a) // 8000
    if n < 1:
        return np.full(1, 0.3, dtype=np.float32)
    rms = np.sqrt((a[: n * 8000].reshape(n, 8000) ** 2).mean(axis=1))
    return np.clip(rms / (np.percentile(rms, 95) + 1e-6), 0, 1.5)


def load_motion(video, duration, on_progress=None, cancelled=lambda: False):
    """Per-second amount of on-screen change (0..1.5). Decodes the whole video once at tiny size."""
    W, H, fps = 96, 54, 2
    cmd = ["ffmpeg", "-v", "error", "-threads", "0", "-i", str(video), "-an", "-sn",
           "-vf", f"fps={fps},scale={W}:{H}:flags=fast_bilinear,format=gray", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if hasattr(cancelled, "attach"):
        cancelled.attach(p)
    size, prev, diffs, expected = W * H, None, [], max(duration * fps, 1)
    try:
        while True:
            if cancelled():
                p.kill()
                raise Cancelled()
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).astype(np.int16)
            if prev is not None:
                diffs.append(float(np.abs(frame - prev).mean()))
            prev = frame
            if on_progress and len(diffs) % 120 == 0:
                on_progress(min(len(diffs) / expected, 1.0))
        p.wait()
        if cancelled():
            raise Cancelled()
    finally:
        if hasattr(cancelled, "detach"):
            cancelled.detach()
        if p.poll() is None:
            p.kill()
    arr = np.array(diffs, dtype=np.float32)
    n = len(arr) // fps
    if n < 1:
        return np.full(1, 0.3, dtype=np.float32)
    per_sec = arr[: n * fps].reshape(n, fps).mean(axis=1)
    return np.clip(per_sec / (np.percentile(per_sec, 95) + 1e-6), 0, 1.5)


def _score(start, end, sl, energy, motion=None):
    dur = max(end - start, 1e-3)
    text = " ".join(s["text"] for s in sl)
    low = text.lower()
    nwords = sum(len(s["words"]) for s in sl)
    if sl:
        first = sl[0]["text"].lower()
        fw = re.sub(r"\W", "", first.split()[0]) if first.split() else ""
        hook = 0.5 if ("?" in sl[0]["text"] or fw in QUESTION_STARTS) else 0.0
        hook += min(0.5, 0.25 * sum(p in first for p in HOOK_PHRASES))
        kw = min(1.0, sum(low.count(p) for p in HOOK_PHRASES) / 5)
        emo = min(1.0, (text.count("!") + text.count("?") + sum(1 for w in text.split() if len(w) > 2 and w.isupper())) / 5)
        num = min(1.0, len(re.findall(r"\d+", text)) / 4)
        pace = 1 - min(1.0, abs(nwords / dur - 2.8) / 2.0)
        done = 1.0 if text.strip()[-1:] in ".!?" else 0.4
        filler = min(1.0, len(FILLERS.findall(text)) / max(nwords / 25, 1) / 3)
    else:                                   # no speech in this window (action / music scene)
        hook = kw = emo = num = filler = 0.0
        pace, done = 0.4, 1.0
    seg_e = energy[int(start): int(end) + 1]
    en = float(min(1.0, seg_e.mean() * 0.8 + seg_e.std() * 0.8)) if len(seg_e) else 0.3
    base = 25 * hook + 15 * kw + 10 * emo + 5 * num + 15 * pace + 20 * en + 10 * done - 10 * filler
    if motion is not None:
        seg_m = motion[int(start): int(end) + 1]
        ms = float(min(1.0, seg_m.mean() * 0.75 + seg_m.max() * 0.25)) if len(seg_m) else 0.0
        base = base * 0.85 + 15 * ms
    return float(max(0.0, min(99.0, base * 1.3)))


def _title(t):
    parts = re.sub(r"\s+", " ", t).strip().rstrip(".,;:!?-").split()
    w = " ".join(parts[:9]) + ("…" if len(parts) > 9 else "")
    return w[:1].upper() + w[1:]


def _shingles(text):
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + 3]) for i in range(max(len(w) - 2, 0))} or set(w)


def _near_duplicate(a, b):
    if not a or not b:
        return False
    return len(a & b) / len(a | b) > 0.3


def _candidates(sents, energy, motion, duration, min_len, max_len):
    out = []                                              # (score, start, end, text)
    for i in range(len(sents)):
        best = None
        for j in range(i, len(sents)):
            d = sents[j]["end"] - sents[i]["start"]
            if d > max_len:
                break
            if d < min_len:
                continue
            sc = _score(sents[i]["start"], sents[j]["end"], sents[i: j + 1], energy, motion)
            if best is None or sc > best[0]:
                best = (sc, i, j)
        if best:
            sc, i, j = best
            out.append((sc, sents[i]["start"], sents[j]["end"], " ".join(x["text"] for x in sents[i: j + 1])))
    if motion is not None:                                # extra fixed-length windows for scenes without speech
        target, t = (min_len + max_len) / 2, 0.0
        while t + min_len <= duration:
            s, e = t, min(duration, t + target)
            sl = [x for x in sents if x["end"] > s and x["start"] < e]
            if sl:
                if sl[0]["start"] < s and s - sl[0]["start"] <= 4:
                    s = sl[0]["start"]
                if sl[-1]["end"] > e and sl[-1]["end"] - e <= 4 and sl[-1]["end"] - s <= max_len:
                    e = sl[-1]["end"]
                sl = [x for x in sents if x["start"] >= s - 0.01 and x["end"] <= e + 0.01]
            out.append((_score(s, e, sl, energy, motion), s, e, " ".join(x["text"] for x in sl)))
            t += 5
    return out


def find_highlights(segments, video, duration, count=6, min_len=20, max_len=60, ai=None, audio_track=0,
                    motion=None, dedupe=True, log=print, on_stage=lambda s: None):
    sents = build_sentences(segments)
    energy = load_energy(video, audio_track)
    cands = _candidates(sents, energy, motion, duration, min_len, max_len)

    pool_size = count * 3 if ai else count
    gap = 8.0 if dedupe else 0.0                          # back-to-back windows are usually the same scene
    chosen, skipped = [], 0
    for sc, s0, e0, text in sorted(cands, key=lambda c: c[0], reverse=True):
        s, e = max(0.0, s0 - 0.15), min(duration, e0 + 0.35)
        sh = _shingles(text)
        if any(not (e <= c["start"] - gap or s >= c["end"] + gap) for c in chosen):
            continue
        if dedupe and any(_near_duplicate(sh, c["_sh"]) for c in chosen):
            skipped += 1
            continue
        chosen.append({"start": s, "end": e, "score": round(sc, 1), "text": text, "title": _title(text) or "Moment",
                       "_sh": sh})
        if len(chosen) >= pool_size:
            break
    if skipped:
        log(f"Skipped {skipped} near-duplicate moment(s)")
        if len(chosen) < min(count, pool_size):
            log(f"Only {len(chosen)} distinct moment(s) found; turn off the duplicate filter to get more")

    if not chosen:                                        # no usable speech: fall back to evenly spaced windows
        win = min((min_len + max_len) / 2, duration)
        n = max(1, min(count, int(duration // win)))
        step = duration / n
        chosen = [{"start": k * step, "end": min(duration, k * step + win), "score": 50.0, "text": "",
                   "title": f"Moment {k + 1}", "_sh": set()} for k in range(n)]

    if ai and len(chosen) > 0:
        on_stage("Ranking with AI")
        annotate(chosen, ai[0], ai[1], log, rerank=True)
    chosen.sort(key=lambda c: c["score"], reverse=True)
    for c in chosen:
        c.pop("_sh", None)
    return chosen[:count]
