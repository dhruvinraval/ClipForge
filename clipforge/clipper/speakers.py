"""Experimental, fully offline speaker labelling.

No neural model: each stretch of speech is described by its voice colour (MFCCs) and pitch, then clustered into
k groups. It works best for 2-3 clearly different voices (for example a man and a woman). Similar voices,
overlapping speech and noisy audio will be confused. Words get a 'spk' number (0 = whoever speaks first)."""
from __future__ import annotations

import wave

import numpy as np

SR = 16000


def _mel(f):
    return 2595 * np.log10(1 + f / 700)


def _imel(m):
    return 700 * (10 ** (m / 2595) - 1)


def _fbank(n=26, nfft=512, lo=80, hi=7600):
    pts = _imel(np.linspace(_mel(lo), _mel(hi), n + 2))
    bins = np.floor((nfft + 1) * pts / SR).astype(int)
    fb = np.zeros((n, nfft // 2 + 1))
    for i in range(1, n + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        for k in range(l, c):
            fb[i - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[i - 1, k] = (r - k) / max(r - c, 1)
    return fb


_FB = _fbank()
_DCT = np.cos(np.pi * np.arange(13)[:, None] * (2 * np.arange(26)[None] + 1) / 52)


def _frames(x, size, hop):
    n = 1 + (len(x) - size) // hop
    return x[np.arange(size)[None] + hop * np.arange(n)[:, None]]


def _features(x):
    """27 numbers: 13 MFCC means, 13 MFCC std-devs, log median pitch (NaN if unvoiced)."""
    if len(x) < 4000:
        return None
    fr = _frames(x, 400, 160) * np.hamming(400)
    mel = np.log(np.abs(np.fft.rfft(fr, 512)) ** 2 @ _FB.T + 1e-10)
    mfcc = mel @ _DCT.T
    pf = _frames(x, 640, 320)
    rms = np.sqrt((pf ** 2).mean(1))
    f0s = []
    for f, e in zip(pf, rms):
        if e < 0.2 * rms.max() or e < 0.005:
            continue
        spec = np.abs(np.fft.rfft(f * np.hanning(640), 1280)) ** 2
        r = np.fft.irfft(spec)[:400]
        lag = 40 + int(np.argmax(r[40:320]))
        if r[lag] / (r[0] + 1e-9) > 0.3:
            f0s.append(SR / lag)
    f0 = np.log(np.median(f0s)) if f0s else np.nan
    return np.concatenate([mfcc.mean(0), mfcc.std(0), [f0]])


def _kmeans(X, k, iters=40):
    c = [X[0]]
    for _ in range(k - 1):
        d = np.min([((X - ci) ** 2).sum(1) for ci in c], axis=0)
        c.append(X[int(d.argmax())])
    C = np.array(c)
    for _ in range(iters):
        lab = ((X[:, None, :] - C[None]) ** 2).sum(2).argmin(1)
        new = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(new, C):
            break
        C = new
    return lab


def label_speakers(wav, segments, k=2):
    """Adds word['spk'] in place. Returns True if labelling was possible."""
    words = [w for s in segments for w in s["words"]]
    chunks, cur = [], []
    for w in words:
        if cur and (w["start"] - cur[-1]["end"] > 0.5 or w["end"] - cur[0]["start"] > 6):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)

    feats, idx = [], []
    with wave.open(str(wav)) as wf:
        for ci, ch in enumerate(chunks):
            if ch[-1]["end"] - ch[0]["start"] < 0.6:
                continue
            wf.setpos(min(int(ch[0]["start"] * SR), wf.getnframes() - 1))
            raw = wf.readframes(int((ch[-1]["end"] - ch[0]["start"]) * SR))
            f = _features(np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768)
            if f is not None:
                feats.append(f)
                idx.append(ci)
    if len(feats) < k * 3:
        return False
    X = np.array(feats)
    X[np.isnan(X[:, -1]), -1] = np.nanmedian(X[:, -1]) if not np.all(np.isnan(X[:, -1])) else 0.0
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    X[:, -1] *= 3                                   # pitch is the most reliable cue: weight it triple
    lab = dict(zip(idx, _kmeans(X, k)))

    remap, last = {}, 0
    for ci, ch in enumerate(chunks):
        raw_l = lab.get(ci)
        if raw_l is None:
            spk = last
        else:
            remap.setdefault(raw_l, len(remap))
            spk = last = remap[raw_l]
        for w in ch:
            w["spk"] = spk
    return True
