from __future__ import annotations

import json
import re
from pathlib import Path

from . import Cancelled
from .ai import annotate, keyword_hashtags, resolve as resolve_ai
from .captions import build_ass, normalize_style, words_in_range
from .download import download
from .export import write_csv, write_srt, write_txt, write_vtt
from .highlights import find_highlights, load_motion
from .render import (ASPECTS, extract_audio, list_audio_tracks, make_thumb, probe, render_clip, smooth_path,
                     track_faces)
from .speakers import label_speakers
from .transcribe import transcribe

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ small helpers
def parse_ranges(text, duration):
    """Manual clip ranges, one per line: '1:20-2:05 optional title'. Times are SS, MM:SS or HH:MM:SS."""
    def sec(s):
        v = 0.0
        for part in s.split(":"):
            v = v * 60 + float(part)
        return v
    out = []
    for line in (text or "").splitlines():
        m = re.match(r"\s*([\d:.]+)\s*(?:-|–|to)\s*([\d:.]+)\s*(.*)", line)
        if not m:
            continue
        s, e = sec(m.group(1)), min(sec(m.group(2)), duration)
        if e - s >= 1:
            out.append({"start": s, "end": e, "score": 0, "text": "", "title": m.group(3).strip() or f"Clip {len(out) + 1}"})
    return out


def _slug(t):
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")[:40] or "clip"


# ------------------------------------------------------------------ transcript store (supports partial coverage)
def _load_tr(job_dir):
    p = Path(job_dir) / "transcript.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"language": None, "duration": 0, "segments": [], "coverage": []}


def _save_tr(job_dir, tr):
    (Path(job_dir) / "transcript.json").write_text(json.dumps(tr))


def _flat(tr):
    out = []
    for w in sorted((w for s in tr["segments"] for w in s["words"]), key=lambda w: w["start"]):
        if out and abs(w["start"] - out[-1]["start"]) < 0.05 and w["word"] == out[-1]["word"]:
            continue
        out.append(w)
    return out


def _covered(cov, s, e):
    return any(a <= s + 0.01 and e <= b + 0.01 for a, b in cov)


def _merge(cov):
    out = []
    for a, b in sorted(cov):
        if out and a <= out[-1][1] + 0.5:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def ensure_words(job_dir, src, opts, dur, start, end, cancelled, on_progress=None):
    """Words for [start, end]. Only transcribes the part that has not been transcribed yet."""
    tr = _load_tr(job_dir)
    if not _covered(tr.get("coverage", []), start, end):
        rs, re_ = max(0.0, start - 1.0), min(dur, end + 1.0)
        wav = Path(job_dir) / f"range_{int(rs)}.wav"
        try:
            extract_audio(src, opts["audio_track"], wav, start=rs, dur=re_ - rs)
            part = transcribe(wav, opts["whisper_model"], opts.get("language"),
                              "translate" if opts.get("translate") else "transcribe", rs, on_progress, cancelled)
        finally:
            wav.unlink(missing_ok=True)
        keep = [s for s in tr["segments"] if not (s["start"] >= rs - 0.01 and s["end"] <= re_ + 0.01)]
        tr["segments"] = sorted(keep + part["segments"], key=lambda s: s["start"])
        tr["language"] = tr.get("language") or part["language"]
        tr["duration"] = dur
        tr["coverage"] = _merge(tr.get("coverage", []) + [[rs, re_]])
        _save_tr(job_dir, tr)
    return _flat(tr)


# ------------------------------------------------------------------ one clip
def render_one(job_dir, src, src_w, src_h, opts, words, pick, n, on_progress, cancelled, log, name=None):
    clips_dir = Path(job_dir) / "clips"
    W, H = ASPECTS[opts["aspect"]]
    cdur = pick["end"] - pick["start"]
    name = name or f"{n:02d}_{_slug(pick['title'])}"
    cw = words_in_range(words, pick["start"], pick["end"])

    ass_path = None
    if opts["captions"] or opts["hook"] or opts["progress_bar"]:
        ass_path = clips_dir / f"{name}.ass"
        ass_path.write_text(build_ass(cw if opts["captions"] else [], W, H, cdur, opts["style"],
                                      pick["title"] if opts["hook"] else None, opts["progress_bar"]),
                            encoding="utf-8")
    track = None
    if opts["layout"] == "smart" and src_w / src_h > W / H + 0.02:
        log("Tracking faces for smart crop")
        try:
            track = smooth_path(track_faces(src, pick["start"], pick["end"], cancelled=cancelled), src_w / 2)
        except Cancelled:
            raise
        except Exception as e:  # e.g. OpenCV build without the Haar detector -> plain centre crop
            log(f"Face tracking unavailable ({type(e).__name__}); using centre crop")

    out = clips_dir / f"{name}.mp4"
    render_clip(src, out, pick["start"], cdur, W, H, opts["layout"], track, ass_path, ROOT / "fonts", opts["gpu"],
                src_w, src_h, on_progress, cancelled, audio_track=opts["audio_track"], target_mb=opts["target_mb"])
    make_thumb(out, clips_dir / f"{name}.jpg")

    subs = []
    if opts["export_subs"] and cw:
        write_srt(cw, clips_dir / f"{name}.srt")
        write_vtt(cw, clips_dir / f"{name}.vtt")
        write_txt(cw, clips_dir / f"{name}.txt")
        subs = [f"{name}.srt", f"{name}.vtt", f"{name}.txt"]
    text = pick.get("text") or " ".join(w["word"] for w in cw)
    return {"index": n, "title": pick["title"], "start": round(pick["start"], 2), "end": round(pick["end"], 2),
            "duration": round(cdur, 1), "score": pick.get("score", 0), "text": text[:400],
            "summary": pick.get("summary", ""), "hashtags": pick.get("hashtags", []),
            "file": f"{name}.mp4", "thumb": f"{name}.jpg", "subs": subs, "rev": 1,
            "size_mb": round(out.stat().st_size / 1048576, 1)}


# ------------------------------------------------------------------ whole job
def run_pipeline(job_dir, opts, update, cancelled):
    job_dir = Path(job_dir)
    clips_dir = job_dir / "clips"
    clips_dir.mkdir(exist_ok=True)
    log = lambda m: update(log=m)  # noqa: E731

    def check():
        if cancelled():
            raise Cancelled()

    # 1 -- source -----------------------------------------------------------
    if opts.get("url"):
        update(stage="Downloading video", progress=0.01, log=f"Downloading {opts['url']}")
        src, title = download(opts["url"], job_dir, lambda f: update(progress=0.01 + 0.09 * f), cancelled)
        update(title=title)
    else:
        src = Path(opts["source"])
    src_w, src_h, dur = probe(src)
    tracks = list_audio_tracks(src)
    if tracks and opts["audio_track"] >= len(tracks):
        log(f"Audio track {opts['audio_track'] + 1} not found; using track 1")
        opts["audio_track"] = 0
    full = dict(opts)
    full["source"] = str(src)
    (job_dir / "opts.json").write_text(json.dumps(full))
    update(duration=dur, progress=0.1, log=f"Source: {src_w}x{src_h}, {dur:.0f}s"
           + (f", audio {tracks[opts['audio_track']]['label']}" if tracks else ", no audio track"))
    check()

    # 2 -- transcript -------------------------------------------------------
    manual = parse_ranges(opts.get("manual"), dur)
    ai = resolve_ai(opts, log) if opts["ai"] != "off" else None
    need_text = opts["captions"] or not manual or opts["export_subs"] or ai is not None
    if need_text and not tracks:
        raise RuntimeError("This video has no audio track to transcribe. Use manual ranges with captions off.")
    words, segments = [], []
    task = "translate" if opts["translate"] else "transcribe"
    if need_text and manual:
        update(stage="Transcribing speech", log=f"Transcribing only your {len(manual)} chosen range(s) ({opts['whisper_model']} model)")
        for i, m in enumerate(manual):
            words = ensure_words(job_dir, src, opts, dur, m["start"], m["end"], cancelled,
                                 lambda f, i=i: update(progress=0.14 + 0.31 * (i + f) / len(manual)))
        if opts["speakers"]:
            log("Speaker labels need full-video mode; skipped for manual ranges")
    elif need_text:
        wav = job_dir / "audio.wav"
        update(stage="Extracting audio", progress=0.10, log="Extracting the audio track")
        extract_audio(src, opts["audio_track"], wav, total=dur, on_progress=lambda f: update(progress=0.10 + 0.04 * f),
                      cancelled=cancelled)
        update(stage="Transcribing speech", log=f"Whisper model: {opts['whisper_model']}"
               + (" (translating to English)" if task == "translate" else "") + " (first run downloads it)")
        tr = transcribe(wav, opts["whisper_model"], opts.get("language"), task, 0.0,
                        lambda f: update(progress=0.14 + 0.31 * f), cancelled)
        tr["coverage"] = [[0.0, dur]]
        if opts["speakers"]:
            update(stage="Labelling speakers", progress=0.45, log="Labelling speakers (experimental)")
            try:
                ok = label_speakers(str(wav), tr["segments"], opts["speaker_count"])
                log("Speaker labels added" if ok else "Not enough speech to tell speakers apart")
            except Exception as e:
                log(f"Speaker labelling skipped ({type(e).__name__}: {str(e)[:100]})")
        _save_tr(job_dir, tr)
        wav.unlink(missing_ok=True)
        segments, words = tr["segments"], _flat(tr)
        update(language=tr["language"], log=f"Transcribed {len(segments)} segments ({tr['language']})")
    check()

    # 3 -- choose clips -----------------------------------------------------
    if manual:
        picks = manual
        update(log=f"Using {len(picks)} manual clip range(s)")
        for p in picks:
            p["text"] = " ".join(w["word"] for w in words_in_range(words, p["start"], p["end"]))
        if ai and any(p["text"] for p in picks):
            update(stage="Writing titles with AI", progress=0.5)
            mine = {i: p["title"] for i, p in enumerate(picks) if not re.fullmatch(r"Clip \d+", p["title"])}
            annotate(picks, ai[0], ai[1], log, rerank=False)
            for i, t in mine.items():
                picks[i]["title"] = t
    else:
        motion = None
        if opts["motion"]:
            update(stage="Analysing motion", progress=0.46, log="Measuring on-screen motion (decodes the video once)")
            motion = load_motion(src, dur, lambda f: update(progress=0.46 + 0.04 * f), cancelled)
        update(stage="Finding highlights", progress=0.5)
        picks = find_highlights(segments, src, dur, opts["count"], opts["min_len"], opts["max_len"], ai,
                                opts["audio_track"], motion, opts["dedupe"], log, lambda s: update(stage=s))
        update(log=f"Selected {len(picks)} highlight(s)")
    if not picks:
        raise RuntimeError("No clips could be found. Try a longer video or use manual ranges.")
    for p in picks:
        p.setdefault("summary", "")
        if not p.get("hashtags"):
            p["hashtags"] = keyword_hashtags(p.get("text", ""))
    update(ranges=[[round(p["start"], 2), round(p["end"], 2)] for p in picks])

    # 4 -- render -----------------------------------------------------------
    clips = []
    base, share = 0.55, 0.45 / len(picks)
    for n, p in enumerate(picks, 1):
        check()
        update(stage=f"Rendering clip {n} of {len(picks)}", progress=base + share * (n - 1),
               log=f"Clip {n}: {p['start']:.1f}s → {p['end']:.1f}s  “{p['title']}”")
        try:
            clip = render_one(job_dir, src, src_w, src_h, opts, words, p, n,
                              lambda f, n=n: update(progress=base + share * (n - 1 + f)), cancelled, log)
        except Cancelled:
            raise
        except Exception as e:  # one bad clip must not throw away the others
            update(log=f"Clip {n} failed and was skipped: {str(e)[:400]}")
            continue
        clips.append(clip)
        update(clips=list(clips))

    if not clips:
        raise RuntimeError("Every clip failed to render. See the activity log for the ffmpeg error.")
    (job_dir / "clips.json").write_text(json.dumps(clips, indent=1))
    write_csv(clips, clips_dir / "clips.csv")
    update(status="done", stage="Done", progress=1.0, log=f"Finished — {len(clips)} clip(s) ready")


# ------------------------------------------------------------------ re-render one clip (trim / style changes)
def rerender(job_dir, clip, changes, log, cancelled):
    job_dir = Path(job_dir)
    op = job_dir / "opts.json"
    if not op.exists():
        raise RuntimeError("This job was made with an older version, so re-rendering is not available for it.")
    opts = json.loads(op.read_text())
    src = Path(opts["source"])
    if not src.exists():
        raise RuntimeError(f"The source video is no longer at {src}.")
    for k in ("aspect", "layout"):
        if changes.get(k):
            opts[k] = changes[k]
    if opts["aspect"] not in ASPECTS:
        raise RuntimeError("Unknown aspect ratio.")
    opts["layout"] = "blur" if opts["layout"] == "blur" else "smart"
    for k in ("captions", "hook", "progress_bar", "export_subs"):
        if k in changes:
            opts[k] = bool(changes[k])
    if "style" in changes:
        opts["style"] = normalize_style(changes["style"])
    src_w, src_h, dur = probe(src)
    start = max(0.0, float(changes.get("start", clip["start"])))
    end = min(dur, float(changes.get("end", clip["end"])))
    if end - start < 1:
        raise RuntimeError("The clip must be at least 1 second long.")
    title = (changes.get("title") or clip["title"]).strip()[:80] or clip["title"]
    words = []
    if opts["captions"] or opts["export_subs"]:
        log("Transcribing any new part of the range")
        words = ensure_words(job_dir, src, opts, dur, start, end, cancelled)
    pick = {"start": start, "end": end, "title": title, "score": clip.get("score", 0), "text": ""}
    new = render_one(job_dir, src, src_w, src_h, opts, words, pick, clip["index"], lambda f: None, cancelled, log,
                     name=Path(clip["file"]).stem)
    new.update(summary=clip.get("summary", ""), hashtags=clip.get("hashtags", []), rev=clip.get("rev", 1) + 1)
    return new
