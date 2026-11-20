"""Export helpers: subtitle files, clip list, GIF / MP3, cover images and platform presets."""
from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path

from .captions import build_cover_ass

# Common defaults, editable in the UI. Platform limits change over time, so check them before posting.
PLATFORMS = {
    "tiktok":   {"label": "TikTok",            "aspect": "9:16", "min_len": 20, "max_len": 60,  "target_mb": 0},
    "reels":    {"label": "Instagram Reels",   "aspect": "9:16", "min_len": 20, "max_len": 90,  "target_mb": 0},
    "shorts":   {"label": "YouTube Shorts",    "aspect": "9:16", "min_len": 20, "max_len": 60,  "target_mb": 0},
    "x":        {"label": "X (Twitter)",       "aspect": "16:9", "min_len": 15, "max_len": 120, "target_mb": 0},
    "discord":  {"label": "Discord (25 MB)",   "aspect": "16:9", "min_len": 10, "max_len": 60,  "target_mb": 24},
    "whatsapp": {"label": "WhatsApp (16 MB)",  "aspect": "9:16", "min_len": 10, "max_len": 60,  "target_mb": 15},
}


def _lines(words, max_chars=38, max_dur=4.0):
    lines, cur = [], []
    for w in words:
        if cur and (sum(len(x["word"]) + 1 for x in cur) + len(w["word"]) > max_chars
                    or w["end"] - cur[0]["start"] > max_dur or re.search(r"[.!?]$", cur[-1]["word"])):
            lines.append(cur)
            cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    return lines


def _t(t, sep):
    ms = int(round(max(t, 0) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _speaker_text(lines):
    """Join words per line, prefixing 'Speaker N:' whenever the speaker changes (only if words carry 'spk')."""
    last = None
    for l in lines:
        txt = " ".join(x["word"] for x in l)
        spk = l[0].get("spk")
        if spk is not None and spk != last:
            txt, last = f"Speaker {spk + 1}: {txt}", spk
        yield txt


def _cues(words, sep):
    lines = _lines(words)
    for i, (l, txt) in enumerate(zip(lines, _speaker_text(lines)), 1):
        yield i, f"{_t(l[0]['start'], sep)} --> {_t(max(l[-1]['end'], l[0]['start'] + 0.3), sep)}", txt


def write_srt(words, path):
    Path(path).write_text("\n".join(f"{i}\n{ts}\n{txt}\n" for i, ts, txt in _cues(words, ",")), encoding="utf-8")


def write_vtt(words, path):
    body = "\n".join(f"{ts}\n{txt}\n" for _, ts, txt in _cues(words, "."))
    Path(path).write_text("WEBVTT\n\n" + body, encoding="utf-8")


def write_txt(words, path):
    Path(path).write_text("\n".join(_speaker_text(_lines(words, 80, 12))) + "\n", encoding="utf-8")


def write_csv(clips, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:      # utf-8-sig so Excel shows accents correctly
        w = csv.writer(f)
        w.writerow(["rank", "title", "start_s", "end_s", "duration_s", "score", "summary", "hashtags", "file"])
        for c in clips:
            w.writerow([c["index"], c["title"], c["start"], c["end"], c["duration"], c["score"],
                        c.get("summary", ""), " ".join(c.get("hashtags", [])), c["file"]])


def _ff(cmd, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or "ffmpeg failed").strip()[-500:])


def make_gif(clip, out, seconds=12.0, start=0.0, width=480, fps=12):
    _ff(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{seconds:.2f}", "-i", str(clip),
         "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse",
         "-loop", "0", str(out)])


def make_mp3(clip, out):
    _ff(["ffmpeg", "-y", "-v", "error", "-i", str(clip), "-vn", "-c:a", "libmp3lame", "-q:a", "2", str(out)])


def make_cover(clip, out, t, text, W, H, font="Arial", position="middle"):
    """Grab one frame of a rendered clip and put big text on it (text drawn through libass)."""
    out = Path(out)
    cwd = out.parent
    vf = []
    if text.strip():
        ass = cwd / (out.stem + ".ass")
        ass.write_text(build_cover_ass(text.strip(), W, H, font, position), encoding="utf-8")
        vf = ["-vf", f"ass='{ass.name}'"]
    _ff(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(t, 0):.2f}", "-i", str(clip), "-frames:v", "1",
         *vf, "-q:v", "2", str(out)], cwd=str(cwd))
    (cwd / (out.stem + ".ass")).unlink(missing_ok=True)
