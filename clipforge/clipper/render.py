"""ffmpeg rendering: face-aware vertical reframing, blur-fit layout, burned-in captions, loudness normalisation."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from . import Cancelled

ASPECTS = {"9:16": (1080, 1920), "1:1": (1080, 1080), "4:5": (1080, 1350), "16:9": (1920, 1080)}


def tools_ok() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def has_nvenc() -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        return "h264_nvenc" in out
    except Exception:
        return False


def probe(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration", "-of", "json", str(path)])
    j = json.loads(out)
    s = j["streams"][0]
    return int(s["width"]), int(s["height"]), float(j["format"]["duration"])


def list_audio_tracks(path):
    """[{'index': 0, 'label': '1: Hindi (aac, 6 ch)'}, ...] in the order ffmpeg's 0:a:N selects them."""
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
            "stream=codec_name,channels:stream_tags=language,title", "-of", "json", str(path)])
        tracks = []
        for i, st in enumerate(json.loads(out).get("streams", [])):
            tags = st.get("tags", {})
            lang = tags.get("language") or tags.get("title") or "unknown language"
            tracks.append({"index": i, "label": f"{i + 1}: {lang} ({st.get('codec_name', '?')}, {st.get('channels', '?')} ch)"})
        return tracks
    except Exception:
        return []


def extract_audio(src, track, out, start=None, dur=None, total=None, on_progress=None, cancelled=lambda: False):
    """Pull one audio track out as 16 kHz mono WAV (what Whisper wants), optionally only a time range."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-progress", "pipe:1", "-nostats"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-map", f"0:a:{track}", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)]
    rc, tail = _run(cmd, dur or total or 1, on_progress, cancelled)
    if rc != 0:
        raise RuntimeError(f"Could not read audio track {track + 1} (does the video have sound?): {tail}")
    return out


# --------------------------------------------------------------------------- face tracking
def track_faces(src, start, end, step=0.5, cancelled=lambda: False):
    """Sample frames and return [(t_rel, face_center_x | None)] using OpenCV's Haar face detector."""
    import cv2
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(str(src))
    pts, t = [], start
    while t <= end:
        if cancelled():
            cap.release()
            raise Cancelled()
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        scale = 480 / w if w > 480 else 1.0
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale != 1.0 else frame
        faces = cascade.detectMultiScale(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 1.2, 5, minSize=(24, 24))
        if len(faces):
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            pts.append((t - start, (x + fw / 2) / scale))
        else:
            pts.append((t - start, None))
        t += step
    cap.release()
    return pts


def smooth_path(pts, default, alpha=0.3):
    """Fill gaps, median-filter spikes, then exponential-smooth so the virtual camera glides."""
    import numpy as np
    xs = [p[1] for p in pts]
    if not pts or all(x is None for x in xs):
        return [(0.0, default)]
    first = next(x for x in xs if x is not None)
    filled, last = [], first
    for x in xs:
        last = x if x is not None else last
        filled.append(last)
    arr = np.array(filled, dtype=float)
    padded = np.pad(arr, 1, mode="edge")
    med = np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0)
    out, cur = [], med[0]
    for v in med:
        cur = alpha * v + (1 - alpha) * cur
        out.append(cur)
    return [(p[0], float(v)) for p, v in zip(pts, out)]


def _lerp_expr(points):
    """Piecewise-linear ffmpeg expression in t, built as a balanced tree.

    ffmpeg's expression parser rejects deeply nested if(); a balanced tree keeps depth ~log2(n)
    however long the clip is. Near-identical keypoints are dropped to keep the expression short."""
    if len(points) == 1:
        return f"{points[0][1]:.1f}"
    kept = [points[0]]
    for p in points[1:-1]:
        if abs(p[1] - kept[-1][1]) > 3:
            kept.append(p)
    kept.append((max(points[-1][0], kept[-1][0] + 0.01), points[-1][1]))
    kept.append((kept[-1][0] + 1000.0, kept[-1][1]))      # hold the last position, never extrapolate
    segs = [(t0, v0, t1, v1) for (t0, v0), (t1, v1) in zip(kept[:-1], kept[1:]) if t1 > t0]

    def build(lo, hi):
        if hi - lo == 1:
            t0, v0, t1, v1 = segs[lo]
            return f"{v0:.1f}+({v1 - v0:.2f})*(t-{t0:.2f})/{t1 - t0:.2f}"
        mid = (lo + hi) // 2
        return f"if(lt(t,{segs[mid][0]:.2f}),{build(lo, mid)},{build(mid, hi)})"

    return build(0, len(segs))


# --------------------------------------------------------------------------- filter graph
def _ff_path(p) -> str:
    return str(p).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def _ass_filter(ass_path, fonts_dir=None):
    """ffmpeg runs with cwd = the folder holding the .ass file, so only plain relative names go into the filter
    (drive letters, backslashes and colons in Windows paths break filter parsing)."""
    fd = ""
    if fonts_dir and Path(fonts_dir).is_dir() and any(Path(fonts_dir).glob("*.[to]tf")):
        try:
            rel = os.path.relpath(fonts_dir, Path(ass_path).parent).replace("\\", "/")
            fd = f":fontsdir='{rel}'"
        except ValueError:
            pass
    return f"ass='{Path(ass_path).name}'{fd}"


def build_graph(src_w, src_h, W, H, layout, track, ass_path, fonts_dir):
    tgt, src = W / H, src_w / src_h
    if abs(src - tgt) < 0.02:
        body = f"scale={W}:{H}:flags=lanczos,setsar=1"
    elif layout == "blur":
        body = (f"split[a][b];"
                f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=40:6,eq=brightness=-0.08[bg];"
                f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1")
    else:  # smart crop
        if src > tgt:
            ch, cw = src_h // 2 * 2, int(src_h * tgt) // 2 * 2
            pts = [(t, min(max(cx - cw / 2, 0), src_w - cw)) for t, cx in track] if track else [(0, (src_w - cw) / 2)]
            x_expr, y_expr = f"clip({_lerp_expr(pts)},0,{src_w - cw})", "0"
        else:
            cw, ch = src_w // 2 * 2, int(src_w / tgt) // 2 * 2
            x_expr, y_expr = "0", f"{(src_h - ch) / 2:.0f}"
        body = f"crop={cw}:{ch}:x='{x_expr}':y='{y_expr}',scale={W}:{H}:flags=lanczos,setsar=1"
    if ass_path:
        body += "," + _ass_filter(ass_path, fonts_dir)
    return f"[0:v]{body}[v]"


def _run(cmd, dur, on_progress, cancelled, cwd=None):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         cwd=cwd, encoding="utf-8", errors="replace")
    tail = []
    if hasattr(cancelled, "attach"):
        cancelled.attach(p)          # lets pause/resume freeze this exact ffmpeg process
    try:
        for line in p.stdout:
            if cancelled():
                p.kill()
                raise Cancelled()
            line = line.strip()
            if line.startswith(("out_time_us=", "out_time_ms=")):
                try:
                    if on_progress:
                        on_progress(min(int(line.split("=")[1]) / 1e6 / max(dur, 0.1), 1.0))
                except ValueError:
                    pass
            elif "=" not in line or line.startswith("["):
                tail.append(line)
        p.wait()
        if cancelled():                   # process was killed by a cancel, not an ffmpeg failure
            raise Cancelled()
    finally:
        if hasattr(cancelled, "detach"):
            cancelled.detach()
    return p.returncode, " | ".join(tail[:10])


def render_clip(src, out, start, dur, W, H, layout, track, ass_path, fonts_dir, gpu,
                src_w, src_h, on_progress=None, cancelled=lambda: False, audio_track=0, target_mb=0):
    graph = build_graph(src_w, src_h, W, H, layout, track, ass_path, fonts_dir)
    af = f"afade=t=in:d=0.05,afade=t=out:st={max(dur - 0.12, 0):.2f}:d=0.12,loudnorm=I=-16:TP=-1.5:LRA=11"
    rate = None
    if target_mb:                          # aim for a file size: derive the video bitrate from the clip length
        vk = max(200, int(target_mb * 8192 * 0.93 / max(dur, 1.0) - 128))
        rate = ["-b:v", f"{vk}k", "-maxrate", f"{int(vk * 1.15)}k", "-bufsize", f"{vk * 2}k"]
    tail = ""
    for use_gpu in ([True, False] if gpu else [False]):
        if use_gpu:
            venc = ["-c:v", "h264_nvenc", "-preset", "p5", *(rate or ["-cq", "21", "-b:v", "0"])]
        else:
            venc = ["-c:v", "libx264", "-preset", "veryfast", *(rate or ["-crf", "20"])]
        cmd = ["ffmpeg", "-y", "-v", "error", "-progress", "pipe:1", "-nostats",
               "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
               "-filter_complex", graph, "-map", "[v]", "-map", f"0:a:{audio_track}?", "-af", af,
               *venc, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k" if target_mb else "160k",
               "-movflags", "+faststart", str(out)]
        rc, tail = _run(cmd, dur, on_progress, cancelled, cwd=str(Path(out).parent))
        if rc == 0:
            return
    raise RuntimeError(f"ffmpeg failed: {tail[:700]}")


def make_thumb(video, out):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "1", "-i", str(video), "-frames:v", "1",
                    "-vf", "scale=360:-2", str(out)], capture_output=True)
