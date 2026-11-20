"""ClipForge — local AI video clipper.  Run:  python app.py   then open http://127.0.0.1:5000"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
import zipfile
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory

from clipper import Cancelled
from clipper.ai import ollama_models
from clipper.captions import STYLES, normalize_style
from clipper.control import Control
from clipper.export import PLATFORMS, make_cover, make_gif, make_mp3, write_csv
from clipper.pipeline import rerender, run_pipeline
from clipper.render import ASPECTS, has_nvenc, list_audio_tracks, probe, tools_ok

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "workspace"
WORK.mkdir(exist_ok=True)
PRESETS = ROOT / "presets.json"

app = Flask(__name__)
JOBS: dict[str, dict] = {}
CTL: dict[str, Control] = {}
LOCK = threading.RLock()
SLOT = threading.Semaphore(1)          # one heavy job at a time; others wait in the queue
ID_RE = re.compile(r"^[a-f0-9]{10}$")

# progress value at which each measurable step ends (used for the time-remaining estimate)
STEP_END = {"Downloading video": 0.10, "Extracting audio": 0.14, "Transcribing speech": 0.45,
            "Analysing motion": 0.50, "render": 1.0}


# ------------------------------------------------------------------ job store
def _save(job):
    with LOCK:
        (WORK / job["id"] / "job.json").write_text(json.dumps(job, indent=1))


def _load_existing():
    for f in WORK.glob("*/job.json"):
        try:
            job = json.loads(f.read_text())
            if job.get("status") in ("queued", "running", "paused"):
                job.update(status="error", error="Interrupted (the app was restarted).")
            for c in job.get("clips", []):
                c["busy"] = False
            JOBS[job["id"]] = job
        except Exception:
            pass


def _updater(job):
    last = [0.0]

    def update(stage=None, progress=None, log=None, **kw):
        with LOCK:
            now = time.time()
            if stage:
                job["stage"] = stage
            if progress is not None:
                job["progress"] = round(progress, 3)
            if log:
                job["log"] = (job["log"] + [log])[-200:]
            job.update(kw)
            if kw.get("status") == "running":
                job["started"] = now
            if kw.get("status") in ("done", "error", "cancelled"):
                job["finished"] = now
                if job.get("paused_at"):           # cancelled while paused: count the last pause too
                    job["paused_total"] += now - job["paused_at"]
                    job["paused_at"] = None

            # time remaining: extrapolate from how fast the current step is moving (pauses excluded)
            step = job.get("stage", "")
            step = "render" if step.startswith("Rendering") else step
            if job.get("_step") != step:
                job.update(_step=step, _step_t=now, _step_p=job["progress"], _step_pt=job["paused_total"], eta=None)
            end = STEP_END.get(step)
            if job.get("status") == "running" and end and end > job["_step_p"]:
                f = (job["progress"] - job["_step_p"]) / (end - job["_step_p"])
                elapsed = now - job["_step_t"] - (job["paused_total"] - job["_step_pt"])
                if f > 0.02 and elapsed > 4:
                    raw = elapsed * (1 - f) / f
                    prev = job.get("eta")
                    job["eta"] = round(raw if prev is None else 0.7 * prev + 0.3 * raw)
                    job["eta_scope"] = "total" if step == "render" else "step"
            elif job.get("status") not in ("running", "paused"):
                job["eta"] = None

            if log or stage or kw or now - last[0] > 1.5:
                _save(job)
                last[0] = now
    return update


def _worker(job, opts):
    update = _updater(job)
    try:
        with SLOT:
            ctl = CTL[job["id"]]
            if ctl.cancelled:
                raise Cancelled()
            update(status="running", stage="Starting", progress=0.0)
            run_pipeline(WORK / job["id"], opts, update, ctl)
    except Cancelled:
        update(status="cancelled", stage="Cancelled")
    except Exception as e:
        traceback.print_exc()
        update(status="error", stage="Failed", error=str(e), log=f"Error: {e}")


def _job_or_404(jid):
    if not ID_RE.match(jid) or jid not in JOBS:
        abort(404)
    return JOBS[jid]


def _clip_or_404(job, n):
    for c in job["clips"]:
        if c["index"] == n:
            return c
    abort(404)


def _set_clip(job, n, **kw):
    with LOCK:
        for c in job["clips"]:
            if c["index"] == n:
                c.update(kw)
        _save(job)


# ------------------------------------------------------------------ pages and settings
@app.get("/")
def index():
    return render_template("index.html", platforms=PLATFORMS)


@app.get("/api/health")
def health():
    return jsonify(ffmpeg=tools_ok(), nvenc=tools_ok() and has_nvenc(),
                   claude=bool(os.environ.get("ANTHROPIC_API_KEY")), ollama=ollama_models())


@app.get("/api/probe")
def probe_path():
    p = Path(request.args.get("path", "").strip().strip("\"'")).expanduser()
    return jsonify(tracks=list_audio_tracks(p) if p.is_file() else [])


def _user_styles():
    try:
        return json.loads(PRESETS.read_text())
    except Exception:
        return {}


@app.get("/api/styles")
def styles():
    rows = [{"id": k, "label": v["label"], "params": v, "builtin": True} for k, v in STYLES.items()]
    rows += [{"id": k, "label": v["label"], "params": v, "builtin": False} for k, v in _user_styles().items()]
    return jsonify(rows)


@app.post("/api/styles")
def save_style():
    body = request.get_json(silent=True) or {}
    params = normalize_style(dict(body.get("params") or {}, label=body.get("label") or "Custom"))
    sid = "u_" + (re.sub(r"[^a-z0-9]+", "-", params["label"].lower()).strip("-") or "style")
    with LOCK:
        data = _user_styles()
        data[sid] = params
        PRESETS.write_text(json.dumps(data, indent=1))
    return jsonify(id=sid)


@app.delete("/api/styles/<sid>")
def delete_style(sid):
    with LOCK:
        data = _user_styles()
        if sid not in data:
            abort(404)
        data.pop(sid)
        PRESETS.write_text(json.dumps(data, indent=1))
    return jsonify(ok=True)


# ------------------------------------------------------------------ jobs
def _int(form, key, default, lo, hi):
    try:
        return max(lo, min(hi, int(float(form.get(key, default)))))
    except (TypeError, ValueError):
        return default


@app.post("/api/jobs")
def create_job():
    if not tools_ok():
        return jsonify(error="ffmpeg/ffprobe not found. Install ffmpeg and restart (see README)."), 400
    f = request.form
    jid = uuid.uuid4().hex[:10]
    d = WORK / jid
    d.mkdir()
    try:
        style = normalize_style(json.loads(f.get("style_json") or "null") or STYLES["hormozi"])
    except ValueError:
        style = normalize_style(STYLES["hormozi"])
    on = lambda k: f.get(k) == "on"  # noqa: E731
    opts = {
        "aspect": f.get("aspect") if f.get("aspect") in ASPECTS else "9:16",
        "layout": "blur" if f.get("layout") == "blur" else "smart",
        "style": style,
        "captions": on("captions"), "hook": on("hook"), "progress_bar": on("progress_bar"), "gpu": on("gpu"),
        "count": _int(f, "count", 6, 1, 20),
        "min_len": _int(f, "min_len", 20, 5, 170),
        "max_len": _int(f, "max_len", 60, 10, 180),
        "audio_track": _int(f, "audio_track", 0, 0, 31),
        "dedupe": on("dedupe"), "motion": on("motion"), "translate": on("translate"),
        "ai": f.get("ai") if f.get("ai") in ("off", "claude", "ollama") else "off",
        "ai_model": (f.get("ai_model") or "").strip()[:60],
        "speakers": on("speakers"), "speaker_count": _int(f, "speaker_count", 2, 2, 5),
        "export_subs": on("export_subs"), "target_mb": _int(f, "target_mb", 0, 0, 2000),
        "whisper_model": f.get("whisper_model") if f.get("whisper_model") in
                         ("tiny", "base", "small", "medium", "large-v3") else "small",
        "language": (f.get("language") or "").strip()[:8] or None,
        "manual": f.get("manual", ""),
    }
    if opts["max_len"] <= opts["min_len"]:
        opts["max_len"] = opts["min_len"] + 10

    upload = request.files.get("file")
    raw = (f.get("source") or f.get("path") or f.get("url") or "").strip().strip("\"'").strip()
    if upload and upload.filename:
        dest = d / f"source{Path(upload.filename).suffix.lower() or '.mp4'}"
        upload.save(dest)
        opts["source"], title = str(dest), Path(upload.filename).stem
    elif raw and re.match(r"^(https?://|www\.)", raw, re.I):          # a link
        url = raw if raw.lower().startswith("http") else "https://" + raw
        opts["url"], title = url, url
    elif raw:                                                          # a path on this computer
        p = Path(raw).expanduser()
        if not p.is_file():
            shutil.rmtree(d, ignore_errors=True)
            msg = "That is a folder, not a video file." if p.is_dir() else (
                f"File not found: {p}. (Links must start with http:// or https://)")
            return jsonify(error=msg), 400
        opts["source"], title = str(p), p.stem
    else:
        shutil.rmtree(d, ignore_errors=True)
        return jsonify(error="Add a video: upload a file, paste a link, or give a file path."), 400

    job = {"id": jid, "title": title, "status": "queued", "stage": "Queued", "progress": 0.0,
           "created": time.time(), "duration": 0, "paused_total": 0.0, "paused_at": None,
           "ranges": [], "clips": [], "log": [], "error": None,
           "opts": {k: v for k, v in opts.items() if k not in ("source", "manual")}}
    CTL[jid] = Control()
    with LOCK:
        JOBS[jid] = job
        _save(job)
    threading.Thread(target=_worker, args=(job, opts), daemon=True).start()
    return jsonify(id=jid)


@app.get("/api/jobs")
def list_jobs():
    rows = sorted(JOBS.values(), key=lambda j: j["created"], reverse=True)
    return jsonify([{k: j[k] for k in ("id", "title", "status", "progress", "created")} for j in rows])


@app.get("/api/jobs/<jid>")
def get_job(jid):
    with LOCK:
        return jsonify(_job_or_404(jid))


@app.post("/api/jobs/<jid>/cancel")
def cancel_job(jid):
    _job_or_404(jid)
    if jid in CTL:
        CTL[jid].cancel()
    return jsonify(ok=True)


@app.post("/api/jobs/<jid>/pause")
def pause_job(jid):
    job = _job_or_404(jid)
    with LOCK:
        if job["status"] != "running" or jid not in CTL:
            return jsonify(error="Only a running job can be paused."), 409
        CTL[jid].pause()
        job.update(status="paused", paused_at=time.time())
        _save(job)
    return jsonify(ok=True)


@app.post("/api/jobs/<jid>/resume")
def resume_job(jid):
    job = _job_or_404(jid)
    with LOCK:
        if job["status"] != "paused" or jid not in CTL:
            return jsonify(error="This job is not paused."), 409
        job["paused_total"] += time.time() - (job.get("paused_at") or time.time())
        job.update(status="running", paused_at=None)
        CTL[jid].resume()
        _save(job)
    return jsonify(ok=True)


@app.delete("/api/jobs/<jid>")
def delete_job(jid):
    _job_or_404(jid)
    if jid in CTL:
        CTL[jid].cancel()
    with LOCK:
        JOBS.pop(jid, None)
    shutil.rmtree(WORK / jid, ignore_errors=True)
    return jsonify(ok=True)


@app.post("/api/jobs/<jid>/open")
def open_folder(jid):
    _job_or_404(jid)
    path = str(WORK / jid / "clips")
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])
    return jsonify(ok=True)


# ------------------------------------------------------------------ per-clip actions
def _rerender_worker(job, n, changes):
    with SLOT:
        try:
            clip = dict(next(c for c in job["clips"] if c["index"] == n))
            new = rerender(WORK / job["id"], clip, changes, lambda m: None, Control())
            new.update(busy=False, error=None)
            with LOCK:
                job["clips"] = [new if c["index"] == n else c for c in job["clips"]]
                _save(job)
                (WORK / job["id"] / "clips.json").write_text(json.dumps(job["clips"], indent=1))
                write_csv(job["clips"], WORK / job["id"] / "clips" / "clips.csv")
        except Exception as e:
            traceback.print_exc()
            _set_clip(job, n, busy=False, error=str(e)[:300])


@app.post("/api/jobs/<jid>/clips/<int:n>/rerender")
def rerender_clip(jid, n):
    job = _job_or_404(jid)
    clip = _clip_or_404(job, n)
    if job["status"] in ("queued", "running", "paused"):
        return jsonify(error="Wait for the job to finish before editing a clip."), 409
    if clip.get("busy"):
        return jsonify(error="This clip is already being re-rendered."), 409
    _set_clip(job, n, busy=True, error=None)
    threading.Thread(target=_rerender_worker, args=(job, n, request.get_json(silent=True) or {}),
                     daemon=True).start()
    return jsonify(ok=True)


@app.post("/api/jobs/<jid>/clips/<int:n>/export")
def export_clip(jid, n):
    job = _job_or_404(jid)
    clip = _clip_or_404(job, n)
    fmt = (request.get_json(silent=True) or {}).get("fmt")
    cd = WORK / jid / "clips"
    src = cd / clip["file"]
    try:
        if fmt == "gif":
            out = cd / f"{src.stem}.gif"
            make_gif(src, out, min(clip["duration"], 12))
        elif fmt == "mp3":
            out = cd / f"{src.stem}.mp3"
            make_mp3(src, out)
        else:
            return jsonify(error="Unknown format."), 400
    except Exception as e:
        return jsonify(error=str(e)), 500
    return jsonify(url=f"/media/{jid}/{out.name}?dl=1")


@app.post("/api/jobs/<jid>/clips/<int:n>/cover")
def cover_clip(jid, n):
    job = _job_or_404(jid)
    clip = _clip_or_404(job, n)
    b = request.get_json(silent=True) or {}
    cd = WORK / jid / "clips"
    src = cd / clip["file"]
    W, H, _ = probe(src)
    out = cd / f"{src.stem}_cover.jpg"
    try:
        make_cover(src, out, float(b.get("t", 1)), str(b.get("text", ""))[:120], W, H,
                   str(b.get("font") or "Arial")[:40], b.get("position") if b.get("position") in
                   ("top", "middle", "bottom") else "middle")
    except Exception as e:
        return jsonify(error=str(e)), 500
    return jsonify(url=f"/media/{jid}/{out.name}?v={int(time.time())}")


@app.get("/api/jobs/<jid>/zip")
def zip_job(jid):
    job = _job_or_404(jid)
    cd = WORK / jid / "clips"
    keep = {".mp4", ".srt", ".vtt", ".txt", ".csv", ".gif", ".mp3"}
    out = WORK / jid / "all_clips.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(cd.iterdir()):
            if f.suffix in keep or f.name.endswith("_cover.jpg"):
                z.write(f, f.name)
    slug = re.sub(r"[^a-z0-9]+", "-", job["title"].lower()).strip("-")[:40] or "clips"
    return send_file(out, as_attachment=True, download_name=f"clipforge-{slug}.zip")


@app.get("/media/<jid>/<path:name>")
def media(jid, name):
    _job_or_404(jid)
    return send_from_directory(WORK / jid / "clips", name, as_attachment=request.args.get("dl") == "1")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not tools_ok():
        print("\n!! ffmpeg / ffprobe not found on PATH. Install ffmpeg first (see README.md).\n")
    _load_existing()
    url = f"http://127.0.0.1:{a.port}"
    print(f"ClipForge running at {url}")
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=a.port, threaded=True, debug=False)
