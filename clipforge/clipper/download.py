from __future__ import annotations
from pathlib import Path

from . import Cancelled


def download(url: str, out_dir: Path, on_progress=None, cancelled=lambda: False):
    """Download a video (YouTube, X, TikTok, Twitch VODs, ... anything yt-dlp supports)."""
    import yt_dlp

    def hook(d):
        if cancelled():
            raise Cancelled()
        if d.get("status") == "downloading" and on_progress:
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                on_progress(d.get("downloaded_bytes", 0) / total)

    opts = {
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    files = [f for f in sorted(out_dir.glob("source.*")) if f.suffix not in (".part", ".ytdl")]
    if not files:
        raise RuntimeError("Download finished but no video file was found.")
    return files[0], info.get("title") or "video"
