# ClipForge

A local video clipper. Give it a long video (file, YouTube/X/TikTok link, or a path on your disk) and it:

1. transcribes the speech with word-level timing (faster-whisper, offline)
2. scores every possible 20–60 s window for hook strength, emotion, pace, audio energy and completeness, then picks the best non-overlapping moments
3. reframes them to 9:16, 4:5, 1:1 or 16:9, either following the speaker's face or on a blurred background
4. burns in animated word-by-word captions (Hormozi, Neon, Boxed, Minimal), a title hook for the first 3 seconds and a progress bar
5. normalises loudness and exports ready-to-post MP4s with thumbnails

Everything runs on your machine. The only optional network feature is Claude re-ranking.

## Setup

You need Python 3.10+ and ffmpeg.

**Install ffmpeg**
- Windows: `winget install Gyan.FFmpeg`
- macOS: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

**Install and run**
```bash
cd clipforge
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
The browser opens at http://127.0.0.1:5000. Options: `--port 8080`, `--no-browser`.

The first run downloads the Whisper model you pick (small is about 460 MB). After that it works offline.

## Using it

**Quick start on Windows:** double-click `ClipForge.bat`. It checks for Python and ffmpeg, installs everything on the first run, and starts the app. On macOS or Linux run `./start.sh`.

- **One video box**: drop a file, paste a link (YouTube and anything yt-dlp supports) or paste a file path. A path is read in place, which is the fastest choice for big movies.
- **Auto mode**: pick a format and caption style, then press *Forge clips*.
- **Manual mode**: list ranges, one per line, for example `1:20-2:05 Why I quit`. Auto-detect is skipped, and only those ranges are transcribed, so it is fast.
- **Pause, resume, cancel** and a live time-remaining estimate work during any step.

### Smart picking and AI
- **Audio track**: for dual-audio films, pick the language to transcribe and to use in the clips. Paste a file path to see the track names.
- **Skip duplicates**: drops moments that repeat lines already used and moments right next to a chosen clip. If few distinct moments exist you get fewer clips than asked, and the log says so.
- **Action scoring** (slower): measures on-screen motion and also considers windows with no dialogue, so fights and chases can be picked. It decodes the whole video once, so expect extra minutes on a film.
- **English captions**: Whisper translates the speech to English, so captions and highlight scoring work on English text. Other target languages are not included.
- **Speaker labels** (experimental): groups voices by tone and pitch with no AI model. Works best for 2 to 3 clearly different voices. Labels appear in the .srt, .vtt and .txt files. Full-video mode only.
- **AI helper**: writes titles, one-line summaries and hashtags and re-ranks the shortlist.
  - *Local model (Ollama)*: fully offline. Install Ollama from ollama.com, then run `ollama pull llama3.2` once. The model list appears in the app. Small models give rougher results than Claude.
  - *Claude (online)*: set `ANTHROPIC_API_KEY` before starting. Only transcript text is sent.
  - With no helper, hashtags still come from the most frequent keywords.

### Captions
Pick a built-in style or press *Design your own*: font, size, colours, outline, position, words per line, box. Save it as a preset (stored in `presets.json`) to reuse it. Put extra `.ttf` or `.otf` files in `fonts/` to use them.

### Editing a finished clip
Press *Edit* on a clip: trim with the start and end handles (plus 0.5 s nudges), change the title, format, layout or caption style, then *Re-render*. Only that clip is redone, and only any newly covered speech is transcribed. The *Cover* tab picks a frame and adds big text for a thumbnail.

### Exports
- **Platform presets** set the format and length limits for TikTok, Reels, Shorts, X, Discord and WhatsApp. They are common defaults, so check each platform's current limits.
- **Target file size** (MB) sets the video bitrate to fit. Very long clips with a tiny target will look rough.
- **.srt, .vtt, .txt** per clip, `clips.csv` for every job (titles, times, scores, summaries, hashtags).
- **GIF** (first 12 s), **MP3** and cover images per clip, and **Download all (.zip)** per job.

## Project layout

```
app.py                 Flask server, job queue, REST API
templates/index.html   Web UI
clipper/
  download.py          yt-dlp downloads
  transcribe.py        faster-whisper, word timestamps
  highlights.py        scoring, selection, optional Claude re-rank
  captions.py          caption styles and the ASS subtitle builder
  ai.py                Ollama / Claude helper, keyword hashtags
  speakers.py          offline speaker labelling
  export.py            subtitles, csv, GIF, MP3, covers, platform presets
  render.py            face tracking, ffmpeg filter graph, encoding
  control.py           pause / resume / cancel
  pipeline.py          orchestrates the steps above, and re-renders one clip
fonts/                 drop .ttf/.otf files here to use them in captions
```

## Customising

- **Add a built-in caption style**: add an entry to `STYLES` in `clipper/captions.py`; it appears in the UI automatically. (Or just save one from the designer.)
- **Change the caption font**: copy a font (for example Montserrat ExtraBold) into `fonts/` and change `font="Arial"` in `build_ass`.
- **Tune highlight scoring**: edit `HOOK_PHRASES` and the weights at the end of `_score` in `clipper/highlights.py`.

## Troubleshooting

- *ffmpeg missing*: install it and restart; the header shows ffmpeg status.
- *Face tracking does nothing*: it uses OpenCV's Haar detector, which finds front-facing faces only. Otherwise it falls back to a centre crop. Pick *Blurred background* for screen recordings or wide shots.
- *YouTube download fails*: run `pip install -U yt-dlp`; sites change often.
- *Slow transcription*: choose a smaller Whisper model.
- Only download or clip content you have the rights to use.
