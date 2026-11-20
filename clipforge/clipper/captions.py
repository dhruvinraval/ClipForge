"""Animated word-by-word captions rendered as an ASS subtitle file (burned in by ffmpeg/libass)."""
from __future__ import annotations

import re

BASE = dict(label="Custom", font="Arial", size=100, primary="FFFFFF", active="FFE14D", outline="000000",
            outline_a=0, outline_w=7, shadow=2, upper=True, box=False, group=3, highlight=True, position="bottom")

STYLES = {
    "hormozi": {**BASE, "label": "Hormozi"},
    "neon":    {**BASE, "label": "Neon", "active": "3DE0FF", "outline": "3A0F6B", "outline_w": 6, "shadow": 0},
    "boxed":   {**BASE, "label": "Boxed", "active": "B8FF4D", "outline_a": 0x55, "outline_w": 14, "shadow": 0,
                "upper": False, "box": True, "group": 4},
    "minimal": {**BASE, "label": "Minimal", "active": "FFFFFF", "outline_w": 3, "shadow": 1, "upper": False,
                "group": 6, "highlight": False},
}


def _hex(v, default):
    v = str(v or "").lstrip("#")
    return v.upper() if re.fullmatch(r"[0-9A-Fa-f]{6}", v) else default


def _int(v, default, lo, hi):
    try:
        return max(lo, min(hi, int(float(v))))
    except (TypeError, ValueError):
        return default


def normalize_style(raw) -> dict:
    """Validate and clamp a caption style coming from the UI (never trust it blindly)."""
    raw = raw if isinstance(raw, dict) else {}
    s = dict(BASE)
    s["label"] = re.sub(r"[^\w .\-]", "", str(raw.get("label") or "Custom"))[:30].strip() or "Custom"
    s["font"] = re.sub(r"[,\\\n\r{}]", "", str(raw.get("font") or "Arial")).strip()[:40] or "Arial"
    s["size"] = _int(raw.get("size"), 100, 50, 170)
    for k in ("primary", "active", "outline"):
        s[k] = _hex(raw.get(k), BASE[k])
    s["outline_a"] = _int(raw.get("outline_a"), 0, 0, 255)
    s["outline_w"] = _int(raw.get("outline_w"), 7, 0, 24)
    s["shadow"] = _int(raw.get("shadow"), 2, 0, 8)
    s["group"] = _int(raw.get("group"), 3, 1, 8)
    for k in ("upper", "box", "highlight"):
        s[k] = raw.get(k) is True or raw.get(k) in (1, "true", "True") if k in raw else BASE[k]
    s["position"] = raw.get("position") if raw.get("position") in ("bottom", "middle", "top") else "bottom"
    return s


def resolve_style(x) -> dict:
    return normalize_style(x if isinstance(x, dict) else STYLES.get(x, STYLES["hormozi"]))


def _col(rgb: str, alpha: int = 0) -> str:
    return f"&H{alpha:02X}{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}"


def _ts(t: float) -> str:
    cs = int(round(max(t, 0) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _esc(t: str) -> str:
    return t.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def words_in_range(all_words, start, end):
    """Words overlapping [start, end], with timestamps shifted to clip-relative time."""
    return [{"word": w["word"], "start": max(w["start"] - start, 0.0), "end": min(w["end"], end) - start}
            for w in all_words if w["end"] > start and w["start"] < end]


def _group(words, max_words, max_chars):
    groups, cur = [], []
    for w in words:
        too_long = cur and sum(len(x["word"]) + 1 for x in cur) + len(w["word"]) > max_chars
        if cur and (len(cur) >= max_words or w["start"] - cur[-1]["end"] > 0.7 or too_long
                    or re.search(r"[.!?]$", cur[-1]["word"])):
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def build_ass(words, W, H, dur, style="hormozi", title=None, progress_bar=True, hook_seconds=3.2) -> str:
    st = resolve_style(style)
    portrait = H > W
    fs = int((88 if portrait else (76 if H == W else 62)) * st["size"] / 100)
    align = {"bottom": 2, "middle": 5, "top": 8}[st["position"]]
    margin_v = {"bottom": int(H * (0.24 if portrait else 0.09)), "middle": 0,
                "top": int(H * (0.2 if title else 0.1))}[st["position"]]
    max_chars = int((20 if portrait or H == W else 34) * min(1.6, 100 / st["size"]))
    font = st["font"]

    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},{fs},{_col(st['primary'])},{_col(st['primary'])},{_col(st['outline'], st['outline_a'])},{_col('000000', 0x60)},-1,0,0,0,100,100,0,0,{3 if st['box'] else 1},{st['outline_w']},{st['shadow']},{align},60,60,{margin_v},1
Style: Hook,{font},{int(fs * 0.78)},{_col('111111')},{_col('111111')},{_col('FFE14D')},{_col('000000', 0x80)},-1,0,0,0,100,100,0,0,3,18,0,8,90,90,{int(H * 0.09)},1
Style: Bar,{font},20,{_col('FFE14D')},{_col('FFE14D')},{_col('FFE14D')},{_col('000000')},0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    if title:
        ev.append(f"Dialogue: 1,{_ts(0)},{_ts(min(hook_seconds, dur))},Hook,,0,0,0,,{{\\fad(180,220)}}{_esc(title)}")

    groups = _group(words, st["group"], max_chars)
    for gi, g in enumerate(groups):
        next_start = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else dur
        for k, w in enumerate(g):
            t0 = w["start"]
            t1 = g[k + 1]["start"] if k + 1 < len(g) else min(w["end"] + 0.3, next_start, dur)
            if t1 <= t0:
                t1 = t0 + 0.12
            parts = []
            for m, x in enumerate(g):
                txt = _esc(x["word"].upper() if st["upper"] else x["word"])
                if st["highlight"] and m == k:
                    parts.append(f"{{\\c{_col(st['active'])}&\\fscx84\\fscy84\\t(0,110,\\fscx112\\fscy112)}}{txt}"
                                 f"{{\\c{_col(st['primary'])}&\\fscx100\\fscy100}}")
                else:
                    parts.append(txt)
            ev.append(f"Dialogue: 0,{_ts(t0)},{_ts(t1)},Cap,,0,0,0,,{' '.join(parts)}")

    if progress_bar and dur > 1:
        y = H - 14
        ev.append(
            f"Dialogue: 2,{_ts(0)},{_ts(dur)},Bar,,0,0,0,,"
            f"{{\\an7\\pos(0,{y})\\clip(0,0,1,{H})\\t(0,{int(dur * 1000)},\\clip(0,0,{W},{H}))\\bord0\\shad0\\p1}}"
            f"m 0 0 l {W} 0 l {W} 14 l 0 14"
        )
    return head + "\n".join(ev) + "\n"


def build_cover_ass(text, W, H, font="Arial", position="middle") -> str:
    """One big text block for a cover image (rendered with a single ffmpeg frame grab)."""
    fs = 96 if H > W else 72
    align = {"top": 8, "middle": 5, "bottom": 2}.get(position, 5)
    mv = int(H * 0.12) if position != "middle" else 0
    font = re.sub(r"[,\\\n\r{}]", "", font) or "Arial"
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cover,{font},{fs},{_col('FFFFFF')},{_col('FFFFFF')},{_col('000000', 0x40)},{_col('000000')},-1,0,0,0,100,100,0,0,3,26,0,{align},100,100,{mv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:01:00.00,Cover,,0,0,0,,{_esc(text)}
"""
