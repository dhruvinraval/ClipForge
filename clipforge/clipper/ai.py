"""Optional AI helpers: Claude (online) or a local model through Ollama (offline).

Both produce, per clip: a virality score, a punchy title, a one-line summary and hashtags.
Without any AI, hashtags still come from a simple keyword counter that needs no model."""
from __future__ import annotations

import json
import os
import re
import urllib.request

STOP = set("""about above after again also always another because been before being both cannot could does doing done
down each else even ever every from gonna have having here into just kind know like little make many more most much
must never only other over really right said same say says should since some still such take than that their them
then there these they thing things think this those through today very want well were what when where which while
whom will with would yeah your you're don't it's i'm that's what's there's""".split())


def keyword_hashtags(text: str, n: int = 5) -> list[str]:
    counts: dict[str, int] = {}
    for w in re.findall(r"[A-Za-z][A-Za-z']{3,}", text.lower()):
        if w not in STOP:
            counts[w] = counts.get(w, 0) + 1
    top = sorted(counts, key=lambda k: (-counts[k], k))[:n]
    return ["#" + w.replace("'", "") for w in top]


def ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen(ollama_host() + "/api/tags", timeout=1.5) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return []


def resolve(opts, log=print):
    """Return (provider, model) or None, logging why an AI helper was skipped."""
    prov = opts.get("ai", "off")
    if prov == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            log("Claude helper skipped: ANTHROPIC_API_KEY is not set")
            return None
        return "claude", opts.get("ai_model") or os.environ.get("CLIPFORGE_LLM_MODEL", "claude-sonnet-5-5")
    if prov == "ollama":
        models = ollama_models()
        if not models:
            log("Local AI skipped: Ollama is not running or has no models (install one with: ollama pull llama3.2)")
            return None
        model = opts.get("ai_model") if opts.get("ai_model") in models else models[0]
        return "ollama", model
    return None


def _ollama_chat(prompt, model):
    body = json.dumps({"model": model, "stream": False, "format": "json",
                       "messages": [{"role": "user", "content": prompt}],
                       "options": {"num_ctx": 8192, "temperature": 0.3}}).encode()
    req = urllib.request.Request(ollama_host() + "/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["message"]["content"]


def _claude_chat(prompt, model):
    import anthropic
    msg = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]).messages.create(
        model=model, max_tokens=3000, messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text


def _parse(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("clips") or next((v for v in data.values() if isinstance(v, list)), [])
    return data


def annotate(cands, provider, model, log=print, rerank=True, keep_titles=False):
    """Fill title / summary / hashtags (and blend in a score when rerank=True) for each candidate, in place."""
    chat = _ollama_chat if provider == "ollama" else _claude_chat
    size = 4 if provider == "ollama" else 10
    limit = 900 if provider == "ollama" else 1500
    ok = 0
    for b in range(0, len(cands), size):
        batch = cands[b:b + size]
        payload = [{"id": i, "seconds": round(c["end"] - c["start"]), "transcript": c["text"][:limit]}
                   for i, c in enumerate(batch)]
        prompt = (
            "You are a short-form video editor. For each clip transcript below return: "
            '"score" (0-100, how likely it is to perform on TikTok/Reels/Shorts: strong opening hook, '
            'self-contained, emotional or insightful payoff), "title" (punchy on-screen title, max 8 words), '
            '"summary" (one sentence, max 20 words) and "hashtags" (4 to 6 relevant hashtags). '
            'Reply with ONLY a JSON object: {"clips":[{"id":0,"score":80,"title":"...","summary":"...",'
            '"hashtags":["#a","#b"]}]}\n\n' + json.dumps(payload))
        try:
            for r in _parse(chat(prompt, model)):
                c = batch[int(r["id"])]
                if rerank and "score" in r:
                    c["score"] = round(0.4 * c["score"] + 0.6 * max(0.0, min(100.0, float(r["score"]))), 1)
                if r.get("title") and not keep_titles:
                    c["title"] = str(r["title"]).strip()[:80]
                if r.get("summary"):
                    c["summary"] = str(r["summary"]).strip()[:200]
                tags = [("#" + str(t).lstrip("#").replace(" ", "")) for t in r.get("hashtags", []) if str(t).strip()]
                if tags:
                    c["hashtags"] = tags[:6]
                ok += 1
        except Exception as e:  # a bad batch must never fail the job
            log(f"AI batch skipped ({type(e).__name__}: {str(e)[:120]})")
    log(f"AI helper ({provider}: {model}) annotated {ok} of {len(cands)} clip(s)")
    return cands
