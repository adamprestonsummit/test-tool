# === app.py (Chunk 1/3) ===============================================
# SEO Audit & Competitor Comparator — Streamlit App (OpenAI optional)
# Run locally:
#   1) pip install -U streamlit requests beautifulsoup4 lxml tldextract plotly python-dateutil pandas
#   2) (optional) export PSI_API_KEY="<your-google-pagespeed-insights-api-key>"
#   3) (optional) export OPENAI_API_KEY="<your-openai-api-key>"
#   4) streamlit run app.py
# ======================================================================

from __future__ import annotations

import os
import re
import io
import csv
import json
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple, Any

import requests
from bs4 import BeautifulSoup
import tldextract

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------- Config -----------------------------
USER_AGENT = "Mozilla/5.0 (compatible; SEOAuditBot/1.1; +https://example.com/audit)"
TIMEOUT = 15
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en"}

# ----------------------------- Basic Helpers -----------------------------
def _get_semrush_key() -> Optional[str]:
    return (
        os.environ.get("SEMRUSH_API_KEY")
        or (st.secrets.get("SEMRUSH_API_KEY") if hasattr(st, "secrets") else None)
    )

def _get_openai_key() -> Optional[str]:
    return (
        os.environ.get("OPENAI_API_KEY")
        or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
    )

def normalize_url(domain_or_url: str) -> str:
    x = (domain_or_url or "").strip()
    if not x:
        return x
    if not x.startswith(("http://", "https://")):
        x = "https://" + x
    return x

def http_get(url: str) -> Tuple[Optional[requests.Response], Optional[str]]:
    try:
        resp = requests.get(url, headers=HEADERS, allow_redirects=True, timeout=TIMEOUT)
        return resp, None
    except requests.RequestException as e:
        return None, str(e)

def get_home_html(url: str) -> Tuple[Optional[str], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "final_url": None,
        "status_code": None,
        "redirects": 0,
        "https": url.startswith("https://"),
        "elapsed_ms": None,
        "page_bytes": None,
        "error": None,
    }
    resp, err = http_get(url)
    if err or not resp:
        meta["error"] = err or "No response"
        return None, meta
    meta["final_url"] = resp.url
    meta["status_code"] = resp.status_code
    meta["redirects"] = len(resp.history)
    meta["https"] = resp.url.startswith("https://")
    meta["elapsed_ms"] = int(resp.elapsed.total_seconds() * 1000)
    meta["page_bytes"] = len(resp.content) if resp.content is not None else None
    if resp.status_code >= 400:
        meta["error"] = f"HTTP {resp.status_code}"
        return None, meta
    # best-effort decoding
    try:
        resp.encoding = resp.apparent_encoding or resp.encoding
        html = resp.text
    except Exception:
        html = resp.content.decode("utf-8", errors="ignore")
    return html, meta

def parse_html(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")

def extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ext.domain

def is_internal(href: str, base_domain: str) -> bool:
    if not href:
        return False
    if href.startswith(("/", "#")):
        return True
    if href.startswith(("mailto:", "tel:")):
        return True
    try:
        d = extract_domain(href)
        return d == base_domain
    except Exception:
        return False

def find_meta_tag(soup: BeautifulSoup, name: str | None = None, prop: str | None = None) -> Optional[str]:
    if name:
        el = soup.find("meta", attrs={"name": name})
        if el and el.get("content"):
            return el["content"].strip()
    if prop:
        el = soup.find("meta", attrs={"property": prop})
        if el and el.get("content"):
            return el["content"].strip()
    return None

def has_json_ld(soup: BeautifulSoup) -> bool:
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            _ = json.loads(tag.string or "{}")
            return True
        except Exception:
            continue
    return False

def get_robots_txt(domain_url: str) -> Dict[str, Any]:
    base = normalize_url(domain_url).rstrip("/")
    robots_url = base + "/robots.txt"
    meta: Dict[str, Any] = {"exists": False, "sitemaps": [], "disallow_count": 0, "error": None}
    resp, err = http_get(robots_url)
    if err or not resp or resp.status_code >= 400:
        meta["error"] = err or (resp and f"HTTP {resp.status_code}")
        return meta
    text = resp.text
    meta["exists"] = True
    sm = re.findall(r"(?i)^sitemap:\s*(\S+)$", text, re.M)
    meta["sitemaps"] = sm
    dis = re.findall(r"(?i)^disallow:\s*(\S*)$", text, re.M)
    meta["disallow_count"] = len(dis)
    return meta

def has_sitemap(domain_url: str, hinted_sitemaps: List[str]) -> bool:
    base = normalize_url(domain_url).rstrip("/")
    candidates = set(hinted_sitemaps or [])
    candidates.add(base + "/sitemap.xml")
    for u in list(candidates):
        resp, err = http_get(u)
        if resp and resp.status_code < 400 and resp.content and b"<urlset" in resp.content[:4096]:
            return True
    return False

# ----------------------------- OpenAI helper (optional) -----------------------------
def ai_analyze_with_openai(
    page_text: str,
    topic_hint: Optional[str] = None,
    extra_context: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
    debug: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Uses the ChatGPT API (OpenAI) to score content AND propose themes, FAQs, and internal link suggestions.
    Returns a dict with 'ai_scores' and 'ai_findings'.
    """
    api_key = _get_openai_key()
    if not api_key:
        if debug: st.write("DEBUG: No OPENAI_API_KEY found")
        return None

    text = (page_text or "").strip()
    if len(text) > 20000:
        text = text[:20000]

    # We ask for structured JSON with explicit fields we’ll render in the UI.
    schema_hint = {
        "ai_scores": {
            "intent_match": 0, "topical_coverage": 0, "eeat": 0, "helpfulness": 0,
            "originality_judgement": 0, "tone_fit": 0, "conversion_copy": 0, "internal_link_opps": 0
        },
        "ai_findings": {
            "missing_subtopics": [],         # content themes to add (strings)
            "weak_sections": [],
            "entities_detected": [],
            "schema_recommendations": [],
            "faq_suggestions": [],           # list[str]
            "internal_link_suggestions": [], # list[{"anchor": str, "target": str}]
            "copy_suggestions": []
        }
    }

    system_text = "You are an SEO content analyst. Return ONLY valid JSON matching the provided schema."

    ctx = json.dumps(extra_context or {}, ensure_ascii=False)
    user_prompt = (
        "Return ONLY compact JSON exactly matching this schema:\n"
        f"{json.dumps(schema_hint)}\n\n"
        f"Business/topic hint: {topic_hint or '(none)'}\n"
        f"Extra context (URL, headings, sample internal anchors): {ctx}\n\n"
        "Populate:\n"
        "- 'missing_subtopics': 5–10 concise content themes the page should add (strings).\n"
        "- 'faq_suggestions': 5–10 concise user-intent questions (strings).\n"
        "- 'internal_link_suggestions': 5–10 objects with keys 'anchor' and 'target' "
        "(target is a probable internal path or slug like '/pricing' or '/guides/x').\n"
        "Score fields (0–100) in 'ai_scores'.\n\n"
        "Text to analyze:\n<<<\n" + text + "\n>>>\n"
    )

    endpoint = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "system", "content": system_text},
                     {"role": "user", "content": user_prompt}],
        "temperature": 0.2,
    }

    try:
        resp = requests.post(endpoint, headers=headers, json=body, timeout=timeout)
        if debug: st.write("DEBUG OpenAI status:", resp.status_code)
        if resp.status_code != 200:
            if debug: st.write("DEBUG OpenAI error body:", resp.text[:800])
            return None
        data = resp.json()
        out_text = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""
        out_text = out_text.strip()
        if out_text.startswith("```"):
            out_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", out_text, flags=re.MULTILINE).strip()
        return json.loads(out_text)
    except Exception as e:
        if debug: st.write("DEBUG exception:", repr(e))
        return None


# ----------------------------- Semrush helpers -----------------------------
SEMRUSH_BASE = "https://api.semrush.com/"

def semrush_get(report_type: str, params: dict, timeout: int = 30) -> list[dict] | None:
    api_key = _get_semrush_key()
    if not api_key:
        return None
    q = {"type": report_type, "key": api_key, **params}
    try:
        r = requests.get(SEMRUSH_BASE, params=q, timeout=timeout)
        if r.status_code != 200 or not r.text or "ERROR" in r.text[:30].upper():
            return None
        buf = io.StringIO(r.text)  # Semrush returns CSV
        rows = list(csv.DictReader(buf))
        return rows
    except Exception:
        return None

def last_full_month_anchor(today: date | None = None) -> date:
    d = today or date.today()
    first_this = d.replace(day=1)
    return (first_this - timedelta(days=1)).replace(day=1)

def months_ago(anchor: date, n: int) -> date:
    y = anchor.year + (anchor.month - 1 - n) // 12
    m = (anchor.month - 1 - n) % 12 + 1
    return date(y, m, 1)

def ai_suggest_keywords(topic_hint: str, n: int = 30) -> list[str]:
    if not topic_hint:
        return []
    prompt = (
        f"Generate a flat list of up to {n} UK English search keywords for the topic: {topic_hint}.\n"
        "Mix head, mid, and long-tail. Return one per line, no numbering."
    )
    api_key = _get_openai_key()
    if not api_key:
        return []
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a concise SEO keyword ideation assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.4
            },
            timeout=30
        )
        if resp.status_code != 200:
            return []
        text = (resp.json().get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""
        kws = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()]
        seen, out = set(), []
        for k in kws:
            kl = k.lower()
            if kl not in seen:
                seen.add(kl); out.append(k)
        return out[:n]
    except Exception:
        return []

def keyword_research_with_volumes(topic_hint: str, database: str = "uk") -> list[dict] | None:
    seeds = ai_suggest_keywords(topic_hint, n=30)
    if not seeds:
        return None
    return semrush_keyword_volumes(seeds, database=database) or []

def semrush_backlinks_overview(target: str, target_type: str = "root_domain") -> dict | None:
    rows = semrush_get("backlinks_overview", {"target": target, "target_type": target_type})
    return rows[0] if rows else None

def semrush_refdomains_count(target: str, target_type: str = "root_domain") -> int | None:
    rows = semrush_get("backlinks_refdomains", {"target": target, "target_type": target_type, "export_columns": "refdomain"})
    return len(rows) if rows is not None else None

def semrush_domain_overview(domain: str, database: str = "uk", display_date: str | None = None) -> dict | None:
    params = {"domain": domain, "database": database, "export_columns": "Dn,Or,Ot"}
    if display_date:
        params["display_date"] = display_date  # YYYY-MM-15
    rows = semrush_get("domain_ranks", params)
    return rows[0] if rows else None

def semrush_url_keywords_count(url: str, database: str = "uk") -> int | None:
    rows = semrush_get("url_organic", {"url": url, "database": database, "display_limit": 1000, "export_columns": "Ph,Po"})
    return len(rows) if rows is not None else None

def semrush_keyword_volumes(keywords: list[str], database: str = "uk") -> list[dict] | None:
    kws = ",".join(kw.strip() for kw in keywords if kw.strip())[:5000]
    rows = semrush_get("phrase_all", {"phrase": kws, "database": database, "export_columns": "Ph,Nq,Kd"})
    return rows

# ----------------------------- Content & Structure Analysis -----------------------------
def visible_text_from_soup(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text

def split_sentences(text: str) -> List[str]:
    bits = re.split(r"(?<=[.!?])\s+", text)
    return [b.strip() for b in bits if b.strip()]

def count_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    vowels = "aeiouy"
    syll = 0
    prev_v = False
    for ch in w:
        v = ch in vowels
        if v and not prev_v:
            syll += 1
        prev_v = v
    if w.endswith("e") and syll > 1:
        syll -= 1
    return max(1, syll)

def flesch_reading_ease(text: str) -> Optional[float]:
    words = re.findall(r"[A-Za-z']+", text)
    sents = split_sentences(text)
    if not words or not sents:
        return None
    w = len(words)
    s = max(1, len(sents))
    syll = sum(count_syllables(tok) for tok in words)
    return 206.835 - 1.015 * (w / s) - 84.6 * (syll / w)

def originality_heuristic(text: str) -> Dict[str, Any]:
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]
    w = len(words)
    unique = len(set(words))
    ttr = (unique / w) if w else 0.0
    trigrams = [tuple(words[i:i + 3]) for i in range(max(0, w - 2))]
    from collections import Counter
    c = Counter(trigrams)
    rep = sum(1 for _, v in c.items() if v > 1)
    rep_ratio = (rep / max(1, len(c))) if c else 0.0
    return {"ttr": round(ttr, 3), "repeated_trigram_ratio": round(rep_ratio, 3)}

def tone_heuristic(text: str) -> Dict[str, Any]:
    tokens = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]
    total = len(tokens)
    exclam = text.count("!")
    adverbs = sum(1 for w in tokens if w.endswith("ly") and len(w) > 3)
    buzz = {"best", "amazing", "revolutionary", "ultimate", "incredible", "guaranteed", "exclusive", "limited"}
    buzz_count = sum(1 for w in tokens if w in buzz)
    you_rate = sum(1 for w in tokens if w in {"you", "your", "yours"})
    return {
        "exclamation_density": round(exclam / max(1, len(split_sentences(text))), 3),
        "adverb_rate": round(adverbs / max(1, total), 3),
        "buzz_rate": round(buzz_count / max(1, total), 3),
        "second_person_rate": round(you_rate / max(1, total), 3),
    }

def heading_audit(soup: BeautifulSoup) -> Dict[str, Any]:
    headings = []
    for level in range(1, 7):
        for h in soup.find_all(f"h{level}"):
            headings.append({"level": level, "text": (h.get_text(" ", strip=True) or "").strip()})
    h1 = sum(1 for h in headings if h["level"] == 1)
    h2 = sum(1 for h in headings if h["level"] == 2)
    empty = sum(1 for h in headings if not h["text"])
    dom_levels = [int(n.name[1]) for n in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])]
    skips = sum(1 for a, b in zip(dom_levels, dom_levels[1:]) if b - a > 1)
    return {"h_total": len(headings), "h1_count": h1, "h2_count": h2, "empty_headings": empty, "level_skips": skips}

def internal_anchor_quality(soup: BeautifulSoup, base_domain: str) -> Dict[str, Any]:
    bad = {"click here", "read more", "learn more", "more", "here"}
    total = 0
    good = 0
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        txt = (a.get_text(" ", strip=True) or "").lower()
        if is_internal(href, base_domain):
            total += 1
            if len(txt) >= 4 and txt not in bad:
                good += 1
    return {"internal_total": total, "descriptive_ratio": round((good / total) if total else 1.0, 3)}

def js_reliance_metrics(soup: BeautifulSoup, html_bytes: Optional[int]) -> Dict[str, Any]:
    scripts = soup.find_all("script")
    ext = sum(1 for s in scripts if s.get("src"))
    inline_chars = sum(len((s.string or "")) for s in scripts if not s.get("src"))
    text_bytes = len(visible_text_from_soup(soup).encode("utf-8", errors="ignore"))
    ratio = (text_bytes / max(1, (html_bytes or 0))) if html_bytes else None
    return {
        "script_count": len(scripts),
        "external_script_count": ext,
        "inline_script_chars": inline_chars,
        "text_to_html_ratio": round(ratio, 3) if ratio is not None else None,
    }
# === app.py (Chunk 2/3) ===============================================

# ----------------------------- Scoring -----------------------------
def clamp(v, lo, hi):
    try:
        return max(lo, min(hi, v))
    except Exception:
        return lo

def score_title(length: int) -> int:
    if length == 0:
        return 0
    if 35 <= length <= 65:
        return 100
    diff = min(abs(50 - length), 50)
    return int(round(100 - (diff * 2)))

def score_meta_desc(length: int) -> int:
    if length == 0:
        return 0
    if 70 <= length <= 160:
        return 100
    diff = min(abs(115 - length), 115)
    return int(round(100 - (diff * 0.8)))

def score_h1(count: int) -> int:
    if count == 0:
        return 0
    if count == 1:
        return 100
    if 2 <= count <= 3:
        return 60
    return 30

def score_links(internal: int, external: int) -> int:
    total = internal + external
    if total == 0:
        return 30
    ratio = internal / total
    ideal = 0.6
    diff = abs(ratio - ideal)
    return int(round(100 - diff * 120))

def score_images_alt(ratio: float) -> int:
    return int(round(ratio * 100))

def score_tech(meta: Dict[str, Any]) -> int:
    pts = 0
    total = 0
    def add(cond, weight):
        nonlocal pts, total
        total += weight
        if cond:
            pts += weight
    add(meta.get("https", False), 2)
    add(meta.get("status_code", 0) < 400, 2)
    add(meta.get("robots_exists", False), 2)
    add(meta.get("sitemap_exists", False), 2)
    add(meta.get("viewport_meta", False), 1)
    add(not meta.get("noindex", False), 2)
    add((meta.get("elapsed_ms") or 9999) < 1200, 1)
    return int(round((pts / max(total, 1)) * 100))

def score_social(meta: Dict[str, Any]) -> int:
    pts = 0
    total = 0
    def add(cond, weight):
        nonlocal pts, total
        total += weight
        if cond:
            pts += weight
    add(meta.get("og_present", False), 1)
    add(meta.get("twitter_present", False), 1)
    add(meta.get("schema_jsonld", False), 1)
    return int(round((pts / max(total, 1)) * 100))

def score_performance(meta: Dict[str, Any]) -> int:
    psi_perf = meta.get("psi_scores", {}).get("performance")
    if psi_perf is not None:
        return int(psi_perf)
    ms = meta.get("elapsed_ms") or 2500
    size = meta.get("page_bytes") or 600000
    s = 100
    if ms > 500:
        s -= min(60, (ms - 500) / 20)
    if size > 200000:
        s -= min(30, (size - 200000) / 20000)
    return int(round(clamp(s, 0, 100)))

def score_readability(fre: Optional[float]) -> int:
    if fre is None:
        return 50
    if 50 <= fre <= 70:
        return 100
    if fre > 70:
        return int(max(70, min(100, 70 + (fre - 70) * 1)))
    return int(max(0, 100 - (50 - fre) * 2))

def score_originality(h: Dict[str, Any]) -> int:
    ttr = h.get("ttr", 0)
    rep = h.get("repeated_trigram_ratio", 0)
    s = 0
    s += min(70, ttr * 100)
    s += max(0, 30 - rep * 100)
    return int(max(0, min(100, s)))

def score_tone(h: Dict[str, Any]) -> int:
    s = 100
    s -= min(30, h.get("exclamation_density", 0) * 60)
    s -= min(30, h.get("buzz_rate", 0) * 200)
    s -= min(10, abs(h.get("adverb_rate", 0) - 0.06) * 100)
    s -= min(10, abs(h.get("second_person_rate", 0) - 0.01) * 100)
    return int(max(0, min(100, s)))

def score_headings(h: Dict[str, Any]) -> int:
    s = 100
    if h.get("h1_count", 0) == 0:
        s -= 40
    if h.get("h1_count", 0) > 1:
        s -= 20
    if h.get("h2_count", 0) == 0:
        s -= 15
    s -= min(25, h.get("level_skips", 0) * 10)
    s -= min(20, h.get("empty_headings", 0) * 5)
    return int(max(0, min(100, s)))

def score_internal_anchor_quality(aq: Dict[str, Any]) -> int:
    return int(round((aq.get("descriptive_ratio", 0) or 0) * 100))

def score_js_reliance(js: Dict[str, Any]) -> int:
    s = 100
    s -= min(40, max(0, js.get("script_count", 0) - 5) * 4)
    s -= min(20, max(0, js.get("external_script_count", 0) - 3) * 3)
    ratio = js.get("text_to_html_ratio")
    if ratio is not None and ratio < 0.2:
        s -= min(30, (0.2 - ratio) * 200)
    return int(max(0, min(100, s)))

def compute_scores(meta: Dict[str, Any]) -> Dict[str, Any]:
    scores: Dict[str, Any] = {}
    scores["score_title"] = score_title(meta.get("title_len", 0))
    scores["score_meta_desc"] = score_meta_desc(meta.get("meta_desc_len", 0))
    scores["score_h1"] = score_h1(meta.get("h1_count", 0))
    scores["score_links"] = score_links(meta.get("internal_links", 0), meta.get("external_links", 0))
    scores["score_images_alt"] = score_images_alt(meta.get("img_alt_ratio", 0))
    scores["score_tech"] = score_tech(meta)
    scores["score_social"] = score_social(meta)
    scores["score_performance"] = score_performance(meta)
    scores["score_readability"] = score_readability(meta.get("readability_fre"))
    scores["score_originality"] = score_originality(meta.get("originality", {}))
    scores["score_tone"] = score_tone(meta.get("tone", {}))
    scores["score_heading_structure"] = score_headings(meta.get("headings", {}))
    scores["score_anchor_quality"] = score_internal_anchor_quality(meta.get("anchor_quality", {}))
    scores["score_js"] = score_js_reliance(meta.get("js_reliance", {}))

    # Optional AI scores (if provided)
    ai_scores = meta.get("ai_scores")
    ai_weight_map: Dict[str, float] = {}
    if isinstance(ai_scores, dict) and ai_scores:
        def cv(x):
            try:
                return int(max(0, min(100, float(x))))
            except Exception:
                return 0
        scores["score_ai_intent"] = cv(ai_scores.get("intent_match"))
        scores["score_ai_coverage"] = cv(ai_scores.get("topical_coverage"))
        scores["score_ai_eeat"] = cv(ai_scores.get("eeat"))
        scores["score_ai_helpfulness"] = cv(ai_scores.get("helpfulness"))
        scores["score_ai_originality"] = cv(ai_scores.get("originality_judgement"))
        scores["score_ai_tone"] = cv(ai_scores.get("tone_fit"))
        scores["score_ai_conversion"] = cv(ai_scores.get("conversion_copy"))
        scores["score_ai_internal_links"] = cv(ai_scores.get("internal_link_opps"))
        ai_weight_map = {
            "score_ai_intent": 0.8,
            "score_ai_coverage": 0.9,
            "score_ai_eeat": 0.8,
            "score_ai_helpfulness": 0.8,
            "score_ai_originality": 0.6,
            "score_ai_tone": 0.4,
            "score_ai_conversion": 0.8,
            "score_ai_internal_links": 0.5,
        }

    weights = {
        "score_title": 1.0,
        "score_meta_desc": 0.9,
        "score_h1": 0.6,
        "score_links": 0.6,
        "score_images_alt": 0.5,
        "score_tech": 1.4,
        "score_social": 0.5,
        "score_performance": 1.4,
        "score_readability": 1.0,
        "score_originality": 0.8,
        "score_tone": 0.5,
        "score_heading_structure": 1.0,
        "score_anchor_quality": 0.7,
        "score_js": 0.9,
        **ai_weight_map,
    }
    total_w = sum(weights.values())
    overall = sum(scores[k] * w for k, w in weights.items() if k in scores)
    scores["overall_score"] = int(round(overall / total_w)) if total_w else 0
    scores["_weights"] = weights
    return scores

# ----------------------------- Recommendations -----------------------------
LOW = 70
RECS = {
    "score_title": "Rewrite the <title> to 35–65 characters, include a primary keyword near the start, and keep it unique.",
    "score_meta_desc": "Write a 70–160 character meta description with a benefit-led CTA and the main keyword naturally.",
    "score_h1": "Use exactly one clear, descriptive H1 that matches the page’s topic.",
    "score_links": "Improve internal linking — add contextual links to key pages; keep internal ~40–85% of all links.",
    "score_images_alt": "Add descriptive, concise alt text to images (skip purely decorative ones).",
    "score_tech": "Fix technical basics: enforce HTTPS, valid status 200, robots.txt, sitemap.xml, mobile viewport, avoid noindex.",
    "score_social": "Add Open Graph & Twitter Card tags; include JSON-LD structured data.",
    "score_performance": "Speed up: compress images, lazy-load media, minify assets, enable caching/HTTP2; fix CWV regressions.",
    "score_readability": "Write in plain language with short sentences and scannable paragraphs; aim Flesch 50–70.",
    "score_originality": "Reduce repetition, add unique insights/data/examples; avoid boilerplate phrasing.",
    "score_tone": "Dial down hype/buzzwords, reduce exclamations; prefer concrete benefits and specifics.",
    "score_heading_structure": "Fix heading hierarchy: one H1, use H2s for sections; avoid skipping levels or empty headings.",
    "score_anchor_quality": "Use descriptive anchor text for internal links (avoid ‘click here’/‘read more’).",
    "score_js": "Trim JavaScript: remove unused scripts, defer non-critical JS, and ensure core content is server-rendered.",
    "score_ai_intent": "Ensure copy answers the primary search intent; add/clarify CTA and key actions.",
    "score_ai_coverage": "Cover missing subtopics with dedicated H2/H3 sections and examples.",
    "score_ai_eeat": "Expose author credentials, cite sources, add about/contact/policy and last-updated info.",
    "score_ai_helpfulness": "Add step-by-step guidance, evidence, data, and illustrative examples.",
    "score_ai_originality": "Add unique POV, proprietary data, or case studies to differentiate content.",
    "score_ai_tone": "Align tone with brand (clear, confident, non-hypey); simplify jargon.",
    "score_ai_conversion": "Sharpen value prop above the fold, add trust signals and specific CTAs.",
    "score_ai_internal_links": "Add contextual internal links to related cornerstone pages with descriptive anchors.",
}

def generate_recommendations(r: Dict[str, Any]) -> List[str]:
    msgs: List[str] = []
    # Generic: any low score
    for key, msg in RECS.items():
        if r.get(key) is not None and r.get(key) < LOW:
            msgs.append(f"{key.replace('score_', '').replace('_', ' ').title()}: {msg}")

    # Specific nudges
    if r.get("title_len", 0) < 35:
        msgs.append("Title is short — expand to include a primary keyword and value prop (35–65 chars).")
    if r.get("title_len", 0) > 65:
        msgs.append("Title is long — trim to ~60 chars to avoid truncation.")
    if r.get("meta_desc_len", 0) == 0:
        msgs.append("Missing meta description — add 70–160 chars compelling summary.")
    if r.get("h1_count", 0) == 0:
        msgs.append("Missing H1 — add one descriptive H1.")
    if r.get("headings", {}).get("level_skips", 0) > 0:
        msgs.append("Heading level skips detected (e.g., H2 → H4) — keep hierarchy sequential.")
    aq = r.get("anchor_quality", {})
    if aq.get("internal_total", 0) >= 5 and (aq.get("descriptive_ratio", 1.0) < 0.7):
        msgs.append("Many internal links use vague anchors — replace with descriptive text.")
    js = r.get("js_reliance", {})
    if js.get("script_count", 0) > 10:
        msgs.append("High script count — audit and remove non-essential JS.")
    if js.get("text_to_html_ratio") is not None and js.get("text_to_html_ratio") < 0.15:
        msgs.append("Very low text-to-HTML ratio — core content may be thin or JS-reliant.")

    # AI findings
    ai = r.get("ai_findings") or {}
    for ms in (ai.get("copy_suggestions") or [])[:8]:
        msgs.append(f"Copy: {ms}")
    for ms in (ai.get("schema_recommendations") or [])[:5]:
        msgs.append(f"Schema: Consider adding {ms} JSON-LD.")
    return msgs

# ----------------------------- Analysis -----------------------------
def analyze_page(
    url: str,
    use_ai: bool = False,
    topic_hint: Optional[str] = None,
    show_ai_debug: bool = False,
) -> Dict[str, Any]:
    html, fetch_meta = get_home_html(url)
    result: Dict[str, Any] = {"_url": url, "_final_url": fetch_meta.get("final_url"), "_domain": extract_domain(url)}
    result.update(fetch_meta)

    if fetch_meta.get("error") or not html:
        # Compute minimal scores even if failed fetch
        result.update(compute_scores(result))
        result["_recommendations"] = generate_recommendations(result)
        result["_issue_count"] = len(result["_recommendations"])
        return result

    soup = parse_html(html)
    base_domain = extract_domain(result.get("_final_url") or url)

    # Title & meta
    title_tag = soup.find("title")
    title = (title_tag.get_text(strip=True) if title_tag else "") or ""
    title_len = len(title)
    meta_desc = find_meta_tag(soup, name="description") or ""
    meta_desc_len = len(meta_desc)

    # H1
    h1s = [h.get_text(strip=True) for h in soup.find_all("h1")]
    h1_count = len(h1s)

    # Canonical
    canonical = ""
    link_canon = soup.find("link", rel=lambda v: v and "canonical" in v)
    if link_canon is not None and link_canon.get("href"):
        canonical = link_canon["href"].strip()

    # Social & schema
    og_title = find_meta_tag(soup, prop="og:title")
    og_desc = find_meta_tag(soup, prop="og:description")
    tw_title = find_meta_tag(soup, name="twitter:title")
    tw_desc = find_meta_tag(soup, name="twitter:description")
    schema_jsonld = has_json_ld(soup)

    # Content quality & structure
    visible_text = visible_text_from_soup(soup)
    fre = flesch_reading_ease(visible_text)
    orig = originality_heuristic(visible_text)
    tone = tone_heuristic(visible_text)
    headings_meta = heading_audit(soup)
    anchor_q = internal_anchor_quality(soup, base_domain)
    js_meta = js_reliance_metrics(soup, fetch_meta.get("page_bytes"))


    # Light context to help AI suggest themes/FAQs/internal links
    headings_texts = [h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2", "h3"])][:30]
    anchor_texts = []
    seen = set()
    for a in a_tags:
        href = a.get("href") or ""
        txt = (a.get_text(" ", strip=True) or "").strip()
    if is_internal(href, base_domain) and len(txt) >= 4:
        t = txt.lower()
        if t not in {"click here","read more","learn more","more","here"} and t not in seen:
            seen.add(t); anchor_texts.append(txt)
    if len(anchor_texts) >= 40:
        break
    extra_ctx = {
        "final_url": result.get("_final_url") or url,
        "domain": base_domain,
        "headings": headings_texts,
        "sample_internal_anchor_texts": anchor_texts[:40],
    }

    # Links
    a_tags = soup.find_all("a")
    internal = 0
    external = 0
    for a in a_tags:
        href = a.get("href") or ""
        if is_internal(href, base_domain):
            internal += 1
        elif href.startswith(("http://", "https://")):
            external += 1

    # Images
    imgs = soup.find_all("img")
    img_count = len(imgs)
    img_alt_with = sum(1 for im in imgs if (im.get("alt") or "").strip())
    img_alt_ratio = (img_alt_with / img_count) if img_count else 1.0

    # Robots & Sitemap
    robots = get_robots_txt(url)
    sitemap_present = has_sitemap(url, robots.get("sitemaps", []))

    # Mobile meta viewport & robots meta
    viewport = soup.find("meta", attrs={"name": "viewport"}) is not None
    robots_meta = find_meta_tag(soup, name="robots") or ""
    is_noindex = "noindex" in robots_meta.lower()

    # PSI (optional)
    cwv: Dict[str, Any] = {}
    psi_scores: Dict[str, Any] = {}
    psi_key = os.environ.get("PSI_API_KEY")
    try:
        if psi_key:
            psi_url = (
                "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
                f"?url={requests.utils.quote(result.get('_final_url') or url, safe='')}"
                f"&key={psi_key}&category=performance&category=seo&category=accessibility&category=pwa&strategy=mobile"
            )
            r, err = http_get(psi_url)
            if r and r.status_code == 200:
                data = r.json()
                lh = data.get("lighthouseResult", {})
                cats = lh.get("categories", {})
                for k, v in cats.items():
                    if isinstance(v, dict) and "score" in v and v["score"] is not None:
                        psi_scores[k] = round(float(v["score"]) * 100)
                metrics = data.get("loadingExperience", {}).get("metrics", {})
                def pct_good(key):
                    m = metrics.get(key, {})
                    d = m.get("distributions", [])
                    good = next((x for x in d if x.get("min") == 0), None)
                    return int(round((good.get("proportion", 0) * 100))) if good else None
                cwv = {
                    "LCP_ms": (lh.get("audits", {}).get("largest-contentful-paint", {}).get("numericValue")),
                    "CLS": (lh.get("audits", {}).get("cumulative-layout-shift", {}).get("numericValue")),
                    "INP_ms": (lh.get("audits", {}).get("interaction-to-next-paint", {}).get("numericValue")),
                    "GOOD_LCP_%": pct_good("LARGEST_CONTENTFUL_PAINT_MS"),
                    "GOOD_CLS_%": pct_good("CUMULATIVE_LAYOUT_SHIFT"),
                    "GOOD_INP_%": pct_good("INTERACTION_TO_NEXT_PAINT"),
                }
    except Exception:
        pass

    result.update({
        "title": title,
        "title_len": title_len,
        "meta_desc_len": meta_desc_len,
        "h1_count": h1_count,
        "canonical": canonical,
        "og_present": bool(og_title or og_desc),
        "twitter_present": bool(tw_title or tw_desc),
        "schema_jsonld": schema_jsonld,
        "internal_links": internal,
        "external_links": external,
        "images": img_count,
        "img_alt_ratio": round(img_alt_ratio, 3),
        "robots_exists": robots.get("exists", False),
        "sitemap_exists": sitemap_present,
        "viewport_meta": viewport,
        "noindex": is_noindex,
        "psi_scores": psi_scores,
        "cwv": cwv,
        "readability_fre": None if fre is None else round(fre, 1),
        "originality": orig,
        "tone": tone,
        "headings": headings_meta,
        "anchor_quality": anchor_q,
        "js_reliance": js_meta,
    })

     # AI analysis (optional, ChatGPT API)
    ai = None
    if use_ai:
        ai = ai_analyze_with_openai(visible_text, topic_hint, extra_context=extra_ctx, debug=show_ai_debug)
    if ai:
        result["ai_scores"] = ai.get("ai_scores")
        result["ai_findings"] = ai.get("ai_findings")
    elif use_ai:
        result["_ai_error"] = "OpenAI returned no result (check key/quotas)."


    # Compute sub-scores & overall
    result.update(compute_scores(result))

    # Recommendations
    recs = generate_recommendations(result)
    result["_recommendations"] = recs
    result["_issue_count"] = len(recs)
    return result
# === app.py (Chunk 3/3) ===============================================

st.set_page_config(page_title="SEO Audit & Competitor Comparator", layout="wide")
st.title("🔎 SEO Audit & Competitor Comparator")
st.caption("Audits your homepage for on-page + technical basics, optional CWV via PSI, competitor comparison, and optional OpenAI for semantic checks.")

# --- Polished detail view styles ---
st.markdown("""
<style>
.badge{display:inline-block;padding:.25rem .55rem;border-radius:999px;font-weight:600;
       border:1px solid rgba(0,0,0,.06);font-size:0.85rem;margin-right:.35rem;margin-bottom:.35rem}
.badge.good{background:#E6F4EA;color:#137333;border-color:#D7EBD8}
.badge.warn{background:#FEF7E0;color:#8F4E00;border-color:#F3E6B1}
.badge.bad{background:#FCE8E6;color:#A50E0E;border-color:#F7C5C0}
.badge.info{background:#EEF2FF;color:#3730A3;border-color:#E0E7FF}

.kv-grid{display:grid;grid-template-columns:1fr 1fr;gap:.4rem 1rem}
.kv-grid .k{opacity:.75}
.kv-grid .v{font-weight:600}

.section-title{font-weight:700;margin:.2rem 0 .5rem 0}
.subtle{opacity:.8}

.score-pill{display:inline-block;padding:.25rem .6rem;border-radius:999px;background:#EEF2FF;
            color:#4338CA;font-weight:700}
.hr{border-top:1px solid rgba(0,0,0,.08);margin:.75rem 0}
.title-small{font-weight:600;opacity:.9}
.code-wrap{white-space:pre-wrap;word-break:break-word;font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
           background:#f7f7f9;border:1px solid #eee;border-radius:8px;padding:.5rem}
</style>
""", unsafe_allow_html=True)

def _chip(text: str, kind: str = "info"):
    st.markdown(f"<span class='badge {kind}'>{text}</span>", unsafe_allow_html=True)

def _kv_section(title: str, pairs: list[tuple[str, str | int | float | None]]):
    with st.container(border=True):
        st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
        rows = []
        for k, v in pairs:
            vv = "—" if v is None or v == "" else v
            rows.append(f"<div class='k'>{k}</div><div class='v'>{vv}</div>")
        st.markdown(f"<div class='kv-grid'>{''.join(rows)}</div>", unsafe_allow_html=True)

def _progress_row(label: str, value_0_to_1: float | None):
    col1, col2 = st.columns([1,3])
    with col1:
        st.markdown(f"<div class='title-small'>{label}</div>", unsafe_allow_html=True)
    with col2:
        pct = 0 if value_0_to_1 is None else max(0, min(1, float(value_0_to_1)))
        st.progress(int(pct * 100))

def _good_bad_chip(ok: bool, label_true="Yes", label_false="No"):
    _chip(label_true if ok else label_false, "good" if ok else "bad")

def _score_chips(res: dict):
    cols = st.columns(6)
    items = [
        ("Overall", res.get("overall_score")),
        ("Tech", res.get("score_tech")),
        ("Perf", res.get("score_performance")),
        ("Readability", res.get("score_readability")),
        ("Headings", res.get("score_heading_structure")),
        ("Anchors", res.get("score_anchor_quality")),
    ]
    for c, (lbl, val) in zip(cols, items):
        with c:
            st.markdown(f"<div class='subtle'>{lbl}</div>", unsafe_allow_html=True)
            st.markdown(f"<span class='score-pill'>{int(val or 0)}</span>", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    default_domain = st.text_input(
        "Your domain or URL",
        placeholder="example.com or https://example.com",
        key="domain_input",
    )
    competitors = st.text_area(
        "Competitors (one per line)",
        placeholder="competitor1.com\ncompetitor2.com",
        key="competitors_input",
    )

    st.subheader("AI Analysis (optional)")
    show_ai_debug = st.checkbox("Show AI debug", value=False, key="ai_debug")
    provider = st.selectbox("AI provider", ["OpenAI", "Off"], index=0, key="ai_provider")
    use_ai = provider != "Off"
    topic_hint = st.text_input("Topic/intent hint (optional)", "", key="ai_topic_hint")


    st.subheader("Semrush (optional)")
    use_semrush = st.checkbox("Fetch Semrush insights", value=False, key="semrush_toggle")
    if use_semrush and not _get_semrush_key():
        st.warning("No SEMRUSH_API_KEY found in Secrets or env.")

    run_btn = st.button("Run audit", type="primary", key="run_audit_btn")

    st.divider()
    st.subheader("Google PSI API (optional)")
    st.write("Set environment var `PSI_API_KEY` before running for Core Web Vitals + Lighthouse scores.")

# --- Main ---
if run_btn and default_domain:
    targets = [normalize_url(default_domain)]
    for line in (competitors or "").splitlines():
        line = line.strip()
        if line:
            targets.append(normalize_url(line))

    st.info(f"Auditing {len(targets)} site(s).")
    results: List[Dict[str, Any]] = []
    progress = st.progress(0.0)
    status = st.empty()

    for i, t in enumerate(targets, start=1):
        status.write(f"Fetching: {t}")
        res = analyze_page(
            t,
            use_ai=use_ai,
            topic_hint=topic_hint,
            show_ai_debug=show_ai_debug,
        )

        if use_semrush:
            domain = res.get("_domain")
            final_url = res.get("_final_url") or res.get("_url")

            bl_dom = semrush_backlinks_overview(domain, "root_domain") or {}
            bl_url = semrush_backlinks_overview(final_url, "url") or {}
            rd_dom_count = semrush_refdomains_count(domain, "root_domain")
            rd_url_count = semrush_refdomains_count(final_url, "url")
            dom_ov = semrush_domain_mom_yoy(domain, "uk") or {}
            url_kw_count = semrush_url_keywords_count(final_url, "uk")

            res["semrush"] = {
                "backlinks_domain": bl_dom,
                "backlinks_url": bl_url,
                "refdomains_domain_count": rd_dom_count,
                "refdomains_url_count": rd_url_count,
                "domain_organic_uk": dom_ov,
                "url_keywords_uk": url_kw_count,
            }

            if topic_hint:
                res["keyword_research"] = keyword_research_with_volumes(topic_hint, "uk")

        results.append(res)
        progress.progress(i / len(targets))

    status.write("Done.")

    # ------------------------ SUMMARY TABLE ------------------------
    flat_results = []
    for res in results:
        sm = res.get("semrush", {})
        row = {
            "Domain": res.get("_domain"),
            "Final URL": res.get("_final_url") or res.get("_url"),
            "HTTP Status": res.get("status_code"),
            "Load (ms)": res.get("elapsed_ms"),
            "Title len": res.get("title_len"),
            "Meta len": res.get("meta_desc_len"),
            "H1s": res.get("h1_count"),
            "Alt ratio": res.get("img_alt_ratio"),
            "Tech": res.get("score_tech"),
            "Perf": res.get("score_performance"),
            "Social": res.get("score_social"),
            "Overall": res.get("overall_score"),

            # Semrush (optional)
            "Backlinks (Domain)": (sm.get("backlinks_domain", {}) or {}).get("backlinks", "N/A"),
            "Backlinks (URL)": (sm.get("backlinks_url", {}) or {}).get("backlinks", "N/A"),
            "Ref Domains (Domain)": sm.get("refdomains_domain_count", "N/A"),
            "Ref Domains (URL)": sm.get("refdomains_url_count", "N/A"),
            "Organic Traffic UK": (sm.get("domain_organic_uk", {}) or {}).get("Ot", "N/A"),
            "MoM Change UK": (sm.get("domain_organic_uk", {}) or {}).get("Ot_mom_%", "N/A"),
            "YoY Change UK": (sm.get("domain_organic_uk", {}) or {}).get("Ot_yoy_%", "N/A"),
            "Keywords (URL, UK)": sm.get("url_keywords_uk", "N/A"),
        }
        flat_results.append(row)

    df_summary = pd.DataFrame(flat_results)

    def _hi_num(val):
        if val == "N/A" or val is None:
            return ""
        try:
            valf = float(val)
        except Exception:
            return ""
        if valf > 1000:
            return "background-color: #4caf50; color: white;"
        elif valf > 500:
            return "background-color: #c8e6c9;"
        else:
            return "background-color: #ffcdd2;"

    def _hi_pct(val):
        if val == "N/A" or val is None:
            return ""
        try:
            valf = float(val)
        except Exception:
            return ""
        if valf > 0:
            return "background-color: #4caf50; color: white;"
        elif valf > -5:
            return "background-color: #fff9c4;"
        else:
            return "background-color: #ffcdd2;"

    semrush_cols_high = [
        "Backlinks (Domain)", "Backlinks (URL)",
        "Ref Domains (Domain)", "Ref Domains (URL)",
        "Organic Traffic UK", "Keywords (URL, UK)"
    ]
    semrush_cols_percent = ["MoM Change UK", "YoY Change UK"]

    styled_df = df_summary.style.applymap(_hi_num, subset=semrush_cols_high)
    styled_df = styled_df.applymap(_hi_pct, subset=semrush_cols_percent)

    st.subheader("Summary")
    st.dataframe(styled_df, use_container_width=True)

    csv_bytes = df_summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Summary CSV",
        data=csv_bytes,
        file_name="seo_audit_summary.csv",
        mime="text/csv",
    )

    # ------------------------ Radar chart ------------------------
    # Base categories always present
    base_cats = [
        ("Title", "score_title"),
        ("Meta", "score_meta_desc"),
        ("H1", "score_h1"),
        ("Links", "score_links"),
        ("Alt", "score_images_alt"),
        ("Tech", "score_tech"),
        ("Perf", "score_performance"),
        ("Readability", "score_readability"),
        ("Originality", "score_originality"),
        ("Tone", "score_tone"),
        ("Headings", "score_heading_structure"),
        ("Anchors", "score_anchor_quality"),
        ("JS", "score_js"),
        ("Social", "score_social"),
    ]
    # Add AI categories only if any result has them
    any_ai = any(r.get("ai_scores") for r in results)
    ai_cats = [
        ("AI Intent", "score_ai_intent"),
        ("AI Coverage", "score_ai_coverage"),
        ("AI E-E-A-T", "score_ai_eeat"),
        ("AI Helpful", "score_ai_helpfulness"),
        ("AI Originality", "score_ai_originality"),
        ("AI Tone", "score_ai_tone"),
        ("AI Conversion", "score_ai_conversion"),
        ("AI Internal Links", "score_ai_internal_links"),
    ] if any_ai else []

    cats = base_cats + ai_cats
    fig = go.Figure()
    theta = [c[0] for c in cats]
    for r in results:
        vals = [int(r.get(c[1], 0) or 0) for c in cats]
        fig.add_trace(go.Scatterpolar(r=vals, theta=theta, fill='toself', name=r.get("_domain")))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=520)
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------ Overall score bar chart ------------------------
    st.subheader("Overall Score Comparison")
    fig2 = px.bar(
        x=[r.get("_domain") for r in results],
        y=[r.get("overall_score") for r in results],
        color=[r.get("_domain") for r in results],
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"x": "Domain", "y": "Overall Score"},
        text=[r.get("overall_score") for r in results],
    )
    fig2.update_traces(textposition="outside")
    fig2.update_yaxes(range=[0, 100])
    st.plotly_chart(fig2, use_container_width=True)

 # ------------------------ Details (polished) ------------------------
    if results:
        st.subheader("Details by Site")
        for res in results:
            header_left = f"{res.get('_domain')} — details"
            header_right = f"Overall: {int(res.get('overall_score') or 0)}"
            with st.expander(f"{header_left}   |   {header_right}"):
                # Top row: chips
                st.markdown("##### Key Scores")
                _score_chips(res)
                st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    def _bullets(items):
        items = items or []
    for it in items[:10]:
        if isinstance(it, dict):
            anchor = it.get("anchor") or it.get("text") or ""
            target = it.get("target") or it.get("slug") or it.get("url") or ""
            if anchor or target:
                st.markdown(f"- **{anchor}** → `{target}`")
        else:
            st.markdown(f"- {it}")


            # 2-column layout of cards
            col1, col2 = st.columns(2)

            with col1:
                # BASICS
                _kv_section("Basics", [
                    ("Final URL", res.get("_final_url")),
                    ("Status", res.get("status_code")),
                    ("HTTPS", "Yes" if res.get("https") else "No"),
                    ("Redirects", res.get("redirects")),
                    ("Load (ms)", res.get("elapsed_ms")),
                    ("HTML bytes", res.get("page_bytes")),
                    ("Noindex", "Yes" if res.get("noindex") else "No"),
                ])

                # ON-PAGE
                with st.container(border=True):
                    st.markdown("<div class='section-title'>On-page</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='title-small'>Title</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='code-wrap'>{(res.get('title') or '')}</div>", unsafe_allow_html=True)
                    _kv_section("On-page — stats", [
                        ("Title length", res.get("title_len")),
                        ("Meta desc length", res.get("meta_desc_len")),
                        ("H1 count", res.get("h1_count")),
                        ("Canonical", res.get("canonical")),
                    ])

                # CONTENT QUALITY
                with st.container(border=True):
                    st.markdown("<div class='section-title'>Content Quality</div>", unsafe_allow_html=True)
                    _kv_section("Readability & Tone", [
                        ("Flesch Reading Ease", res.get("readability_fre")),
                        ("Originality (TTR)", (res.get("originality") or {}).get("ttr")),
                        ("Repeated 3-grams", (res.get("originality") or {}).get("repeated_trigram_ratio")),
                        ("Exclam/100 sents", (res.get("tone") or {}).get("exclamation_density")),
                        ("Buzz rate", (res.get("tone") or {}).get("buzz_rate")),
                    ])

            with col2:
                # HEADING STRUCTURE
                _kv_section("Heading Structure", [
                    ("Total headings", (res.get("headings") or {}).get("h_total")),
                    ("H1 count", (res.get("headings") or {}).get("h1_count")),
                    ("H2 count", (res.get("headings") or {}).get("h2_count")),
                    ("Empty headings", (res.get("headings") or {}).get("empty_headings")),
                    ("Level skips", (res.get("headings") or {}).get("level_skips")),
                ])

                # LINKS & ANCHORS
                with st.container(border=True):
                    st.markdown("<div class='section-title'>Links & Anchors</div>", unsafe_allow_html=True)
                    _kv_section("Link counts", [
                        ("Internal links", res.get("internal_links")),
                        ("External links", res.get("external_links")),
                        ("Images", res.get("images")),
                    ])
                    _progress_row("Alt text ratio", res.get("img_alt_ratio"))
                    _progress_row("Descriptive anchor ratio", (res.get("anchor_quality") or {}).get("descriptive_ratio"))

                # JS RELIANCE
                _kv_section("JS Reliance", [
                    ("Script count", (res.get("js_reliance") or {}).get("script_count")),
                    ("External scripts", (res.get("js_reliance") or {}).get("external_script_count")),
                    ("Inline script chars", (res.get("js_reliance") or {}).get("inline_script_chars")),
                    ("Text / HTML ratio", (res.get("js_reliance") or {}).get("text_to_html_ratio")),
                ])

                # ROBOTS / SITEMAP / SOCIAL
                with st.container(border=True):
                    st.markdown("<div class='section-title'>Robots, Sitemap & Social</div>", unsafe_allow_html=True)
                    st.markdown("**robots.txt** ", unsafe_allow_html=True)
                    _good_bad_chip(bool(res.get("robots_exists")), "Present", "Missing")
                    st.markdown("**sitemap** ", unsafe_allow_html=True)
                    _good_bad_chip(bool(res.get("sitemap_exists")), "Present", "Missing")
                    st.markdown("**Open Graph / Twitter / JSON-LD**", unsafe_allow_html=True)
                    _chip("OG", "good" if res.get("og_present") else "bad")
                    _chip("Twitter", "good" if res.get("twitter_present") else "bad")
                    _chip("JSON-LD", "good" if res.get("schema_jsonld") else "bad")

                # PSI / CWV
                if res.get("psi_scores") or res.get("cwv"):
                    with st.container(border=True):
                        st.markdown("<div class='section-title'>Lighthouse & Core Web Vitals (mobile)</div>", unsafe_allow_html=True)
                        if res.get("psi_scores"):
                            _kv_section("Lighthouse categories", [(k.upper(), v) for k, v in (res.get("psi_scores") or {}).items()])
                        if res.get("cwv"):
                            _kv_section("CWV", [
                                ("LCP (ms)", (res.get("cwv") or {}).get("LCP_ms")),
                                ("CLS", (res.get("cwv") or {}).get("CLS")),
                                ("INP (ms)", (res.get("cwv") or {}).get("INP_ms")),
                                ("Good LCP %", (res.get("cwv") or {}).get("GOOD_LCP_%")),
                                ("Good CLS %", (res.get("cwv") or {}).get("GOOD_CLS_%")),
                                ("Good INP %", (res.get("cwv") or {}).get("GOOD_INP_%")),
                            ])

            # AI (optional)
    if res.get("ai_scores") or res.get("_ai_error"):
        with st.container(border=True):
            st.markdown("<div class='section-title'>AI Analysis (ChatGPT)</div>", unsafe_allow_html=True)
        if res.get("ai_scores"):
            _kv_section("Scores (AI)", [(k.replace("score_ai_","").upper(), v)
                                        for k, v in res.items() if k.startswith("score_ai_")])
        ai_f = res.get("ai_findings") or {}
        if ai_f:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Content themes to add**")
                _bullets(ai_f.get("missing_subtopics"))
            with c2:
                st.markdown("**FAQ suggestions**")
                _bullets(ai_f.get("faq_suggestions"))
            with c3:
                st.markdown("**Internal link suggestions**")
                # Works with either list[str] or list[{anchor,target}]
                _bullets(ai_f.get("internal_link_suggestions"))
        if res.get("_ai_error"):
            _chip(res["_ai_error"], "warn")


            # Recommendations (always last)
            with st.container(border=True):
                st.markdown("<div class='section-title'>Recommendations</div>", unsafe_allow_html=True)
                recs = res.get("_recommendations", []) or []
                if recs:
                    for m in recs:
                        st.write("•", m)
                else:
                    _chip("No critical issues detected", "good")


    # ------------------------ Downloads ------------------------
    st.subheader("Export")
    json_blob = json.dumps(results, indent=2)
    st.download_button(
        "Download JSON",
        data=json_blob,
        file_name="seo_audit_results.json",
        mime="application/json"
    )

    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    header = [
        "domain", "final_url", "overall", "title", "title_len", "meta_desc_len",
        "h1_count", "internal", "external", "img_alt_ratio", "tech", "perf",
        "social", "status", "load_ms", "page_bytes", "issues", "ai"
    ]
    writer.writerow(header)
    for r in results:
        writer.writerow([
            r.get("_domain"),
            r.get("_final_url"),
            r.get("overall_score"),
            r.get("title"),
            r.get("title_len"),
            r.get("meta_desc_len"),
            r.get("h1_count"),
            r.get("internal_links"),
            r.get("external_links"),
            r.get("img_alt_ratio"),
            r.get("score_tech"),
            r.get("score_performance"),
            r.get("score_social"),
            r.get("status_code"),
            r.get("elapsed_ms"),
            r.get("page_bytes"),
            r.get("_issue_count"),
            "Yes" if r.get("ai_scores") else "No",
        ])
    st.download_button(
        "Download CSV",
        data=csv_buf.getvalue(),
        file_name="seo_audit_results.csv",
        mime="text/csv"
    )

else:
    st.info("Enter your domain and any competitors, choose AI if desired, then click **Run audit**.")

# ----------------------------- Notes -----------------------------
# This is a lightweight checker focused on homepage-level signals. For crawling and JS-rendered content at scale,
# integrate with a crawler or a headless browser. PSI integration is optional via PSI_API_KEY env var.
