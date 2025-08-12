# SEO Audit & Competitor Comparator — Streamlit App (with optional Gemini AI)
# ------------------------------------------------
# Run locally:
#   1) pip install -U streamlit requests beautifulsoup4 lxml tldextract plotly python-dateutil
#   2) (optional, for AI) pip install -U google-generativeai
#   3) (optional) export PSI_API_KEY="<your-google-pagespeed-insights-api-key>"
#   4) (optional, for AI) export GEMINI_API_KEY="<your-gemini-api-key>"  # or GOOGLE_API_KEY
#   5) streamlit run app.py
#
# What it does:
# - Audits a domain's homepage for on-page SEO and technical basics
# - (Optionally) pulls Core Web Vitals + Lighthouse via PageSpeed Insights API
# - Compares multiple competitors side-by-side
# - Shows radar + bar charts and detailed factor breakdowns
# - Lets you download CSV/JSON of results
# - Generates prescriptive recommendations for low-scoring areas
# - NEW (optional): Runs Gemini AI for semantic content checks (intent, coverage, E-E-A-T, etc.)

import os
import re
import json
from typing import Dict, List, Optional, Tuple, Any

import requests
from bs4 import BeautifulSoup
import tldextract

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

def _get_semrush_key() -> Optional[str]:
    """
    Get the Semrush API key from environment variables or Streamlit secrets.
    """
    return (
        os.environ.get("SEMRUSH_API_KEY")
        or (st.secrets.get("SEMRUSH_API_KEY") if hasattr(st, "secrets") else None)
    )


# Optional AI SDK (Gemini)
try:
    import google.generativeai as genai  # type: ignore
    _GENAI_AVAILABLE = True
except Exception:
    genai = None
    _GENAI_AVAILABLE = False

USER_AGENT = "Mozilla/5.0 (compatible; SEOAuditBot/1.1; +https://example.com/audit)"
TIMEOUT = 15
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en"}

# ----------------------------- Basic Helpers -----------------------------

AI_ENV_VARS = ["GEMINI_API_KEY", "GOOGLE_API_KEY"]  # either works


def normalize_url(domain_or_url: str) -> str:
    x = domain_or_url.strip()
    if not x:
        return x
    if not x.startswith("http://") and not x.startswith("https://"):
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
    if href.startswith("/") or href.startswith("#"):
        return True
    if href.startswith("mailto:") or href.startswith("tel:"):
        return True
    try:
        d = extract_domain(href)
        return d == base_domain
    except Exception:
        return False


def find_meta_tag(soup: BeautifulSoup, name: str = None, prop: str = None) -> Optional[str]:
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
    base = normalize_url(domain_url)
    if base.endswith("/"):
        base = base[:-1]
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
    base = normalize_url(domain_url)
    if base.endswith("/"):
        base = base[:-1]
    candidates = set(hinted_sitemaps)
    candidates.add(base + "/sitemap.xml")
    for u in list(candidates):
        resp, err = http_get(u)
        if resp and resp.status_code < 400 and resp.content and b"<urlset" in resp.content[:4096]:
            return True
    return False

# ----------------------------- AI (Gemini) Helpers -----------------------------

def _get_openai_key() -> str | None:
    return (
        os.environ.get("OPENAI_API_KEY")
        or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
    )

def _get_gemini_key() -> Optional[str]:
    # Prefer env vars if set, else fall back to Streamlit secrets
    return (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or (st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None)
        or (st.secrets.get("GOOGLE_API_KEY") if hasattr(st, "secrets") else None)
    )


def _get_openai_key() -> Optional[str]:
    return (
        os.environ.get("OPENAI_API_KEY")
        or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
    )

def ai_analyze_with_openai(page_text: str, topic_hint: Optional[str] = None, timeout: int = 30, debug: bool = False) -> Optional[Dict[str, Any]]:
    api_key = _get_openai_key()
    if not api_key:
        if debug: st.write("DEBUG: No OPENAI_API_KEY found")
        return None

    text = (page_text or "").strip()
    if len(text) > 20000:
        text = text[:20000]

    schema_hint = {
        "ai_scores": {
            "intent_match": 0, "topical_coverage": 0, "eeat": 0, "helpfulness": 0,
            "originality_judgement": 0, "tone_fit": 0, "conversion_copy": 0, "internal_link_opps": 0
        },
        "ai_findings": {
            "missing_subtopics": [], "weak_sections": [], "entities_detected": [],
            "schema_recommendations": [], "faq_suggestions": [],
            "internal_link_suggestions": [], "copy_suggestions": []
        }
    }

    system_text = (
        "You are an SEO content analyst. Return ONLY valid JSON matching the provided schema."
    )
    user_prompt = (
        "Return ONLY compact JSON (no prose) exactly matching this schema:\n"
        f"{json.dumps(schema_hint)}\n\n"
        f"Business/topic hint: {topic_hint or '(none)'}\n"
        "Text to analyze:\n<<<\n" + text + "\n>>>\n\n"
        "Score intent_match, topical_coverage, eeat, helpfulness, originality_judgement, "
        "tone_fit, conversion_copy, internal_link_opps (0–100 each). Return JSON only."
    )

    endpoint = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": "gpt-4o-mini",  # good availability & price; switch to 'gpt-4o' if you have access
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    try:
        resp = requests.post(endpoint, headers=headers, json=body, timeout=timeout)
        if debug: st.write("DEBUG OpenAI status:", resp.status_code)
        if resp.status_code != 200:
            if debug: st.write("DEBUG OpenAI error body:", resp.text[:800])
            return None
        data = resp.json()
        out_text = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        if not out_text:
            if debug: st.write("DEBUG: Empty message content", data)
            return None

        out_text = out_text.strip()
        if out_text.startswith("```"):
            out_text = re.sub(r"^```(?:json)?|```$", "", out_text, flags=re.MULTILINE).strip()

        return json.loads(out_text)
    except Exception as e:
        if debug: st.write("DEBUG exception:", repr(e))
        return None

# ---------- Semrush helpers ----------
import csv
import io
from datetime import date, timedelta

def _get_semrush_key() -> str | None:
    return (
        os.environ.get("SEMRUSH_API_KEY")
        or (st.secrets.get("SEMRUSH_API_KEY") if hasattr(st, "secrets") else None)
    )

SEMRUSH_BASE = "https://api.semrush.com/"

def semrush_get(report_type: str, params: dict, timeout: int = 30) -> list[dict] | None:
    """
    Generic Semrush fetcher. Returns a list of dict rows (CSV → dict).
    """
    api_key = _get_semrush_key()
    if not api_key:
        return None
    q = {"type": report_type, "key": api_key, **params}
    try:
        r = requests.get(SEMRUSH_BASE, params=q, timeout=timeout)
        if r.status_code != 200 or not r.text or "ERROR" in r.text[:30].upper():
            return None
        # Semrush returns CSV
        buf = io.StringIO(r.text)
        rows = list(csv.DictReader(buf))
        return rows
    except Exception:
        return None

def last_full_month_anchor(today: date | None = None) -> date:
    d = today or date.today()
    first_this = d.replace(day=1)
    return (first_this - timedelta(days=1)).replace(day=1)  # first day of last full month

def months_ago(anchor: date, n: int) -> date:
    y = anchor.year + (anchor.month - 1 - n) // 12
    m = (anchor.month - 1 - n) % 12 + 1
    return date(y, m, 1)

def ai_suggest_keywords(topic_hint: str, n: int = 30) -> list[str]:
    """
    Uses your existing OpenAI setup to propose seed keywords.
    Keep it simple + cheap; return a flat list of strings.
    """
    if not topic_hint:
        return []
    prompt = (
        f"Generate a flat list of up to {n} UK English search keywords for the topic: {topic_hint}.\n"
        "Mix head, mid, and long-tail. Return one per line, no numbering."
    )
    # Reuse your OpenAI call (chat completions) here to keep code DRY
    api_key = os.environ.get("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
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
        # de-dupe and trim
        seen, out = set(), []
        for k in kws:
            k_low = k.lower()
            if k_low not in seen:
                seen.add(k_low); out.append(k)
        return out[:n]
    except Exception:
        return []

def keyword_research_with_volumes(topic_hint: str, database: str = "uk") -> list[dict] | None:
    seeds = ai_suggest_keywords(topic_hint, n=30)
    if not seeds:
        return None
    vols = semrush_keyword_volumes(seeds, database=database) or []
    return vols

# ---------- Specific wrappers (common reports) ----------

def semrush_backlinks_overview(target: str, target_type: str = "root_domain") -> dict | None:
    """
    Backlink overview for domain/root_domain/subdomain/url.
    Common columns (vary by plan): 'backlinks', 'refdomains', 'ascore'
    """
    # Backlinks API endpoints vary by account; this one works for most:
    rows = semrush_get(
        "backlinks_overview",
        {"target": target, "target_type": target_type}
    )
    return rows[0] if rows else None

def semrush_refdomains_count(target: str, target_type: str = "root_domain") -> int | None:
    rows = semrush_get(
        "backlinks_refdomains",
        {"target": target, "target_type": target_type, "export_columns": "refdomain"}
    )
    return len(rows) if rows is not None else None

def semrush_domain_overview(domain: str, database: str = "uk", display_date: str | None = None) -> dict | None:
    """
    Domain analytics (organic keywords & estimated traffic). 
    Legacy Analytics API often exposes: type=domain_ranks (current) and 'display_date' for history.
    Returns the first row with columns like: Dn (domain), Or (organic keywords), Ot (organic traffic).
    """
    params = {
        "domain": domain,
        "database": database,
        "export_columns": "Dn,Or,Ot"  # Domain, Organic keywords, Organic traffic (est.)
    }
    if display_date:
        params["display_date"] = display_date  # YYYY-MM-15 (Semrush expects a month day; 15 is common)
    rows = semrush_get("domain_ranks", params)
    return rows[0] if rows else None

def semrush_url_keywords_count(url: str, database: str = "uk") -> int | None:
    """
    Count keywords a specific URL ranks for (approximation).
    type=url_organic returns rows per keyword → count = len(rows).
    """
    rows = semrush_get(
        "url_organic",
        {"url": url, "database": database, "display_limit": 1_000, "export_columns": "Ph,Po"}
    )
    return len(rows) if rows is not None else None

def semrush_keyword_volumes(keywords: list[str], database: str = "uk") -> list[dict] | None:
    """
    Batch volumes for up to ~100 keywords.
    'phrase_all' typically returns: Ph (phrase), Nq (volume), Co (competition), Kd (difficulty) depending on plan.
    """
    kws = ",".join(kw.strip() for kw in keywords if kw.strip())[:5000]  # keep URL small
    rows = semrush_get(
        "phrase_all",
        {"phrase": kws, "database": database, "export_columns": "Ph,Nq,Kd"}
    )
    return rows

def semrush_domain_mom_yoy(domain: str, database: str = "uk") -> dict | None:
    """
    Pull current, previous month, and previous year (same month) Or/Ot and compute deltas.
    """
    anchor = last_full_month_anchor()
    this_m = anchor.strftime("%Y-%m-15")
    prev_m = months_ago(anchor, 1).strftime("%Y-%m-15")
    prev_y = months_ago(anchor, 12).strftime("%Y-%m-15")

    cur = semrush_domain_overview(domain, database, this_m) or {}
    mo  = semrush_domain_overview(domain, database, prev_m) or {}
    yo  = semrush_domain_overview(domain, database, prev_y) or {}

    def as_int(x): 
        try: return int(float(x))
        except: return None

    Or_cur, Ot_cur = as_int(cur.get("Or")), as_int(cur.get("Ot"))
    Or_mo,  Ot_mo  = as_int(mo.get("Or")),  as_int(mo.get("Ot"))
    Or_yo,  Ot_yo  = as_int(yo.get("Or")),  as_int(yo.get("Ot"))

    def delta(cur, old):
        if cur is None or old is None: return None
        if old == 0: return None
        return round((cur - old) / old * 100, 1)

    return {
        "Or": Or_cur, "Ot": Ot_cur,
        "Or_mom_%": delta(Or_cur, Or_mo), "Ot_mom_%": delta(Ot_cur, Ot_mo),
        "Or_yoy_%": delta(Or_cur, Or_yo), "Ot_yoy_%": delta(Ot_cur, Ot_yo),
        "_dates": {"this": this_m, "mom": prev_m, "yoy": prev_y}
    }

# ----------------------------- Content & Structure Analysis Helpers -----------------------------

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

    # Optional AI scores
    ai_scores = meta.get("ai_scores")
    ai_weight_map: Dict[str, float] = {}
    if isinstance(ai_scores, dict) and ai_scores:
        # clamp and include
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

LOW = 70  # threshold for "low" score guidance

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
    # AI-driven recs (labels map to score keys)
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

    # Specific, data-driven nudges
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

    # AI-specific suggestions
    ai = r.get("ai_findings") or {}
    for ms in ai.get("copy_suggestions", [])[:8]:
        msgs.append(f"Copy: {ms}")
    for ms in ai.get("schema_recommendations", [])[:5]:
        msgs.append(f"Schema: Consider adding {ms} JSON-LD.")
    return msgs

# ----------------------------- Analysis -----------------------------

from typing import Any  # make sure Any is imported
def analyze_page(
    url: str,
    use_ai: bool = False,
    topic_hint: Optional[str] = None,
    show_ai_debug: bool = False,     # <-- add this
) -> Dict[str, Any]:
    html, fetch_meta = get_home_html(url)
    result: Dict[str, Any] = {"_url": url, "_final_url": fetch_meta.get("final_url"), "_domain": extract_domain(url)}
    result.update(fetch_meta)

    if fetch_meta.get("error") or not html:
        return result

    soup = parse_html(html)
    base_domain = extract_domain(result.get("_final_url") or url)

    # Title & meta
    title_tag = soup.find("title")
    title = (title_tag.get_text(strip=True) if title_tag else None) or ""
    title_len = len(title)
    meta_desc = find_meta_tag(soup, name="description") or ""
    meta_desc_len = len(meta_desc)

    # H1
    h1s = [h.get_text(strip=True) for h in soup.find_all("h1")]
    h1_count = len(h1s)

    # Canonical
    canonical = None
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

    # Links
    a_tags = soup.find_all("a")
    internal = 0
    external = 0
    for a in a_tags:
        href = a.get("href") or ""
        if is_internal(href, base_domain):
            internal += 1
        elif href.startswith("http://") or href.startswith("https://"):
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
        "canonical": canonical or "",
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

       # AI analysis (optional)
    if use_ai:
        ai = ai_analyze_with_openai(visible_text, topic_hint)
    if ai:
        result["ai_scores"] = ai.get("ai_scores")
        result["ai_findings"] = ai.get("ai_findings")
    else:
        result["_ai_error"] = "OpenAI returned no result (check key/quotas)."

    # Compute sub-scores & overall
    result.update(compute_scores(result))

    # Recommendations
    recs = generate_recommendations(result)
    result["_recommendations"] = recs
    result["_issue_count"] = len(recs)

    return result

# ----------------------------- UI -----------------------------

st.set_page_config(page_title="SEO Audit & Competitor Comparator", layout="wide")

st.title("🔎 SEO Audit & Competitor Comparator")
st.caption("Audits your homepage for on-page + technical basics, optional CWV via PSI, competitor comparison, and optional Gemini AI for semantic checks.")

# --- Sidebar (define ALL inputs first) ---
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

    st.subheader("AI Analysis")
    show_ai_debug = st.checkbox("Show AI debug", value=False, key="ai_debug")
    provider = st.selectbox("AI provider", ["OpenAI (ChatGPT)", "Off"], index=0, key="ai_provider")
    use_ai = provider != "Off"
    topic_hint = st.text_input("Topic/intent hint (optional)", "", key="ai_topic_hint")

    st.subheader("Semrush (optional)")
    use_semrush = st.checkbox("Fetch Semrush insights", value=False, key="semrush_toggle")
    if use_semrush and not _get_semrush_key():
        st.warning("No SEMRUSH_API_KEY found in Secrets.")

    # IMPORTANT: define the button LAST, right here
    run_btn = st.button("Run audit", type="primary", key="run_audit_btn")

    st.divider()
    st.subheader("Google PSI API (optional)")
    st.write("Set environment var `PSI_API_KEY` before running for Core Web Vitals + Lighthouse scores.")



# --- Main trigger (must be BELOW the sidebar block) ---
if run_btn and default_domain:
    # Build targets list
    targets = [normalize_url(default_domain)]
    for line in (competitors or "").splitlines():
        line = line.strip()
        if line:
            targets.append(normalize_url(line))

    st.info(f"Auditing {len(targets)} site(s). This may take ~5–30s each depending on response time, PSI, and AI.")

    results: List[Dict[str, Any]] = []
    progress = st.progress(0.0)
    status = st.empty()

    for i, t in enumerate(targets, start=1):
        status.write(f"Fetching: {t}")

        # Core page analysis (includes AI if enabled)
        res = analyze_page(
            t,
            use_ai=use_ai,
            topic_hint=topic_hint,
            show_ai_debug=show_ai_debug,
        )

        
        # Semrush extras (optional)
        if use_semrush:
            domain = res.get("_domain")
            final_url = res.get("_final_url") or res.get("_url")

            # Backlinks & ref domains
            bl_dom = semrush_backlinks_overview(domain, "root_domain") or {}
            bl_url = semrush_backlinks_overview(final_url, "url") or {}
            rd_dom_count = semrush_refdomains_count(domain, "root_domain")
            rd_url_count = semrush_refdomains_count(final_url, "url")

            # Domain organic metrics (UK) with MoM/YoY deltas
            dom_ov = semrush_domain_mom_yoy(domain, "uk") or {}

            # URL-level keyword count (UK)
            url_kw_count = semrush_url_keywords_count(final_url, "uk")

            res["semrush"] = {
                "backlinks_domain": bl_dom,
                "backlinks_url": bl_url,
                "refdomains_domain_count": rd_dom_count,
                "refdomains_url_count": rd_url_count,
                "domain_organic_uk": dom_ov,
                "url_keywords_uk": url_kw_count,
            }

            # AI → Semrush volumes for suggested keywords (if you provided a hint)
            if topic_hint:
                res["keyword_research"] = keyword_research_with_volumes(topic_hint, "uk")

        results.append(res)
        progress.progress(i / len(targets))

    status.write("Done.")
    # ↓ Keep your Summary / Radar / Details / Downloads sections after this

    # ----- Summary Table -----
    st.subheader("Summary")

    def row_from(r: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "Domain": r.get("_domain"),
            "Final URL": r.get("_final_url"),
            "HTTP": r.get("status_code"),
            "Load (ms)": r.get("elapsed_ms"),
            "Title len": r.get("title_len"),
            "Meta len": r.get("meta_desc_len"),
            "H1s": r.get("h1_count"),
            "Alt %": int(round((r.get("img_alt_ratio", 0) or 0) * 100)),
            "Tech": r.get("score_tech"),
            "Perf": r.get("score_performance"),
            "Social": r.get("score_social"),
            "Overall": r.get("overall_score"),
            "Issues": r.get("_issue_count"),
            "AI?": "Yes" if r.get("ai_scores") else "No",
            "Error": r.get("error"),
        }

    table_rows = [row_from(r) for r in results]
    st.dataframe(table_rows, use_container_width=True)

    # ----- Radar chart of category scores -----
    st.subheader("Category Radar")
    base_cats = [
        ("Title", "score_title"),
        ("Meta", "score_meta_desc"),
        ("H1", "score_h1"),
        ("Links", "score_links"),
        ("Alt", "score_images_alt"),
        ("Tech", "score_tech"),
        ("Social", "score_social"),
        ("Perf", "score_performance"),
        ("Readability", "score_readability"),
        ("Originality", "score_originality"),
        ("Tone", "score_tone"),
        ("Headings", "score_heading_structure"),
        ("Anchors", "score_anchor_quality"),
        ("JS Reliance", "score_js"),
    ]
    ai_cats = [
        ("AI Intent", "score_ai_intent"),
        ("AI Coverage", "score_ai_coverage"),
        ("AI E-E-A-T", "score_ai_eeat"),
        ("AI Helpfulness", "score_ai_helpfulness"),
        ("AI Originality", "score_ai_originality"),
        ("AI Tone", "score_ai_tone"),
        ("AI Conversion", "score_ai_conversion"),
        ("AI Int. Links", "score_ai_internal_links"),
    ] if any(r.get("ai_scores") for r in results) else []

    cats = base_cats + ai_cats

    fig = go.Figure()
    theta = [c[0] for c in cats]
    for r in results:
        vals = [r.get(c[1], 0) for c in cats]
        fig.add_trace(go.Scatterpolar(r=vals, theta=theta, fill='toself', name=r.get("_domain")))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=520)
    st.plotly_chart(fig, use_container_width=True)

    # ----- Overall score bar chart -----
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

    # ----- Detail expanders -----
    st.subheader("Details by Site")
    for r in results:
        with st.expander(f"{r.get('_domain')} — details"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Basics**")
                st.write({
                    "Final URL": r.get("_final_url"),
                    "Status": r.get("status_code"),
                    "HTTPS": r.get("https"),
                    "Redirects": r.get("redirects"),
                    "Load (ms)": r.get("elapsed_ms"),
                    "HTML bytes": r.get("page_bytes"),
                    "Noindex": r.get("noindex"),
                })
                st.markdown("**On-page**")
                st.write({
                    "Title": r.get("title"),
                    "Title length": r.get("title_len"),
                    "Meta desc length": r.get("meta_desc_len"),
                    "H1 count": r.get("h1_count"),
                    "Canonical": r.get("canonical"),
                })
                st.markdown("**Content Quality**")
                st.write({
                    "Flesch Reading Ease": r.get("readability_fre"),
                    "Originality (TTR)": r.get("originality", {}).get("ttr"),
                    "Repeated 3-grams": r.get("originality", {}).get("repeated_trigram_ratio"),
                    "Tone (exclam/100sents)": r.get("tone", {}).get("exclamation_density"),
                    "Tone (buzz rate)": r.get("tone", {}).get("buzz_rate"),
                })
                st.markdown("**Content Stats**")
                st.write({
                    "Links (internal/external)": f"{r.get('internal_links')}/{r.get('external_links')}",
                    "Images": r.get("images"),
                    "Alt ratio": r.get("img_alt_ratio"),
                })
            with col2:
                st.markdown("**Headings**")
                st.write(r.get("headings"))
                st.markdown("**Internal Link Anchors**")
                st.write(r.get("anchor_quality"))
                st.markdown("**JS Reliance**")
                st.write(r.get("js_reliance"))
                st.markdown("**Robots/Sitemap**")
                st.write({
                    "robots.txt": r.get("robots_exists"),
                    "sitemap": r.get("sitemap_exists"),
                })
                st.markdown("**Social/Schema**")
                st.write({
                    "Open Graph": r.get("og_present"),
                    "Twitter Cards": r.get("twitter_present"),
                    "JSON-LD schema": r.get("schema_jsonld"),
                })
                if r.get("psi_scores"):
                    st.markdown("**Lighthouse Categories (mobile)**")
                    st.write(r.get("psi_scores"))
                if r.get("cwv"):
                    st.markdown("**Core Web Vitals (mobile)**")
                    st.write(r.get("cwv"))

            if r.get("ai_scores"):
                st.markdown("**AI Analysis **")
                st.write(r.get("ai_scores"))
                ai_f = r.get("ai_findings") or {}
                if ai_f:
                    if ai_f.get("missing_subtopics"):
                        st.write({"Missing subtopics": ai_f.get("missing_subtopics")})
                    if ai_f.get("copy_suggestions"):
                        st.write({"Copy suggestions": ai_f.get("copy_suggestions")[:8]})
                    if ai_f.get("schema_recommendations"):
                        st.write({"Schema recommendations": ai_f.get("schema_recommendations")})
                    if ai_f.get("faq_suggestions"):
                        st.write({"FAQ suggestions": ai_f.get("faq_suggestions")[:5]})
                    if ai_f.get("internal_link_suggestions"):
                        st.write({"Internal link suggestions": ai_f.get("internal_link_suggestions")[:8]})
            elif r.get("_ai_error"):
                        st.info(f"AI note: {r['_ai_error']}")

            st.markdown("**Recommendations**")
            recs = r.get("_recommendations", [])
            if recs:
                for m in recs:
                    st.write("• ", m)
            else:
                st.write("No critical issues detected. Nice!")

    if r.get("semrush"):
        st.markdown("**Semrush Insights**")
        sm = r["semrush"]

    colA, colB = st.columns(2)
    with colA:
        st.write({"Backlinks (domain)": semrush.get("backlinks_domain")})
        st.write({"Ref domains (domain)": semrush.get("refdomains_domain_count")})
        st.write({"Backlinks (URL)": semrush.get("backlinks_url")})
        st.write({"Ref domains (URL)": semrush.get("refdomains_url_count")})

    with colB:
        dom = semrush.get("domain_organic_uk") or {}
        st.write({
            "UK Organic keywords": dom.get("Or"),
            "UK Organic traffic": dom.get("Ot"),
            "MoM % (KW/Traffic)": (dom.get("Or_mom_%"), dom.get("Ot_mom_%")),
            "YoY % (KW/Traffic)": (dom.get("Or_yoy_%"), dom.get("Ot_yoy_%")),
            "Periods (this/mom/yoy)": dom.get("_dates"),
        })
        st.write({"URL UK Keyword count (approx)": semrush.get("url_keywords_uk")})

    if r.get("keyword_research"):
        st.markdown("**AI Keyword Research + Volumes (UK)**")
        st.dataframe(r["keyword_research"], use_container_width=True)


     # ----- Downloads -----
    st.subheader("Export")
    json_blob = json.dumps(results, indent=2)
    st.download_button(
        "Download JSON",
        data=json_blob,
        file_name="seo_audit_results.json",
        mime="application/json"
    )

    import csv
    import io
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
# integrate with a crawler or a headless browser. PSI integration is optional and requires an API key.
