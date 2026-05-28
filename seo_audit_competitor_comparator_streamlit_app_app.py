from __future__ import annotations

import os
import re
import io
import csv
import json
import zipfile
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

import requests
from bs4 import BeautifulSoup
import tldextract

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
USER_AGENT = "Mozilla/5.0 (compatible; SEOAuditBot/2.0; +https://example.com/audit)"
TIMEOUT = 15
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-GB,en;q=0.9"}
SCORE_THRESHOLD = 70          # below this → flagged as issue
PRIORITY_HIGH   = 60
PRIORITY_MED    = 75

# ─────────────────────────────────────────────
# Key helpers
# ─────────────────────────────────────────────
def _key(name: str) -> Optional[str]:
    return (
        os.environ.get(name)
        or (st.secrets.get(name) if hasattr(st, "secrets") and name in st.secrets else None)
    )

def _get_semrush_key() -> Optional[str]:  return _key("SEMRUSH_API_KEY")
def _get_gemini_key()  -> Optional[str]:  return _key("GEMINI_API_KEY")

# ─────────────────────────────────────────────
# HTTP / URL utils
# ─────────────────────────────────────────────
def normalize_url(x: str) -> str:
    x = (x or "").strip()
    if not x: return x
    if not x.startswith(("http://", "https://")): x = "https://" + x
    return x

def http_get(url: str) -> Tuple[Optional[requests.Response], Optional[str]]:
    try:
        r = requests.get(url, headers=HEADERS, allow_redirects=True, timeout=TIMEOUT)
        return r, None
    except requests.RequestException as e:
        return None, str(e)

def get_html(url: str) -> Tuple[Optional[str], Dict[str, Any]]:
    meta: Dict[str, Any] = {"final_url": None, "status_code": None, "redirects": 0,
                             "https": url.startswith("https://"), "elapsed_ms": None,
                             "page_bytes": None, "error": None}
    resp, err = http_get(url)
    if err or not resp:
        meta["error"] = err or "No response"; return None, meta
    meta.update({"final_url": resp.url, "status_code": resp.status_code,
                  "redirects": len(resp.history), "https": resp.url.startswith("https://"),
                  "elapsed_ms": int(resp.elapsed.total_seconds() * 1000),
                  "page_bytes": len(resp.content) if resp.content else None})
    if resp.status_code >= 400:
        meta["error"] = f"HTTP {resp.status_code}"; return None, meta
    try:
        resp.encoding = resp.apparent_encoding or resp.encoding
        return resp.text, meta
    except Exception:
        return resp.content.decode("utf-8", errors="ignore"), meta

def parse_html(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")

def extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain

def is_internal(href: str, base: str) -> bool:
    if not href: return False
    if href.startswith(("/", "#", "mailto:", "tel:")): return True
    try: return extract_domain(href) == base
    except: return False

def find_meta(soup: BeautifulSoup, name=None, prop=None) -> Optional[str]:
    if name:
        el = soup.find("meta", attrs={"name": name})
        if el and el.get("content"): return el["content"].strip()
    if prop:
        el = soup.find("meta", attrs={"property": prop})
        if el and el.get("content"): return el["content"].strip()
    return None

def has_json_ld(soup: BeautifulSoup, raw_html: str = "") -> bool:
    for tag in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            json.loads(tag.string or "{}"); return True
        except: pass
    # Fallback regex
    if raw_html:
        import re as _re
        blobs = _re.findall(
            r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            raw_html, _re.DOTALL | _re.IGNORECASE
        )
        for b in blobs:
            try: json.loads(b.strip()); return True
            except: pass
    return False

def get_json_ld_types(soup: BeautifulSoup, raw_html: str = "") -> List[str]:
    """Extract all schema @type values. Falls back to regex on raw HTML for JS-injected schema."""
    types = []

    def _collect_types(obj, depth=0):
        if depth > 5: return
        if isinstance(obj, dict):
            t = obj.get("@type")
            if t:
                if isinstance(t, list):
                    for x in t: types.append(str(x))
                else:
                    types.append(str(t))
            for v in obj.values():
                _collect_types(v, depth+1)
        elif isinstance(obj, list):
            for item in obj:
                _collect_types(item, depth+1)

    # Strategy 1: BeautifulSoup parsed tags
    for tag in soup.find_all("script", {"type": "application/ld+json"}):
        txt = tag.string or tag.get_text()
        if txt and txt.strip():
            try:
                _collect_types(json.loads(txt))
            except Exception:
                pass

    # Strategy 2: regex on raw HTML (catches lazy-loaded / escaped schema)
    if not types and raw_html:
        import re as _re
        blobs = _re.findall(
            r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            raw_html, _re.DOTALL | _re.IGNORECASE
        )
        for blob in blobs:
            blob = blob.strip()
            if not blob: continue
            try:
                _collect_types(json.loads(blob))
            except Exception:
                # Try stripping HTML entities / CDATA
                cleaned = _re.sub(r"<!\[CDATA\[|\]\]>", "", blob).strip()
                try: _collect_types(json.loads(cleaned))
                except: pass

    # Deduplicate, preserve order
    seen = set(); result = []
    for t in types:
        if t not in seen: seen.add(t); result.append(t)
    return result

def get_robots_txt(url: str) -> Dict[str, Any]:
    meta = {"exists": False, "sitemaps": [], "disallow_count": 0, "error": None}
    robots_url = normalize_url(url).rstrip("/") + "/robots.txt"
    resp, err = http_get(robots_url)
    if err or not resp or resp.status_code >= 400:
        meta["error"] = err or f"HTTP {resp.status_code if resp else '?'}"; return meta
    meta["exists"] = True
    meta["sitemaps"] = re.findall(r"(?i)^sitemap:\s*(\S+)$", resp.text, re.M)
    meta["disallow_count"] = len(re.findall(r"(?i)^disallow:\s*(\S*)$", resp.text, re.M))
    return meta

def has_sitemap(url: str, hinted: List[str]) -> bool:
    base = normalize_url(url).rstrip("/")
    for u in set(hinted) | {base + "/sitemap.xml"}:
        resp, _ = http_get(u)
        if resp and resp.status_code < 400 and resp.content and b"<urlset" in resp.content[:4096]:
            return True
    return False

# ─────────────────────────────────────────────
# Content analysis helpers
# ─────────────────────────────────────────────
def visible_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript"]): tag.extract()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True))

def split_sentences(text: str) -> List[str]:
    return [b.strip() for b in re.split(r"(?<=[.!?])\s+", text) if b.strip()]

def count_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w: return 0
    prev_v = False; syll = 0
    for ch in w:
        v = ch in "aeiouy"
        if v and not prev_v: syll += 1
        prev_v = v
    if w.endswith("e") and syll > 1: syll -= 1
    return max(1, syll)

def flesch_reading_ease(text: str) -> Optional[float]:
    words = re.findall(r"[A-Za-z']+", text)
    sents = split_sentences(text)
    if not words or not sents: return None
    w, s = len(words), max(1, len(sents))
    syll = sum(count_syllables(t) for t in words)
    return 206.835 - 1.015 * (w / s) - 84.6 * (syll / w)

def originality_heuristic(text: str) -> Dict[str, Any]:
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]
    w = len(words); unique = len(set(words)); ttr = (unique / w) if w else 0.0
    trigrams = [tuple(words[i:i+3]) for i in range(max(0, w-2))]
    c = Counter(trigrams)
    rep = sum(1 for _, v in c.items() if v > 1)
    rep_ratio = (rep / max(1, len(c))) if c else 0.0
    return {"ttr": round(ttr, 3), "repeated_trigram_ratio": round(rep_ratio, 3)}

def tone_heuristic(text: str) -> Dict[str, Any]:
    tokens = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]
    total = len(tokens)
    buzz = {"best", "amazing", "revolutionary", "ultimate", "incredible", "guaranteed", "exclusive", "limited"}
    return {
        "exclamation_density": round(text.count("!") / max(1, len(split_sentences(text))), 3),
        "adverb_rate": round(sum(1 for w in tokens if w.endswith("ly") and len(w) > 3) / max(1, total), 3),
        "buzz_rate": round(sum(1 for w in tokens if w in buzz) / max(1, total), 3),
        "second_person_rate": round(sum(1 for w in tokens if w in {"you", "your", "yours"}) / max(1, total), 3),
    }

def heading_audit(soup: BeautifulSoup) -> Dict[str, Any]:
    headings = []
    for level in range(1, 7):
        for h in soup.find_all(f"h{level}"):
            headings.append({"level": level, "text": h.get_text(" ", strip=True).strip()})
    h1 = sum(1 for h in headings if h["level"] == 1)
    h2 = sum(1 for h in headings if h["level"] == 2)
    empty = sum(1 for h in headings if not h["text"])
    dom = [int(n.name[1]) for n in soup.find_all(["h1","h2","h3","h4","h5","h6"])]
    skips = sum(1 for a, b in zip(dom, dom[1:]) if b - a > 1)
    return {"h_total": len(headings), "h1_count": h1, "h2_count": h2,
            "empty_headings": empty, "level_skips": skips, "all_headings": headings}

def anchor_quality(soup: BeautifulSoup, base: str) -> Dict[str, Any]:
    bad = {"click here","read more","learn more","more","here"}
    total = good = 0
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        txt = (a.get_text(" ", strip=True) or "").lower()
        if is_internal(href, base):
            total += 1
            if len(txt) >= 4 and txt not in bad: good += 1
    return {"internal_total": total, "descriptive_ratio": round((good/total) if total else 1.0, 3)}

def js_reliance(soup: BeautifulSoup, html_bytes: Optional[int]) -> Dict[str, Any]:
    scripts = soup.find_all("script")
    ext = sum(1 for s in scripts if s.get("src"))
    inline_chars = sum(len(s.string or "") for s in scripts if not s.get("src"))
    text_bytes = len(visible_text(soup).encode("utf-8", errors="ignore"))
    ratio = (text_bytes / max(1, html_bytes or 0)) if html_bytes else None
    return {"script_count": len(scripts), "external_script_count": ext,
            "inline_script_chars": inline_chars,
            "text_to_html_ratio": round(ratio, 3) if ratio is not None else None}

# ─────────────────────────────────────────────
# AEO-specific checks
# ─────────────────────────────────────────────
def aeo_checks(soup: BeautifulSoup, visible: str, raw_html: str = "") -> Dict[str, Any]:
    """Answer Engine Optimisation signals"""
    # FAQ patterns
    faq_schema = any(t in ["FAQPage", "QAPage"] for t in get_json_ld_types(soup))
    faq_headings = bool(re.search(r"(?i)(frequently asked|FAQ|Q:|A:)", visible))
    question_marks = visible.count("?")

    # HowTo / Steps
    howto_schema = any(t in ["HowTo", "Recipe"] for t in get_json_ld_types(soup))
    how_to_headings = bool(re.search(r"(?i)(how to|step \d|step-by-step)", visible))

    # Featured snippet bait
    has_list_items = bool(soup.find("ul") or soup.find("ol"))
    definition_patterns = bool(re.search(r"(?i)(\bis\b.*defined as|\bmeaning of\b|\bwhat is\b)", visible[:3000]))

    # Entities & structured data
    schema_types = get_json_ld_types(soup, raw_html)
    breadcrumb = "BreadcrumbList" in schema_types
    review_schema = any("Review" in t or "AggregateRating" in t for t in schema_types)
    local_business = "LocalBusiness" in schema_types or "Organization" in schema_types
    article_schema = any("Article" in t or "BlogPosting" in t or "NewsArticle" in t for t in schema_types)

    # E-E-A-T signals
    author_signals = bool(re.search(r"(?i)(author|written by|by\s+[A-Z][a-z]+|last updated)", visible))
    cite_signals = bool(soup.find_all("cite") or re.search(r"(?i)(source:|according to|research shows|study)", visible))
    about_page = bool(re.search(r"(?i)/about|/team|/contact", str(soup)))

    # Passage / direct answer
    short_answer_signal = bool(re.search(r"(?i)(in short|in summary|the answer is|to summarize|tldr|tl;dr)", visible))

    return {
        "faq_schema": faq_schema, "faq_headings": faq_headings,
        "question_count": question_marks,
        "howto_schema": howto_schema, "how_to_headings": how_to_headings,
        "has_lists": has_list_items, "definition_patterns": definition_patterns,
        "schema_types": schema_types, "breadcrumb_schema": breadcrumb,
        "review_schema": review_schema, "local_business_schema": local_business,
        "article_schema": article_schema,
        "author_signals": author_signals, "cite_signals": cite_signals,
        "short_answer_signals": short_answer_signal,
    }

# ─────────────────────────────────────────────
# Semrush helpers
# ─────────────────────────────────────────────
SEMRUSH_BASE = "https://api.semrush.com/analytics/v1/"
SEMRUSH_DATA_BASE = "https://api.semrush.com/"   # older data.semrush endpoints

def semrush_get(report_type: str, params: dict, timeout: int = 30, base: str = SEMRUSH_BASE) -> Optional[List[dict]]:
    key = _get_semrush_key()
    if not key: return None
    q = {"type": report_type, "key": key, **params}
    try:
        r = requests.get(base, params=q, timeout=timeout)
        if r.status_code != 200 or not r.text or "ERROR" in r.text[:30].upper():
            return None
        rows = list(csv.DictReader(io.StringIO(r.text), delimiter=";"))
        return rows
    except Exception:
        return None

def _as_int(x) -> Optional[int]:
    try: return int(float(x))
    except: return None

def last_full_month(today: date = None) -> date:
    d = today or date.today()
    return (d.replace(day=1) - timedelta(days=1)).replace(day=1)

def months_ago(anchor: date, n: int) -> date:
    y = anchor.year + (anchor.month - 1 - n) // 12
    m = (anchor.month - 1 - n) % 12 + 1
    return date(y, m, 1)

def semrush_domain_overview(domain: str, database: str = "uk", display_date: str = None) -> Optional[dict]:
    params = {"domain": domain, "database": database,
              "export_columns": "Dn,Rk,Or,Ot,Oc,Ad,At,Ac,FKn,FKnp"}
    if display_date: params["display_date"] = display_date
    rows = semrush_get("domain_ranks", params, base=SEMRUSH_DATA_BASE)
    return rows[0] if rows else None

def semrush_backlinks_overview(target: str, target_type: str = "root_domain") -> Optional[dict]:
    rows = semrush_get("backlinks_overview", {"target": target, "target_type": target_type})
    if not rows: return None
    row = rows[0]
    # Normalise column name variants
    if "total" in row and "backlinks" not in row:     row["backlinks"] = row["total"]
    if "domains_num" in row and "refdomains" not in row: row["refdomains"] = row["domains_num"]
    if "ips_num" in row and "refips" not in row:      row["refips"] = row["ips_num"]
    return row

def semrush_top_anchors(target: str, limit: int = 10) -> Optional[List[dict]]:
    return semrush_get("backlinks_anchors",
                       {"target": target, "target_type": "root_domain",
                        "display_limit": limit, "export_columns": "anchor,domains_num,backlinks"})

def semrush_url_keywords(url: str, database: str = "uk", limit: int = 50) -> Optional[List[dict]]:
    return semrush_get("url_organic",
                       {"url": url, "database": database,
                        "display_limit": limit, "export_columns": "Ph,Po,Nq,Kd,Tr,Ur"},
                       base=SEMRUSH_DATA_BASE)

def semrush_domain_keywords(domain: str, database: str = "uk", limit: int = 20) -> Optional[List[dict]]:
    return semrush_get("domain_organic",
                       {"domain": domain, "database": database,
                        "display_limit": limit, "export_columns": "Ph,Po,Nq,Kd,Tr,Ur"},
                       base=SEMRUSH_DATA_BASE)

def semrush_mom_yoy(domain: str, database: str = "uk") -> Optional[dict]:
    anchor = last_full_month()
    this_m  = anchor.strftime("%Y-%m-15")
    prev_m  = months_ago(anchor, 1).strftime("%Y-%m-15")
    prev_y  = months_ago(anchor, 12).strftime("%Y-%m-15")
    cur = semrush_domain_overview(domain, database, this_m) or {}
    mo  = semrush_domain_overview(domain, database, prev_m) or {}
    yo  = semrush_domain_overview(domain, database, prev_y) or {}
    def delta(cur_v, old_v):
        if cur_v is None or old_v is None or old_v == 0: return None
        return round((cur_v - old_v) / old_v * 100, 1)
    Or_c = _as_int(cur.get("Or")); Ot_c = _as_int(cur.get("Ot"))
    Or_m = _as_int(mo.get("Or"));  Ot_m = _as_int(mo.get("Ot"))
    Or_y = _as_int(yo.get("Or"));  Ot_y = _as_int(yo.get("Ot"))
    return {"Or": Or_c, "Ot": Ot_c,
            "Rk": _as_int(cur.get("Rk")),
            "Ad": _as_int(cur.get("Ad")), "At": _as_int(cur.get("At")),
            "Or_mom_%": delta(Or_c, Or_m), "Ot_mom_%": delta(Ot_c, Ot_m),
            "Or_yoy_%": delta(Or_c, Or_y), "Ot_yoy_%": delta(Ot_c, Ot_y),
            "_dates": {"this": this_m, "mom": prev_m, "yoy": prev_y}}

# ─────────────────────────────────────────────
# Gemini AI analysis
# ─────────────────────────────────────────────
def _gemini_generate(prompt: str, debug: bool = False) -> Optional[str]:
    """Call Gemini 2.5 Flash and return the raw text response."""
    import google.generativeai as genai
    api_key = _get_gemini_key()
    if not api_key:
        if debug: st.write("DEBUG: No GEMINI_API_KEY")
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        if debug: st.write("Gemini error:", repr(e))
        return None

def ai_analyze(page_text: str, topic_hint: str = None,
               extra_context: dict = None, debug: bool = False) -> Optional[dict]:
    api_key = _get_gemini_key()
    if not api_key:
        if debug: st.write("DEBUG: No GEMINI_API_KEY")
        return None
    text = (page_text or "").strip()[:20000]
    schema = {
        "ai_scores": {
            "intent_match": 0, "topical_coverage": 0, "eeat": 0, "helpfulness": 0,
            "originality": 0, "tone_fit": 0, "conversion_copy": 0, "internal_link_opps": 0,
            "aeo_readiness": 0, "content_depth": 0
        },
        "ai_findings": {
            "missing_subtopics": [], "weak_sections": [], "entities_detected": [],
            "schema_recommendations": [], "faq_suggestions": [],
            "internal_link_suggestions": [], "copy_suggestions": [],
            "aeo_opportunities": [], "competitor_gap_notes": []
        }
    }
    prompt = (
        "You are an SEO and AEO content analyst. Return ONLY compact JSON matching this schema "
        "(no markdown fences, no preamble, no explanation):\n"
        f"{json.dumps(schema)}\n\n"
        f"Topic hint: {topic_hint or '(none)'}\n"
        f"Context (URL, headings, schema types): {json.dumps(extra_context or {})}\n\n"
        "SCORING RULES:\n"
        "- All scores 0-100.\n"
        "- aeo_readiness: how well the page lets AI/search extract a direct factual answer.\n"
        "- intent_match: does the content directly serve the user's search intent?\n"
        "- eeat: evidence of Experience, Expertise, Authoritativeness, Trustworthiness.\n"
        "- content_depth: breadth and depth of topic coverage vs what a user needs.\n\n"
        "LIST FIELDS — populate each with 5-8 concrete, page-specific, actionable items:\n"
        "- missing_subtopics: specific content themes this page is missing (e.g. 'Price comparison table').\n"
        "- faq_suggestions: user questions this page should answer directly (e.g. 'How long does installation take?').\n"
        "- aeo_opportunities: specific structural changes to win featured snippets or AI Overviews "
        "(e.g. 'Add a 50-word definition of [term] in a <p> tag directly under the H1').\n"
        "- schema_recommendations: specific schema types not yet present (e.g. 'FAQPage', 'HowTo').\n"
        "- internal_link_suggestions: objects with {anchor, target} for useful internal links.\n"
        "- copy_suggestions: specific copy improvements for conversion or clarity.\n\n"
        "Text to analyse:\n<<<\n" + text + "\n>>>"
    )
    raw = _gemini_generate(prompt, debug=debug)
    if not raw: return None
    try:
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
        start = raw.find("{"); end = raw.rfind("}")
        if start != -1 and end > start: raw = raw[start:end+1]
        return json.loads(raw)
    except Exception as e:
        if debug: st.write("Gemini JSON parse error:", repr(e))
        return None

def ai_keyword_seeds(topic: str, n: int = 30) -> List[str]:
    if not _get_gemini_key() or not topic: return []
    prompt = (
        f"Generate {n} UK English search keywords for the topic: {topic}.\n"
        "Mix head terms, mid-tail, and long-tail queries. Return one keyword per line, no numbering, no bullets."
    )
    raw = _gemini_generate(prompt)
    if not raw: return []
    kws = [ln.strip("-• ").strip() for ln in raw.splitlines() if ln.strip()]
    seen, out = set(), []
    for k in kws:
        if k.lower() not in seen: seen.add(k.lower()); out.append(k)
    return out[:n]

# ─────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────
def clamp(v, lo, hi):
    try: return max(lo, min(hi, v))
    except: return lo

def score_title(l: int) -> int:
    if l == 0: return 0
    if 35 <= l <= 65: return 100
    return int(round(clamp(100 - abs(50-l)*2, 0, 100)))

def score_meta_desc(l: int) -> int:
    if l == 0: return 0
    if 70 <= l <= 160: return 100
    return int(round(clamp(100 - abs(115-l)*0.8, 0, 100)))

def score_h1(n: int) -> int:
    return {0: 0, 1: 100}.get(n, 60 if n <= 3 else 30)

def score_links(internal: int, external: int) -> int:
    total = internal + external
    if total == 0: return 30
    return int(round(clamp(100 - abs(internal/total - 0.6)*120, 0, 100)))

def score_tech(m: dict) -> int:
    pts = total = 0
    def add(c, w):
        nonlocal pts, total; total += w
        if bool(c): pts += w
    sc = m.get("status_code")
    add(isinstance(sc, (int,float)) and sc < 400, 2)
    add(m.get("https"), 2)
    add(m.get("robots_exists"), 2)
    add(m.get("sitemap_exists"), 2)
    add(m.get("viewport_meta"), 1)
    add(not m.get("noindex"), 2)
    add((m.get("elapsed_ms") or 9999) < 1200, 1)
    add(m.get("canonical"), 1)
    return int(round((pts/max(total,1))*100))

def score_social(m: dict) -> int:
    pts = sum([bool(m.get("og_present")), bool(m.get("twitter_present")), bool(m.get("schema_jsonld"))])
    return int(round((pts/3)*100))

def score_performance(m: dict) -> int:
    psi = (m.get("psi_scores") or {}).get("performance")
    if psi is not None: return int(psi)
    ms = m.get("elapsed_ms") or 2500
    size = m.get("page_bytes") or 600000
    s = 100
    if ms > 500: s -= min(60, (ms-500)/20)
    if size > 200000: s -= min(30, (size-200000)/20000)
    return int(round(clamp(s, 0, 100)))

def score_readability(fre) -> int:
    if fre is None: return 50
    if 50 <= fre <= 70: return 100
    if fre > 70: return int(max(70, min(100, 70+(fre-70)*1)))
    return int(max(0, 100-(50-fre)*2))

def score_headings(h: dict) -> int:
    s = 100
    if h.get("h1_count",0) == 0: s -= 40
    if h.get("h1_count",0) > 1: s -= 20
    if h.get("h2_count",0) == 0: s -= 15
    s -= min(25, h.get("level_skips",0)*10)
    s -= min(20, h.get("empty_headings",0)*5)
    return int(max(0, min(100, s)))

def score_aeo(aeo: dict) -> int:
    """Score AEO readiness 0-100"""
    pts = 0
    if aeo.get("faq_schema"): pts += 20
    elif aeo.get("faq_headings"): pts += 10
    if aeo.get("howto_schema"): pts += 15
    elif aeo.get("how_to_headings"): pts += 7
    if aeo.get("has_lists"): pts += 10
    if aeo.get("definition_patterns"): pts += 10
    if aeo.get("breadcrumb_schema"): pts += 10
    if aeo.get("article_schema"): pts += 10
    if aeo.get("author_signals"): pts += 10
    if aeo.get("cite_signals"): pts += 10
    if aeo.get("short_answer_signals"): pts += 5
    # Bonus for multiple schema types
    if len(aeo.get("schema_types",[])) >= 3: pts += 5
    return min(100, pts)

def compute_scores(m: dict) -> dict:
    s = {
        "score_title":        score_title(m.get("title_len",0)),
        "score_meta_desc":    score_meta_desc(m.get("meta_desc_len",0)),
        "score_h1":           score_h1(m.get("h1_count",0)),
        "score_links":        score_links(m.get("internal_links",0), m.get("external_links",0)),
        "score_img_alt":      int(round((m.get("img_alt_ratio",0) or 0)*100)),
        "score_tech":         score_tech(m),
        "score_social":       score_social(m),
        "score_performance":  score_performance(m),
        "score_readability":  score_readability(m.get("readability_fre")),
        "score_originality":  _score_orig(m.get("originality",{})),
        "score_tone":         _score_tone(m.get("tone",{})),
        "score_headings":     score_headings(m.get("headings",{})),
        "score_anchor":       int(round((m.get("anchor_quality",{}).get("descriptive_ratio",0) or 0)*100)),
        "score_js":           _score_js(m.get("js_reliance",{})),
        "score_aeo":          score_aeo(m.get("aeo",{})),
    }
    # AI scores
    ai_weight_map = {}
    ai_scores = m.get("ai_scores")
    if isinstance(ai_scores, dict) and ai_scores:
        def cv(x):
            try: return int(max(0, min(100, float(x))))
            except: return 0
        ai_map = {
            "score_ai_intent":       ("intent_match",       0.8),
            "score_ai_coverage":     ("topical_coverage",   0.9),
            "score_ai_eeat":         ("eeat",               0.8),
            "score_ai_helpfulness":  ("helpfulness",        0.8),
            "score_ai_originality":  ("originality",        0.5),
            "score_ai_tone":         ("tone_fit",           0.4),
            "score_ai_conversion":   ("conversion_copy",    0.8),
            "score_ai_internal":     ("internal_link_opps", 0.5),
            "score_ai_aeo":          ("aeo_readiness",      0.9),
            "score_ai_depth":        ("content_depth",      0.7),
        }
        for key, (field, weight) in ai_map.items():
            s[key] = cv(ai_scores.get(field))
            ai_weight_map[key] = weight

    weights = {
        "score_title": 1.0, "score_meta_desc": 0.9, "score_h1": 0.6,
        "score_links": 0.6, "score_img_alt": 0.5, "score_tech": 1.4,
        "score_social": 0.5, "score_performance": 1.4, "score_readability": 1.0,
        "score_originality": 0.7, "score_tone": 0.4, "score_headings": 1.0,
        "score_anchor": 0.7, "score_js": 0.9, "score_aeo": 1.2,
        **ai_weight_map
    }
    total_w = sum(weights.values())
    s["overall_score"] = int(round(sum(s.get(k,0)*w for k,w in weights.items()) / (total_w or 1)))
    s["_weights"] = weights
    return s

def _score_orig(h: dict) -> int:
    return int(max(0, min(100, min(70, h.get("ttr",0)*100) + max(0, 30 - h.get("repeated_trigram_ratio",0)*100))))

def _score_tone(h: dict) -> int:
    s = 100
    s -= min(30, h.get("exclamation_density",0)*60)
    s -= min(30, h.get("buzz_rate",0)*200)
    s -= min(10, abs(h.get("adverb_rate",0)-0.06)*100)
    return int(max(0, min(100, s)))

def _score_js(js: dict) -> int:
    s = 100
    s -= min(40, max(0, js.get("script_count",0)-5)*4)
    s -= min(20, max(0, js.get("external_script_count",0)-3)*3)
    ratio = js.get("text_to_html_ratio")
    if ratio is not None and ratio < 0.2: s -= min(30, (0.2-ratio)*200)
    return int(max(0, min(100, s)))

# ─────────────────────────────────────────────
# Recommendations engine
# ─────────────────────────────────────────────
SCORE_RECS = {
    "score_title":       ("On-Page",      "HIGH",   "Rewrite <title> to 35–65 chars with primary keyword near the start. Unique per page."),
    "score_meta_desc":   ("On-Page",      "HIGH",   "Write a 70–160 char meta description with a benefit-led CTA and the main keyword naturally included."),
    "score_h1":          ("On-Page",      "HIGH",   "Use exactly one descriptive H1 that matches the page's primary search intent."),
    "score_links":       ("On-Page",      "MEDIUM", "Improve internal linking — add contextual links to key pages; aim for internal links to be ~60% of all links."),
    "score_img_alt":     ("On-Page",      "MEDIUM", "Add descriptive, concise alt text to images. Decorative images can use empty alt=''."),
    "score_headings":    ("On-Page",      "MEDIUM", "Fix heading hierarchy: one H1 → sequential H2s → H3s. Avoid skipping levels or empty headings."),
    "score_anchor":      ("On-Page",      "MEDIUM", "Replace vague link anchors ('click here', 'read more') with descriptive keyword-rich text."),
    "score_tech":        ("Technical",    "HIGH",   "Fix technical basics: enforce HTTPS, valid 200 status, robots.txt, sitemap.xml, mobile viewport, avoid noindex."),
    "score_js":          ("Technical",    "MEDIUM", "Trim JavaScript: remove unused scripts, defer non-critical JS, ensure core content is server-rendered."),
    "score_performance": ("Performance",  "HIGH",   "Speed up page: compress images, lazy-load media, minify CSS/JS, enable HTTP/2 & caching. Fix Core Web Vitals regressions."),
    "score_social":      ("Social/Schema","MEDIUM", "Add Open Graph & Twitter Card meta tags. Implement JSON-LD structured data (at minimum Organization/WebPage)."),
    "score_readability": ("Content",      "MEDIUM", "Write in plain language: shorter sentences, simpler words. Aim for Flesch Reading Ease 50–70."),
    "score_originality": ("Content",      "MEDIUM", "Reduce repetition; add unique insights, original data, examples, or proprietary analysis."),
    "score_tone":        ("Content",      "LOW",    "Dial down hype/buzzwords and exclamations. Prefer concrete benefits and specifics over superlatives."),
    "score_aeo":         ("AEO",          "HIGH",   "Improve Answer Engine Optimisation: add FAQPage/HowTo schema, use Q&A heading patterns, include direct answer summaries."),
    "score_ai_intent":   ("Content",      "HIGH",   "Ensure copy directly answers the primary search intent; add/clarify CTA and user-action paths."),
    "score_ai_coverage": ("Content",      "HIGH",   "Cover missing subtopics with dedicated H2/H3 sections, examples and supporting evidence."),
    "score_ai_eeat":     ("Content",      "HIGH",   "Expose author credentials, cite sources, add about/contact/policy pages and 'last updated' timestamps."),
    "score_ai_helpfulness":("Content",    "MEDIUM", "Add step-by-step guidance, evidence, data, worked examples, and comparisons."),
    "score_ai_aeo":      ("AEO",          "HIGH",   "Structure content so AI/voice search can extract direct answers: short definitions, numbered steps, quick summaries."),
    "score_ai_depth":    ("Content",      "MEDIUM", "Increase content depth: go beyond surface-level coverage with detailed explanations and supporting detail."),
    "score_ai_conversion":("On-Page",     "MEDIUM", "Sharpen the value proposition above the fold; add trust signals (reviews, accreditations, stats) and clear CTAs."),
}

PRIORITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

def generate_recommendations(result: dict, competitor_results: List[dict] = None) -> List[dict]:
    recs = []
    for key, (category, priority, msg) in SCORE_RECS.items():
        score_val = result.get(key)
        if score_val is None: continue
        if score_val >= SCORE_THRESHOLD: continue

        # Check if competitors do better
        comp_avg = None
        comp_note = ""
        if competitor_results:
            comp_scores = [c.get(key) for c in competitor_results if c.get(key) is not None]
            if comp_scores:
                comp_avg = round(sum(comp_scores)/len(comp_scores))
                if comp_avg > score_val + 10:
                    comp_note = f" (Competitors avg {comp_avg}/100 — they are outperforming you here.)"
                    # Elevate priority if competitors are beating you
                    if priority == "LOW": priority = "MEDIUM"
                    elif priority == "MEDIUM": priority = "HIGH"

        recs.append({
            "category":   category,
            "priority":   priority,
            "score":      score_val,
            "comp_avg":   comp_avg,
            "message":    msg + comp_note,
            "score_key":  key,
        })

    # Specific nudges
    def nudge(cat, pri, msg):
        recs.append({"category": cat, "priority": pri, "score": None, "comp_avg": None, "message": msg, "score_key": "specific"})

    if result.get("title_len",0) == 0:      nudge("On-Page","HIGH","Missing <title> tag — add one with primary keyword.")
    if result.get("meta_desc_len",0) == 0:  nudge("On-Page","HIGH","Missing meta description — add 70–160 char compelling summary.")
    if result.get("h1_count",0) == 0:       nudge("On-Page","HIGH","No H1 tag found — add one descriptive H1.")
    if result.get("title_len",0) > 65:      nudge("On-Page","MEDIUM","Title over 65 chars — trim to avoid SERP truncation.")
    if (result.get("headings",{}) or {}).get("level_skips",0) > 0:
        nudge("On-Page","MEDIUM","Heading level skips detected (e.g. H2→H4). Keep hierarchy sequential.")
    if (result.get("js_reliance",{}) or {}).get("script_count",0) > 10:
        nudge("Technical","MEDIUM","High script count (>10). Audit and remove non-essential JS.")
    aeo = result.get("aeo",{}) or {}
    if not aeo.get("faq_schema") and not aeo.get("faq_headings"):
        nudge("AEO","HIGH","No FAQ content or FAQPage schema detected. Add Q&A sections to capture featured snippets and AI answers.")
    if not aeo.get("author_signals"):
        nudge("Content","HIGH","No author/E-E-A-T signals detected. Add author byline, credentials, and publication date.")
    if not aeo.get("breadcrumb_schema"):
        nudge("Technical","MEDIUM","No BreadcrumbList schema. Add breadcrumb structured data for enhanced SERP appearance.")

    # AI findings as recs
    ai_f = result.get("ai_findings") or {}
    for s in (ai_f.get("schema_recommendations") or [])[:5]:
        nudge("Social/Schema","MEDIUM",f"Add {s} JSON-LD schema markup.")
    for s in (ai_f.get("aeo_opportunities") or [])[:5]:
        # Only add if it's substantive (more than just "voice search for X")
        if s and len(s) > 20:
            nudge("AEO","HIGH",f"AEO: {s}")

    # Sort by priority then score (worst first)
    recs.sort(key=lambda r: (PRIORITY_ORDER.get(r["priority"],99), r["score"] or 0))
    return recs

# ─────────────────────────────────────────────
# PSI / CWV
# ─────────────────────────────────────────────
def fetch_psi(url: str, strategy: str = "mobile", debug: bool = False) -> Tuple[dict, dict, dict]:
    psi_scores, cwv = {}, {}
    psi_status = {"ok": False, "strategy": strategy}
    psi_key = _key("PSI_API_KEY")
    if not psi_key:
        psi_status["api_error"] = "Missing PSI_API_KEY"; return psi_scores, cwv, psi_status
    try:
        strategies = [strategy] if strategy == "desktop" else ["mobile","desktop"]
        for strat in strategies:
            r = requests.get(
                "https://www.googleapis.com/pagespeedonline/v5/runPagespeed",
                params={"url": url, "key": psi_key,
                        "category": ["performance","seo","accessibility","best-practices"],
                        "strategy": strat},
                timeout=60
            )
            psi_status.update({"http_status": r.status_code, "strategy": strat})
            if r.status_code != 200:
                psi_status["api_error"] = r.text[:200]; continue
            data = r.json()
            if "error" in data:
                psi_status["api_error"] = (data.get("error",{}) or {}).get("message"); continue
            lh = data.get("lighthouseResult",{})
            if not lh: continue
            for k, v in (lh.get("categories",{}) or {}).items():
                if isinstance(v,dict) and v.get("score") is not None:
                    psi_scores[k] = round(float(v["score"])*100)
            metrics = data.get("loadingExperience",{}).get("metrics",{}) or {}
            def pct_good(key):
                d = (metrics.get(key,{}) or {}).get("distributions",[])
                g = next((x for x in d if x.get("min")==0), None)
                return int(round(g.get("proportion",0)*100)) if g else None
            audits = lh.get("audits",{}) or {}
            cwv = {
                "LCP_ms": (audits.get("largest-contentful-paint",{}) or {}).get("numericValue"),
                "CLS":    (audits.get("cumulative-layout-shift",{}) or {}).get("numericValue"),
                "INP_ms": (audits.get("interaction-to-next-paint",{}) or {}).get("numericValue"),
                "GOOD_LCP_%": pct_good("LARGEST_CONTENTFUL_PAINT_MS"),
                "GOOD_CLS_%": pct_good("CUMULATIVE_LAYOUT_SHIFT"),
                "GOOD_INP_%": pct_good("INTERACTION_TO_NEXT_PAINT"),
            }
            psi_status["ok"] = True; break
    except Exception as e:
        psi_status["exception"] = repr(e)
    return psi_scores, cwv, psi_status

# ─────────────────────────────────────────────
# Full page analysis
# ─────────────────────────────────────────────
def analyze_page(url: str, use_ai: bool = False, topic_hint: str = None,
                 psi_strategy: str = "mobile", debug_ai: bool = False,
                 debug_psi: bool = False) -> Dict[str, Any]:
    html, fetch_meta = get_html(url)
    result: Dict[str, Any] = {"_url": url, "_final_url": fetch_meta.get("final_url"),
                               "_domain": extract_domain(url)}
    result.update(fetch_meta)
    if fetch_meta.get("error") or not html:
        result.update(compute_scores(result))
        result["_recommendations"] = generate_recommendations(result)
        result["_issue_count"] = len(result["_recommendations"])
        return result

    soup = parse_html(html)
    base = extract_domain(result.get("_final_url") or url)

    # ── On-page basics ──
    title_tag = soup.find("title")
    title = (title_tag.get_text(strip=True) if title_tag else "") or ""
    meta_desc = find_meta(soup, name="description") or ""
    canonical_tag = soup.find("link", rel=lambda v: v and "canonical" in v)
    canonical = (canonical_tag.get("href","").strip() if canonical_tag else "") or ""
    h1s = [h.get_text(strip=True) for h in soup.find_all("h1")]
    robots_meta = find_meta(soup, name="robots") or ""
    viewport = soup.find("meta", attrs={"name":"viewport"}) is not None

    # ── Social / schema ──
    og_present = bool(find_meta(soup, prop="og:title") or find_meta(soup, prop="og:description"))
    tw_present  = bool(find_meta(soup, name="twitter:title") or find_meta(soup, name="twitter:description"))
    schema_jsonld = has_json_ld(soup, html or "")

    # ── Content ──
    vis = visible_text(soup)
    fre  = flesch_reading_ease(vis)
    orig = originality_heuristic(vis)
    tone = tone_heuristic(vis)
    hdgs = heading_audit(soup)
    anch = anchor_quality(soup, base)
    js   = js_reliance(soup, fetch_meta.get("page_bytes"))
    aeo  = aeo_checks(soup, vis, html or "")

    # ── Links / images ──
    a_tags = soup.find_all("a")
    int_links = ext_links = 0
    for a in a_tags:
        href = a.get("href") or ""
        if is_internal(href, base): int_links += 1
        elif href.startswith(("http://","https://")): ext_links += 1
    imgs = soup.find_all("img")
    img_count = len(imgs)
    img_alt_ratio = sum(1 for im in imgs if (im.get("alt") or "").strip()) / img_count if img_count else 1.0

    # ── Robots / sitemap ──
    robots = get_robots_txt(url)
    sitemap = has_sitemap(url, robots.get("sitemaps",[]))

    # ── PSI ──
    psi_scores, cwv, psi_status = fetch_psi(result.get("_final_url") or url, psi_strategy, debug_psi)
    if debug_psi: st.write({"PSI_STATUS": psi_status})

    result.update({
        "title": title, "title_len": len(title),
        "meta_desc": meta_desc, "meta_desc_len": len(meta_desc),
        "canonical": canonical,
        "h1_count": len(h1s), "h1_texts": h1s,
        "og_present": og_present, "twitter_present": tw_present,
        "schema_jsonld": schema_jsonld,
        "internal_links": int_links, "external_links": ext_links,
        "images": img_count, "img_alt_ratio": round(img_alt_ratio, 3),
        "robots_exists": robots.get("exists",False),
        "sitemap_exists": sitemap,
        "viewport_meta": viewport,
        "noindex": "noindex" in robots_meta.lower(),
        "readability_fre": None if fre is None else round(fre,1),
        "originality": orig, "tone": tone, "headings": hdgs,
        "anchor_quality": anch, "js_reliance": js,
        "aeo": aeo,
        "psi_scores": psi_scores, "cwv": cwv, "psi_status": psi_status,
    })

    # ── AI analysis ──
    if use_ai:
        headings_texts = [h.get_text(" ",strip=True) for h in soup.find_all(["h1","h2","h3"])][:30]
        extra_ctx = {"final_url": result.get("_final_url") or url, "domain": base,
                     "headings": headings_texts, "schema_types": aeo.get("schema_types",[])}
        ai_out = ai_analyze(vis, topic_hint, extra_ctx, debug=debug_ai)
        if ai_out:
            result["ai_scores"] = ai_out.get("ai_scores")
            result["ai_findings"] = ai_out.get("ai_findings")
        else:
            result["_ai_error"] = "OpenAI returned no result (check key/quotas)."

    result.update(compute_scores(result))
    result["_recommendations"] = generate_recommendations(result)
    result["_issue_count"] = len(result["_recommendations"])
    return result

# ─────────────────────────────────────────────
# CWV grading helpers
# ─────────────────────────────────────────────
def cwv_grade(metric: str, val) -> str:
    try: v = float(val)
    except: return "warn"
    if metric == "CLS": return "good" if v<=0.10 else ("warn" if v<=0.25 else "bad")
    if metric == "LCP_ms": return "good" if v<=2500 else ("warn" if v<=4000 else "bad")
    if metric == "INP_ms": return "good" if v<=200 else ("warn" if v<=500 else "bad")
    return "warn"

def score_colour(s) -> str:
    try: v = int(s)
    except: return "#aaa"
    if v >= 80: return "#22c55e"
    if v >= 60: return "#f59e0b"
    return "#ef4444"

def score_band(s) -> str:
    try: v = int(s)
    except: return "warn"
    if v >= 80: return "good"
    if v >= 60: return "warn"
    return "bad"

# ─────────────────────────────────────────────
# Export helpers
# ─────────────────────────────────────────────
def results_to_csv(results: List[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    headers = ["domain","url","overall","title_len","meta_desc_len","h1_count",
               "internal_links","external_links","img_alt_ratio","score_tech",
               "score_performance","score_social","score_aeo","score_readability",
               "score_headings","score_anchor","schema_types",
               "status_code","load_ms","page_bytes","issue_count"]
    w.writerow(headers)
    for r in results:
        aeo = r.get("aeo",{}) or {}
        w.writerow([
            r.get("_domain"), r.get("_final_url"), r.get("overall_score"),
            r.get("title_len"), r.get("meta_desc_len"), r.get("h1_count"),
            r.get("internal_links"), r.get("external_links"), r.get("img_alt_ratio"),
            r.get("score_tech"), r.get("score_performance"), r.get("score_social"),
            r.get("score_aeo"), r.get("score_readability"), r.get("score_headings"),
            r.get("score_anchor"), "|".join(aeo.get("schema_types",[])),
            r.get("status_code"), r.get("elapsed_ms"), r.get("page_bytes"),
            r.get("_issue_count"),
        ])
    return buf.getvalue()

def recs_to_csv(results: List[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["domain","category","priority","score","comp_avg","recommendation"])
    for r in results:
        domain = r.get("_domain","")
        for rec in (r.get("_recommendations") or []):
            w.writerow([domain, rec["category"], rec["priority"],
                        rec.get("score",""), rec.get("comp_avg",""), rec["message"]])
    return buf.getvalue()

def build_zip(results: List[dict]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("audit_scores.csv", results_to_csv(results))
        zf.writestr("audit_recommendations.csv", recs_to_csv(results))
        zf.writestr("audit_full.json", json.dumps(results, indent=2, default=str))
    return buf.getvalue()

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
st.set_page_config(page_title="SEO & AEO Audit Tool", layout="wide", page_icon="🔍")

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Score ring colours driven by data attributes */
.metric-card {
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 12px;
  padding: 1rem 1.2rem;
  text-align: center;
  position: relative;
}
.metric-card .label { color: #94a3b8; font-size: 0.7rem; text-transform: uppercase; letter-spacing: .08em; margin-bottom: .3rem; }
.metric-card .value { font-size: 2rem; font-weight: 700; line-height: 1; }
.metric-card .sub   { color: #64748b; font-size: 0.7rem; margin-top: .2rem; }

.badge { display:inline-block; padding:.2rem .55rem; border-radius:999px; font-size:.78rem;
         font-weight:600; margin:.18rem; border:1px solid transparent; }
.badge-good { background:#052e16; color:#4ade80; border-color:#166534; }
.badge-warn { background:#422006; color:#fbbf24; border-color:#854d0e; }
.badge-bad  { background:#450a0a; color:#f87171; border-color:#991b1b; }
.badge-info { background:#0c1a4b; color:#93c5fd; border-color:#1e40af; }
.badge-neu  { background:#1e293b; color:#94a3b8; border-color:#334155; }

.rec-card {
  background:#0f172a; border:1px solid #1e293b; border-radius:10px;
  padding:.85rem 1rem; margin-bottom:.5rem;
}
.rec-card .rec-cat { font-size:.68rem; text-transform:uppercase; letter-spacing:.08em; color:#64748b; margin-bottom:.2rem; }
.rec-card .rec-msg { color:#e2e8f0; font-size:.88rem; line-height:1.5; }
.rec-card .rec-score { font-family:'JetBrains Mono',monospace; font-size:.8rem; color:#94a3b8; }

.pill-high   { background:#450a0a; color:#f87171; border:1px solid #991b1b; border-radius:999px; padding:.15rem .6rem; font-size:.72rem; font-weight:700; }
.pill-medium { background:#422006; color:#fbbf24; border:1px solid #854d0e; border-radius:999px; padding:.15rem .6rem; font-size:.72rem; font-weight:700; }
.pill-low    { background:#052e16; color:#4ade80; border:1px solid #166534; border-radius:999px; padding:.15rem .6rem; font-size:.72rem; font-weight:700; }

.section-header { font-size:1rem; font-weight:700; color:#1e293b; margin: .8rem 0 .5rem 0; padding-bottom:.3rem; border-bottom:1px solid rgba(0,0,0,0.12); }
.kv-row { display:flex; justify-content:space-between; padding:.2rem 0; border-bottom:1px solid rgba(0,0,0,0.06); }
.kv-row .k { color:#64748b; font-size:.82rem; }
.kv-row .v { color:#1e293b; font-size:.82rem; font-weight:600; font-family:'JetBrains Mono',monospace; }

.comp-banner { background:#1e293b; border-radius:8px; padding:.6rem 1rem; margin:.4rem 0; font-size:.82rem; color:#94a3b8; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🔍 SEO & AEO Audit")
    st.markdown("---")

    client_url = st.text_input("Client URL *", placeholder="https://example.com/page",
                                help="The page you want to audit and improve")
    competitors_raw = st.text_area("Competitor URLs (one per line)",
                                    placeholder="https://competitor1.com/page\nhttps://competitor2.com/page",
                                    height=100)

    st.markdown("---")
    st.markdown("#### AI Analysis (Gemini 2.5 Flash)")
    gemini_ok = bool(_get_gemini_key())
    st.caption("✅ Gemini key found" if gemini_ok else "❌ No key — set GEMINI_API_KEY")
    use_ai   = st.checkbox("Enable Gemini AI analysis", value=False, disabled=not gemini_ok)
    topic_hint = st.text_input("Topic / intent hint", placeholder="e.g. buy running shoes UK",
                                disabled=not use_ai)
    debug_ai = st.checkbox("Debug AI", value=False)

    st.markdown("---")
    st.markdown("#### PageSpeed Insights")
    psi_ok = bool(_key("PSI_API_KEY"))
    st.caption("✅ PSI key found" if psi_ok else "❌ No PSI key — set PSI_API_KEY")
    psi_strategy = st.selectbox("Strategy", ["mobile","desktop"])
    debug_psi = st.checkbox("Debug PSI", value=False)

    st.markdown("---")
    st.markdown("#### Semrush")
    sm_ok = bool(_get_semrush_key())
    st.caption("✅ Semrush key found" if sm_ok else "❌ No key — set SEMRUSH_API_KEY")
    use_semrush = st.checkbox("Fetch Semrush data", value=False)
    semrush_db  = st.selectbox("Database", ["uk","us","au","ca","de","fr"], index=0)

    st.markdown("---")
    run_btn = st.button("▶ Run Audit", type="primary", use_container_width=True)

# ── Header ──
st.markdown("# 🔍 SEO & AEO Audit Dashboard")
st.caption("On-page · Technical · AEO · E-E-A-T · Content · Performance · Semrush · Competitor comparison")

if not client_url:
    st.info("Enter your client URL in the sidebar (and competitor URLs to benchmark against), then click **Run Audit**.")
    st.stop()

# ── Run audit when button clicked — store in session_state ──
if run_btn:
    targets = [normalize_url(client_url)]
    for line in (competitors_raw or "").splitlines():
        line = line.strip()
        if line: targets.append(normalize_url(line))

    is_client_flags = [True] + [False]*(len(targets)-1)
    label_list = ["🟢 Client"] + [f"🔵 Comp {i}" for i in range(1, len(targets))]

    st.markdown(f"**Auditing {len(targets)} URL(s)...**")
    progress = st.progress(0.0)
    status_ph = st.empty()
    _results: List[Dict[str, Any]] = []

    for i, (url, is_cli, lbl) in enumerate(zip(targets, is_client_flags, label_list), 1):
        status_ph.write(f"⏳ {lbl}: {url}")
        res = analyze_page(url, use_ai=use_ai, topic_hint=topic_hint,
                           psi_strategy=psi_strategy, debug_ai=debug_ai, debug_psi=debug_psi)
        res["_label"] = lbl; res["_is_client"] = is_cli

        if use_semrush and sm_ok:
            _domain = res.get("_domain","")
            final_url = res.get("_final_url") or url
            res["semrush"] = {
                "domain_organic":  semrush_mom_yoy(_domain, semrush_db),
                "backlinks":       semrush_backlinks_overview(_domain, "root_domain"),
                "top_anchors":     semrush_top_anchors(_domain, 10),
                "url_keywords":    semrush_url_keywords(final_url, semrush_db, 50),
                "domain_keywords": semrush_domain_keywords(_domain, semrush_db, 20),
            }
            if topic_hint:
                seeds = ai_keyword_seeds(topic_hint, 20)
                if seeds:
                    kw_str = ",".join(seeds[:20])
                    res["semrush"]["keyword_research"] = semrush_get(
                        "phrase_all", {"phrase": kw_str, "database": semrush_db, "export_columns": "Ph,Nq,Kd"},
                        base=SEMRUSH_DATA_BASE
                    )

        _results.append(res)
        progress.progress(i / len(targets))

    status_ph.empty(); progress.empty()

    _client = _results[0]
    _comps  = _results[1:]
    _client["_recommendations"] = generate_recommendations(_client, _comps)

    # Persist to session state
    st.session_state["audit_results"]  = _results
    st.session_state["audit_topic"]    = topic_hint
    st.session_state["audit_semrush"]  = use_semrush

# ── Render from session state (survives widget reruns) ──
if "audit_results" not in st.session_state:
    st.stop()

results = st.session_state["audit_results"]
client = results[0]
comps  = results[1:]

# ─────────────────────────────────────────────
# DASHBOARD TABS
# ─────────────────────────────────────────────
col_hdr, col_clear = st.columns([5,1])
with col_clear:
    if st.button("🗑️ Clear & reset", use_container_width=True):
        for k in ["audit_results","audit_topic","audit_semrush"]:
            st.session_state.pop(k, None)
        st.rerun()

tabs = st.tabs(["📊 Overview", "🔎 Page Details", "⚡ AEO & Schema",
                "📈 Semrush", "🏆 Recommendations", "📥 Export"])

# ──────────────────────────────────────────────
# TAB 1 — OVERVIEW DASHBOARD
# ──────────────────────────────────────────────
with tabs[0]:
    st.markdown("### Score Overview")

    SCORE_GROUPS = [
        ("Overall",        "overall_score"),
        ("Technical",      "score_tech"),
        ("Performance",    "score_performance"),
        ("Content",        "score_readability"),
        ("AEO",            "score_aeo"),
        ("Social/Schema",  "score_social"),
        ("On-Page",        "score_title"),
        ("Headings",       "score_headings"),
    ]

    # Score cards row for client
    cols = st.columns(len(SCORE_GROUPS))
    for col, (lbl, key) in zip(cols, SCORE_GROUPS):
        val = client.get(key) or 0
        colour = score_colour(val)
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">{lbl}</div>
              <div class="value" style="color:{colour}">{val}</div>
              <div class="sub">/100</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Radar chart — all sites
    st.markdown("### Competitor Comparison Radar")
    radar_labels = ["Technical","Performance","AEO","Content","Social","Headings","Anchors","Overall"]
    radar_keys   = ["score_tech","score_performance","score_aeo","score_readability",
                    "score_social","score_headings","score_anchor","overall_score"]
    fig_radar = go.Figure()
    colours = [
        ("#22d3ee", "rgba(34,211,238,0.15)"),
        ("#f472b6", "rgba(244,114,182,0.15)"),
        ("#a78bfa", "rgba(167,139,250,0.15)"),
        ("#fb923c", "rgba(251,146,60,0.15)"),
        ("#4ade80", "rgba(74,222,128,0.15)"),
        ("#fbbf24", "rgba(251,191,36,0.15)"),
    ]
    for r, (col, fill) in zip(results, colours):
        vals = [r.get(k) or 0 for k in radar_keys]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=radar_labels+[radar_labels[0]],
            fill="toself", fillcolor=fill, line=dict(color=col, width=2),
            name=r.get("_label","")
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(range=[0,100], gridcolor="#1e293b"),
                   angularaxis=dict(gridcolor="#1e293b"), bgcolor="#0f172a"),
        paper_bgcolor="#0a0f1e", font=dict(color="#e2e8f0", size=11),
        legend=dict(bgcolor="#0f172a", bordercolor="#1e293b"),
        margin=dict(l=60,r=60,t=40,b=40), height=380
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Bar chart — score comparison table
    st.markdown("### Score Breakdown by Site")
    score_df_data = {"Metric": [l for l,_ in SCORE_GROUPS]}
    for r in results:
        score_df_data[r.get("_label",r.get("_domain",""))] = [r.get(k) or 0 for _,k in SCORE_GROUPS]
    score_df = pd.DataFrame(score_df_data).set_index("Metric")
    fig_bar = px.bar(score_df.reset_index().melt(id_vars="Metric", var_name="Site", value_name="Score"),
                     x="Metric", y="Score", color="Site", barmode="group",
                     color_discrete_sequence=[c[0] for c in colours],
                     template="plotly_dark")
    fig_bar.update_layout(paper_bgcolor="#0a0f1e", plot_bgcolor="#0a0f1e",
                           legend=dict(bgcolor="#0f172a"), height=300,
                           margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

    # Quick summary table
    st.markdown("### Quick Stats")
    summary_rows = []
    for r in results:
        aeo_d = r.get("aeo",{}) or {}
        summary_rows.append({
            "Site": r.get("_label",""), "URL": r.get("_final_url") or r.get("_url",""),
            "Overall": r.get("overall_score"), "Tech": r.get("score_tech"),
            "Perf": r.get("score_performance"), "AEO": r.get("score_aeo"),
            "Status": r.get("status_code"), "Load ms": r.get("elapsed_ms"),
            "Schema types": ", ".join(aeo_d.get("schema_types",[])) or "—",
            "Issues": r.get("_issue_count"),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────
# TAB 2 — PAGE DETAILS
# ──────────────────────────────────────────────
with tabs[1]:
    sel_label = st.selectbox("Select site to inspect", [r.get("_label","") for r in results])
    res = next((r for r in results if r.get("_label") == sel_label), results[0])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>On-Page Basics</div>", unsafe_allow_html=True)
        for k, v in [
            ("Title", res.get("title") or "—"),
            ("Title length", f"{res.get('title_len',0)} chars"),
            ("Meta description", (res.get("meta_desc") or "")[:80] + ("…" if len(res.get("meta_desc",""))>80 else "") or "—"),
            ("Meta desc length", f"{res.get('meta_desc_len',0)} chars"),
            ("H1 count", res.get("h1_count","—")),
            ("H1 text", (res.get("h1_texts") or ["—"])[0][:60]),
            ("Canonical", res.get("canonical") or "—"),
            ("Noindex", "⚠️ YES" if res.get("noindex") else "No"),
        ]:
            st.markdown(f"<div class='kv-row'><span class='k'>{k}</span><span class='v'>{v}</span></div>",
                        unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Technical</div>", unsafe_allow_html=True)
        for k, v in [
            ("HTTPS", "✅" if res.get("https") else "❌"),
            ("Status code", res.get("status_code","—")),
            ("Redirects", res.get("redirects","—")),
            ("Load time", f"{res.get('elapsed_ms','—')} ms"),
            ("Page size", f"{round((res.get('page_bytes') or 0)/1024)} KB"),
            ("robots.txt", "✅" if res.get("robots_exists") else "❌"),
            ("Sitemap", "✅" if res.get("sitemap_exists") else "❌"),
            ("Viewport meta", "✅" if res.get("viewport_meta") else "❌"),
        ]:
            st.markdown(f"<div class='kv-row'><span class='k'>{k}</span><span class='v'>{v}</span></div>",
                        unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-header'>Content Quality</div>", unsafe_allow_html=True)
        for k, v in [
            ("Flesch Reading Ease", res.get("readability_fre","—")),
            ("Word diversity (TTR)", (res.get("originality",{}) or {}).get("ttr","—")),
            ("Repeated 3-grams", (res.get("originality",{}) or {}).get("repeated_trigram_ratio","—")),
            ("Buzz word rate", (res.get("tone",{}) or {}).get("buzz_rate","—")),
            ("Exclamation density", (res.get("tone",{}) or {}).get("exclamation_density","—")),
        ]:
            st.markdown(f"<div class='kv-row'><span class='k'>{k}</span><span class='v'>{v}</span></div>",
                        unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Heading Structure</div>", unsafe_allow_html=True)
        hdgs = res.get("headings",{}) or {}
        for k, v in [
            ("Total headings", hdgs.get("h_total","—")),
            ("H1 count", hdgs.get("h1_count","—")),
            ("H2 count", hdgs.get("h2_count","—")),
            ("Empty headings", hdgs.get("empty_headings","—")),
            ("Level skips", hdgs.get("level_skips","—")),
        ]:
            st.markdown(f"<div class='kv-row'><span class='k'>{k}</span><span class='v'>{v}</span></div>",
                        unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Links & Images</div>", unsafe_allow_html=True)
        for k, v in [
            ("Internal links", res.get("internal_links","—")),
            ("External links", res.get("external_links","—")),
            ("Images", res.get("images","—")),
            ("Alt text coverage", f"{round((res.get('img_alt_ratio',0) or 0)*100)}%"),
            ("Descriptive anchors", f"{round(((res.get('anchor_quality',{}) or {}).get('descriptive_ratio',0) or 0)*100)}%"),
        ]:
            st.markdown(f"<div class='kv-row'><span class='k'>{k}</span><span class='v'>{v}</span></div>",
                        unsafe_allow_html=True)

    # CWV
    cwv_d = res.get("cwv",{}) or {}
    psi_d = res.get("psi_scores",{}) or {}
    if cwv_d or psi_d:
        st.markdown("<div class='section-header'>Core Web Vitals & Lighthouse</div>", unsafe_allow_html=True)
        cw1, cw2 = st.columns(2)
        with cw1:
            if psi_d:
                for metric, val in psi_d.items():
                    grade = "good" if val>=90 else ("warn" if val>=50 else "bad")
                    cls_map = {"good":"badge-good","warn":"badge-warn","bad":"badge-bad"}
                    st.markdown(f"<span class='badge {cls_map[grade]}'>{metric.upper()}: {val}</span>",
                                unsafe_allow_html=True)
        with cw2:
            if cwv_d:
                for label, key in [("LCP","LCP_ms"),("CLS","CLS"),("INP","INP_ms")]:
                    v = cwv_d.get(key)
                    if v is not None:
                        g = cwv_grade(key, v)
                        cls_map = {"good":"badge-good","warn":"badge-warn","bad":"badge-bad"}
                        st.markdown(f"<span class='badge {cls_map[g]}'>{label}: {round(float(v),2)}</span>",
                                    unsafe_allow_html=True)

    # AI analysis
    ai_f = res.get("ai_findings") or {}
    ai_s = res.get("ai_scores") or {}
    if ai_s or ai_f:
        st.markdown("<div class='section-header'>AI Content Analysis</div>", unsafe_allow_html=True)
        if ai_s:
            ai_cols = st.columns(5)
            for i, (lbl, key) in enumerate([
                ("Intent","intent_match"),("Coverage","topical_coverage"),
                ("E-E-A-T","eeat"),("Helpfulness","helpfulness"),
                ("AEO","aeo_readiness"),("Depth","content_depth"),
                ("Conversion","conversion_copy"),("Tone","tone_fit"),
            ]):
                val = ai_s.get(key)
                if val is None: continue
                col = ai_cols[i % 5]
                colour = score_colour(val)
                with col:
                    st.markdown(f"""<div class="metric-card">
                      <div class="label">{lbl}</div>
                      <div class="value" style="color:{colour}">{val}</div></div>""",
                        unsafe_allow_html=True)
        if ai_f.get("missing_subtopics"):
            st.markdown("**Content themes to add**")
            for x in (ai_f.get("missing_subtopics") or [])[:8]: st.markdown(f"- {x}")
        if ai_f.get("faq_suggestions"):
            st.markdown("**FAQ / AEO question suggestions**")
            for x in (ai_f.get("faq_suggestions") or [])[:8]: st.markdown(f"- {x}")
        if ai_f.get("aeo_opportunities"):
            st.markdown("**AEO opportunities**")
            for x in (ai_f.get("aeo_opportunities") or [])[:8]: st.markdown(f"- {x}")
        if ai_f.get("internal_link_suggestions"):
            st.markdown("**Internal link suggestions**")
            for it in (ai_f.get("internal_link_suggestions") or [])[:8]:
                if isinstance(it, dict):
                    st.markdown(f"- **{it.get('anchor','')}** → `{it.get('target','')}`")
                else:
                    st.markdown(f"- {it}")

# ──────────────────────────────────────────────
# TAB 3 — AEO & SCHEMA
# ──────────────────────────────────────────────
with tabs[2]:
    st.markdown("### AEO & Structured Data Comparison")
    st.caption("Answer Engine Optimisation signals for featured snippets, AI Overviews, and voice search")

    aeo_checks_list = [
        ("FAQPage Schema",      "faq_schema"),
        ("FAQ Headings/Content","faq_headings"),
        ("HowTo Schema",        "howto_schema"),
        ("How-To Headings",     "how_to_headings"),
        ("List Content",        "has_lists"),
        ("Definition Patterns", "definition_patterns"),
        ("BreadcrumbList",      "breadcrumb_schema"),
        ("Article Schema",      "article_schema"),
        ("Review Schema",       "review_schema"),
        ("LocalBusiness/Org",   "local_business_schema"),
        ("Author Signals",      "author_signals"),
        ("Citation Signals",    "cite_signals"),
        ("Direct Answer Signals","short_answer_signals"),
    ]

    aeo_rows = {"Signal": [l for l,_ in aeo_checks_list]}
    for r in results:
        aeo_d = r.get("aeo",{}) or {}
        cells = []
        for _, key in aeo_checks_list:
            cells.append("✅" if aeo_d.get(key) else "❌")
        aeo_rows[r.get("_label","")] = cells

    aeo_df = pd.DataFrame(aeo_rows).set_index("Signal")
    st.dataframe(aeo_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### Schema Types Detected")
    for r in results:
        aeo_d = r.get("aeo",{}) or {}
        types = aeo_d.get("schema_types",[])
        lbl = r.get("_label","")
        if types:
            badges = " ".join(f"<span class='badge badge-info'>{t}</span>" for t in types)
            st.markdown(f"**{lbl}**: {badges}", unsafe_allow_html=True)
        else:
            st.markdown(f"**{lbl}**: <span class='badge badge-bad'>No JSON-LD detected</span>",
                        unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### AEO Score Breakdown")
    aeo_score_data = {"Site": [r.get("_label","") for r in results],
                      "AEO Score": [r.get("score_aeo",0) for r in results],
                      "Questions on page": [r.get("aeo",{}).get("question_count",0) for r in results]}
    aeo_fig = px.bar(pd.DataFrame(aeo_score_data), x="Site", y="AEO Score",
                     color="Site", color_discrete_sequence=["#22d3ee","#f472b6","#a78bfa","#fb923c"],
                     template="plotly_dark", range_y=[0,100])
    aeo_fig.update_layout(paper_bgcolor="#0a0f1e", plot_bgcolor="#0a0f1e", height=280,
                           margin=dict(l=20,r=20,t=20,b=20), showlegend=False)
    st.plotly_chart(aeo_fig, use_container_width=True)

# ──────────────────────────────────────────────
# TAB 4 — SEMRUSH
# ──────────────────────────────────────────────
with tabs[3]:
    if not use_semrush:
        st.info("Enable 'Fetch Semrush data' in the sidebar and re-run the audit to see organic, backlink and keyword data.")
    elif not sm_ok:
        st.warning("No SEMRUSH_API_KEY found. Add it to Streamlit secrets or as an environment variable.")
    else:
        st.markdown("### Organic Visibility")
        org_rows = []
        for r in results:
            sm = r.get("semrush",{}) or {}
            dom_ov = sm.get("domain_organic") or {}
            def _fmt(v): return "—" if v is None else v
            org_rows.append({
                "Site":              r.get("_label",""),
                "Organic Keywords":  _fmt(dom_ov.get("Or")),
                "Organic Traffic":   _fmt(dom_ov.get("Ot")),
                "Domain Rank":       _fmt(dom_ov.get("Rk")),
                "Traffic MoM %":     _fmt(dom_ov.get("Ot_mom_%")),
                "Traffic YoY %":     _fmt(dom_ov.get("Ot_yoy_%")),
                "Keywords MoM %":    _fmt(dom_ov.get("Or_mom_%")),
            })
        has_organic = any(
            org_rows[i].get("Organic Keywords","—") != "—"
            for i in range(len(org_rows))
        )
        if has_organic:
            st.dataframe(pd.DataFrame(org_rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No Semrush organic data returned for these domains. Check your API key, domain format (e.g. example.com not https://example.com), and that the SEMRUSH_API_KEY has the Analytics API enabled.")

        st.markdown("---")
        st.markdown("### Backlink Overview")
        bl_rows = []
        for r in results:
            sm = r.get("semrush",{}) or {}
            bl = sm.get("backlinks") or {}
            bl_rows.append({
                "Site": r.get("_label",""),
                "Total Backlinks": bl.get("backlinks","—"),
                "Referring Domains": bl.get("refdomains","—"),
                "Referring IPs": bl.get("refips","—"),
                "Follow": bl.get("follows","—"),
                "Nofollow": bl.get("nofollows","—"),
            })
        if any(r.get("Total Backlinks","—") != "—" for r in bl_rows):
            st.dataframe(pd.DataFrame(bl_rows), use_container_width=True, hide_index=True)

        # URL keywords for client
        st.markdown("---")
        st.markdown("### Client URL — Top Ranking Keywords")
        cli_sm = client.get("semrush",{}) or {}
        url_kws = cli_sm.get("url_keywords") or []
        if url_kws:
            kw_df = pd.DataFrame(url_kws)
            kw_df.columns = [c.strip() for c in kw_df.columns]
            st.dataframe(kw_df.head(50), use_container_width=True, hide_index=True)
        else:
            st.info("No URL-level keyword data returned for this page.")

        # Keyword research
        kw_research = cli_sm.get("keyword_research") or []
        if kw_research:
            st.markdown("---")
            st.markdown("### Keyword Research (topic-based)")
            st.caption(f"Based on topic hint: *{topic_hint}*")
            kw_res_df = pd.DataFrame(kw_research)
            if not kw_res_df.empty:
                st.dataframe(kw_res_df.head(50), use_container_width=True, hide_index=True)

        # Top anchors
        cli_anchors = cli_sm.get("top_anchors") or []
        if cli_anchors:
            st.markdown("---")
            st.markdown("### Client Domain — Top Backlink Anchors")
            st.dataframe(pd.DataFrame(cli_anchors).head(10), use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────
# TAB 5 — RECOMMENDATIONS
# ──────────────────────────────────────────────
with tabs[4]:
    st.markdown("### Prioritised Recommendations for Client")
    st.caption("Sorted by impact. Competitor data is used to elevate priority where you're being outperformed.")

    recs = client.get("_recommendations") or []
    if not recs:
        st.success("🎉 No critical issues detected for the client page.")
    else:
        # Filters
        f1, f2 = st.columns(2)
        with f1:
            cat_filter = st.multiselect("Filter by category",
                                         sorted(set(r["category"] for r in recs)), default=[])
        with f2:
            pri_filter = st.multiselect("Filter by priority",
                                         ["HIGH","MEDIUM","LOW"], default=[])

        filtered = [r for r in recs
                    if (not cat_filter or r["category"] in cat_filter)
                    and (not pri_filter or r["priority"] in pri_filter)]

        # Summary bar
        highs   = sum(1 for r in recs if r["priority"]=="HIGH")
        mediums = sum(1 for r in recs if r["priority"]=="MEDIUM")
        lows    = sum(1 for r in recs if r["priority"]=="LOW")
        st.markdown(
            f"<span class='pill-high'>HIGH: {highs}</span> &nbsp;"
            f"<span class='pill-medium'>MEDIUM: {mediums}</span> &nbsp;"
            f"<span class='pill-low'>LOW: {lows}</span>",
            unsafe_allow_html=True
        )
        st.markdown("")

        for rec in filtered:
            pri_cls = {"HIGH":"pill-high","MEDIUM":"pill-medium","LOW":"pill-low"}.get(rec["priority"],"pill-low")
            comp_note = f" &nbsp;|&nbsp; Comp avg: <b>{rec['comp_avg']}</b>" if rec.get("comp_avg") else ""
            score_txt = f"Score: <b>{rec['score']}/100</b>" if rec.get("score") is not None else ""
            st.markdown(f"""
            <div class="rec-card">
              <div class="rec-cat">{rec['category']} &nbsp;
                <span class="{pri_cls}">{rec['priority']}</span>
              </div>
              <div class="rec-msg">{rec['message']}</div>
              <div class="rec-score">{score_txt}{comp_note}</div>
            </div>""", unsafe_allow_html=True)

    # Competitor recs overview
    if comps:
        st.markdown("---")
        st.markdown("### Competitor Issue Summary")
        st.caption("Issues found on competitor pages — opportunities to maintain your advantage or learn from theirs")
        for comp in comps:
            with st.expander(f"{comp.get('_label','')} — {comp.get('_domain','')}"):
                comp_recs = comp.get("_recommendations") or []
                if comp_recs:
                    for rec in comp_recs[:15]:
                        pri_cls = {"HIGH":"pill-high","MEDIUM":"pill-medium","LOW":"pill-low"}.get(rec["priority"],"pill-low")
                        st.markdown(f"""<div class="rec-card">
                          <div class="rec-cat">{rec['category']} <span class="{pri_cls}">{rec['priority']}</span></div>
                          <div class="rec-msg">{rec['message']}</div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.success("No critical issues detected for this competitor.")

# ──────────────────────────────────────────────
# TAB 6 — EXPORT
# ──────────────────────────────────────────────
with tabs[5]:
    st.markdown("### Download Audit Results")
    st.caption("All exports include client and competitor data")

    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        st.markdown("#### 📋 Scores CSV")
        st.caption("Score breakdown for all audited URLs")
        st.download_button("Download Scores CSV",
                            data=results_to_csv(results),
                            file_name="seo_aeo_scores.csv",
                            mime="text/csv",
                            use_container_width=True)

    with dl2:
        st.markdown("#### 🎯 Recommendations CSV")
        st.caption("All prioritised recommendations for every URL")
        st.download_button("Download Recommendations CSV",
                            data=recs_to_csv(results),
                            file_name="seo_aeo_recommendations.csv",
                            mime="text/csv",
                            use_container_width=True)

    with dl3:
        st.markdown("#### 📦 Full ZIP")
        st.caption("Scores, recommendations and raw JSON in one file")
        st.download_button("Download Full ZIP",
                            data=build_zip(results),
                            file_name="seo_aeo_audit.zip",
                            mime="application/zip",
                            use_container_width=True)

    st.markdown("---")
    st.markdown("#### Raw JSON")
    with st.expander("Show raw audit data (JSON)"):
        st.json(results)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("SEO & AEO Audit Tool · On-page · Technical · AEO · E-E-A-T · Core Web Vitals · Semrush · OpenAI · v2.0")
