# SEO Audit & Competitor Comparator — Streamlit App
# ------------------------------------------------
# Run locally:
#   1) pip install -U streamlit requests beautifulsoup4 lxml tldextract plotly python-dateutil
#   2) (optional) export PSI_API_KEY="<your-google-pagespeed-insights-api-key>"
#   3) streamlit run app.py
#
# What it does:
# - Audits a domain's homepage for on-page SEO and technical basics
# - (Optionally) pulls Core Web Vitals + Lighthouse category scores via Google PageSpeed Insights API
# - Compares multiple competitors side-by-side
# - Shows radar + bar charts and detailed factor breakdowns
# - Lets you download CSV/JSON of results

import os
import re
import json
import time
import math
import queue
import threading
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
import tldextract

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dateutil import tz

USER_AGENT = "Mozilla/5.0 (compatible; SEOAuditBot/1.0; +https://example.com/audit)"
TIMEOUT = 15
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en"}

# ----------------------------- Helpers -----------------------------

def normalize_url(domain_or_url: str) -> str:
    x = domain_or_url.strip()
    if not x:
        return x
    if not x.startswith("http://") and not x.startswith("https://"):
        x = "https://" + x
    # Strip path/query for audit focus on homepage
    return x


def http_get(url: str) -> Tuple[Optional[requests.Response], Optional[str]]:
    try:
        resp = requests.get(url, headers=HEADERS, allow_redirects=True, timeout=TIMEOUT)
        return resp, None
    except requests.RequestException as e:
        return None, str(e)


def get_home_html(url: str) -> Tuple[Optional[str], Dict]:
    meta = {
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


def get_robots_txt(domain_url: str) -> Dict:
    base = normalize_url(domain_url)
    if base.endswith("/"):
        base = base[:-1]
    robots_url = base + "/robots.txt"
    meta = {"exists": False, "sitemaps": [], "disallow_count": 0, "error": None}
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
    # Try direct /sitemap.xml if not hinted
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


def analyze_page(url: str) -> Dict:
    html, fetch_meta = get_home_html(url)
    result = {"_url": url, "_final_url": fetch_meta.get("final_url"), "_domain": extract_domain(url)}
    result.update(fetch_meta)

    if fetch_meta.get("error") or not html:
        # Bail early, keep meta only
        return result

    soup = parse_html(html)

    # Title
    title_tag = soup.find("title")
    title = (title_tag.get_text(strip=True) if title_tag else None) or ""
    title_len = len(title)

    # Meta description
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

    # Open Graph / Twitter
    og_title = find_meta_tag(soup, prop="og:title")
    og_desc = find_meta_tag(soup, prop="og:description")
    tw_title = find_meta_tag(soup, name="twitter:title")
    tw_desc = find_meta_tag(soup, name="twitter:description")

    # JSON-LD Schema
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
    base_domain = extract_domain(result.get("_final_url") or url)
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

    # Mobile meta viewport
    viewport = soup.find("meta", attrs={"name": "viewport"}) is not None

    # Noindex
    robots_meta = find_meta_tag(soup, name="robots") or ""
    is_noindex = "noindex" in robots_meta.lower()

    # Performance proxy: response time + PSI if available
    cwv = {}
    psi_scores = {}
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
                # CWV - pick percent-good or numeric where available
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

    # Compute sub-scores (0-100)
    result.update(compute_scores(result))
    return result

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
    if not w: return 0
    vowels = "aeiouy"
    syll = 0
    prev_v = False
    for ch in w:
        v = ch in vowels
        if v and not prev_v: syll += 1
        prev_v = v
    if w.endswith("e") and syll > 1: syll -= 1
    return max(1, syll)

def flesch_reading_ease(text: str) -> Optional[float]:
    words = re.findall(r"[A-Za-z']+", text)
    sents = split_sentences(text)
    if not words or not sents: return None
    w = len(words); s = max(1, len(sents))
    syll = sum(count_syllables(tok) for tok in words)
    return 206.835 - 1.015 * (w/s) - 84.6 * (syll/w)

def originality_heuristic(text: str) -> Dict:
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]
    w = len(words); unique = len(set(words))
    ttr = (unique / w) if w else 0.0
    trigrams = [tuple(words[i:i+3]) for i in range(max(0, w-2))]
    from collections import Counter
    c = Counter(trigrams)
    rep = sum(1 for k, v in c.items() if v > 1)
    rep_ratio = (rep / max(1, len(c))) if c else 0.0
    return {"ttr": round(ttr, 3), "repeated_trigram_ratio": round(rep_ratio, 3)}

def tone_heuristic(text: str) -> Dict:
    tokens = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]
    total = len(tokens)
    exclam = text.count("!")
    adverbs = sum(1 for w in tokens if w.endswith("ly") and len(w) > 3)
    buzz = {"best","amazing","revolutionary","ultimate","incredible","guaranteed","exclusive","limited"}
    buzz_count = sum(1 for w in tokens if w in buzz)
    you_rate = sum(1 for w in tokens if w in {"you","your","yours"})
    return {
        "exclamation_density": round(exclam / max(1, len(split_sentences(text))), 3),
        "adverb_rate": round(adverbs / max(1, total), 3),
        "buzz_rate": round(buzz_count / max(1, total), 3),
        "second_person_rate": round(you_rate / max(1, total), 3),
    }

def heading_audit(soup: BeautifulSoup) -> Dict:
    headings = []
    for level in range(1, 7):
        for h in soup.find_all(f"h{level}"):
            headings.append({"level": level, "text": (h.get_text(" ", strip=True) or "").strip()})
    h1 = sum(1 for h in headings if h["level"] == 1)
    h2 = sum(1 for h in headings if h["level"] == 2)
    empty = sum(1 for h in headings if not h["text"])
    dom_levels = [int(n.name[1]) for n in soup.find_all(["h1","h2","h3","h4","h5","h6"])]
    skips = sum(1 for a,b in zip(dom_levels, dom_levels[1:]) if b - a > 1)
    return {"h_total": len(headings), "h1_count": h1, "h2_count": h2, "empty_headings": empty, "level_skips": skips}

def internal_anchor_quality(soup: BeautifulSoup, base_domain: str) -> Dict:
    bad = {"click here","read more","learn more","more","here"}
    total = 0; good = 0
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        txt = (a.get_text(" ", strip=True) or "").lower()
        if is_internal(href, base_domain):
            total += 1
            if len(txt) >= 4 and txt not in bad: good += 1
    return {"internal_total": total, "descriptive_ratio": round((good/total) if total else 1.0, 3)}

def js_reliance_metrics(soup: BeautifulSoup, html_bytes: Optional[int]) -> Dict:
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


# ----------------------------- Scoring -----------------------------

def clamp(v, lo, hi):
    try:
        return max(lo, min(hi, v))
    except Exception:
        return lo


def score_title(length: int) -> int:
    # Ideal 35-65 chars
    if length == 0:
        return 0
    if 35 <= length <= 65:
        return 100
    # penalize outside range
    diff = min(abs(50 - length), 50)
    return int(round(100 - (diff * 2)))


def score_meta_desc(length: int) -> int:
    # Ideal 70-160
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
    # favor healthy internal linking (40-85%)
    ideal = 0.6
    diff = abs(ratio - ideal)
    return int(round(100 - diff * 120))


def score_images_alt(ratio: float) -> int:
    return int(round(ratio * 100))


def score_tech(meta: Dict) -> int:
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
    # Response time under 1000ms
    add((meta.get("elapsed_ms") or 9999) < 1200, 1)
    return int(round((pts / max(total, 1)) * 100))


def score_social(meta: Dict) -> int:
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


def score_performance(meta: Dict) -> int:
    # Prefer PSI performance score if available; otherwise proxy from elapsed_ms and page_bytes
    psi_perf = meta.get("psi_scores", {}).get("performance")
    if psi_perf is not None:
        return int(psi_perf)
    # fallback simple heuristic
    ms = meta.get("elapsed_ms") or 2500
    size = meta.get("page_bytes") or 600000
    s = 100
    # penalize slow
    if ms > 500:
        s -= min(60, (ms - 500) / 20)
    # penalize large HTML
    if size > 200000:
        s -= min(30, (size - 200000) / 20000)
    return int(round(clamp(s, 0, 100)))


def score_readability(fre: Optional[float]) -> int:
    if fre is None: return 50  # unknown
    # Flesch: 90-100 (very easy) .. 0-30 (college). Aim for 50–70 for web.
    if 50 <= fre <= 70: return 100
    if fre > 70: return int(max(70, min(100, 70 + (fre-70)*1)))   # slightly reward easier
    # below 50: penalize
    return int(max(0, 100 - (50 - fre) * 2))

def score_originality(h: Dict) -> int:
    ttr = h.get("ttr", 0); rep = h.get("repeated_trigram_ratio", 0)
    s = 0
    s += min(70, ttr * 100)           # encourage lexical variety
    s += max(0, 30 - rep * 100)       # penalize repetition
    return int(max(0, min(100, s)))

def score_tone(h: Dict) -> int:
    # Reward low exclamation/buzz; neutral adverb/2nd-person rates
    s = 100
    s -= min(30, h.get("exclamation_density", 0) * 60)
    s -= min(30, h.get("buzz_rate", 0) * 200)
    # gentle nudges
    s -= min(10, abs(h.get("adverb_rate", 0) - 0.06) * 100)
    s -= min(10, abs(h.get("second_person_rate", 0) - 0.01) * 100)
    return int(max(0, min(100, s)))

def score_headings(h: Dict) -> int:
    s = 100
    if h.get("h1_count", 0) == 0: s -= 40
    if h.get("h1_count", 0) > 1: s -= 20
    if h.get("h2_count", 0) == 0: s -= 15
    s -= min(25, h.get("level_skips", 0) * 10)
    s -= min(20, h.get("empty_headings", 0) * 5)
    return int(max(0, min(100, s)))

def score_internal_anchor_quality(aq: Dict) -> int:
    return int(round((aq.get("descriptive_ratio", 0) or 0) * 100))

def score_js_reliance(js: Dict) -> int:
    # Fewer scripts & decent text/HTML ratio is better
    s = 100
    s -= min(40, max(0, js.get("script_count", 0) - 5) * 4)
    s -= min(20, max(0, js.get("external_script_count", 0) - 3) * 3)
    ratio = js.get("text_to_html_ratio")
    if ratio is not None and ratio < 0.2:
        s -= min(30, (0.2 - ratio) * 200)
    return int(max(0, min(100, s)))

def compute_scores(meta: Dict) -> Dict:
    scores = {}
    scores["score_title"] = score_title(meta.get("title_len", 0))
    scores["score_meta_desc"] = score_meta_desc(meta.get("meta_desc_len", 0))
    scores["score_h1"] = score_h1(meta.get("h1_count", 0))
    scores["score_links"] = score_links(meta.get("internal_links", 0), meta.get("external_links", 0))
    scores["score_images_alt"] = score_images_alt(meta.get("img_alt_ratio", 0))
    scores["score_tech"] = score_tech(meta)
    scores["score_social"] = score_social(meta)
    scores["score_performance"] = score_performance(meta)

    # NEW
    scores["score_readability"] = score_readability(meta.get("readability_fre"))
    scores["score_originality"] = score_originality(meta.get("originality", {}))
    scores["score_tone"] = score_tone(meta.get("tone", {}))
    scores["score_heading_structure"] = score_headings(meta.get("headings", {}))
    scores["score_anchor_quality"] = score_internal_anchor_quality(meta.get("anchor_quality", {}))
    scores["score_js"] = score_js_reliance(meta.get("js_reliance", {}))

    # Weighted overall
    weights = {
        "score_title": 1.0,
        "score_meta_desc": 0.9,
        "score_h1": 0.6,
        "score_links": 0.6,
        "score_images_alt": 0.5,
        "score_tech": 1.4,
        "score_social": 0.5,
        "score_performance": 1.4,
        # NEW weights
        "score_readability": 1.0,
        "score_originality": 0.8,
        "score_tone": 0.5,
        "score_heading_structure": 1.0,
        "score_anchor_quality": 0.7,
        "score_js": 0.9,
    }
    total_w = sum(weights.values())
    overall = sum(scores[k] * w for k, w in weights.items())
    scores["overall_score"] = int(round(overall / total_w))
    scores["_weights"] = weights
    return scores



# ----------------------------- UI -----------------------------
st.set_page_config(page_title="SEO Audit & Competitor Comparator", layout="wide")

st.title("🔎 SEO Audit & Competitor Comparator")
st.caption("Audits your homepage for on-page + technical basics, pulls optional Core Web Vitals via PageSpeed Insights, and compares against competitors.")

with st.sidebar:
    st.header("Settings")
    default_domain = st.text_input("Your domain or URL", placeholder="example.com or https://example.com")
    competitors = st.text_area("Competitors (one per line)", placeholder="competitor1.com\ncompetitor2.com")
    run_btn = st.button("Run audit", type="primary")
    st.divider()
    st.subheader("Google PSI API (optional)")
    st.write("Set environment var `PSI_API_KEY` before running for Core Web Vitals + Lighthouse scores.")

if run_btn and default_domain:
    targets = [normalize_url(default_domain)]
    for line in (competitors or "").splitlines():
        line = line.strip()
        if line:
            targets.append(normalize_url(line))

    st.info(f"Auditing {len(targets)} site(s). This may take ~5–30s each depending on response time and PSI.")

    results: List[Dict] = []

    progress = st.progress(0.0)
    status = st.empty()

    for i, t in enumerate(targets, 1):
        status.write(f"Fetching: {t}")
        res = analyze_page(t)
        results.append(res)
        progress.progress(i / len(targets))

    status.write("Done.")

    # ----- Summary Table -----
    st.subheader("Summary")
    def row_from(r: Dict) -> Dict:
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
            "Error": r.get("error"),
        }

    table_rows = [row_from(r) for r in results]
    st.dataframe(table_rows, use_container_width=True)

    # ----- Radar chart of category scores -----
    st.subheader("Category Radar")
    cats = [
        ("Title", "score_title"),
        ("Meta", "score_meta_desc"),
        ("H1", "score_h1"),
        ("Links", "score_links"),
        ("Alt", "score_images_alt"),
        ("Tech", "score_tech"),
        ("Social", "score_social"),
        ("Perf", "score_performance"),
    ]
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
        labels={"x": "Domain", "y": "Overall Score"},
        text=[r.get("overall_score") for r in results],
    )
    fig2.update_traces(textposition='outside')
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
                st.markdown("**Content Stats**")
                st.write({
                    "Links (internal/external)": f"{r.get('internal_links')}/{r.get('external_links')}",
                    "Images": r.get("images"),
                    "Alt ratio": r.get("img_alt_ratio"),
                })
            with col2:
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

    # ----- Downloads -----
    st.subheader("Export")
    json_blob = json.dumps(results, indent=2)
    st.download_button("Download JSON", data=json_blob, file_name="seo_audit_results.json", mime="application/json")

    import csv
    import io
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    header = [
        "domain","final_url","overall","title","title_len","meta_desc_len","h1_count","internal","external","img_alt_ratio","tech","perf","social","status","load_ms","page_bytes"
    ]
    writer.writerow(header)
    for r in results:
        writer.writerow([
            r.get("_domain"), r.get("_final_url"), r.get("overall_score"), r.get("title"), r.get("title_len"), r.get("meta_desc_len"),
            r.get("h1_count"), r.get("internal_links"), r.get("external_links"), r.get("img_alt_ratio"), r.get("score_tech"),
            r.get("score_performance"), r.get("score_social"), r.get("status_code"), r.get("elapsed_ms"), r.get("page_bytes")
        ])
    st.download_button("Download CSV", data=csv_buf.getvalue(), file_name="seo_audit_results.csv", mime="text/csv")

else:
    st.info("Enter your domain and any competitors, then click **Run audit**.")

# ----------------------------- Notes -----------------------------
# This is a lightweight checker focused on homepage-level signals. For crawling, duplicate content, JS-rendered content,
# and deeper issues, integrate with a crawler (e.g., Screaming Frog API) or run this from a headless browser (Playwright/Puppeteer)
# to capture rendered DOM and network requests. The PSI integration is optional and requires an API key.
