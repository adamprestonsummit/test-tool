import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any
import math

# --- URL normalizer ---
def normalize_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url.strip().rstrip("/")

# --- Get Semrush key ---
def _get_semrush_key():
    return st.secrets.get("SEMRUSH_API_KEY")

# --- Semrush scoring helpers ---
def score_semrush_backlinks(bl_dom: dict | None, rd_dom_count: int | None) -> int:
    bl = 0
    ascore = None
    if isinstance(bl_dom, dict):
        try:
            bl = int(float(bl_dom.get("backlinks") or 0))
        except Exception:
            bl = 0
        try:
            ascore_val = bl_dom.get("ascore") or bl_dom.get("authority_score")
            ascore = int(float(ascore_val)) if ascore_val is not None else None
        except Exception:
            ascore = None
    rd = int(rd_dom_count or 0)
    bl_s = min(100, int((math.log10(bl + 1) / 3.0) * 100))
    rd_s = min(100, int((math.log10(rd + 1) / 3.0) * 100))
    parts = [bl_s, rd_s]
    if ascore is not None:
        parts.append(max(0, min(100, int(ascore))))
    return int(sum(parts) / len(parts)) if parts else 0

def score_semrush_organic_trend(dom_ov: dict | None) -> int:
    def pct(v):
        if v is None:
            return None
        try:
            return float(str(v).replace("%", "").strip())
        except Exception:
            return None
    if not isinstance(dom_ov, dict):
        return 50
    moms = []
    for k in ("Or_mom_%", "Ot_mom_%"):
        pv = pct(dom_ov.get(k))
        if pv is not None:
            moms.append(pv)
    mom = sum(moms) / len(moms) if moms else 0.0
    mom = max(-50.0, min(50.0, mom))
    return int(round((mom + 50.0) / 100.0 * 100))

# --- Mock analysis & Semrush calls (replace with real implementations) ---
def analyze_page(url: str, use_ai=False, topic_hint=None, show_ai_debug=False):
    return {
        "_domain": url.split("//")[-1].split("/")[0],
        "_final_url": url,
        "_url": url,
        "overall_score": 80,
        "performance_score": 75,
        "accessibility_score": 88,
        "best_practices_score": 70,
        "seo_score": 85,
        "score_title": 90,
        "score_meta_desc": 85,
        "score_h1": 80,
        "score_links": 75,
        "score_images_alt": 70,
        "score_tech": 85,
        "score_social": 65,
        "score_performance": 75,
        "score_readability": 70,
        "score_originality": 60,
        "score_tone": 80,
        "score_heading_structure": 75,
        "score_anchor_quality": 70,
        "score_js": 65,
        "semrush": {},
    }

def semrush_backlinks_overview(target, scope):
    return {"backlinks": 120, "authority_score": 55}

def semrush_refdomains_count(target, scope):
    return 30

def semrush_domain_mom_yoy(domain, country_code):
    return {"Or_mom_%": "+5%", "Ot_mom_%": "+7%"}

def semrush_url_keywords_count(url, country_code):
    return 45

def keyword_research_with_volumes(topic, country_code):
    return [{"keyword": topic + " example", "volume": 1000}]

# ----- Sidebar -----
with st.sidebar:
    st.header("Settings")
    default_domain = st.text_input("Your domain or URL", placeholder="example.com or https://example.com")
    competitors = st.text_area("Competitors (one per line)", placeholder="competitor1.com\ncompetitor2.com")

    st.subheader("AI Analysis")
    show_ai_debug = st.checkbox("Show AI debug", value=False)
    provider = st.selectbox("AI provider", ["OpenAI (ChatGPT)", "Off"], index=0)
    use_ai = provider != "Off"
    topic_hint = st.text_input("Topic/intent hint (optional)", "")

    st.subheader("Semrush (optional)")
    use_semrush = st.checkbox("Fetch Semrush insights", value=False)
    if use_semrush and not _get_semrush_key():
        st.warning("No SEMRUSH_API_KEY found in Secrets.")

    run_btn = st.button("Run audit", type="primary")
# ----- Main Execution -----
if run_btn and default_domain:
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

        # Core page analysis
        res = analyze_page(
            t,
            use_ai=use_ai,
            topic_hint=topic_hint,
            show_ai_debug=show_ai_debug,
        )

        # ----- Semrush extras -----
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

            # Add normalized scores for radar chart
            res["score_semrush_backlinks"] = score_semrush_backlinks(bl_dom, rd_dom_count)
            res["score_semrush_trend"] = score_semrush_organic_trend(dom_ov)
        else:
            res["score_semrush_backlinks"] = 0
            res["score_semrush_trend"] = 50  # neutral

        # Optional AI keyword research
        if topic_hint:
            res["keyword_research"] = keyword_research_with_volumes(topic_hint, "uk")

        results.append(res)
        progress.progress(i / len(targets))

    status.write("Done.")
if "results" in locals() and results and isinstance(results, list):

    # ----- Category Radar -----
    cats = [
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
        ("Backlinks", "score_semrush_backlinks"),     # Semrush score
        ("Organic Trend", "score_semrush_trend"),     # Semrush score
    ]

    fig = go.Figure()
    theta = [c[0] for c in cats]
    color_palette = px.colors.qualitative.Safe  # Distinct colors
    for idx, r in enumerate(results):
        vals = [int(r.get(c[1], 0) or 0) for c in cats]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=theta, fill='toself',
            name=r.get("_domain"),
            line_color=color_palette[idx % len(color_palette)]
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=520)
    st.plotly_chart(fig, use_container_width=True)

    # ----- Summary Table -----
    st.subheader("Summary Scores")
    import pandas as pd
    summary_data = []
    for r in results:
        summary_data.append({
            "Domain": r.get("_domain"),
            "Overall": r.get("overall_score"),
            "Backlinks": r.get("score_semrush_backlinks"),
            "Organic Trend": r.get("score_semrush_trend"),
        })
    df_summary = pd.DataFrame(summary_data)

    # Conditional coloring
    def color_backlinks(val):
        color = "red" if val < 40 else "orange" if val < 70 else "green"
        return f"background-color: {color}; color: white;"

    def color_trend(val):
        color = "red" if val < 40 else "orange" if val < 70 else "green"
        return f"background-color: {color}; color: white;"

    st.dataframe(
        df_summary.style.applymap(color_backlinks, subset=["Backlinks"])
                        .applymap(color_trend, subset=["Organic Trend"]),
        use_container_width=True
    )

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

            with col2:
                st.markdown("**Semrush Insights**")
                sm = r.get("semrush", {})
                st.write({
                    "Backlinks (domain)": sm.get("backlinks_domain", {}).get("backlinks", "n/a"),
                    "Referring domains (domain)": sm.get("refdomains_domain_count", "n/a"),
                    "Authority Score": sm.get("backlinks_domain", {}).get("authority_score", "n/a"),
                    "Organic traffic UK (MoM/YoY)": sm.get("domain_organic_uk", {}),
                    "Keyword count UK (URL)": sm.get("url_keywords_uk", "n/a"),
                })

            # AI Insights
            if r.get("ai_scores"):
                st.markdown("**AI Analysis**")
                st.write(r.get("ai_scores"))

            # Keyword research
            if r.get("keyword_research"):
                st.markdown("**AI Keyword Research + Volumes (UK)**")
                st.dataframe(r["keyword_research"], use_container_width=True)

else:
    st.warning("No results to display yet. Please run an audit first.")
