# ----- Imports -----
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any

# ===== Helper functions =====
def normalize_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url.strip().rstrip("/")

def _get_semrush_key():
    return st.secrets.get("SEMRUSH_API_KEY")

# Placeholder functions (replace with real API calls)
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
        "content_depth_score": 65 if use_ai else None,
        "entity_coverage_score": 60 if use_ai else None,
        "semrush": {},
    }

# Mock Semrush functions (replace with actual Semrush API calls)
def semrush_backlinks_overview(target, scope):
    return {"backlinks": 100, "authority_score": 55}

def semrush_refdomains_count(target, scope):
    return 25

def semrush_domain_mom_yoy(domain, country_code):
    return {"mom_change": "+5%", "yoy_change": "+12%"}

def semrush_url_keywords_count(url, country_code):
    return 45

def keyword_research_with_volumes(topic, country_code):
    return [{"keyword": topic + " example", "volume": 1000}]

# ----- Sidebar -----
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

    run_btn = st.button("Run audit", type="primary", key="run_audit_btn")
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

            # Add Semrush scores to radar chart metrics
            res["semrush_backlinks_score"] = bl_dom.get("backlinks", 0)
            res["semrush_traffic_change_score"] = (
                float(dom_ov.get("mom_change", "0").replace("%", "")) if "mom_change" in dom_ov else 0
            )

        # Optional AI keyword research
        if topic_hint:
            res["keyword_research"] = keyword_research_with_volumes(topic_hint, "uk")

        results.append(res)
        progress.progress(i / len(targets))

    status.write("Done.")
# ----- Only proceed if results exist -----
if "results" in locals() and results and isinstance(results, list):

    # Base categories
    base_cats = [
        ("Performance", "performance_score"),
        ("Accessibility", "accessibility_score"),
        ("Best Practices", "best_practices_score"),
        ("SEO", "seo_score"),
        ("Backlinks", "semrush_backlinks_score"),           # NEW from Semrush
        ("Traffic Change %", "semrush_traffic_change_score") # NEW from Semrush
    ]

    # AI categories if available
    ai_cats = []
    if any(r.get("ai_scores") for r in results):
        ai_cats.extend([
            ("Content Depth", "content_depth_score"),
            ("Entity Coverage", "entity_coverage_score"),
        ])

    cats = base_cats + ai_cats

    # ----- Radar Chart -----
    fig = go.Figure()
    theta = [c[0] for c in cats]
    for r in results:
        vals = [r.get(c[1], 0) or 0 for c in cats]
        fig.add_trace(go.Scatterpolar(r=vals, theta=theta, fill='toself', name=r.get("_domain")))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(100, max([max([v or 0 for v in [r.get(c[1]) for c in cats]]) for r in results]))])),
        showlegend=True,
        height=520
    )
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

            # ----- Semrush metrics -----
            if r.get("semrush"):
                sm = r["semrush"]
                st.markdown("**Semrush Insights**")
                st.write({
                    "Backlinks (domain)": sm.get("backlinks_domain", {}).get("backlinks", "n/a"),
                    "Backlinks (URL)": sm.get("backlinks_url", {}).get("backlinks", "n/a"),
                    "Referring domains (domain)": sm.get("refdomains_domain_count", "n/a"),
                    "Referring domains (URL)": sm.get("refdomains_url_count", "n/a"),
                    "Authority Score": sm.get("backlinks_domain", {}).get("authority_score", "n/a"),
                    "Organic traffic UK (MoM/YoY)": sm.get("domain_organic_uk", {}),
                    "Keyword count UK (URL)": sm.get("url_keywords_uk", "n/a"),
                })

            # ----- AI Insights -----
            if r.get("ai_scores"):
                st.markdown("**AI Analysis**")
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

            # ----- Recommendations -----
            st.markdown("**Recommendations**")
            recs = r.get("_recommendations", [])
            if recs:
                for m in recs:
                    st.write("• ", m)
            else:
                st.write("No critical issues detected. Nice!")

            # ----- Keyword research -----
            if r.get("keyword_research"):
                st.markdown("**AI Keyword Research + Volumes (UK)**")
                st.dataframe(r["keyword_research"], use_container_width=True)

else:
    st.warning("No results to display yet. Please run an audit first.")
