"""Finviz-inspired dark financial screener theme."""

import streamlit as st

THEME = {
    "bg_page": "#1a1e26",
    "bg_surface": "#222832",
    "bg_surface_elevated": "#2a303c",
    "bg_surface_muted": "#303642",
    "bg_surface_dark": "#151920",
    "bg_nav": "#2c323d",
    "bg_filter": "#1e232c",
    "border_default": "#3d4654",
    "border_subtle": "#323945",
    "border_active": "#4aa3ff",
    "text_primary": "#e8ecf3",
    "text_secondary": "#b8c0cc",
    "text_muted": "#8b95a7",
    "accent_blue": "#4aa3ff",
    "accent_blue_hover": "#79bdff",
    "card_bg": "#131824",
    "card_inner_bg": "#182033",
    "card_border": "#2A3245",
    "nav_active": "#1E88E5",
    "nav_active_hover": "#2196F3",
    "nav_active_border": "#64B5F6",
    "filter_green": "#22c55e",
    "filter_green_hover": "#16a34a",
    "positive": "#22c55e",
    "negative": "#ff4d5e",
    "neutral": "#8b95a7",
    "table_header_bg": "#2c323d",
    "table_row_even": "#1e232c",
    "table_row_odd": "#252b35",
    "table_row_hover": "#323945",
    "link": "#4aa3ff",
    "font_family": "Arial, Helvetica, Inter, system-ui, sans-serif",
    "font_mono": "'Roboto Mono', Consolas, monospace",
}


def get_global_css() -> str:
    t = THEME
    return f"""
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {{
        --bg-page: {t['bg_page']};
        --bg-surface: {t['bg_surface']};
        --bg-nav: {t['bg_nav']};
        --border: {t['border_default']};
        --text: {t['text_primary']};
        --text-muted: {t['text_muted']};
        --blue: {t['accent_blue']};
        --green: {t['filter_green']};
        --red: {t['negative']};
    }}

    /* ── Base ── */
    .stApp {{
        background-color: var(--bg-page) !important;
        color: var(--text) !important;
        font-family: {t['font_family']} !important;
        font-size: 13px !important;
    }}

    .main .block-container {{
        padding: 1.5rem 1rem 0.75rem 1rem !important;
        max-width: 100% !important;
    }}

    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {{
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
        overflow: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        border: none !important;
    }}

    #MainMenu, footer {{
        visibility: hidden !important;
        display: none !important;
    }}

    .stApp > header {{
        display: none !important;
    }}

    .block-container {{
        padding-top: 1.5rem !important;
    }}

    /* Collapse sidebar — Finviz uses full-width filters */
    section[data-testid="stSidebar"] {{
        display: none !important;
    }}

    [data-testid="collapsedControl"] {{
        display: none !important;
    }}

    h1, h2, h3, h4, p, li, label {{
        color: var(--text) !important;
        font-family: {t['font_family']} !important;
    }}

    /* ── Finviz nav bar ── */
    .fv-nav {{
        background: {t['bg_nav']};
        border-bottom: 2px solid {t['border_default']};
        padding: 0 12px;
        margin: -0.4rem -1rem 0 -1rem;
        display: flex;
        align-items: stretch;
        min-height: 38px;
        gap: 0;
    }}

    .fv-nav-brand {{
        display: flex;
        align-items: center;
        padding: 0 14px 0 4px;
        font-size: 15px;
        font-weight: 700;
        color: {t['accent_blue']} !important;
        letter-spacing: -0.02em;
        border-right: 1px solid {t['border_default']};
        margin-right: 4px;
        white-space: nowrap;
    }}

    .fv-nav-item {{
        display: flex;
        align-items: center;
        padding: 0 12px;
        font-size: 12px;
        font-weight: 600;
        color: {t['text_secondary']} !important;
        border-bottom: 2px solid transparent;
        margin-bottom: -2px;
        white-space: nowrap;
    }}

    .fv-nav-item.active {{
        color: #fff !important;
        border-bottom-color: {t['accent_blue']};
        background: rgba(74,163,255,0.08);
    }}

    .fv-nav-spacer {{ flex: 1; }}

    .fv-nav-meta {{
        display: flex;
        align-items: center;
        font-size: 11px;
        color: {t['text_muted']};
        padding: 0 8px;
        font-family: {t['font_mono']};
    }}

    /* ── Filter panel (Finviz screener box) ── */
    .fv-filter-panel {{
        background: {t['bg_filter']};
        border: 1px solid {t['border_default']};
        margin: 8px 0 6px 0;
    }}

    .fv-filter-header {{
        background: {t['bg_surface_muted']};
        border-bottom: 1px solid {t['border_default']};
        padding: 5px 10px;
        font-size: 11px;
        font-weight: 700;
        color: {t['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}

    .fv-filter-body {{
        padding: 6px 8px 8px 8px;
    }}

    .fv-results-bar {{
        background: {t['bg_surface_muted']};
        border: 1px solid {t['border_default']};
        padding: 5px 10px;
        font-size: 12px;
        color: {t['text_secondary']};
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 6px 0;
    }}

    .fv-results-count {{
        font-weight: 700;
        color: {t['filter_green']};
        font-family: {t['font_mono']};
    }}

    /* ── Compact filter labels ── */
    .fv-label {{
        font-size: 11px !important;
        font-weight: 600 !important;
        color: {t['text_muted']} !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 2px !important;
        padding-top: 2px;
    }}

    /* Tighten all widget spacing in filter area */
    .filter-zone [data-testid="stVerticalBlock"] > div {{
        gap: 0.35rem !important;
    }}

    .filter-zone .stSlider {{
        padding-top: 0 !important;
        margin-top: -4px !important;
    }}

    .filter-zone label[data-testid="stWidgetLabel"] {{
        font-size: 11px !important;
        color: {t['text_muted']} !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0 !important;
        min-height: unset !important;
    }}

    .filter-zone .stCheckbox label span {{
        font-size: 12px !important;
        color: {t['text_secondary']} !important;
    }}

    /* ── Inputs ── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stNumberInput input,
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {{
        background-color: {t['bg_surface']} !important;
        border: 1px solid {t['border_default']} !important;
        color: {t['text_primary']} !important;
        border-radius: 3px !important;
        min-height: 28px !important;
        font-size: 12px !important;
    }}

    .stNumberInput input {{
        padding: 2px 8px !important;
    }}

    /* Multiselect tags — blue not red */
    span[data-baseweb="tag"] {{
        background-color: #24324a !important;
        border: 1px solid #3d5a80 !important;
        border-radius: 3px !important;
    }}

    span[data-baseweb="tag"] span,
    span[data-baseweb="tag"] svg {{
        color: {t['accent_blue']} !important;
        fill: {t['accent_blue']} !important;
    }}

    /* Sliders — blue track, blue thumb (override Streamlit red) */
    .stSlider [data-baseweb="slider"] > div > div[style*="background"] {{
        background: {t['accent_blue']} !important;
    }}

    .stSlider div[data-testid="stTickBar"] > div {{
        background: {t['accent_blue']} !important;
    }}

    .stSlider [data-baseweb="slider"] > div > div {{
        background-color: {t['border_default']} !important;
    }}

    .stSlider [data-baseweb="slider"] div[role="slider"] {{
        background-color: {t['accent_blue']} !important;
        border: 2px solid #fff !important;
        box-shadow: none !important;
    }}

    .stSlider [data-testid="stThumbValue"] {{
        font-size: 11px !important;
        color: {t['text_muted']} !important;
    }}

    /* Checkboxes */
    .stCheckbox [data-testid="stCheckbox"] {{
        background-color: {t['bg_surface']} !important;
        border-color: {t['border_default']} !important;
    }}

    /* ── Finviz green Filter button ── */
    .fv-filter-btn .stButton > button {{
        background: {t['filter_green']} !important;
        border: 1px solid {t['filter_green_hover']} !important;
        color: #fff !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        border-radius: 3px !important;
        min-height: 30px !important;
        padding: 4px 20px !important;
        width: 100% !important;
        letter-spacing: 0.02em;
    }}

    .fv-filter-btn .stButton > button:hover {{
        background: {t['filter_green_hover']} !important;
    }}

    /* Secondary buttons */
    .stButton > button {{
        background: {t['bg_surface_muted']} !important;
        color: {t['text_secondary']} !important;
        border: 1px solid {t['border_default']} !important;
        border-radius: 3px !important;
        font-size: 12px !important;
        min-height: 28px !important;
        font-weight: 600 !important;
    }}

    .stButton > button:hover {{
        border-color: {t['accent_blue']} !important;
        color: {t['text_primary']} !important;
    }}

    .stDownloadButton > button {{
        background: {t['bg_surface']} !important;
        color: {t['accent_blue']} !important;
        border: 1px solid {t['border_default']} !important;
        font-size: 11px !important;
        min-height: 26px !important;
        padding: 2px 10px !important;
    }}

    /* ── Tabs (Finviz filter category tabs) ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0 !important;
        background: {t['bg_surface_muted']} !important;
        padding: 0 !important;
        border-radius: 0 !important;
        border-bottom: 1px solid {t['border_default']} !important;
        min-height: 30px !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        color: {t['text_muted']} !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 6px 14px !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        height: 30px !important;
        border-bottom: 2px solid transparent !important;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        color: {t['text_primary']} !important;
        background: rgba(255,255,255,0.04) !important;
    }}

    .stTabs [aria-selected="true"] {{
        color: {t['accent_blue']} !important;
        background: {t['bg_filter']} !important;
        border-bottom: 2px solid {t['accent_blue']} !important;
    }}

    .stTabs [data-baseweb="tab-panel"] {{
        padding: 8px 4px 4px 4px !important;
        background: {t['bg_filter']} !important;
    }}

    /* Results tabs */
    .results-zone .stTabs [data-baseweb="tab-list"] {{
        background: {t['bg_page']} !important;
    }}

    /* ── Data table ── */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {t['border_default']} !important;
        border-radius: 0 !important;
        font-size: 12px !important;
    }}

    div[data-testid="stDataFrame"] table {{
        font-size: 12px !important;
        font-family: {t['font_family']} !important;
    }}

    /* ── KPI strip ── */
    .fv-kpi-strip {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 4px;
        margin: 4px 0;
    }}

    .fv-kpi {{
        background: {t['bg_surface']};
        border: 1px solid {t['border_default']};
        padding: 6px 10px;
    }}

    .fv-kpi-label {{
        font-size: 10px;
        color: {t['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }}

    .fv-kpi-val {{
        font-size: 16px;
        font-weight: 700;
        color: {t['text_primary']};
        font-family: {t['font_mono']};
        margin-top: 2px;
    }}

    .fv-kpi-val.pos {{ color: {t['positive']}; }}
    .fv-kpi-val.neg {{ color: {t['negative']}; }}

    /* ── Empty state ── */
    .fv-empty {{
        border: 1px dashed {t['border_default']};
        padding: 20px;
        text-align: center;
        color: {t['text_muted']};
        font-size: 12px;
        margin-top: 8px;
        background: {t['bg_surface']};
    }}

    .fv-footer {{
        text-align: center;
        color: {t['text_muted']};
        font-size: 10px;
        padding: 8px 0 2px;
        border-top: 1px solid {t['border_subtle']};
        margin-top: 12px;
    }}

    .section-header {{
        font-size: 11px;
        font-weight: 700;
        color: {t['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 6px 0 4px 0;
    }}

    /* Progress */
    .stProgress > div > div > div {{
        background-color: {t['filter_green']} !important;
    }}

    /* Alerts compact */
    .stAlert {{
        padding: 6px 10px !important;
        font-size: 12px !important;
        border-radius: 3px !important;
    }}

    hr {{
        border-color: {t['border_subtle']} !important;
        margin: 4px 0 !important;
    }}

    /* ── Functional top navigation (scoped via st.container(key="top_nav")) ── */
    .st-key-top_nav .stButton > button {{
        background: {t['bg_surface_muted']} !important;
        color: {t['text_secondary']} !important;
        border: 1px solid {t['border_default']} !important;
        border-radius: 4px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        min-height: 34px !important;
        box-shadow: none !important;
    }}

    .st-key-top_nav .stButton > button:hover {{
        border-color: {t['nav_active']} !important;
        color: {t['text_primary']} !important;
    }}

    .st-key-top_nav .stButton > button[data-testid="stBaseButton-primary"] {{
        background: {t['nav_active']} !important;
        border: 1px solid {t['nav_active_border']} !important;
        color: #fff !important;
        font-weight: 700 !important;
        box-shadow: 0 0 0 1px rgba(30,136,229,0.35), 0 2px 8px rgba(30,136,229,0.25) !important;
    }}

    .st-key-top_nav .stButton > button[data-testid="stBaseButton-primary"]:hover {{
        background: {t['nav_active_hover']} !important;
        color: #fff !important;
    }}

    .fv-nav-brand-inline {{
        font-size: 15px;
        font-weight: 700;
        color: {t['accent_blue']};
        letter-spacing: -0.02em;
    }}

    /* ── Card containers ── */
    .st-key-deep_dive_banner,
    div[class*="st-key-quality_card_"],
    div[data-testid="stMetric"] {{
        border: 1px solid {t['card_border']} !important;
        background-color: {t['card_bg']} !important;
        border-radius: 8px !important;
        padding: 1.25rem !important;
    }}

    .st-key-deep_dive_banner {{
        padding: 0.75rem 1.25rem !important;
    }}

    .fv-kpi {{
        border: 1px solid {t['card_border']} !important;
        background-color: {t['card_bg']} !important;
        border-radius: 8px !important;
    }}

    .st-key-open_deep_dive button {{
        background: {t['nav_active']} !important;
        border: 1px solid {t['nav_active_border']} !important;
        color: #fff !important;
        font-weight: 700 !important;
    }}

    .st-key-open_deep_dive button:hover {{
        background: {t['nav_active_hover']} !important;
    }}

    /* Selected-ticker banner */
    .dd-banner {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 6px 22px;
    }}
    .dd-title {{ font-size: 13px; color: {t['text_secondary']}; margin-right: 4px; }}
    .dd-title b {{ color: {t['accent_blue']}; }}
    .dd-stat {{ display: flex; flex-direction: column; line-height: 1.3; }}
    .dd-stat-label {{
        font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; color: {t['text_muted']};
    }}
    .dd-stat-value {{
        font-size: 13px; font-weight: 600; color: {t['text_primary']}; font-family: {t['font_mono']};
    }}

    /* Earnings quality tiles */
    .fq-tile {{
        background: {t['card_inner_bg']};
        border: 1px solid {t['card_border']};
        border-radius: 8px;
        padding: 1rem 1.1rem;
        min-height: 182px;
    }}
    .fq-tile-label, .fq-cat-head {{
        font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;
        color: {t['text_muted']};
    }}
    .fq-tile-value {{
        font-size: 30px; font-weight: 700; color: {t['text_primary']};
        font-family: {t['font_mono']}; margin: 4px 0 10px;
    }}
    .fq-tile-value.green {{ color: {t['positive']}; }}
    .fq-tile-value.red {{ color: {t['negative']}; }}
    .fq-tile-value.muted {{ font-size: 18px; color: {t['text_muted']}; }}
    .fq-tile-denom {{ font-size: 16px; color: {t['text_muted']}; margin-left: 2px; }}
    .fq-tile-sub {{ font-size: 12px; color: {t['text_secondary']}; line-height: 1.5; margin-top: 8px; }}

    .fq-segments {{ display: grid; grid-template-columns: repeat(9, 1fr); gap: 4px; }}
    .fq-seg {{ height: 8px; border-radius: 2px; background: {t['border_default']}; }}
    .fq-seg.pass {{ background: {t['positive']}; }}
    .fq-seg.fail {{ background: {t['negative']}; }}

    .fq-gauge {{ position: relative; display: flex; height: 8px; margin: 2px 0 4px; }}
    .fq-gauge .zone {{ flex: 1; }}
    .fq-gauge .zone.strong {{ background: {t['positive']}; border-radius: 4px 0 0 4px; }}
    .fq-gauge .zone.healthy {{ background: rgba(34,197,94,0.35); }}
    .fq-gauge .zone.risk {{ background: {t['negative']}; border-radius: 0 4px 4px 0; }}
    .fq-marker {{
        position: absolute; top: -4px; width: 4px; height: 16px; border-radius: 2px;
        background: #fff; transform: translateX(-50%); box-shadow: 0 0 0 2px {t['card_inner_bg']};
    }}
    .fq-gauge-scale {{
        display: flex; justify-content: space-between; margin-bottom: 10px;
        font-size: 10px; color: {t['text_muted']}; font-family: {t['font_mono']};
    }}

    .fq-badge {{
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 12px; font-weight: 600; border: 1px solid;
    }}
    .fq-badge.green {{ color: {t['positive']}; background: rgba(34,197,94,0.12); }}
    .fq-badge.red {{ color: {t['negative']}; background: rgba(255,77,94,0.12); }}

    .fq-cat {{
        background: {t['card_inner_bg']};
        border: 1px solid {t['card_border']};
        border-radius: 8px;
        padding: 0.85rem;
        margin-top: 0.75rem;
    }}
    .fq-cat-head {{ display: flex; justify-content: space-between; margin-bottom: 8px; }}
    .fq-cat-count {{ color: {t['text_primary']}; font-family: {t['font_mono']}; }}

    .fq-pill {{
        display: flex; align-items: flex-start; gap: 8px;
        padding: 6px 10px; margin-bottom: 6px; border-radius: 6px; border: 1px solid {t['card_border']};
    }}
    .fq-pill.pass {{ border-color: rgba(34,197,94,0.45); background: rgba(34,197,94,0.08); }}
    .fq-pill.fail {{ border-color: rgba(255,77,94,0.45); background: rgba(255,77,94,0.08); }}
    .fq-pill-icon {{ width: 14px; font-weight: 800; color: {t['text_muted']}; }}
    .fq-pill.pass .fq-pill-icon {{ color: {t['positive']}; }}
    .fq-pill.fail .fq-pill-icon {{ color: {t['negative']}; }}
    .fq-pill-text {{ display: flex; flex-direction: column; min-width: 0; }}
    .fq-pill-label {{ font-size: 12px; font-weight: 600; color: {t['text_primary']}; }}
    .fq-pill-value {{
        font-size: 11px; color: {t['text_muted']}; font-family: {t['font_mono']};
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }}
    .fq-note {{ font-size: 11px; color: {t['text_muted']}; margin-top: 6px; line-height: 1.5; }}

    .fv-nav-meta-bar {{
        font-size: 10px;
        color: {t['text_muted']};
        text-align: right;
        padding: 2px 4px 4px 0;
        font-family: {t['font_mono']};
    }}

    div[data-testid="stRadio"] > div[role="radiogroup"] {{
        gap: 0 !important;
        flex-wrap: wrap;
    }}

    div[data-testid="stRadio"] label {{
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        padding: 6px 12px !important;
        margin: 0 !important;
        min-height: 30px !important;
    }}

    div[data-testid="stRadio"] label:hover {{
        background: rgba(255,255,255,0.04) !important;
    }}

    div[data-testid="stRadio"] label[data-checked="true"],
    div[data-testid="stRadio"] label:has(input:checked) {{
        border-bottom-color: {t['accent_blue']} !important;
        color: #fff !important;
    }}

    div[data-testid="stRadio"] label p {{
        font-size: 12px !important;
        font-weight: 600 !important;
    }}

    .fv-news-item {{
        background: {t['bg_surface']};
        border: 1px solid {t['border_default']};
        padding: 8px 10px;
        margin-bottom: 6px;
        border-radius: 3px;
    }}

    .fv-news-title {{
        font-size: 13px;
        font-weight: 600;
        color: {t['text_primary']};
        margin-bottom: 4px;
    }}

    .fv-news-meta {{
        font-size: 11px;
        color: {t['text_muted']};
        margin-bottom: 4px;
    }}

    .fv-news-link {{
        font-size: 11px;
        color: {t['accent_blue']};
        text-decoration: none;
        font-weight: 600;
    }}
    """


NAV_PAGES = ["Screener", "Charts"]


def _set_active_page(page: str) -> None:
    st.session_state.active_page = page


def render_top_nav() -> str:
    """Render working Streamlit navigation and return the active page."""
    if st.session_state.get("active_page") not in NAV_PAGES:
        st.session_state.active_page = "Screener"

    with st.container(key="top_nav"):
        brand_col, nav_col, _spacer = st.columns([1.3, 2.2, 6.5], vertical_alignment="center")
        with brand_col:
            st.markdown('<div class="fv-nav-brand-inline">StockTerminal</div>', unsafe_allow_html=True)
        with nav_col:
            for col, page in zip(st.columns(len(NAV_PAGES)), NAV_PAGES):
                with col:
                    st.button(
                        page,
                        type="primary" if st.session_state.active_page == page else "secondary",
                        use_container_width=True,
                        key=f"nav_btn_{page.lower()}",
                        on_click=_set_active_page,
                        args=(page,),
                    )
    return st.session_state.active_page


def finviz_nav(active: str = "screener") -> str:
    """Deprecated decorative nav — use render_top_nav() instead."""
    t = THEME
    label_map = {p.lower(): p for p in NAV_PAGES}
    active_label = label_map.get(active, "Screener")
    nav_items = ""
    for page in NAV_PAGES:
        cls = "fv-nav-item active" if page == active_label else "fv-nav-item"
        nav_items += f'<div class="{cls}">{page}</div>'
    return f"""
    <div class="fv-nav">
        <div class="fv-nav-brand">StockTerminal</div>
        {nav_items}
        <div class="fv-nav-spacer"></div>
        <div class="fv-nav-meta">US · CA · LIVE</div>
    </div>
    """


def filter_panel_open(title: str = "Stock Screener Filters") -> str:
    return f"""
    <div class="fv-filter-panel">
        <div class="fv-filter-header">
            <span>{title}</span>
            <span style="font-size:10px;color:{THEME['text_muted']};font-weight:400;">
                Set criteria below, then click Filter
            </span>
        </div>
        <div class="fv-filter-body">
    """


def filter_panel_close() -> str:
    return "</div></div>"


def results_bar(count: int = 0, message: str = "") -> str:
    t = THEME
    count_html = f'<span class="fv-results-count">{count}</span> results' if count else message
    return f'<div class="fv-results-bar"><span>{count_html}</span><span style="font-size:11px;color:{t["text_muted"]}">Powered by Yahoo Finance</span></div>'


def kpi_strip(cards: list) -> str:
    """cards: list of (label, value, css_class optional)"""
    items = ""
    for card in cards:
        label, value = card[0], card[1]
        css = card[2] if len(card) > 2 else ""
        items += f"""
        <div class="fv-kpi">
            <div class="fv-kpi-label">{label}</div>
            <div class="fv-kpi-val {css}">{value}</div>
        </div>"""
    return f'<div class="fv-kpi-strip">{items}</div>'


def format_market_cap_display(mc):
    if mc is None or (isinstance(mc, float) and (mc != mc)):
        return "Data unavailable"
    if mc >= 1e12:
        return f"${mc/1e12:.2f}T"
    if mc >= 1e9:
        return f"${mc/1e9:.2f}B"
    if mc >= 1e6:
        return f"${mc/1e6:.2f}M"
    return f"${mc:,.0f}"


def format_pct(value, include_sign: bool = True) -> str:
    if value is None or (isinstance(value, float) and (value != value)):
        return "-"
    sign = "+" if include_sign and value > 0 else ""
    return f"{sign}{value:.2f}%"


def delta_class(value) -> str:
    if value is None or (isinstance(value, float) and (value != value)):
        return ""
    if value > 0:
        return "pos"
    if value < 0:
        return "neg"
    return ""


def apply_dark_plotly_layout(fig, title: str = None, height: int = None):
    t = THEME
    layout_kwargs = dict(
        paper_bgcolor=t["bg_surface"],
        plot_bgcolor=t["bg_surface"],
        font=dict(family="Arial, sans-serif", color=t["text_secondary"], size=11),
        margin=dict(l=36, r=16, t=40, b=32),
        legend=dict(
            bgcolor=t["bg_surface_elevated"],
            bordercolor=t["border_default"],
            borderwidth=1,
            font=dict(color=t["text_secondary"], size=10),
        ),
        colorway=[t["accent_blue"], t["positive"], t["negative"], "#38bdf8", t["neutral"]],
    )
    if title:
        layout_kwargs["title"] = dict(text=title, font=dict(color=t["text_primary"], size=13))
    if height:
        layout_kwargs["height"] = height
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(
        gridcolor=t["border_subtle"],
        linecolor=t["border_default"],
        tickfont=dict(color=t["text_muted"], size=10),
        zerolinecolor=t["border_subtle"],
    )
    fig.update_yaxes(
        gridcolor=t["border_subtle"],
        linecolor=t["border_default"],
        tickfont=dict(color=t["text_muted"], size=10),
        zerolinecolor=t["border_subtle"],
    )
    return fig


# Backward-compatible aliases
terminal_header = finviz_nav
status_strip = results_bar

def kpi_card(label, value, delta="", delta_type="neutral"):
    css = {"positive": "pos", "negative": "neg"}.get(delta_type, "")
    return f'<div class="fv-kpi"><div class="fv-kpi-label">{label}</div><div class="fv-kpi-val {css}">{value}</div></div>'

def delta_type_for_value(value):
    if value is None or (isinstance(value, float) and (value != value)):
        return "neutral"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "neutral"
