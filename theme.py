"""Dark financial terminal design tokens and UI helpers."""

THEME = {
    "bg_page": "#1f242c",
    "bg_surface": "#242a33",
    "bg_surface_elevated": "#2c323d",
    "bg_surface_muted": "#303642",
    "bg_surface_dark": "#191e26",
    "border_default": "#424a57",
    "border_subtle": "#343b46",
    "border_active": "#2f80ed",
    "text_primary": "#f2f4f8",
    "text_secondary": "#c5ccd8",
    "text_muted": "#8f98a8",
    "accent_blue": "#2f80ed",
    "accent_blue_hover": "#3b93ff",
    "positive": "#22c55e",
    "negative": "#ff4d5e",
    "neutral": "#8b95a7",
    "table_header_bg": "#252b35",
    "table_row_even": "#222832",
    "table_row_odd": "#2b303a",
    "table_row_hover": "#343b48",
    "font_family": "Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    "font_mono": "Inter, 'Roboto Mono', 'SF Mono', Consolas, monospace",
}


def get_global_css() -> str:
    t = THEME
    return f"""
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {{
        --bg-page: {t['bg_page']};
        --bg-surface: {t['bg_surface']};
        --bg-surface-elevated: {t['bg_surface_elevated']};
        --bg-surface-muted: {t['bg_surface_muted']};
        --bg-surface-dark: {t['bg_surface_dark']};
        --border-default: {t['border_default']};
        --border-subtle: {t['border_subtle']};
        --border-active: {t['border_active']};
        --text-primary: {t['text_primary']};
        --text-secondary: {t['text_secondary']};
        --text-muted: {t['text_muted']};
        --accent-blue: {t['accent_blue']};
        --accent-blue-hover: {t['accent_blue_hover']};
        --positive: {t['positive']};
        --negative: {t['negative']};
        --neutral: {t['neutral']};
        --font-family: {t['font_family']};
        --font-mono: {t['font_mono']};
    }}

    .stApp {{
        background-color: var(--bg-page);
        color: var(--text-primary);
        font-family: var(--font-family);
    }}

    .main .block-container {{
        padding-top: 0.75rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }}

    h1, h2, h3, h4, h5, h6, p, li, label, span {{
        color: var(--text-primary);
        font-family: var(--font-family);
    }}

    .stMarkdown small {{
        color: var(--text-muted);
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: var(--bg-surface-dark);
        border-right: 1px solid var(--border-default);
    }}

    section[data-testid="stSidebar"] .block-container {{
        padding-top: 0.75rem;
        padding-left: 0.75rem;
        padding-right: 0.75rem;
    }}

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        font-size: 13px;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.25rem;
    }}

    section[data-testid="stSidebar"] hr {{
        border-color: var(--border-subtle);
        margin: 0.5rem 0;
    }}

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stCheckbox label {{
        font-size: 12px !important;
        color: var(--text-muted) !important;
        font-weight: 500;
    }}

    /* Inputs */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stNumberInput input,
    div[data-baseweb="select"] > div {{
        background-color: #222832 !important;
        border: 1px solid #4a5361 !important;
        color: var(--text-primary) !important;
        border-radius: 6px !important;
        min-height: 32px;
        font-size: 13px !important;
    }}

    .stSlider [data-baseweb="slider"] div[role="slider"] {{
        background-color: var(--accent-blue) !important;
    }}

    .stSlider [data-baseweb="slider"] > div > div {{
        background-color: var(--accent-blue) !important;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: var(--accent-blue) !important;
        color: #ffffff !important;
        border: 1px solid var(--accent-blue-hover) !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        min-height: 34px;
        padding: 0.35rem 0.75rem !important;
    }}

    .stButton > button:hover {{
        background-color: var(--accent-blue-hover) !important;
        border-color: var(--accent-blue-hover) !important;
    }}

    .stDownloadButton > button {{
        background-color: var(--bg-surface-muted) !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: 6px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }}

    .stDownloadButton > button:hover {{
        background-color: var(--bg-surface-elevated) !important;
        border-color: var(--border-active) !important;
        color: var(--text-primary) !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background-color: #252b35;
        padding: 4px;
        border-radius: 6px;
        border: 1px solid var(--border-default);
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: var(--bg-surface-muted);
        color: var(--text-secondary);
        border: 1px solid var(--border-default);
        border-radius: 6px;
        padding: 6px 14px;
        font-size: 13px;
        font-weight: 600;
        height: 32px;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: #24324a !important;
        color: #ffffff !important;
        border-color: var(--border-active) !important;
    }}

    /* Metrics override */
    div[data-testid="stMetric"] {{
        background-color: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: 6px;
        padding: 10px 12px;
    }}

    div[data-testid="stMetric"] label {{
        font-size: 11px !important;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        font-size: 20px !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        font-family: var(--font-mono);
    }}

    /* Dataframes */
    div[data-testid="stDataFrame"] {{
        border: 1px solid var(--border-default);
        border-radius: 6px;
        overflow: hidden;
    }}

    /* Alerts */
    .stAlert {{
        border-radius: 6px;
        font-size: 13px;
        border: 1px solid var(--border-default);
    }}

    div[data-testid="stNotification"] {{
        background-color: var(--bg-surface);
    }}

    /* Progress */
    .stProgress > div > div {{
        background-color: var(--accent-blue);
    }}

    /* Hide Streamlit chrome for denser terminal feel */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header[data-testid="stHeader"] {{
        background-color: transparent;
    }}

    /* Custom terminal components */
    .terminal-topbar {{
        background: {t['bg_surface_muted']};
        border: 1px solid {t['border_default']};
        border-radius: 6px;
        padding: 10px 16px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        min-height: 44px;
    }}

    .terminal-topbar-title {{
        font-size: 16px;
        font-weight: 700;
        color: {t['text_primary']};
        letter-spacing: -0.01em;
    }}

    .terminal-topbar-subtitle {{
        font-size: 12px;
        color: {t['text_muted']};
        margin-top: 2px;
    }}

    .terminal-status-strip {{
        background: {t['bg_surface']};
        border: 1px solid {t['border_subtle']};
        border-radius: 6px;
        padding: 6px 12px;
        margin-bottom: 12px;
        font-size: 12px;
        color: {t['text_secondary']};
    }}

    .kpi-card {{
        background: {t['bg_surface']};
        border: 1px solid {t['border_default']};
        border-radius: 6px;
        padding: 12px 14px;
        text-align: left;
        min-height: 72px;
    }}

    .kpi-label {{
        font-size: 11px;
        font-weight: 600;
        color: {t['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }}

    .kpi-value {{
        font-size: 22px;
        font-weight: 700;
        color: {t['text_primary']};
        font-family: {t['font_mono']};
        line-height: 1.2;
    }}

    .kpi-delta-positive {{
        font-size: 12px;
        color: {t['positive']};
        font-weight: 600;
        margin-top: 4px;
    }}

    .kpi-delta-negative {{
        font-size: 12px;
        color: {t['negative']};
        font-weight: 600;
        margin-top: 4px;
    }}

    .kpi-delta-neutral {{
        font-size: 12px;
        color: {t['neutral']};
        margin-top: 4px;
    }}

    .welcome-panel {{
        background: {t['bg_surface']};
        border: 1px solid {t['border_default']};
        border-radius: 6px;
        padding: 20px 24px;
        margin-top: 8px;
    }}

    .welcome-panel h2 {{
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 8px;
    }}

    .welcome-panel p, .welcome-panel li {{
        font-size: 13px;
        color: {t['text_secondary']};
        line-height: 1.5;
    }}

    .section-header {{
        font-size: 14px;
        font-weight: 600;
        color: {t['text_secondary']};
        margin: 8px 0 6px 0;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}

    .terminal-footer {{
        text-align: center;
        color: {t['text_muted']};
        font-size: 11px;
        padding: 12px 0 4px 0;
        border-top: 1px solid {t['border_subtle']};
        margin-top: 16px;
    }}

    .filter-group-label {{
        font-size: 11px;
        font-weight: 600;
        color: {t['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 10px 0 4px 0;
    }}
    """


def terminal_header(title: str = "Stock Screener", subtitle: str = "Market Analysis Terminal") -> str:
    return f"""
    <div class="terminal-topbar">
        <div>
            <div class="terminal-topbar-title">{title}</div>
            <div class="terminal-topbar-subtitle">{subtitle}</div>
        </div>
        <div style="font-size:11px;color:{THEME['text_muted']};font-family:{THEME['font_mono']};">
            LIVE DATA
        </div>
    </div>
    """


def status_strip(message: str) -> str:
    return f'<div class="terminal-status-strip">{message}</div>'


def kpi_card(label: str, value: str, delta: str = "", delta_type: str = "neutral") -> str:
    delta_class = {
        "positive": "kpi-delta-positive",
        "negative": "kpi-delta-negative",
    }.get(delta_type, "kpi-delta-neutral")
    delta_html = f'<div class="{delta_class}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
  """


def format_pct(value, include_sign: bool = True) -> str:
    if value is None or (isinstance(value, float) and (value != value)):
        return "N/A"
    sign = "+" if include_sign and value > 0 else ""
    return f"{sign}{value:.2f}%"


def delta_type_for_value(value) -> str:
    if value is None or (isinstance(value, float) and (value != value)):
        return "neutral"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "neutral"


def apply_dark_plotly_layout(fig, title: str = None, height: int = None):
    t = THEME
    layout_kwargs = dict(
        paper_bgcolor=t["bg_surface"],
        plot_bgcolor=t["bg_surface"],
        font=dict(family="Inter, sans-serif", color=t["text_secondary"], size=12),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            bgcolor=t["bg_surface_elevated"],
            bordercolor=t["border_default"],
            borderwidth=1,
            font=dict(color=t["text_secondary"], size=11),
        ),
        colorway=[t["accent_blue"], t["positive"], t["negative"], "#38bdf8", t["neutral"]],
    )
    if title:
        layout_kwargs["title"] = dict(text=title, font=dict(color=t["text_primary"], size=14))
    if height:
        layout_kwargs["height"] = height
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(
        gridcolor=t["border_subtle"],
        linecolor=t["border_default"],
        tickfont=dict(color=t["text_muted"], size=11),
        titlefont=dict(color=t["text_muted"], size=11),
    )
    fig.update_yaxes(
        gridcolor=t["border_subtle"],
        linecolor=t["border_default"],
        tickfont=dict(color=t["text_muted"], size=11),
        titlefont=dict(color=t["text_muted"], size=11),
    )
    return fig
