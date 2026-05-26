"""
 — Single unified UI module.

Replaces the duplicate ui_components.py + ui_phase1.py pair.
Exports: inject_css, render_header, kpi_card, panel, small_note, warning_note
"""

from contextlib import contextmanager
import streamlit as st


def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── App background ── */
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(52,211,153,0.10), transparent 28%),
                radial-gradient(circle at top right, rgba(96,165,250,0.08), transparent 28%),
                linear-gradient(135deg, #0b0f1a 0%, #111827 52%, #020617 100%);
            color: #e2e8f0;
        }

        section[data-testid="stSidebar"] {
            background: #0b0f1a;
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        h1, h2, h3 { color: #f8fafc; letter-spacing: -0.03em; }

        /* ── Mission header ── */
        .mission-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 18px 22px;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            background: rgba(17,24,39,0.92);
            margin-bottom: 18px;
            box-shadow: 0 0 30px rgba(52,211,153,0.08);
        }
        .mission-title {
            font-size: 26px;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 4px;
        }
        .mission-subtitle { font-size: 14px; color: #94a3b8; }
        .status-pill {
            font-family: monospace;
            font-size: 12px;
            padding: 7px 11px;
            border-radius: 999px;
            border: 1px solid rgba(52,211,153,0.35);
            color: #34d399;
            background: rgba(52,211,153,0.10);
            white-space: nowrap;
        }

        /* ── KPI strip ── */
        .kpi-strip {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
            margin-bottom: 18px;
        }
        .kpi-card {
            background: rgba(17,24,39,0.92);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 16px 18px;
            min-height: 110px;
            position: relative;
            overflow: hidden;
        }
        .kpi-card::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: #34d399;
        }
        .kpi-card.blue::before  { background: #60a5fa; }
        .kpi-card.amber::before { background: #fbbf24; }
        .kpi-card.red::before   { background: #f87171; }
        .kpi-card.white::before { background: #cbd5e1; }

        .kpi-label {
            font-family: monospace;
            font-size: 11px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }
        .kpi-value {
            font-family: monospace;
            font-size: 26px;
            font-weight: 800;
            color: #34d399;
            line-height: 1.1;
        }
        .kpi-card.blue  .kpi-value { color: #60a5fa; }
        .kpi-card.amber .kpi-value { color: #fbbf24; }
        .kpi-card.red   .kpi-value { color: #f87171; }
        .kpi-card.white .kpi-value { color: #e2e8f0; }
        .kpi-sub { font-size: 12px; color: #94a3b8; margin-top: 6px; }

        /* Workload status colours */
        .status-low    { color: #34d399; }
        .status-medium { color: #fbbf24; }
        .status-high   { color: #f87171; }

        /* ── Generic panel ── */
        .phase-panel {
            background: rgba(17,24,39,0.88);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 18px;
            margin-bottom: 16px;
        }
        .panel-title {
            font-family: monospace;
            font-size: 11px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 12px;
            font-weight: 700;
        }

        /* ── Notes ── */
        .small-note {
            background: rgba(52,211,153,0.08);
            border: 1px solid rgba(52,211,153,0.25);
            border-radius: 14px;
            padding: 14px 16px;
            color: #cbd5e1;
            font-size: 14px;
            line-height: 1.55;
            margin-bottom: 16px;
        }
        .warning-note {
            background: rgba(251,191,36,0.09);
            border: 1px solid rgba(251,191,36,0.28);
            border-radius: 14px;
            padding: 14px 16px;
            color: #fde68a;
            font-size: 14px;
            line-height: 1.55;
            margin-bottom: 16px;
        }

        /* ── Misc ── */
        div[data-testid="stMetric"] {
            background: rgba(17,24,39,0.90);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 14px;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }

        @media (max-width: 1100px) {
            .kpi-strip { grid-template-columns: repeat(2, 1fr); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="mission-header">
            <div>
                <div class="mission-title">✈ Green Arrival Intelligence — ESSA</div>
                <div class="mission-subtitle">
                    Research prototype for workload-aware sustainable arrival operations at
                    Stockholm Arlanda Airport
                </div>
            </div>
            <div class="status-pill">SCAT · OpenSky · PHASE 1</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, subtitle: str = "", color: str = "green") -> None:
    """Render a single KPI card. value may contain safe HTML (e.g. a <span>)."""
    st.markdown(
        f"""
        <div class="kpi-card {color}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def panel(title: str):
    """Context manager that wraps content in a styled phase-panel div."""
    st.markdown(
        f'<div class="phase-panel"><div class="panel-title">{title}</div>',
        unsafe_allow_html=True,
    )
    yield
    st.markdown("</div>", unsafe_allow_html=True)


def small_note(text: str) -> None:
    st.markdown(f'<div class="small-note">{text}</div>', unsafe_allow_html=True)


def warning_note(text: str) -> None:
    st.markdown(f'<div class="warning-note">{text}</div>', unsafe_allow_html=True)
