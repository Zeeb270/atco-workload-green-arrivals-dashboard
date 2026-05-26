import streamlit as st


def inject_phase1_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(52, 211, 153, 0.10), transparent 28%),
                radial-gradient(circle at top right, rgba(96, 165, 250, 0.08), transparent 28%),
                linear-gradient(135deg, #0b0f1a 0%, #111827 52%, #020617 100%);
            color: #e2e8f0;
        }

        section[data-testid="stSidebar"] {
            background: #0b0f1a;
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        h1, h2, h3 {
            color: #f8fafc;
            letter-spacing: -0.03em;
        }

        .mission-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 18px 22px;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            background: rgba(17, 24, 39, 0.92);
            margin-bottom: 18px;
            box-shadow: 0 0 30px rgba(52,211,153,0.08);
        }

        .mission-title {
            font-size: 28px;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 4px;
        }

        .mission-subtitle {
            font-size: 14px;
            color: #94a3b8;
        }

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

        .kpi-strip {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
            margin-bottom: 18px;
        }

        .kpi-card {
            background: rgba(17, 24, 39, 0.92);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 16px 18px;
            min-height: 112px;
            position: relative;
            overflow: hidden;
        }

        .kpi-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: #34d399;
        }

        .kpi-card.blue::before { background: #60a5fa; }
        .kpi-card.amber::before { background: #fbbf24; }
        .kpi-card.red::before { background: #f87171; }
        .kpi-card.white::before { background: #cbd5e1; }

        .kpi-label {
            font-family: monospace;
            font-size: 11px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 10px;
        }

        .kpi-value {
            font-family: monospace;
            font-size: 27px;
            font-weight: 800;
            color: #34d399;
            line-height: 1.1;
        }

        .kpi-card.blue .kpi-value { color: #60a5fa; }
        .kpi-card.amber .kpi-value { color: #fbbf24; }
        .kpi-card.red .kpi-value { color: #f87171; }
        .kpi-card.white .kpi-value { color: #e2e8f0; }

        .kpi-sub {
            font-size: 12px;
            color: #94a3b8;
            margin-top: 7px;
        }

        .phase-panel {
            background: rgba(17, 24, 39, 0.88);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 18px;
            margin-bottom: 16px;
        }

        .panel-title {
            font-family: monospace;
            font-size: 12px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 12px;
            font-weight: 700;
        }

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

        div[data-testid="stMetric"] {
            background: rgba(17, 24, 39, 0.90);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 14px;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }

        @media (max-width: 1100px) {
            .kpi-strip {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def mission_header():
    st.markdown(
        """
        <div class="mission-header">
            <div>
                <div class="mission-title">Green Arrival Optimization Demo</div>
                <div class="mission-subtitle">
                    Historic SCAT-based prototype for Stockholm Arlanda / ESSA arrival strategy evaluation
                </div>
            </div>
            <div class="status-pill">SCAT · HISTORIC DATA · PHASE 1</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label, value, subtitle="", color="green"):
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


def small_note(text):
    st.markdown(
        f"""
        <div class="small-note">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def warning_note(text):
    st.markdown(
        f"""
        <div class="warning-note">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_title(text):
    st.markdown(f'<div class="panel-title">{text}</div>', unsafe_allow_html=True)
