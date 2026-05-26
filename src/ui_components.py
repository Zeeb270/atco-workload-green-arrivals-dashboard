import streamlit as st


def inject_global_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.08), transparent 25%),
                linear-gradient(135deg, #020617 0%, #0f172a 48%, #111827 100%);
            color: #e5e7eb;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.22);
        }

        .hero-card {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.88));
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 26px;
            padding: 34px;
            margin-bottom: 22px;
            box-shadow: 0 0 38px rgba(56, 189, 248, 0.10);
        }

        .hero-title {
            font-size: 44px;
            font-weight: 900;
            color: #f8fafc;
            margin-bottom: 6px;
            letter-spacing: -0.03em;
        }

        .hero-subtitle {
            font-size: 18px;
            color: #94a3b8;
            line-height: 1.6;
            max-width: 980px;
        }

        .pill {
            display: inline-block;
            padding: 7px 13px;
            border-radius: 999px;
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.35);
            color: #7dd3fc;
            font-size: 13px;
            font-weight: 700;
            margin-right: 8px;
            margin-bottom: 8px;
        }

        .kpi-card {
            background: rgba(15, 23, 42, 0.90);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 22px;
            padding: 22px;
            min-height: 126px;
            box-shadow: 0 0 28px rgba(15, 23, 42, 0.35);
        }

        .kpi-label {
            color: #94a3b8;
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 10px;
        }

        .kpi-value {
            color: #38bdf8;
            font-size: 32px;
            font-weight: 900;
            letter-spacing: -0.02em;
        }

        .kpi-caption {
            color: #cbd5e1;
            font-size: 13px;
            margin-top: 8px;
        }

        .section-panel {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 22px;
            padding: 22px;
            margin-top: 16px;
            margin-bottom: 16px;
        }

        .research-note {
            background: rgba(2, 6, 23, 0.72);
            border-left: 4px solid #38bdf8;
            border-radius: 16px;
            padding: 18px 20px;
            color: #cbd5e1;
            margin-top: 14px;
            margin-bottom: 18px;
        }

        .warning-note {
            background: rgba(120, 53, 15, 0.28);
            border-left: 4px solid #f59e0b;
            border-radius: 16px;
            padding: 18px 20px;
            color: #fde68a;
            margin-top: 14px;
            margin-bottom: 18px;
        }

        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            padding: 16px;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
        }

        button[kind="primary"] {
            border-radius: 14px;
        }

        h1, h2, h3 {
            color: #f8fafc;
            letter-spacing: -0.02em;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def hero():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Green Arrival Intelligence Dashboard</div>
            <div class="hero-subtitle">
                A research prototype for evaluating environmentally efficient arrival strategies
                at Stockholm Arlanda using real or simulated air traffic data.
            </div>
            <br>
            <span class="pill">Stockholm Arlanda / ESSA</span>
            <span class="pill">Green Arrival Optimization</span>
            <span class="pill">SCAT-ready</span>
            <span class="pill">Environmental Strategy Evaluation</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def kpi_card(label, value, caption=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def research_note(text):
    st.markdown(
        f"""
        <div class="research-note">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )


def warning_note(text):
    st.markdown(
        f"""
        <div class="warning-note">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )
