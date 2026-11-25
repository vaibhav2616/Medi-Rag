"""
MediRAG — Interactive Clinical Reference Assistant (Streamlit Web UI)
Level 1 of 3-Tier Progressive RAG Portfolio
Features:
- Two-stage LCEL chain with conversational query reformulation
- Universal AI API key box (Auto-detect / Gemini / Groq / OpenAI-compatible / Local Offline)
- Modern lightweight, low-token models (gemini-2.5-flash-lite, llama-3.1-8b-instant, gpt-4.1-nano)
- Live thinking indicator with exact elapsed time (e.g., "Thought for 0.38s")
- Editorial healthcare UI with Light & Dark themes (warm alabaster & obsidian-sage palettes)
- Strict token-saving out-of-scope negative rejection guardrails & source citations
"""

import html as html_lib
import importlib.util
import os
import sys
import time
from typing import Tuple

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# Ensure current module directory is on sys.path
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from chain import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GROQ_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_PERSIST_DIR,
    build_medirag_chain,
    detect_provider_from_key,
    get_embedding_model,
    get_llm,
    get_vectorstore,
    run_medirag_query,
)
from ingest import (
    EXTENDED_CORPUS_JSON_PATH,
    FALLBACK_NIH_RECORDS,
    build_and_persist_chroma,
    build_documents,
    chunk_documents,
    load_verified_json_corpus,
)

# Page configuration
st.set_page_config(
    page_title="MediRAG | Clinical Reference Assistant",
    page_icon=":material/medical_services:",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEME_TOKENS = {
    "Light": {
        "bg": "#F7F5F0",
        "sidebar_bg": "#EFECE4",
        "surface": "#FFFFFF",
        "surface_alt": "#F2EFE8",
        "border": "#E4E0D8",
        "accent": "#2D5A4C",
        "accent_hover": "#23473C",
        "accent_soft": "#EBF2EF",
        "accent_text": "#2D5A4C",
        "text": "#1C1917",
        "text_muted": "#57534E",
        "warn_bg": "#FFF7ED",
        "warn_border": "#FED7AA",
        "warn_text": "#9A3412",
        "pill_border": "#C7DBD3",
    },
    "Dark": {
        "bg": "#121413",
        "sidebar_bg": "#161917",
        "surface": "#1A1D1B",
        "surface_alt": "#222624",
        "border": "#2A2E2B",
        "accent": "#52997E",
        "accent_hover": "#62AD90",
        "accent_soft": "#1E2B26",
        "accent_text": "#6EE7B7",
        "text": "#F2EFE9",
        "text_muted": "#A8A29E",
        "warn_bg": "#2A1810",
        "warn_border": "#7C2D12",
        "warn_text": "#FDBA74",
        "pill_border": "#2D473D",
    },
}


def apply_editorial_theme(mode: str) -> None:
    """Applies a calm, editorial healthcare theme (Light or Dark) with zero AI neon/purple gradients."""
    t = THEME_TOKENS.get(mode, THEME_TOKENS["Light"])
    st.html(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {{
            --mr-bg: {t['bg']};
            --mr-sidebar: {t['sidebar_bg']};
            --mr-surface: {t['surface']};
            --mr-surface-alt: {t['surface_alt']};
            --mr-border: {t['border']};
            --mr-accent: {t['accent']};
            --mr-accent-hover: {t['accent_hover']};
            --mr-accent-soft: {t['accent_soft']};
            --mr-accent-text: {t['accent_text']};
            --mr-text: {t['text']};
            --mr-muted: {t['text_muted']};
            --mr-warn-bg: {t['warn_bg']};
            --mr-warn-border: {t['warn_border']};
            --mr-warn-text: {t['warn_text']};
            --mr-pill-border: {t['pill_border']};
            --background-color: {t['bg']};
            --secondary-background-color: {t['surface']};
            --text-color: {t['text']};
            --primary-color: {t['accent']};
        }}

        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
            background-color: var(--mr-bg) !important;
            color: var(--mr-text) !important;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }}

        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] strong,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary p,
        [data-testid="stStatusWidget"] label,
        [data-testid="stStatusWidget"] p {{
            color: var(--mr-text) !important;
        }}

        [data-testid="stHeader"] {{
            background-color: transparent !important;
        }}

        [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {{
            background-color: var(--mr-sidebar) !important;
            border-right: 1px solid var(--mr-border) !important;
        }}

        [data-testid="stSidebar"] * {{
            color: var(--mr-text);
        }}

        h1, .mr-editorial-title {{
            font-family: 'Instrument Serif', Georgia, 'Times New Roman', serif !important;
            font-weight: 400 !important;
            letter-spacing: -0.02em !important;
            color: var(--mr-text) !important;
        }}

        code, pre, .mr-mono {{
            font-family: 'JetBrains Mono', monospace !important;
        }}

        /* Cards, metrics, expanders & bordered containers */
        [class*="st-key-mr_card_"],
        [class*="st-key-mr_note_"],
        [class*="st-key-mr_doc_"],
        [data-testid="stExpander"],
        [data-testid="stExpander"] details {{
            background-color: var(--mr-surface) !important;
            border-color: var(--mr-border) !important;
            border-radius: 12px !important;
        }}
        [data-testid="stExpanderDetails"] {{
            background-color: var(--mr-surface) !important;
            color: var(--mr-text) !important;
        }}

        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary:hover,
        [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] summary p {{
            background-color: var(--mr-surface) !important;
            color: var(--mr-text) !important;
        }}
        [data-testid="stExpander"] svg {{
            color: var(--mr-text) !important;
            fill: var(--mr-text) !important;
        }}

        /* Alerts / Warnings (out-of-scope & info banners) */
        [data-testid="stAlertContainer"] {{
            background-color: var(--mr-warn-bg) !important;
            border: 1px solid var(--mr-warn-border) !important;
            color: var(--mr-warn-text) !important;
            border-radius: 10px !important;
        }}
        [data-testid="stAlertContainer"] *,
        [data-testid="stAlertContent"],
        [data-testid="stAlertContent"] *,
        [data-testid="stAlertContainer"] [data-testid="stMarkdownContainer"] p {{
            color: var(--mr-warn-text) !important;
        }}
        [data-testid="stAlertContainer"] svg {{
            color: var(--mr-warn-text) !important;
            fill: currentColor !important;
        }}

        /* Prevent double-nested border inside st.container(border=True) > st.metric */
        [data-testid="stMetric"] {{
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }}

        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] p,
        [data-testid="stCaptionContainer"],
        .stCaption {{
            color: var(--mr-muted) !important;
        }}

        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] > div {{
            color: var(--mr-text) !important;
            font-weight: 600 !important;
        }}

        [data-testid="stMetricDelta"] {{
            display: inline-flex !important;
            align-items: center !important;
            background-color: var(--mr-accent-soft) !important;
            color: var(--mr-accent-text) !important;
            border: 1px solid var(--mr-pill-border) !important;
            border-radius: 999px !important;
            padding: 0.15rem 0.55rem !important;
            width: fit-content !important;
        }}
        [data-testid="stMetricDelta"] > div,
        [data-testid="stMetricDelta"] span,
        [data-testid="stMetricDelta"] svg {{
            background-color: transparent !important;
            color: var(--mr-accent-text) !important;
            fill: var(--mr-accent-text) !important;
        }}

        /* Inputs, password eye-toggle & selects (Streamlit 1.67.0 React Aria + BaseWeb fallback) */
        [data-testid="stTextInputRootElement"],
        [data-testid="stTextInputRootElement"] > div,
        [data-baseweb="input"],
        [data-baseweb="base-input"] {{
            background-color: var(--mr-surface) !important;
            color: var(--mr-text) !important;
            border-color: var(--mr-border) !important;
            border-radius: 8px !important;
        }}
        [data-testid="stTextInputField"],
        [data-testid="stTextInputRootElement"] input,
        [data-baseweb="input"] input,
        .stTextInput input {{
            background-color: transparent !important;
            color: var(--mr-text) !important;
            caret-color: var(--mr-text) !important;
            -webkit-text-fill-color: var(--mr-text) !important;
        }}
        [data-testid="stTextInputField"]::placeholder,
        [data-testid="stTextInputRootElement"] input::placeholder,
        [data-baseweb="input"] input::placeholder,
        .stTextInput input::placeholder {{
            color: var(--mr-muted) !important;
            -webkit-text-fill-color: var(--mr-muted) !important;
            opacity: 1 !important;
        }}
        [data-testid="stTextInputRootElement"] button,
        [data-testid="stTextInput"] button,
        [data-baseweb="input"] button {{
            background-color: transparent !important;
            color: var(--mr-muted) !important;
            border: none !important;
        }}
        [data-testid="stTextInputRootElement"] button:hover,
        [data-testid="stTextInput"] button:hover,
        [data-baseweb="input"] button:hover {{
            background-color: var(--mr-surface-alt) !important;
            color: var(--mr-text) !important;
        }}
        [data-testid="stTextInputRootElement"] button svg,
        [data-testid="stTextInput"] button svg,
        [data-baseweb="input"] button svg {{
            color: inherit !important;
            fill: currentColor !important;
        }}

        [data-testid="stSelectbox"] > div > div,
        [data-baseweb="select"] > div,
        .stSelectbox div[data-baseweb="select"] > div {{
            background-color: var(--mr-surface) !important;
            color: var(--mr-text) !important;
            border-color: var(--mr-border) !important;
            border-radius: 8px !important;
        }}
        [data-testid="stSelectbox"] input,
        [data-testid="stSelectbox"] span,
        [data-testid="stSelectbox"] div,
        [data-baseweb="select"] span,
        [data-baseweb="select"] div,
        [data-baseweb="select"] input,
        .stSelectbox div[data-baseweb="select"] span {{
            color: var(--mr-text) !important;
            -webkit-text-fill-color: var(--mr-text) !important;
        }}
        [data-testid="stSelectbox"] button {{
            background-color: transparent !important;
            color: var(--mr-text) !important;
        }}
        [data-testid="stSelectbox"] svg,
        [data-baseweb="select"] svg,
        .stSelectbox div[data-baseweb="select"] svg {{
            color: var(--mr-text) !important;
            fill: currentColor !important;
        }}

        /* Selectbox dropdown popover portal (Streamlit 1.67.0 stSelectboxVirtualDropdown + BaseWeb fallback) */
        [data-testid="stSelectboxVirtualDropdown"],
        [data-testid="stSelectboxVirtualDropdown"] > div,
        [data-baseweb="popover"],
        [data-baseweb="popover"] > div,
        [data-baseweb="menu"],
        [role="listbox"] {{
            background-color: var(--mr-surface) !important;
            color: var(--mr-text) !important;
            border-color: var(--mr-border) !important;
            border-radius: 8px !important;
        }}
        [data-testid="stSelectboxVirtualDropdown"] {{
            border: 1px solid var(--mr-border) !important;
        }}
        [data-testid="stSelectboxVirtualDropdown"] [role="option"],
        [role="option"],
        [data-baseweb="menu"] li {{
            background-color: var(--mr-surface) !important;
            color: var(--mr-text) !important;
        }}
        [data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
        [data-testid="stSelectboxVirtualDropdown"] [role="option"][data-hovered="true"],
        [data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused="true"],
        [data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"],
        [role="option"]:hover,
        [role="option"][aria-selected="true"],
        [data-baseweb="menu"] li:hover,
        [data-baseweb="menu"] li[aria-selected="true"] {{
            background-color: var(--mr-accent-soft) !important;
            color: var(--mr-accent-text) !important;
        }}

        /* Segmented control, pills & all buttons (Streamlit 1.67.0 stButtonGroup + stBaseButton + BaseWeb) */
        [data-testid="stButtonGroup"],
        [data-baseweb="button-group"] {{
            background-color: transparent !important;
            border-color: var(--mr-border) !important;
        }}
        [data-testid^="stBaseButton-"],
        [data-testid="stBaseButton-segmented_control"],
        [data-testid="stBaseButton-pills"],
        [data-testid="stBaseButton-secondary"],
        [data-testid="stButtonGroup"] button,
        [data-testid="stButtonGroup"] [role="radio"],
        [data-testid="stButtonGroup"] [role="option"],
        [data-baseweb="button-group"] button,
        [data-testid="stPills"] button,
        .stButton > button,
        [data-testid="stSidebar"] .stButton > button {{
            background-color: var(--mr-surface) !important;
            color: var(--mr-text) !important;
            border: 1px solid var(--mr-border) !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: border-color 0.15s ease, background-color 0.15s ease, color 0.15s ease;
        }}
        [data-testid^="stBaseButton-"] *,
        [data-testid^="stBaseButton-"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stButtonGroup"] button *,
        [data-testid="stButtonGroup"] button [data-testid="stMarkdownContainer"] p,
        [data-baseweb="button-group"] button *,
        [data-testid="stPills"] button *,
        [data-testid="stPills"] span,
        .stButton > button *,
        .stButton > button [data-testid="stMarkdownContainer"] p {{
            color: inherit !important;
        }}
        [data-testid^="stBaseButton-"]:hover,
        [data-testid="stBaseButton-segmented_control"]:hover,
        [data-testid="stBaseButton-pills"]:hover,
        [data-testid="stBaseButton-secondary"]:hover,
        [data-testid="stButtonGroup"] button:hover,
        [data-baseweb="button-group"] button:hover,
        [data-testid="stPills"] button:hover,
        .stButton > button:hover,
        [data-testid="stSidebar"] .stButton > button:hover {{
            background-color: var(--mr-accent-soft) !important;
            color: var(--mr-accent-text) !important;
            border-color: var(--mr-accent) !important;
        }}
        [data-testid="stBaseButton-primary"],
        button[kind="primary"] {{
            background-color: var(--mr-accent) !important;
            color: #FFFFFF !important;
            border-color: var(--mr-accent) !important;
        }}
        [data-testid="stBaseButton-primary"] *,
        button[kind="primary"] * {{
            color: #FFFFFF !important;
        }}
        [data-testid="stBaseButton-segmented_controlActive"],
        [data-testid="stBaseButton-pillsActive"],
        [data-testid="stButtonGroup"] button[aria-checked="true"],
        [data-testid="stButtonGroup"] button[aria-pressed="true"],
        [data-testid="stButtonGroup"] button[aria-selected="true"],
        [data-baseweb="button-group"] button[aria-checked="true"],
        [data-baseweb="button-group"] button[aria-pressed="true"],
        [data-testid="stPills"] button[aria-selected="true"],
        [data-testid="stPills"] button[aria-pressed="true"] {{
            background-color: var(--mr-accent-soft) !important;
            color: var(--mr-accent-text) !important;
            border-color: var(--mr-accent) !important;
        }}
        [data-testid="stBaseButton-segmented_controlActive"] *,
        [data-testid="stBaseButton-segmented_controlActive"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stBaseButton-pillsActive"] *,
        [data-testid="stBaseButton-pillsActive"] [data-testid="stMarkdownContainer"] p {{
            color: var(--mr-accent-text) !important;
        }}

        /* Sidebar muted captions, tooltips & header controls */
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stTooltipIcon"] svg {{
            color: var(--mr-muted) !important;
            fill: var(--mr-muted) !important;
        }}
        [data-testid="stHeader"] button,
        [data-testid="stHeader"] svg {{
            color: var(--mr-text) !important;
            fill: currentColor !important;
        }}

        /* Chat messages & bottom input */
        [data-testid="stChatMessage"] {{
            background-color: var(--mr-surface) !important;
            border: 1px solid var(--mr-border) !important;
            border-radius: 12px !important;
            padding: 1rem 1.15rem !important;
            margin-bottom: 0.65rem !important;
        }}

        [data-testid="stBottom"], [data-testid="stBottom"] > div {{
            background-color: var(--mr-bg) !important;
        }}

        [data-testid="stChatInput"],
        [data-testid="stChatInput"] > div {{
            background-color: var(--mr-surface) !important;
            border-color: var(--mr-border) !important;
            border-radius: 12px !important;
            color: var(--mr-text) !important;
        }}
        [data-testid="stChatInput"] > div {{
            border: 1px solid var(--mr-border) !important;
        }}
        [data-testid="stChatInputTextArea"],
        [data-testid="stChatInput"] textarea {{
            background-color: transparent !important;
            color: var(--mr-text) !important;
            caret-color: var(--mr-text) !important;
            -webkit-text-fill-color: var(--mr-text) !important;
        }}
        [data-testid="stChatInputTextArea"]::placeholder,
        [data-testid="stChatInput"] textarea::placeholder {{
            color: var(--mr-muted) !important;
            -webkit-text-fill-color: var(--mr-muted) !important;
            opacity: 1 !important;
        }}
        [data-testid="stChatInputSubmitButton"] {{
            background-color: var(--mr-accent) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
        }}
        [data-testid="stChatInputSubmitButton"]:hover:not(:disabled) {{
            background-color: var(--mr-accent-hover) !important;
            color: #FFFFFF !important;
        }}
        [data-testid="stChatInputSubmitButton"]:disabled {{
            background-color: var(--mr-surface-alt) !important;
            color: var(--mr-muted) !important;
            opacity: 0.85 !important;
        }}
        [data-testid="stChatInputSubmitButton"] svg {{
            color: inherit !important;
            fill: currentColor !important;
        }}

        /* Editorial citation pills */
        .mr-citation-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.65rem;
        }}
        .mr-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.24rem 0.65rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 500;
            background-color: var(--mr-accent-soft);
            color: var(--mr-accent-text);
            border: 1px solid var(--mr-pill-border);
        }}
        .mr-pill-source {{
            background-color: var(--mr-surface-alt);
            color: var(--mr-muted);
            border: 1px solid var(--mr-border);
        }}

        /* Grounded header banner */
        .mr-hero {{
            padding: 0.35rem 0 0.75rem 0;
        }}
        .mr-hero-eyebrow {{
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--mr-accent-text);
            margin-bottom: 0.25rem;
        }}
        .mr-hero-sub {{
            color: var(--mr-muted);
            font-size: 0.95rem;
            line-height: 1.5;
            max-width: 72ch;
        }}
        </style>
        """
    )


@st.cache_resource(show_spinner="Initializing clinical vector store & embeddings...")
def load_or_init_retriever() -> Tuple[object, int]:
    """Initializes ChromaDB vector store or seeds it with clinical corpus if empty."""
    persist_dir = DEFAULT_PERSIST_DIR
    embeddings = get_embedding_model()

    needs_seed = False
    chunk_count = 0
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        needs_seed = True
    else:
        try:
            vs = get_vectorstore(persist_dir=persist_dir, embedding_model=embeddings)
            chunk_count = vs._collection.count()
            if chunk_count < 30:
                needs_seed = True
        except Exception:
            needs_seed = True

    if needs_seed:
        records = load_verified_json_corpus(EXTENDED_CORPUS_JSON_PATH) or FALLBACK_NIH_RECORDS
        docs = build_documents(records)
        chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=120)
        vectorstore = build_and_persist_chroma(chunks, persist_dir=persist_dir, embedding_model=embeddings)
        chunk_count = len(chunks)
    else:
        vectorstore = get_vectorstore(persist_dir=persist_dir, embedding_model=embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 4}), chunk_count


def render_citations(citations: list) -> None:
    """Renders editorial citation pills matching the active Light/Dark palette."""
    if not citations:
        return
    pills_html = []
    for cit in citations:
        focus = html_lib.escape(str(cit.get("focus", "Clinical Evidence")))
        qtype = html_lib.escape(str(cit.get("qtype", "inquiry")))
        source = html_lib.escape(str(cit.get("source", "NIH")))
        pills_html.append(
            f'<span class="mr-pill">{focus} · {qtype}</span>'
            f'<span class="mr-pill mr-pill-source">{source}</span>'
        )
    st.html(f'<div class="mr-citation-row">{"".join(pills_html)}</div>')


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "sample_prompt" not in st.session_state:
    st.session_state.sample_prompt = None
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light"

# Sidebar Control Panel
with st.sidebar:
    st.markdown("### :material/medical_services: MediRAG")
    st.caption("Guarded Clinical Reference · Level 1")

    theme_mode = st.segmented_control(
        "Appearance",
        ["Light", "Dark"],
        key="theme_mode",
        width="stretch",
    )
    active_theme = theme_mode or "Light"

    st.markdown("#### AI engine & universal key")
    provider_label = st.selectbox(
        "Provider mode",
        [
            "Auto-Detect from API Key",
            "Local Offline Mode (No Key)",
            "Google Gemini",
            "Groq Cloud",
            "OpenAI / Compatible (OpenRouter, DeepSeek)",
        ],
        index=0,
        key="provider_mode",
        help="Paste any AI API key below to auto-detect, or use Local Offline Mode with zero keys.",
    )

    provider_map = {
        "Auto-Detect from API Key": "auto",
        "Local Offline Mode (No Key)": "local",
        "Google Gemini": "gemini",
        "Groq Cloud": "groq",
        "OpenAI / Compatible (OpenRouter, DeepSeek)": "openai",
    }
    selected_provider = provider_map[provider_label]

    active_api_key = None
    custom_base_url = None
    if selected_provider != "local":
        universal_key = st.text_input(
            "AI API key (any provider)",
            value="",
            type="password",
            key="universal_api_key",
            placeholder="AIza... / gsk_... / sk-... (optional)",
            help="Supports Google Gemini, Groq, OpenAI, OpenRouter, DeepSeek, or leave blank for Local Offline Mode.",
        )
        active_api_key = universal_key.strip() or None

        if selected_provider == "auto":
            if active_api_key:
                resolved_provider = detect_provider_from_key(active_api_key)
                if active_api_key.startswith("sk-or-"):
                    st.caption(":material/key: Detected provider: **OPENROUTER (OpenAI-compatible)**")
                else:
                    st.caption(f":material/key: Detected provider: **{resolved_provider.upper()}**")
            elif os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
                resolved_provider = "gemini"
                st.caption(":material/key: Using environment `GEMINI_API_KEY`")
            elif os.getenv("GROQ_API_KEY"):
                resolved_provider = "groq"
                st.caption(":material/key: Using environment `GROQ_API_KEY`")
            elif os.getenv("OPENAI_API_KEY"):
                resolved_provider = "openai"
                st.caption(":material/key: Using environment `OPENAI_API_KEY`")
            else:
                resolved_provider = "local"
                st.caption(":material/offline_bolt: No key pasted — running in Local Offline Mode.")
        else:
            resolved_provider = selected_provider

        if resolved_provider == "openai" and active_api_key and importlib.util.find_spec("langchain_openai") is None:
            st.warning(
                "`langchain-openai` is not installed in this environment. "
                "Install it (`pip install langchain-openai`) for OpenAI / OpenRouter / DeepSeek inference; "
                "falling back to Local Offline Mode.",
                icon=":material/info:",
            )
    else:
        resolved_provider = "local"
        st.caption(":material/offline_bolt: Running locally with deterministic clinical synthesis.")

    # Lightweight, low-token model presets
    model_presets = {
        "gemini": [DEFAULT_GEMINI_MODEL, "gemini-2.5-flash", "Custom model ID..."],
        "groq": [DEFAULT_GROQ_MODEL, "llama-3.3-70b-versatile", "gemma2-9b-it", "Custom model ID..."],
        "openai": [DEFAULT_OPENAI_MODEL, "gpt-4o-mini", "gpt-4.1-mini", "Custom model ID..."],
        "local": ["mock-clinical-fallback"],
    }
    available_models = model_presets.get(resolved_provider, ["mock-clinical-fallback"])
    chosen_model = st.selectbox(
        "Lightweight model",
        available_models,
        index=0,
        key="model_choice",
        help="Defaults to fast, low-token-consuming text models.",
    )
    if chosen_model == "Custom model ID...":
        model_option = st.text_input(
            "Custom model ID",
            value=DEFAULT_GEMINI_MODEL if resolved_provider == "gemini" else DEFAULT_OPENAI_MODEL,
            key="custom_model_id",
        ).strip() or available_models[0]
    else:
        model_option = chosen_model

    with st.expander("Model & token controls", expanded=False):
        if resolved_provider == "openai":
            custom_base_url = (
                st.text_input(
                    "Custom Base URL (optional)",
                    value="",
                    key="custom_base_url",
                    placeholder="https://openrouter.ai/api/v1",
                    help="Leave blank for OpenAI, or enter OpenRouter / DeepSeek / Ollama endpoint.",
                ).strip()
                or None
            )
        max_tokens = st.slider(
            "Max output tokens",
            128,
            1024,
            512,
            64,
            key="max_tokens_slider",
            help="Caps output length to keep responses concise and token usage low.",
        )
        temperature = st.slider(
            "Sampling temperature",
            0.0,
            0.5,
            0.0,
            0.05,
            key="temperature_slider",
            help="0.0 enforces strict factual consistency and zero hallucination.",
        )

    if st.button("Clear chat history", icon=":material/delete:", width="stretch", key="clear_chat_btn"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("#### Quick test queries")
    sample_queries = [
        ("What is a virus?", "what is a virus"),
        ("What is bacteria?", "what is bacteria"),
        ("Malaria (1-word)", "malaria"),
        ("High blood pressure symptoms", "What are the symptoms of high blood pressure?"),
        ("Type 2 diabetes treatment", "How is Type 2 Diabetes treated?"),
        ("Bitcoin price (out-of-scope)", "What is the price of Bitcoin?"),
    ]
    for label, prompt_val in sample_queries:
        if st.button(label, key=f"btn_{label}", icon=":material/search:", width="stretch"):
            st.session_state.sample_prompt = prompt_val

# Apply editorial Light / Dark theme
apply_editorial_theme(active_theme)

# Load retriever and LLM chain
try:
    retriever, chunk_count = load_or_init_retriever()
    llm = get_llm(
        model_name=model_option,
        temperature=temperature,
        provider=selected_provider if selected_provider != "auto" else "auto",
        api_key=active_api_key,
        base_url=custom_base_url,
        max_tokens=max_tokens,
    )
    rag_chain = build_medirag_chain(retriever=retriever, llm=llm)
except Exception as e:
    st.error(f"Failed to initialize MediRAG chain: {e}", icon=":material/error:")
    st.stop()

# Editorial Hero Header
st.html(
    """
    <div class="mr-hero">
        <div class="mr-hero-eyebrow">Verified Clinical Reference · Strict Grounded RAG</div>
        <h1 class="mr-editorial-title" style="margin:0; font-size:2.35rem; line-height:1.15;">
            MediRAG Clinical Assistant
        </h1>
        <p class="mr-hero-sub" style="margin:0.4rem 0 0 0;">
            Evidence-based answers grounded strictly in indexed NIH, CDC, and NIAID records.
            Topics outside the vector store are refused in both Local and Cloud API modes to prevent hallucination.
        </p>
    </div>
    """
)

col1, col2, col3 = st.columns(3)
with col1:
    with st.container(border=True, key="mr_card_engine"):
        if llm.__class__.__name__ == "MockClinicalLLM":
            engine_label = "Local Offline"
            engine_delta = "Zero API cost"
        else:
            engine_label = f"{resolved_provider.title()}"
            engine_delta = model_option
        st.metric("Active engine", engine_label, delta=engine_delta)
with col2:
    with st.container(border=True, key="mr_card_chunks"):
        st.metric("Vector store chunks", f"{chunk_count:,}", delta="NIH · CDC · NIAID")
with col3:
    with st.container(border=True, key="mr_card_policy"):
        st.metric("Grounding policy", "Strict RAG", delta="Zero extrapolation")

# Suggestion chips when chat is empty
SUGGESTIONS = {
    ":material/coronavirus: What is a virus?": "what is a virus",
    ":material/microbiology: What is bacteria?": "what is bacteria",
    ":material/vaccines: How do vaccines work?": "What is immunity and how do vaccines work?",
    ":material/favorite: High blood pressure symptoms": "What are the symptoms of high blood pressure?",
}

prompt = st.chat_input("Ask a clinical question (e.g., what is a virus, malaria, asthma)...")

if st.session_state.sample_prompt:
    prompt = st.session_state.sample_prompt
    st.session_state.sample_prompt = None

if not st.session_state.messages and not prompt:
    empty_state_slot = st.empty()
    with empty_state_slot.container():
        with st.container(border=True, key="mr_note_grounding"):
            st.markdown(
                "**Strict Grounding Note:** MediRAG only answers topics present in its clinical vector store "
                "(2,560+ NIH/CDC MedQuAD records + foundational biology entries like *virus*, *bacteria*, *infection*, "
                "*fever*, *inflammation*, and *vaccines*). Connecting a cloud API key improves natural synthesis, "
                "while still strictly refusing unindexed or non-medical questions."
            )
        selected_chip = st.pills(
            "Try asking",
            list(SUGGESTIONS.keys()),
            label_visibility="collapsed",
            key="empty_state_pills",
        )
        if selected_chip:
            prompt = SUGGESTIONS[selected_chip]
    if prompt:
        empty_state_slot.empty()

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            if msg.get("thinking_time") is not None:
                st.caption(f":material/timer: Thought for {msg['thinking_time']:.2f}s")
            if msg.get("is_rejected"):
                st.warning(msg["content"], icon=":material/shield:")
            else:
                st.markdown(msg["content"])
                render_citations(msg.get("citations", []))

# Handle new user prompt
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        start_time = time.perf_counter()
        with st.status(
            ":shimmer[Searching clinical evidence & verifying guardrails...]",
            type="compact",
        ) as status:
            result = run_medirag_query(
                chain=rag_chain,
                query=prompt,
                chat_history=st.session_state.chat_history,
            )
            elapsed = time.perf_counter() - start_time
            status.update(
                label=f"Thought for {elapsed:.2f}s",
                state="complete",
                expanded=False,
            )

        answer = result["answer"]
        citations = result["citations"]
        is_rejected = result["is_rejected"]
        context_docs = result.get("context", [])

        if is_rejected:
            st.warning(answer, icon=":material/shield:")
        else:
            st.markdown(answer)
            render_citations(citations)

            if context_docs:
                with st.expander("View retrieved clinical reference passages", icon=":material/article:"):
                    for idx, doc in enumerate(context_docs, 1):
                        with st.container(border=True, key=f"mr_doc_{idx}_{len(st.session_state.messages)}"):
                            st.caption(f"Document #{idx} · Source: {doc.metadata.get('source', 'NIH')}")
                            st.markdown(doc.page_content)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "is_rejected": is_rejected,
            "thinking_time": elapsed,
        }
    )
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append(AIMessage(content=answer))
