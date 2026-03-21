import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

# On Streamlit Cloud, secrets are in st.secrets not .env
# This ensures GROQ_API_KEY is available as an env var in both environments
try:
    import streamlit as _st
    if "GROQ_API_KEY" in _st.secrets:
        os.environ["GROQ_API_KEY"] = _st.secrets["GROQ_API_KEY"]
except Exception:
    pass
from utils.logger import log
from agent.generate_answer import GENERATE_PROMPT

# Only import get_retriever on local — avoids KeyError on Streamlit Cloud
import os as _os
IS_LOCAL = not _os.path.exists("/mount/src")
if IS_LOCAL:
    from agent.retriever_tool import get_retriever
import time

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AgentRAG",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM CSS — Dark terminal/agentic theme
# -----------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Syne:wght@400;600;700;800&display=swap');

/* ── Base ── */
html, body, [data-testid="stApp"] {
    background-color: #0a0a0f;
    color: #e2e8f0;
    font-family: 'Syne', sans-serif;
}

/* ── Hide default streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Force sidebar always visible — hide collapse arrow button ── */
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="stSidebarContent"] { display: block !important; }
section[data-testid="stSidebar"] { 
    transform: none !important;
    min-width: 280px !important;
    width: 280px !important;
}

/* ── Expander — Terminal Logs ── */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: #020207 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #6366f1 !important;
    padding: 8px 12px !important;
    background: transparent !important;
    border: none !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    color: #a5b4fc !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
    fill: #6366f1 !important;
}

/* ── Main container ── */
[data-testid="stAppViewContainer"] {
    background: #0a0a0f;
}
[data-testid="stMain"] {
    background: #0a0a0f;
}

/* ── HEADER BANNER ── */
.agent-byline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #475569;
    letter-spacing: 1.5px;
    margin-top: 2px;
}
.agent-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 28px 8px 12px 8px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 24px;
}
.agent-logo {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    box-shadow: 0 0 24px rgba(99,102,241,0.4);
    flex-shrink: 0;
}
.agent-title-block { display: flex; flex-direction: column; gap: 2px; }
.agent-title {
    font-family: 'Syne', sans-serif;
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #a5b4fc, #e2e8f0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.agent-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #6366f1;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.agent-byline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #475569;
    letter-spacing: 1px;
    margin-top: 2px;
}
.agent-status-pill {
    margin-left: auto;
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.3);
    color: #4ade80;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    padding: 4px 12px;
    border-radius: 20px;
    text-transform: uppercase;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 8px 0 !important;
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"] .stMarkdown,
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] {
    background: #1e1b4b;
    border: 1px solid #312e81;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #111827 !important;
    border: 1px solid #374151 !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
}
[data-testid="stChatInputSubmitButton"] svg {
    fill: #6366f1 !important;
}

/* ── Status widget ── */
[data-testid="stStatus"] {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: #94a3b8 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #070710 !important;
    border-right: 1px solid #1e293b !important;
}
[data-testid="stSidebar"] * {
    font-family: 'JetBrains Mono', monospace;
}

/* ── Sidebar header ── */
.sidebar-header {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6366f1;
    padding: 8px 0 4px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sidebar-dot {
    width: 6px; height: 6px;
    background: #6366f1;
    border-radius: 50%;
    box-shadow: 0 0 8px #6366f1;
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── Log terminal box ── */
.log-terminal {
    background: #020207;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    color: #4ade80;
    line-height: 1.7;
    max-height: 420px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
}
.log-terminal .log-time { color: #475569; }
.log-terminal .log-msg  { color: #4ade80; }

/* ── Clear button ── */
[data-testid="stSidebar"] button {
    background: #1e1b4b !important;
    border: 1px solid #312e81 !important;
    color: #a5b4fc !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    border-radius: 8px !important;
    padding: 6px 14px !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebar"] button:hover {
    background: #312e81 !important;
    border-color: #6366f1 !important;
    box-shadow: 0 0 12px rgba(99,102,241,0.3) !important;
}

/* ── Divider ── */
hr {
    border-color: #1e293b !important;
    margin: 12px 0 !important;
}

/* ── Stats row ── */
.stats-row {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
}
.stat-chip {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 10px;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
    flex: 1;
    text-align: center;
}
.stat-chip span {
    display: block;
    color: #a5b4fc;
    font-size: 13px;
    font-weight: 600;
}

/* ── Welcome screen ── */
.welcome-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 20px;
    text-align: center;
    gap: 16px;
}
.welcome-hex {
    font-size: 52px;
    filter: drop-shadow(0 0 20px rgba(99,102,241,0.6));
    animation: float 4s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-10px); }
}
.welcome-title {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #e2e8f0;
}
.welcome-desc {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #475569;
    max-width: 400px;
    line-height: 1.8;
}
.suggestion-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-top: 8px;
}
.suggestion-chip {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 20px;
    padding: 6px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #64748b;
    cursor: pointer;
}
.suggestion-chip:hover {
    border-color: #6366f1;
    color: #a5b4fc;
}

/* ── Suggestion buttons styled as chips ── */
div[data-testid="stHorizontalBlock"] button {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 20px !important;
    color: #64748b !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    padding: 6px 14px !important;
    transition: all 0.2s !important;
}
div[data-testid="stHorizontalBlock"] button:hover {
    border-color: #6366f1 !important;
    color: #a5b4fc !important;
    background: #1e1b4b !important;
}

/* ── Header compact ── */
.agent-header {
    padding: 16px 8px 12px 8px !important;
    margin-bottom: 16px !important;
}

/* ── User chat bubble ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #0f0f1a !important;
    border: 1px solid #1e1b4b !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin-bottom: 8px !important;
}

/* ── Assistant chat bubble ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: #0a0a14 !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin-bottom: 8px !important;
}

/* ── Fix chat input border — remove pink/red ── */
[data-testid="stChatInput"] textarea {
    background: #111827 !important;
    color: #e2e8f0 !important;
    border: none !important;
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stBottom"] {
    background: #0a0a0f !important;
    border-top: 1px solid #1e293b !important;
    padding-top: 8px !important;
}
div[data-testid="stChatInput"] {
    border: 1px solid #374151 !important;
    border-radius: 12px !important;
    background: #111827 !important;
}
div[data-testid="stChatInput"]:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
}

/* ── File uploader — dark themed ── */
[data-testid="stFileUploader"] {
    background: #0f172a !important;
    border: 1px dashed #1e293b !important;
    border-radius: 10px !important;
    padding: 8px !important;
}
[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
    padding: 8px !important;
}
[data-testid="stFileUploader"] section > div {
    background: transparent !important;
}
[data-testid="stFileUploader"] span {
    color: #475569 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    color: #475569 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    font-size: 10px !important;
}
/* Browse files button */
[data-testid="stFileUploaderDropzone"] button {
    background: #1e1b4b !important;
    border: 1px solid #312e81 !important;
    color: #a5b4fc !important;
    font-size: 10px !important;
    padding: 4px 12px !important;
    border-radius: 6px !important;
}
/* Uploaded file pill */
[data-testid="stFileUploaderFile"] {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploaderFileName"] {
    color: #a5b4fc !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
}
[data-testid="stFileUploaderFileData"] {
    color: #475569 !important;
    font-size: 9px !important;
}

/* ── Source citations ── */
.source-section {
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid #1e293b;
}
.source-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: #334155;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.source-link {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #6366f1;
    text-decoration: none;
    display: inline-block;
    margin-right: 12px;
    margin-top: 2px;
    transition: color 0.2s;
}
.source-link:hover { color: #a5b4fc; }
.source-file {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #475569;
    display: inline-block;
    margin-right: 12px;
    margin-top: 2px;
}

/* ── Query history buttons ── */
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid #1e293b !important;
    color: #64748b !important;
    font-size: 10px !important;
    text-align: left !important;
    padding: 4px 10px !important;
    border-radius: 6px !important;
    margin-bottom: 3px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"]:hover {
    border-color: #6366f1 !important;
    color: #a5b4fc !important;
    background: #0f0f1a !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    color: #64748b !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: #6366f1 !important;
    color: #a5b4fc !important;
    background: #1e1b4b !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# CACHED RESOURCES
# -----------------------------

@st.cache_resource
def load_graph():
    from agent.graph import graph
    return graph

@st.cache_resource
def load_streaming_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=300,
        api_key=os.getenv("GROQ_API_KEY")
    )

app = load_graph()
streaming_llm = load_streaming_llm()

# -----------------------------
# SESSION STATE
# -----------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "retrieval_count" not in st.session_state:
    st.session_state.retrieval_count = 0
if "full_messages" not in st.session_state:
    # Stores actual LangChain message objects for conversation memory
    # Sliding window of last 10 messages to stay within LLM context limits
    st.session_state.full_messages = []
if "selected_suggestion" not in st.session_state:
    st.session_state.selected_suggestion = None
if "uploaded_files" not in st.session_state:
    # Tracks filenames already indexed — prevents duplicate ingestion
    st.session_state.uploaded_files = set()
if "query_history" not in st.session_state:
    # Stores last 10 queries for quick re-run from sidebar
    st.session_state.query_history = []

# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:

    st.markdown('<div class="sidebar-header"><div class="sidebar-dot"></div>AGENT MONITOR</div>', unsafe_allow_html=True)

    # Stats chips
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-chip"><span>{st.session_state.query_count}</span>Queries</div>
        <div class="stat-chip"><span>{st.session_state.retrieval_count}</span>Retrievals</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⟳  Clear Session"):
        st.session_state.chat_history = []
        st.session_state.debug_logs = []
        st.session_state.query_count = 0
        st.session_state.retrieval_count = 0
        st.session_state.full_messages = []
        st.session_state.selected_suggestion = None
        st.session_state.uploaded_files = set()
        st.session_state.query_history = []
        st.rerun()

    st.divider()

    st.divider()

    if IS_LOCAL:
        st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:11px;color:#6366f1;letter-spacing:2px;text-transform:uppercase;margin:4px 0 8px 0;">📁 Upload Documents</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
    else:
        uploaded = None
        st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:10px;color:#334155;margin:4px 0;">📁 File upload available in local version only.</p>', unsafe_allow_html=True)

    if uploaded:
        from split.splitter import split_documents
        from langchain_core.documents import Document
        import tempfile, os

        new_files = [f for f in uploaded if f.name not in st.session_state.uploaded_files]

        if new_files:
            with st.spinner(f"Indexing {len(new_files)} file(s)..."):
                all_new_docs = []
                for file in new_files:
                    try:
                        # Save uploaded file to temp location
                        suffix = ".pdf" if file.name.endswith(".pdf") else ".txt"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(file.read())
                            tmp_path = tmp.name

                        # Load using appropriate loader
                        if suffix == ".pdf":
                            from langchain_community.document_loaders import PyPDFLoader
                            loader = PyPDFLoader(tmp_path)
                        else:
                            from langchain_community.document_loaders import TextLoader
                            loader = TextLoader(tmp_path)

                        docs = loader.load()

                        # Set source metadata to original filename
                        for doc in docs:
                            doc.metadata["source"] = file.name

                        all_new_docs.extend(docs)
                        os.unlink(tmp_path)  # clean up temp file

                    except Exception as e:
                        st.error(f"Failed to load {file.name}: {e}")

                if all_new_docs:
                    # Split into chunks
                    chunks = split_documents(all_new_docs)

                    # Append to existing FAISS index
                    retriever_instance = get_retriever()
                    added = retriever_instance.add_documents(chunks)

                    # Mark files as indexed
                    for file in new_files:
                        st.session_state.uploaded_files.add(file.name)

                    st.success(f"✅ Indexed {added} chunks from {len(new_files)} file(s)")

    # Show already indexed files
    if st.session_state.uploaded_files:
        st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:10px;color:#475569;margin:4px 0;">Indexed files:</p>', unsafe_allow_html=True)
        for fname in st.session_state.uploaded_files:
            st.markdown(f'<p style="font-family:JetBrains Mono,monospace;font-size:10px;color:#4ade80;margin:2px 0;">⬡ {fname}</p>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:11px;color:#6366f1;letter-spacing:2px;text-transform:uppercase;margin:4px 0 8px 0;">🕐 Query History</p>', unsafe_allow_html=True)

    if st.session_state.query_history:
        for i, q in enumerate(reversed(st.session_state.query_history[-10:])):
            truncated = q[:35] + "..." if len(q) > 35 else q
            if st.button(truncated, key=f"qhist_{i}", use_container_width=True):
                st.session_state.selected_suggestion = q
                st.rerun()
    else:
        st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:10px;color:#334155;margin:4px 0;">// no queries yet</p>', unsafe_allow_html=True)

    st.divider()

    # Export chat button
    st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:11px;color:#6366f1;letter-spacing:2px;text-transform:uppercase;margin:4px 0 8px 0;">💾 Export</p>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        # Build plain text export
        export_lines = ["AgentRAG — Chat Export", "Built by Amitoj Singh", "=" * 40, ""]
        for role, message in st.session_state.chat_history:
            prefix = "You" if role == "user" else "AgentRAG"
            export_lines.append(f"{prefix}:")
            export_lines.append(message)
            export_lines.append("")
        export_text = "\n".join(export_lines)
        st.download_button(
            label="⬇  Download Chat",
            data=export_text,
            file_name="agentrag_chat.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:10px;color:#334155;margin:4px 0;">// no chat to export</p>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:11px;color:#6366f1;letter-spacing:2px;text-transform:uppercase;margin:4px 0 8px 0;">📋 Terminal Logs</p>', unsafe_allow_html=True)
    log_placeholder = st.empty()

def render_logs():
    """Render logs as styled terminal HTML."""
    logs = st.session_state.debug_logs[-40:]
    if not logs:
        html = '<div class="log-terminal" style="color:#334155;">// no logs yet</div>'
    else:
        lines = ""
        for entry in logs:
            # Split [HH:MM:SS] from message
            if entry.startswith("[") and "]" in entry:
                ts = entry[:10]
                msg = entry[11:]
                lines += f'<span class="log-time">{ts}</span> <span class="log-msg">{msg}</span>\n'
            else:
                lines += f'<span class="log-msg">{entry}</span>\n'
        html = f'<div class="log-terminal">{lines}</div>'
    log_placeholder.markdown(html, unsafe_allow_html=True)

render_logs()

# -----------------------------
# HEADER
# -----------------------------

st.markdown("""
<div class="agent-header">
    <div class="agent-logo">⬡</div>
    <div class="agent-title-block">
        <div class="agent-title">AgentRAG</div>
        <div class="agent-subtitle">Retrieval-Augmented Intelligence</div>
        <div class="agent-byline">Built by Amitoj Singh</div>
    </div>
    <div class="agent-status-pill">● ONLINE</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# HELPER: log to terminal + UI
# -----------------------------

def ui_log(message: str):
    entry = log(message)
    st.session_state.debug_logs.append(entry)
    render_logs()  # update sidebar live after every log

# -----------------------------
# CHAT HISTORY or WELCOME SCREEN
# -----------------------------

SUGGESTIONS = [
    "What is agentic AI?",
    "How does RAG work?",
    "Agentic vs Generative AI",
    "What are AI agents?",
]

if not st.session_state.chat_history:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-hex">⬡</div>
        <div class="welcome-title">What do you want to know?</div>
        <div class="welcome-desc">
            I retrieve answers from your knowledge base using FAISS vector search
            and local LLMs. No cloud. No data leaving your machine.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Clickable suggestion buttons styled as chips
    cols = st.columns(len(SUGGESTIONS))
    for i, suggestion in enumerate(SUGGESTIONS):
        with cols[i]:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                st.session_state.selected_suggestion = suggestion
                st.rerun()
else:
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

# -----------------------------
# USER INPUT & STREAMING RESPONSE
# -----------------------------

# Always render chat_input so it appears after suggestion completes
typed_prompt = st.chat_input("Ask anything about your knowledge base...")

# Use suggestion if clicked, otherwise use typed input
if st.session_state.selected_suggestion:
    prompt = st.session_state.selected_suggestion
    st.session_state.selected_suggestion = None
else:
    prompt = typed_prompt

if prompt:

    # Clear welcome screen by adding to history
    st.session_state.chat_history.append(("user", prompt))
    st.session_state.query_count += 1

    # Add to query history (avoid duplicates, keep last 10)
    if prompt not in st.session_state.query_history:
        st.session_state.query_history.append(prompt)
        if len(st.session_state.query_history) > 10:
            st.session_state.query_history.pop(0)

    # Add current question to conversation memory
    st.session_state.full_messages.append(HumanMessage(content=prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        # ── Step 1: Routing + Retrieval ──
        with st.status("⬡  Routing query through agent graph...", expanded=False) as status:
            ui_log("Graph invoked — routing query")

            # Build clean memory window for Groq tool calling
            # Groq fails if history contains:
            # 1. ToolMessages (raw retrieval results)
            # 2. AIMessages with tool_calls (prior routing decisions)
            # Only pass plain HumanMessages and plain AIMessages (text only)
            from langchain_core.messages import ToolMessage
            clean_history = []
            for m in st.session_state.full_messages[-10:]:
                if isinstance(m, ToolMessage):
                    continue  # skip tool results
                if isinstance(m, AIMessage) and m.tool_calls:
                    continue  # skip AI messages that contain tool calls
                clean_history.append(m)

            # Always ensure current question is the last message
            if not clean_history or clean_history[-1].content != prompt:
                clean_history.append(HumanMessage(content=prompt))

            memory_window = clean_history
            ui_log(f"Memory: {len(memory_window)} messages in context")

            result = app.invoke({
                "messages": memory_window,
                "doc_grade": None,
                "rewrite_count": 0
            })

            status.update(label="✅  Graph complete", state="complete")
            ui_log("Graph completed")

        # ── Step 2: Extract context ──
        context = ""
        sources = []
        retrieved = False

        for msg in result["messages"]:
            if hasattr(msg, "type") and msg.type == "tool":
                raw = msg.content
                try:
                    import ast
                    parsed = ast.literal_eval(raw)
                    context = parsed.get("text", "")
                    num_chunks = len(parsed.get("chunks", []))
                    sources = parsed.get("sources", [])
                    chunks = parsed.get("chunks", [])
                    retrieved = True
                    st.session_state.retrieval_count += 1
                    ui_log(f"Retrieval triggered — {num_chunks} chunks fetched")
                    ui_log(f"Sources: {sources}")
                    ui_log(f"Context length: {len(context)} chars")
                    # Log each chunk preview in debug panel
                    for ci, chunk in enumerate(chunks):
                        preview = chunk[:120].replace("\n", " ")
                        ui_log(f"  Chunk {ci+1}: {preview}...")
                except Exception:
                    context = raw
                    sources = []
                    retrieved = True
                    st.session_state.retrieval_count += 1
                    ui_log(f"Retrieval triggered — context length: {len(context)} chars")
                break

        if not retrieved:
            ui_log("No retrieval — LLM answering directly")

        # ── Step 3: Stream answer ──
        if context:
            from agent.generate_answer import GENERATE_PROMPT

            # Build conversation history string from last 6 messages (3 exchanges)
            history_msgs = st.session_state.full_messages[-7:-1]  # exclude current question
            history = "\n".join([
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                for m in history_msgs
            ]) if history_msgs else "No previous conversation."

            stream_prompt = GENERATE_PROMPT.format(
                question=prompt,
                history=history,
                context=context[:1200]
            )
            ui_log("Streaming answer from retrieved context...")
            ai_message = st.write_stream(
                chunk.content
                for chunk in streaming_llm.stream(stream_prompt)
                if chunk.content
            )
        else:
            ui_log("Streaming direct answer...")
            ai_message = st.write_stream(
                chunk.content
                for chunk in streaming_llm.stream(prompt)
                if chunk.content
            )

        # ── Show source citations below answer ──
        if sources:
            from urllib.parse import urlparse
            links_html = ""
            for src in sources:
                if src.startswith("http"):
                    domain = urlparse(src).netloc.replace("www.", "")
                    links_html += f"<a href='{src}' target='_blank' class='source-link'>⬡ {domain}</a>"
                else:
                    links_html += f"<span class='source-file'>⬡ {src}</span>"
            st.markdown(
                f"<div class='source-section'>"
                f"<div class='source-label'>Sources</div>"
                f"{links_html}"
                f"</div>",
                unsafe_allow_html=True
            )

        ui_log("Answer complete")
        render_logs()

    final_answer = ai_message or "I couldn't generate a response."
    st.session_state.chat_history.append(("assistant", final_answer))

    # Add AI response to conversation memory
    st.session_state.full_messages.append(AIMessage(content=final_answer))