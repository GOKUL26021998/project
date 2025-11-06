# app.py
"""
Main Streamlit app providing UI for:
- OpenAI API key validation
- File uploads and embedding
- ChatGPT-like conversation
- Multi-model selection
- Chat history management
"""

import streamlit as st
from openai_key import validate_openai_key
from file_handler import load_document, chunk_and_embed
from chat_engine import chat_with_docs

# -------------------- Streamlit UI Layout --------------------

st.set_page_config(page_title="LangChain GenAI Chat", layout="wide")
st.title("💬 LangChain Generative AI Chat")

# -------------------- Session Initialization --------------------

if "history" not in st.session_state:
    st.session_state["history"] = []
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""
if "model" not in st.session_state:
    st.session_state["model"] = "gpt-4o"

# -------------------- Sidebar: Settings --------------------

st.sidebar.header("⚙️ Settings")

# Input: OpenAI API Key
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

# Dropdown: Select Model
model_choice = st.sidebar.selectbox(
    "Choose OpenAI Model:",
    options=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    index=0
)
st.session_state["model"] = model_choice

# Validate API Key
if api_key and validate_openai_key(api_key):
    st.session_state["api_key"] = api_key
else:
    st.stop()

# -------------------- File Upload Section --------------------

uploaded_file = st.file_uploader(
    "📂 Upload a document (PDF, Word, CSV, Excel, JSON)",
    type=["pdf", "csv", "xls", "xlsx", "json", "doc", "docx"]
)

if uploaded_file:
    docs = load_document(uploaded_file)
    if docs:
        num_chunks = chunk_and_embed(docs, st.session_state["api_key"])
        st.success(f"✅ Successfully embedded {num_chunks} chunks from your file.")

# -------------------- Chat Interface --------------------

st.subheader("💬 Chat with Your Data")

# Button: Create a new chat session
if st.button("🆕 New Chat"):
    st.session_state["history_archive"] = st.session_state.get("history_archive", [])
    st.session_state["history_archive"].append(st.session_state["history"])
    st.session_state["history"] = []
    st.success("✨ New chat session started!")

# Input: User query
user_input = st.text_input("Ask a question about your documents:")

# Generate AI response
if user_input:
    response, st.session_state["history"] = chat_with_docs(
        user_input,
        st.session_state["api_key"],
        st.session_state["model"],
        st.session_state["history"]
    )
    st.markdown(f"**Assistant:** {response}")

# Display current chat messages
if st.session_state["history"]:
    with st.expander("📜 Current Chat History"):
        for chat in st.session_state["history"]:
            st.write(f"**You:** {chat['user']}")
            st.write(f"**Assistant:** {chat['assistant']}")

# -------------------- Sidebar: Chat History Archive --------------------

if "history_archive" in st.session_state and st.session_state["history_archive"]:
    with st.sidebar.expander("🕓 Past Chat Sessions"):
        for i, session in enumerate(st.session_state["history_archive"], 1):
            st.write(f"**Chat {i}:** {len(session)} messages")
