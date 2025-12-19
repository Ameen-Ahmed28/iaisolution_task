import streamlit as st
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from multi_modal_rag_system import MultiModalRAGSystem


# Page config
st.set_page_config(
    page_title="DocuMind AI | Multi-Modal RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern dark theme
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Global white text */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div,
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
    .stTextInput label, .stSelectbox label, .stSlider label,
    .stRadio label, .stCheckbox label, .stFileUploader label,
    [data-testid="stWidgetLabel"], [data-testid="stCaptionContainer"] {
        color: #ffffff !important;
    }
    
    /* Slightly dimmer for secondary text */
    .stCaption, caption, .stApp small {
        color: #c0c0d0 !important;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #a0a0b0;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-2px);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: #a0a0b0;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    
    /* Chat styling */
    .chat-message {
        padding: 1rem 1.5rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        animation: fadeIn 0.3s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
        border-bottom-right-radius: 4px;
    }
    
    .assistant-message {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #e0e0e0;
        margin-right: 20%;
        border-bottom-left-radius: 4px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #a0a0b0;
        font-weight: 500;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: transparent !important;
        border: none !important;
        border-radius: 12px;
        padding: 0;
    }
    
    /* Inner drop zone - black with white border */
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section > div,
    [data-testid="stFileDropzone"],
    [data-testid="stFileUploader"] [data-testid="stFileDropzoneInstructions"] {
        background: #000000 !important;
        border: 2px solid #ffffff !important;
        border-radius: 12px !important;
    }
    
    [data-testid="stFileUploader"] section > div:first-child {
        background: #000000 !important;
        border: 2px solid #ffffff !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: rgba(46, 204, 113, 0.2);
        border: 1px solid rgba(46, 204, 113, 0.5);
        border-radius: 8px;
    }
    
    .stError {
        background: rgba(231, 76, 60, 0.2);
        border: 1px solid rgba(231, 76, 60, 0.5);
        border-radius: 8px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.5);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(102, 126, 234, 0.8);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_rag_system():
    """Initialize RAG system once and cache it"""
    try:
        rag = MultiModalRAGSystem()
        return rag
    except Exception as e:
        st.error(f"❌ Failed to initialize: {str(e)}")
        st.info("💡 Make sure GROQ_API_KEY is set in your .env file")
        st.stop()


def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 DocuMind AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Modal Document Intelligence powered by Groq</p>', unsafe_allow_html=True)

    # Initialize RAG
    rag = init_rag_system()

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Retrieval settings
        k_docs = st.slider("📚 Documents to retrieve", 2, 10, 4)
        
        st.divider()
        
        # Features section
        st.markdown("""
        ### 🎯 Features
        - 📑 **PDF & Image Processing**
        - 🔍 **Semantic Search**
        - 🧠 **Groq LLM (Ultra-fast)**
        - 📝 **Source Citations**
        - 📤 **Batch Upload**
        - 💾 **Persistent Storage**
        """)
        
        st.divider()
        
        # API Status
        st.markdown("### 🔌 API Status")
        if os.getenv("GROQ_API_KEY"):
            st.success("✅ Groq API Connected")
        else:
            st.error("❌ Groq API Key Missing")
        
        st.divider()
        
        # Stats
        st.markdown("### 📊 Current Session")
        st.metric("Documents Indexed", len(rag.docstore))

    # Main content - Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["💬 Chat", "📤 Upload Documents", "📊 Knowledge Base", "ℹ️ About"]
    )

    # ============= TAB 1: Chat =============
    with tab1:
        st.markdown("### 💬 Ask Questions About Your Documents")
        
        # Check if documents exist
        if not rag.docstore:
            st.info("📭 No documents indexed yet. Upload documents in the '📤 Upload Documents' tab to get started!")
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    st.caption(f"📚 Sources: {', '.join(set(message['sources']))}")

        # Input
        query = st.chat_input("Ask a question about your documents...")

        if query:
            # User message
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        result = rag.qa_pipeline(query, k=k_docs)

                        # Answer
                        st.markdown(result["answer"])

                        # Sources
                        if result["sources"]:
                            st.caption(
                                f"📚 **Sources:** {', '.join(set(result['sources']))}"
                            )

                        # Store in session
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": result["answer"],
                                "sources": result["sources"],
                            }
                        )

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    # ============= TAB 2: Upload Documents (Multiple Files) =============
    with tab2:
        st.markdown("### 📤 Upload & Index Documents")
        st.markdown("Upload multiple PDF or image files to build your knowledge base.")
        
        # Multiple file uploader
        uploaded_files = st.file_uploader(
            "Choose files (PDF, PNG, JPG, TXT, MD, CSV)",
            type=["pdf", "png", "jpg", "jpeg", "txt", "md", "csv"],
            accept_multiple_files=True,
            help="You can select multiple files at once. Supports PDF, images, and text files.",
        )

        if uploaded_files:
            st.markdown(f"**📁 Selected {len(uploaded_files)} file(s):**")
            
            # Show file list
            for f in uploaded_files:
                st.markdown(f"- {f.name} ({f.size / 1024:.1f} KB)")
            
            # Process button
            if st.button("🚀 Process All Files", use_container_width=True):
                # Create temp directory
                temp_path = Path("./temp_upload")
                temp_path.mkdir(exist_ok=True)
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_docs = 0
                total_chunks = 0
                processed_files = []
                failed_files = []
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.markdown(f"📥 Processing: **{uploaded_file.name}** ({idx + 1}/{len(uploaded_files)})")
                    
                    # Save temp file
                    file_path = temp_path / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        # Ingest
                        docs = rag.ingest_document(str(file_path))
                        
                        # Chunk
                        chunked = rag.chunk_documents(docs, chunk_size=512, chunk_overlap=100)
                        
                        # Index
                        rag.add_documents_to_index(chunked)
                        
                        total_docs += len(docs)
                        total_chunks += len(chunked)
                        processed_files.append(uploaded_file.name)
                        
                    except Exception as e:
                        failed_files.append((uploaded_file.name, str(e)))
                
                progress_bar.progress(1.0)
                status_text.empty()
                
                # Summary
                st.markdown("---")
                st.markdown("### ✅ Processing Complete!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(processed_files)}</div>
                        <div class="metric-label">Files Processed</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{total_docs}</div>
                        <div class="metric-label">Elements Extracted</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{total_chunks}</div>
                        <div class="metric-label">Chunks Indexed</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show processed files
                if processed_files:
                    with st.expander("✅ Successfully Processed Files"):
                        for fname in processed_files:
                            st.markdown(f"- ✅ {fname}")
                
                # Show failed files
                if failed_files:
                    with st.expander("❌ Failed Files"):
                        for fname, error in failed_files:
                            st.markdown(f"- ❌ {fname}: {error}")

    # ============= TAB 3: Knowledge Base =============
    with tab3:
        st.markdown("### 📊 Knowledge Base Overview")

        if rag.docstore:
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(rag.docstore)}</div>
                    <div class="metric-label">Total Documents</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                try:
                    vector_count = len(rag.vectorstore.get()["ids"])
                except:
                    vector_count = 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{vector_count}</div>
                    <div class="metric-label">Vector Embeddings</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                avg_len = int(sum(len(d.page_content) for d in rag.docstore.values()) / len(rag.docstore))
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{avg_len}</div>
                    <div class="metric-label">Avg. Doc Length</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            
            # Document list
            st.markdown("### 📑 Indexed Documents")
            
            # Group by source
            sources = {}
            for doc_id, doc in rag.docstore.items():
                source = doc.metadata.get('source', 'Unknown')
                if source not in sources:
                    sources[source] = []
                sources[source].append(doc)
            
            for source, docs in sources.items():
                with st.expander(f"📄 {source} ({len(docs)} chunks)"):
                    for i, doc in enumerate(docs[:5]):
                        st.markdown(f"**Chunk {i+1}** - {doc.metadata.get('element_type', 'Text')}")
                        st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        st.markdown("---")
                    if len(docs) > 5:
                        st.info(f"Showing 5 of {len(docs)} chunks")

            # Clear database button
            st.markdown("---")
            if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
                rag.clear_all()
                st.success("✅ Knowledge base cleared!")
                st.rerun()

        else:
            st.info("📭 No documents indexed yet. Upload documents to get started!")

    # ============= TAB 4: About =============
    with tab4:
        st.markdown("### ℹ️ About DocuMind AI")

        st.markdown("""
        <div class="glass-card">
            <h3>🏗️ Architecture</h3>
            <p>DocuMind AI is a production-ready Multi-Modal RAG system that combines:</p>
            <ul>
                <li><strong>Document Processing</strong> - PyMuPDF for PDF parsing, PIL + Tesseract for OCR</li>
                <li><strong>Embeddings</strong> - HuggingFace BGE-small (local, no API needed)</li>
                <li><strong>Vector Database</strong> - ChromaDB for semantic search</li>
                <li><strong>LLM Inference</strong> - Groq API (ultra-fast, free tier available)</li>
                <li><strong>Persistence</strong> - Documents survive app restarts!</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <h3>🚀 Key Features</h3>
            <ul>
                <li>✅ Multi-file batch upload</li>
                <li>✅ PDF, Image, and Text file support</li>
                <li>✅ Automatic table extraction</li>
                <li>✅ OCR for scanned documents</li>
                <li>✅ Semantic search with RAG</li>
                <li>✅ Source citations in answers</li>
                <li>✅ Ultra-fast Groq LLM inference</li>
                <li>✅ <strong>Persistent storage (survives restarts)</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <h3>📊 Performance</h3>
            <ul>
                <li><strong>Document Processing:</strong> 1-3 sec per document</li>
                <li><strong>Query Latency:</strong> 0.5-2 sec per question</li>
                <li><strong>Embedding Model:</strong> BAAI/bge-small-en-v1.5</li>
                <li><strong>LLM:</strong> Groq Llama 3.3 70B (500+ tokens/sec)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()