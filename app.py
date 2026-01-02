import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
import tempfile
import os
# Page Config
st.set_page_config(page_title="RAG Similarity Visualizer", layout="wide")
st.title("🔍 RAG Similarity Visualizer")
st.markdown("### Peek inside the 'Black Box' of Vector Search")
# Initialize session state for persistent storage across reruns
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "docs" not in st.session_state:
    st.session_state.docs = []
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None
# 1. Setup Embeddings (Using a free, local model)
@st.cache_resource
def load_embeddings():
    """Load embeddings model with caching to avoid reloading."""
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception as e:
        st.error(f"Failed to load embeddings model: {e}")
        return None
def process_pdf(uploaded_file):
    """Process uploaded PDF and create vector store."""
    try:
        # Create a temporary file to store the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        # Load and Split
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        if not pages:
            st.error("Could not extract any content from the PDF.")
            return None, []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        docs = text_splitter.split_documents(pages)
        if not docs:
            st.error("No text chunks were created from the document.")
            return None, []
        # Create Vector Store
        embeddings = load_embeddings()
        if embeddings is None:
            return None, []
        vectorstore = FAISS.from_documents(docs, embeddings)
        # Cleanup temp file
        os.unlink(tmp_path)
        return vectorstore, docs
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None, []
def display_results(results_with_scores):
    """Display search results with visual formatting."""
    st.subheader("🎯 Top 5 Retrieved Chunks")
    for i, (doc, score) in enumerate(results_with_scores):
        # Normalize score for a "Match %" feel
        # FAISS L2 distance: lower is better, typically 0-2 range for normalized embeddings
        confidence = max(0, min(100, 100 - (score * 50)))
        with st.expander(f"📄 Chunk #{i+1} | Match Score: {confidence:.1f}%", expanded=(i == 0)):
            st.markdown(f"**Content:**")
            st.write(doc.page_content)
            st.progress(confidence / 100)
            
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"📏 Raw L2 Distance: {score:.4f}")
            with col2:
                if hasattr(doc, 'metadata') and doc.metadata:
                    st.caption(f"📄 Page: {doc.metadata.get('page', 'N/A')}")
def display_statistics(vectorstore, query, total_docs):
    """Display statistical analysis of the retrieval."""
    st.divider()
    st.subheader("📊 Statistical Analysis of Retrieval")
    try:
        all_docs = vectorstore.similarity_search_with_score(query, k=total_docs)
        scores = [s for d, s in all_docs]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Chunks", len(scores))
        with col2:
            st.metric("Min Distance", f"{min(scores):.4f}")
        with col3:
            st.metric("Max Distance", f"{max(scores):.4f}")
        chart_data = pd.DataFrame({
            "Chunk Index": range(len(scores)),
            "Distance": scores
        })
        
        st.area_chart(chart_data.set_index("Chunk Index"))
        st.caption(
            "📈 This chart shows the distribution of 'distance' across your entire document. "
            "A sharp drop-off at the start indicates a very precise match."
        )
    except Exception as e:
        st.warning(f"Could not generate statistics: {e}")
# Main Application Flow
embeddings = load_embeddings()
if embeddings is None:
    st.error("❌ Could not load the embeddings model. Please check your installation.")
    st.stop()
# 2. File Upload Section
st.sidebar.header("📁 Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload a PDF to analyze", type="pdf")
# Process file only when a new file is uploaded
if uploaded_file is not None:
    # Check if this is a new file
    if st.session_state.current_file_name != uploaded_file.name:
        with st.spinner("🔄 Processing PDF..."):
            vectorstore, docs = process_pdf(uploaded_file)
            
            if vectorstore is not None:
                st.session_state.vectorstore = vectorstore
                st.session_state.docs = docs
                st.session_state.file_processed = True
                st.session_state.current_file_name = uploaded_file.name
                st.success(f"✅ Successfully processed **{len(docs)}** chunks from '{uploaded_file.name}'!")
            else:
                st.session_state.file_processed = False
    else:
        # Same file, already processed
        if st.session_state.file_processed:
            st.info(f"📄 Using previously processed file: '{uploaded_file.name}' ({len(st.session_state.docs)} chunks)")
# 3. Query Section - Only show if we have a processed file
if st.session_state.file_processed and st.session_state.vectorstore is not None:
    st.divider()
    query = st.text_input("💬 Enter a question to test the retrieval:", placeholder="e.g., What skills does this person have?")
    if query and query.strip():
        with st.spinner("🔍 Searching..."):
            try:
                # Perform Similarity Search with Scores
                results_with_scores = st.session_state.vectorstore.similarity_search_with_score(
                    query.strip(), k=5
                )
                if results_with_scores:
                    display_results(results_with_scores)
                    display_statistics(
                        st.session_state.vectorstore,
                        query.strip(),
                        len(st.session_state.docs)
                    )
                else:
                    st.warning("No results found for your query.")
            except Exception as e:
                st.error(f"Error during search: {e}")
else:
    st.info("👆 Please upload a PDF file in the sidebar to get started!")
# Footer
st.sidebar.divider()
st.sidebar.caption("Built with LangChain, FAISS & Streamlit")
