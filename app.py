import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd

# Page Config
st.set_page_config(page_title="RAG Similarity Visualizer", layout="wide")
st.title("🔍 RAG Similarity Visualizer")
st.markdown("### Peek inside the 'Black Box' of Vector Search")

# 1. Setup Embeddings (Using a free, local model to get you started immediately)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

# 2. File Upload
uploaded_file = st.file_file_uploader("Upload a PDF to analyze", type="pdf")

if uploaded_file:
    # Save temp file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Load and Split
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(pages)
    
    # Create Vector Store
    vectorstore = FAISS.from_documents(docs, embeddings)
    st.success(f"Successfully processed {len(docs)} chunks!")

    # 3. Query Section
    query = st.text_input("Enter a question to test the retrieval:")
    
    if query:
        # Perform Similarity Search with Scores
        # Scores in FAISS are L2 distance (Lower is better/more similar)
        results_with_scores = vectorstore.similarity_search_with_score(query, k=5)
        
        st.subheader("Top 5 Retrieved Chunks")
        
        # Display as a Visual List
        for i, (doc, score) in enumerate(results_with_scores):
            # Normalize score for a "Match %" feel (Simple approximation)
            confidence = max(0, 100 - (score * 50)) 
            
            with st.expander(f"Chunk #{i+1} | Match Score: {confidence:.2f}%"):
                st.write(f"**Content:** {doc.page_content}")
                st.progress(confidence / 100)
                st.caption(f"Raw L2 Distance: {score:.4f}")

        # 4. The "Statistics" View (Leveraging your Resume's strength)
        st.divider()
        st.subheader("Statistical Analysis of retrieval")
        all_docs = vectorstore.similarity_search_with_score(query, k=len(docs))
        scores = [s for d, s in all_docs]
        
        chart_data = pd.DataFrame({"Distance": scores})
        st.area_chart(chart_data)
        st.caption("This chart shows the distribution of 'distance' across your entire document. A sharp drop-off at the start indicates a very precise match.")
