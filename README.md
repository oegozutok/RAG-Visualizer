# 🔍 RAG Similarity Visualizer

### Peek inside the "Black Box" of Vector Search
Enterprise AI requires more than just answers; it requires **explainability**. This tool provides a diagnostic layer for Retrieval-Augmented Generation (RAG) by visualizing the statistical distance between user queries and document embeddings.

## 🚀 Features
* **Document Ingestion:** Processes PDF files into manageable chunks using Recursive Character Splitting.
* **Vector Search:** Utilizes FAISS (Facebook AI Similarity Search) for high-performance L2 distance calculations.
* **Similarity Heatmap:** Visualizes retrieval confidence scores to help debug "bad" AI responses.
* **Statistical Distribution:** Charts the distance of the entire document corpus relative to the query.

## 🛠️ Tech Stack
* **Language:** Python
* **Orchestration:** LangChain
* **Vector Store:** FAISS
* **Embeddings:** HuggingFace (all-MiniLM-L6-v2)
* **Interface:** Streamlit
* **Data Handling:** Pandas, NumPy

## 📦 Installation & Setup
1. **Clone the repo:** `git clone https://github.com/oegozutok/RAG-Visualizer.git`
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Run the app:** `streamlit run app.py`
