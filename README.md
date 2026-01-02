# 🔍 RAG Similarity Visualizer

A Streamlit application that lets you peek inside the "black box" of vector search by visualizing how RAG (Retrieval Augmented Generation) works.

## Features

- **PDF Upload**: Upload any PDF document (like a resume)
- **Semantic Search**: Ask questions and see which chunks are retrieved
- **Visual Scoring**: See match percentages and L2 distances
- **Statistical Analysis**: View distance distribution across all chunks

## Installation

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## How It Works

1. **Upload** a PDF file (e.g., your resume)
2. The app splits the document into chunks and creates embeddings
3. **Ask a question** to see which chunks are most relevant
4. View the **match scores** and **statistical analysis**

## Key Fixes Made

- ✅ Proper session state management (prevents infinite loops)
- ✅ Correct embeddings model path
- ✅ Temporary file cleanup
- ✅ Error handling throughout
- ✅ Visual improvements and UX enhancements
