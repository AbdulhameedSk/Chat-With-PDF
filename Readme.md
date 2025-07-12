

## 🧠 Chat with PDF using Google Gemini Pro (2.5) + LangChain

This project allows you to upload any PDF file, automatically split it into chunks, embed the content using HuggingFace, and ask natural language questions using **Google's Gemini Pro (2.5)**.

---

### 🚀 Features

* ✅ Upload and parse any PDF file
* ✅ Chunk large text with overlap for context preservation
* ✅ Embed PDF chunks using HuggingFace Sentence Transformers
* ✅ Store and search with FAISS vector index
* ✅ Ask questions using Google Gemini Pro (2.5)
* ✅ Returns answers grounded in the PDF content

---

### 📦 Dependencies

Install all required packages:

```bash
!pip install langchain
!pip install google-generativeai
!pip install pypdf
!pip install chromadb
!pip install faiss-cpu
!pip install sentence-transformers
```

---

### 🔑 Configuration

Set your Gemini API Key:

```python
import google.generativeai as genai
genai.configure(api_key="your-api-key-here")
```

---

### 📁 Steps to Run

#### 1. Upload the PDF

```python
from google.colab import files
uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]
```

#### 2. Load PDF Content

```python
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader(pdf_path)
pages = loader.load()
```

#### 3. Split into Chunks

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(pages)
```

#### 4. Create Vector Embeddings

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding)
```

#### 5. Ask Questions Using Gemini Pro

```python
llm = genai.GenerativeModel(model_name="gemini-2.5-pro")

def ask_pdf(query):
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {query}
Answer:"""
    response = llm.generate_content(prompt)
    return response.text

# Example:
ask_pdf("Summarize the document")
```

---

### 💡 Notes

* Make sure `vectorstore` is defined **after** the PDF is split.
* `gemini-2.5-pro` may change to `gemini-1.5-pro` depending on API availability.
* Your HuggingFace model (`all-MiniLM-L6-v2`) is a good lightweight encoder for PDF Q\&A.
* You can extend this to support multi-turn chat or integrate with `Gradio`.

---

### 📜 License

MIT License – free to use, modify, and share.

---

