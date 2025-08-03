
# Fine-Tuning vs Retrieval-Augmented Generation (RAG)

This project highlights the concepts of **Fine-Tuning** and **Retrieval-Augmented Generation (RAG)** in the domain of Natural Language Processing (NLP), explaining their key differences and typical use-cases.

## What Are Fine-Tuning and RAG?

### Fine-Tuning

**Definition:**  
Fine-tuning is the process of taking a pre-trained AI language model (like GPT, BERT, etc.) and adapting it to perform better on a specific task or domain by training it further on a smaller, task-specific dataset.

**Use-case Example:**  
Suppose you have a generic GPT model that you want to specialize to answer medical questions—you collect many QA pairs from the medical field, and train the model further on these.

**Features:**

- The model learns patterns from your dataset, adjusting many internal parameters.
- Resulting model is customized and will always use this “absorbed” data, even for unrelated queries.

### Retrieval-Augmented Generation (RAG)

**Definition:**  
RAG is a way to enhance language models by allowing them to search external sources (like documents or databases) in real time when answering a query.

**How it works:**

- When a question is asked, RAG retrieves related documents (using search, embeddings, or other IR methods).
- The language model uses these retrieved pieces as context to generate a more accurate and grounded answer.

**Features:**

- The base model is not retrained—instead, it gets access to fresh or updated information on demand.
- Very good for scenarios where data changes frequently or the knowledge base is too large to train directly into a model.

## Differences Between Fine-Tuning and RAG

| Feature            | Fine-Tuning                           | RAG (Retrieval-Augmented Generation)      |
|--------------------|-------------------------------------|-------------------------------------------|
| Adaptation         | Retrains model weights on new data  | Uses retrieval to bring in external data  |
| Update Knowledge   | Requires retraining for new info    | Just update the database, no retraining   |
| Latency            | Answers are fast after training     | Potential extra time for retrieval step   |
| Flexibility        | Stuck with the data at train time   | Can use up-to-date or very large datasets |
| Use-case           | Stable, specialized knowledge       | Dynamic, large, or constantly changing info|

## Visual Explanation

### Fine-Tuning and Retrieval-Augmented Generation (RAG) Flowchart

<img src="image.png">

## Summary

- **Fine-Tuning** is suitable when you want a model specialized for a fixed domain or task, with embedded knowledge.
- **RAG** is ideal for systems requiring up-to-date answers from large or frequently changing datasets, leveraging external context retrieval without retraining the language model.

## How to Use This Repository

*(If your project contains code)*

- This repository contains both a React frontend and Python backend implementing a basic RAG system.
- Use the React app to submit questions.
- The Python backend retrieves relevant documents and generates responses based on RAG.



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


