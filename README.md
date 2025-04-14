# 🧭 RAG Explorer – Tiny RAG App with Local Data

A lightweight Retrieval-Augmented Generation (RAG) app that uses **LlamaIndex**, **OpenAI API**, and **Streamlit** to answer questions based on your local markdown/text files.

---

## 🚀 Features

- Load local `.md`, `.txt`, or `.csv` files
- Ask natural language questions
- Uses GPT to respond based on retrieved content
- Minimal UI with Streamlit

---

## 📦 Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your OpenAI API key to .env
streamlit run app.py
```

---

## 🗂️ Project Structure

```
rag_explorer/
├── app.py
├── data/
│   └── sample.md
├── .env.example
├── requirements.txt
└── README.md
```
