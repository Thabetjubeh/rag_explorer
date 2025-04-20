import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import Document
import pandas as pd


ALLOWED_EXTENSIONS = [".md", ".txt", ".csv"]

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        ext = os.path.splitext(filename)[1]
        if ext.lower() not in ALLOWED_EXTENSIONS:
            continue

        filepath = os.path.join(directory, filename)

        if ext == ".csv":
            try:
                df = pd.read_csv(filepath)
                text = df.to_string(index=False)
                documents.append(Document(text=text, metadata={"filename": filename}))
            except Exception as e:
                st.warning(f"⚠️ Could not read CSV: {filename} – {e}")
        else:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                    documents.append(Document(text=text, metadata={"filename": filename}))
            except Exception as e:
                st.warning(f"⚠️ Could not read file: {filename} – {e}")
    
    return documents


# Set up LLM and LlamaIndex
llm = OpenAI(api_key=openai_api_key, temperature=0.3, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)



# Load documents from /data
st.sidebar.title("📂 Data Loader")
data_path = "data"
st.sidebar.write(f"Using data from: `{data_path}/`")

documents = load_documents(data_path)

# Show loaded files
if not documents:
    st.warning("⚠️ No supported files found in `/data`.")
else:
    st.sidebar.markdown("### ✅ Loaded Files")
    for doc in documents:
        st.sidebar.write(f"- {doc.metadata.get('filename', 'unknown')}")

# Build index
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine(similarity_top_k=1)

# Streamlit UI
st.title("🧭 RAG Explorer")
st.write("Ask a question about your local data below:")

query = st.text_input("💬 Enter your question")
if query:
    with st.spinner("Thinking..."):
        response = query_engine.query(query)

        # Show context
        st.markdown("### 📄 Retrieved Context")
        if hasattr(response, "source_nodes") and response.source_nodes:
            for node in response.source_nodes:
                text = node.node.text.strip()
                st.markdown(f"```text\n{text[:500]}\n...```")
        else:
            st.info("No source nodes available")

        # Show answer
        st.markdown("### 🧠 Answer")
        st.write(response.response)

