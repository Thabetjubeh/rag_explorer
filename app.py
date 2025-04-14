import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.llms.openai import OpenAI


# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up LLM and LlamaIndex
llm = OpenAI(api_key=openai_api_key, temperature=0.3, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)



# Load documents from /data
st.sidebar.title("📂 Data Loader")
data_path = "data"
st.sidebar.write(f"Using data from: `{data_path}/`")

documents = SimpleDirectoryReader(data_path).load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()

# Streamlit UI
st.title("🧭 RAG Explorer")
st.write("Ask a question about your local data below:")

query = st.text_input("💬 Enter your question")
if query:
    with st.spinner("Thinking..."):
        response = query_engine.query(query)
        st.markdown("### 🧠 Answer")
        st.write(response.response)
