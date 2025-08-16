import asyncio
import streamlit as st
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# -------------------------------
# Fix asyncio loop for Streamlit
# -------------------------------
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

# -------------------------------
# Streamlit UI Styling
# -------------------------------
st.set_page_config(page_title="RAGenius", page_icon="📘", layout="centered")

# Custom CSS (no extra div wrapper now ✅)
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            color: #2c3e50;
            font-size: 30px !important;
            font-weight: 700 !important;
            margin-bottom: 20px;
        }
        .stFileUploader {
            background-color: #ffffff;
            border: 2px dashed #4a90e2;
            border-radius: 12px;
            padding: 15px;
        }
        .stChatInputContainer {
            background-color: #eef2f7;
            border-radius: 12px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Main UI
# -------------------------------
st.markdown("<h1 class='main-title'>📖 RAG x Gemini:🔎Retrieval with Brilliance</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])
query = st.chat_input("💬 Ask a question about the PDF:")

# -------------------------------
# Core Logic (unchanged)
# -------------------------------
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("✅ PDF uploaded successfully!")

    loader = PyPDFLoader(tmp_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(data)

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        timeout=120
    )

    all_texts = [doc.page_content for doc in docs]
    batch_size = 5
    all_embeddings = []
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i+batch_size]
        embeddings = embedding_model.embed_documents(batch_texts)
        all_embeddings.extend(embeddings)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None
    )

    system_prompt = (
        "🤖 You are an assistant for **question-answering** tasks. "
        "Use the following retrieved context to answer clearly. "
        "If you don't know, say ❌ 'I don’t know'. "
        "Keep answers short (≤ 5 sentences), and use bullet points if useful."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    if query:
        with st.spinner("✨ Thinking with Gemini..."):
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            response = rag_chain.invoke({"input": query})

        st.markdown("### 📌 Answer")
        st.info(response["answer"])
