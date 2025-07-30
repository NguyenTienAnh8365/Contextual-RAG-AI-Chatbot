# === IMPORT CÁC THƯ VIỆN CẦN THIẾT ===
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
import uuid
from process_data import processed_docs
import logging
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from retriever_and_vectordb import retriever
import os

# === EMBEDDINGS MODEL ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

# === THÊM UUID DUY NHẤT ===
def add_unique_metadata(docs):
    for doc in docs:
        if "uuid" not in doc.metadata:
            doc.metadata["uuid"] = str(uuid.uuid4())
    return docs

# === KHỞI TẠO LLM VỚI OLLAMA LOCAL ===
def get_llm_local(retriever):

    # Ollama model
    llm = ChatOllama(
        model="your model from local",
        temperature=0,
        streaming=True
    )

    # Định nghĩa hàm định dạng tài liệu
    def format_docs(processed_docs):
        return "\n\n".join(doc.page_content for doc in processed_docs)

    # Kéo prompt từ hub
    prompt = hub.pull("rlm/rag-prompt")

    # Cập nhât chuỗi RAG
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain

# === HỰC HIỆN TRUYỀN ĐỐI SỐ ===
docs = processed_docs
embeddings = embeddings
retriever = retriever
