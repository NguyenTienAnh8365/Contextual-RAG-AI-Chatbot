# === IMPORT CÁC THƯ VIỆN CẦN THIẾT ===
from langchain_openai import ChatOpenAI
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import logging
from process_data import processed_docs
from langchain_huggingface import HuggingFaceEmbeddings
import uuid
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from retriever_and_vectordb import retriever

# === TẢI CÁC BIẾN MÔI TRƯỜNG ===
load_dotenv()

# === LẤY API KEY Ừ BIẾN MÔI TRƯỜNG ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY not found in environment variables")

# === EMBEDDINGS MODEL ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

# === THÊM UUID DUY NHẤT ===
def add_unique_metadata(docs):
    for doc in docs:
        if "uuid" not in doc.metadata:
            doc.metadata["uuid"] = str(uuid.uuid4())
    return docs

# === KHỞI TẠO MÔ HÌNH LLM ===
def get_llm_and_agent(retriever, model_choice=None):
    """
    Khởi tạo Language Model và Agent với cấu hình cụ thể.
    Args:
        retriever: Retriever đã được cấu hình để tìm kiếm thông tin.
        model_choice: Chọn model ('gpt4', 'grok'), tự động phát hiện nếu không được chỉ định.
    """
    # Tự động phát hiện model nếu không có lựa chọn
    if not model_choice:
        if OPENAI_API_KEY:
            model_choice = "gpt4"
        elif XAI_API_KEY:
            model_choice = "grok"
        else:
            raise ValueError("Không tìm thấy API key hợp lệ. Vui lòng cung cấp OPENAI_API_KEY hoặc XAI_API_KEY.")

    try:
        # LLM model
        # GPT-4
        if model_choice == "gpt4":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY không khả dụng.")

            llm = ChatOpenAI(
                temperature=0,
                streaming=True,
                model='gpt-4o-mini',
                api_key=OPENAI_API_KEY
            )

        # Grok AI
        elif model_choice == "grok":
            if not XAI_API_KEY:
                raise ValueError("XAI_API_KEY không khả dụng.")

            llm = ChatOpenAI(
                temperature=0,
                streaming=True,
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                api_key=XAI_API_KEY,
                base_url="https://api.groq.com/openai/v1"
            )

        # Hoặc mô hình khác...(thêm ở đây)
        else:
            raise ValueError(f"Model '{model_choice}' không được hỗ trợ.")

        # Định nghĩa hàm định dạng tài liệu
        def format_docs(processed_docs):
            return "\n\n".join(doc.page_content for doc in processed_docs)

        # Kéo prompt từ hub
        prompt = hub.pull("rlm/rag-prompt")

        # Cập nhật chuỗi RAG
        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )
        return rag_chain

    except Exception as e:
        raise ValueError(f"Lỗi khi khởi tạo LLM và Agent: {e}")

# === THỰC HIỆN TRUYỀN ĐỐI SỐ
docs = processed_docs
embeddings = embeddings
retriever = retriever
