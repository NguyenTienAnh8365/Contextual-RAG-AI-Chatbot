# === IMPORT CÁC THƯ VIỆN CẦN THIẾT ===
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from process_data import processed_docs
import uuid
import logging

# === TẢI CÁC BIẾN MÔI TRƯỜNG ===
load_dotenv()

# === THIẾT LẬP LOGGING ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === EMBEDDINGS MODEL ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

# === THÊM UUID DUY NHẤT ===
def add_unique_metadata(docs):
    for doc in docs:
        if "uuid" not in doc.metadata:
            doc.metadata["uuid"] = str(uuid.uuid4())
    return docs

# === KHỞI TẠO RETRIEVER ===
def get_retriever(docs, embeddings, persist_directory: str = "chroma_db") -> EnsembleRetriever:
    """
    Tạo retriever kết hợp từ Chroma và BM25 với dữ liệu đầu vào.
    """
    os.makedirs(persist_directory, exist_ok=True)

    # Xử lý đầu vào (Nếu không có docs hoặc docs không phải là một danh sách, tạo tài liệu mặc định)
    if not docs or (isinstance(docs, list) and len(docs) == 0):
        logging.info("Không có dữ liệu đầu vào.")
        try:
            # Tạo tài liệu mặc định
            default_doc = [
                Document(
                    page_content="Không có dữ liệu hợp lệ để khởi tạo retriever.",
                    metadata={"source": "error"}
                )
            ]
            # Trả về BM25Retriever mặc định với các tài liệu mặc định
            return BM25Retriever.from_documents(default_doc)

        except Exception as e:
            logging.error(f"Lỗi khi xử lý dữ liệu đầu vào: {e}")
            return BM25Retriever.from_documents(default_doc)

    try:
        # Thêm metadata duy nhất cho mỗi tài liệu
        docs = add_unique_metadata(docs)

        # Tạo vector database và retriever từ Chroma
        vector_db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
        chroma_retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )   # Vector search

        logging.info(f"Vector DB type: {type(vector_db)}")

        # Tìm các tài liệu để xây dựng BM25
        documents = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in vector_db.similarity_search("", k=100)
        ]

        if not documents:
            raise ValueError("Không tìm thấy documents để tạo BM25 retriever.")

        # Tạo BM25 retriever từ các tài liệu đã tải
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 4  # Giới hạn số lượng kết quả trả về từ BM25

        # Kết hợp hai retriever với tỷ lệ trọng số
        ensemble_retriever = EnsembleRetriever(
            retrievers=[chroma_retriever, bm25_retriever],
            weights=[0.7, 0.3]  # 70% từ Chroma, 30% từ BM25
        )

        logging.info("Retriever đã được tạo và kết hợp thành công.")
        return ensemble_retriever

    except Exception as e:
        logging.error(f"Lỗi khi khởi tạo retriever: {str(e)}")

        # Trả về retriever với document mặc định nếu có lỗi
        default_doc = [
            Document(
                page_content="Có lỗi xảy ra khi kết nối database. Vui lòng thử lại sau.",
                metadata={"source": "error"}
            )
        ]
        return BM25Retriever.from_documents(default_doc)

# === THỰC HIỆN TRUYỀN ĐỐI SỐ ===
docs = processed_docs
embeddings = embeddings
retriever = get_retriever(docs, embeddings)
