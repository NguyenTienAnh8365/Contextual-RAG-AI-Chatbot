# === IMPORT CÁC THƯ VIỆN CẦN THIẾT ===
import streamlit as st
from dotenv import load_dotenv
from agent import get_llm_and_agent
from local import get_llm_local as get_ollama_local
from retriever_and_vectordb import retriever
import os
import logging
import time
from FlagEmbedding import FlagReranker

# === TẢI CÁC BIẾN MÔI TRƯỜNG ===
load_dotenv()

# === LẤY API KEY Ừ BIẾN MÔI TRƯỜNG ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY not found in environment variables")

# === CẤU HÌNH TRANG WEB ===
def setup_page():
    """
    Cấu hình cơ bản cho trang web.
    """
    st.set_page_config(page_title="AI Assistant", page_icon="💬", layout="wide")

# === KHỞI TẠO ỨNG DỤNG ===
def initialize_app():
    """
    Khởi tạo các cài đặt cần thiết.
    """
    load_dotenv()
    setup_page()

# === THANH CÔNG CỤ BÊN TRÁI ===
def setup_sidebar():
    """
    Tạo giao diện thanh bên trái.
    """
    with st.sidebar:
        st.title("⚙️ Cấu hình")
        st.markdown("---")

        # Chọn embeddings model
        embeddings_choice = st.radio("🔤 Chọn Embedding Model:", ["Sentence-Transformer"])

        # Chọn AI Model
        model_choice = st.radio("🤖 Chọn AI Model:", ["GPT-4", "Grok", "Ollama (Local)"])

        return embeddings_choice, model_choice

# === GIAO DIỆN CHAT ===
def setup_chat_interface(model_choice):
    """
    Cấu hình giao diện chat.
    """
    st.title("💬 AI Assistant")
    st.caption({
        "GPT-4": "🚀 Hỗ trợ bởi GPT-4",
        "Grok": "🚀 Hỗ trợ bởi X.AI Grok",
        "Ollama (Local)": "🚀 Hỗ trợ bởi Ollama"
    }.get(model_choice, "🚀 Trợ lý AI của bạn"))

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

# Tạo reranker từ mô hình FlagReranker
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

# === XỬ LÝ ĐẦU VÀO NGƯỜI DÙNG ===
def handle_user_input_with_reranking(agent_executor, retriever):
    """
    Xử lý đầu vào từ người dùng và hiển thị câu trả lời với reranking.
    """
    if prompt := st.chat_input("Hỏi tôi điều gì bạn mong muốn!"):
        st.session_state.messages.append({"role": "human", "content": prompt})
        st.chat_message("human").markdown(prompt)

        # Cập nhật hiệu ứng cho trạng thái suy nghĩ
        response_placeholder = st.chat_message("assistant").empty()
        thinking_texts = ["💭 Đang suy nghĩ", "💭 Đang suy nghĩ.", "💭 Đang suy nghĩ..", "💭 Đang suy nghĩ..."]

        # Tạo hiệu ứng "..." chuyển động
        for _ in range(25):
            for text in thinking_texts:
                response_placeholder.markdown(text)  # Cập nhật trạng thái
                time.sleep(0.25)  # độ trễ giữa các lần thay đổi

        try:
            # Truyền câu hỏi vào agent executor và lấy câu trả lời từ LLM
            USER_QUESTION = prompt
            
            # Lấy các tài liệu liên quan từ retriever
            documents = retriever.get_relevant_documents(USER_QUESTION)

            # Chọn tài liệu có điểm rerank cao nhất
            reranker_scores = reranker.compute_score([[USER_QUESTION, doc.page_content] for doc in documents], normalize=True)

            # Chuyển đổi điểm thành tỷ lệ phần trăm (0-100%)
            reranker_percentage_scores = [score * 100 for score in reranker_scores]

            # In ra điểm số dưới dạng tỷ lệ phần trăm
            for i, score in enumerate(reranker_percentage_scores):
                st.markdown(f"Điểm số giữa câu hỏi và tài liệu {i + 1}: {score:.2f}%")

            # Chọn tài liệu có điểm rerank cao nhất để dựa vào đó tạo câu trả lời từ LLM
            best_answer_index = reranker_scores.index(max(reranker_scores))
            best_document = documents[best_answer_index].page_content  # Chọn tài liệu có điểm cao nhất

            # Sử dụng tài liệu có điểm cao nhất để tạo câu trả lời từ LLM
            # Truyền câu hỏi và tài liệu vào LLM để có câu trả lời phù hợp
            if hasattr(agent_executor, 'invoke'):
                output_stream = agent_executor.invoke(f"{USER_QUESTION}", stream=True)
            else:
                raise ValueError("Agent Executor không có phương thức 'invoke'.")

            # Câu trả lời cuối
            final_answer = ""
            for part in output_stream:
                final_answer += part  # Trả lời từng phần trong stream
                response_placeholder.markdown(f"🗣️ Câu trả lời: {final_answer}") # Cập nhật câu trả lời cuối cùng
                time.sleep(0.01)  # thời gian trễ giữa các từ

            # Lưu câu trả lời vào session
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

            # Tính toán lại điểm rerank giữa câu trả lời từ LLM và tài cu hỏi từ người dùng
            final_reranker_score = reranker.compute_score([[USER_QUESTION, final_answer], [USER_QUESTION, best_document]], normalize=True)

            # In ra điểm số (kết hợp)
            st.markdown(f"Điểm số cho câu trả lời từ LLM: {((final_reranker_score[0] * 100 * 0.8 + final_reranker_score[1] * 100 * 0.2)):.2f}%")

        except Exception as e:
            response_placeholder.markdown(f"❌ Lỗi: {e}")
            st.error(f"Lỗi trong khi xử lý: {e}")

# === CHẠY ỨNG DỤNG ===
def main():

    initialize_app()

    embeddings_choice, model_choice = setup_sidebar()

    try:
        # Tạo agent cho mô hình đã chọn
        if model_choice == "Ollama (Local)":
            agent_executor = get_ollama_local(retriever)  # Tạo agent cho Ollama local
        elif model_choice == "Grok":
            agent_executor = get_llm_and_agent(retriever, model_choice="grok")  # Tạo agent cho Grok
        elif model_choice == "GPT-4":
            agent_executor = get_llm_and_agent(retriever, model_choice="gpt4")  # Tạo agent cho GPT-4
        else:
            raise ValueError(f"Mô hình '{model_choice}' không được hỗ trợ.")

        # Cấu hình giao diện chat
        setup_chat_interface(model_choice)

        # Xử lý đầu vào của người dùng với streaming
        handle_user_input_with_reranking(agent_executor, retriever)

    except Exception as e:
        logging.error(f"Lỗi khi khởi chạy ứng dụng: {e}")
        st.error(f"Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    main()
