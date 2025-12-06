import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import sqlite3
from datetime import datetime
import os
import re

# 1. CẤU HÌNH & TỪ ĐIỂN DỮ LIỆU
MODEL_NAME = "vinai/phobert-base-v2"

# Cấu hình hiển thị lịch sử
INITIAL_LIMIT = 5
STEP_LOAD = 5

# Từ điển viết tắt
ABBREVIATIONS = {
    "ko": "không", "k": "không", "kh": "không", "khong": "không", "hok": "không",
    "dc": "được", "duoc": "được", "dk": "được",
    "rat": "rất", "rit": "rất",
    "wa": "quá", "qua": "quá",
    "lam": "lắm", "lun": "luôn",
    "bh": "bây giờ", "h": "giờ", "r": "rồi",
    "vui": "vui", "hp": "hạnh phúc", "happy": "hạnh phúc",
    "buon": "buồn", "sad": "buồn", "sau": "sầu",
    "iu": "yêu", "yeu": "yêu", "thik": "thích",
    "ok": "tốt", "good": "tốt", "bad": "tệ"
}

# 1. NHÓM ĐẶC BIỆT: Phủ định của tiêu cực 
KEYWORDS_SPECIAL_POSITIVE = [
    "không buồn", "hết buồn", "chẳng buồn", "chả buồn",
    "không sao", "không việc gì", "không đau", "hết đau", "đỡ đau",
    "không tệ", "không mệt", "đỡ rồi", "đỡ nhiều",
    "không xấu", "không sợ", "hết sợ", "không dở"
]

# 2. NHÓM TIÊU CỰC (Bao gồm cả "không vui", "không thích")
KEYWORDS_NEGATIVE = [
    "không vui", "không thích", "không ngon", "không tốt", "không hay", "không đẹp",
    "chả vui", "chả thích", "chẳng vui", "chẳng thích", "k vui", "ko vui",
    "không hài lòng", "không ổn", "không dám",
    "buồn", "chán", "mệt", "đau", "khổ", "tệ", "xấu", "ghét", "sợ", "khóc", 
    "dở", "kém", "bực", "cáu", "nhục", "fail", "thất vọng", "cô đơn",
    "trễ", "muộn", "chậm", "tắc đường", "kẹt xe", "rớt", "tạch", "trượt", "hỏng",      
    "mất", "lạc", "trộm", "cướp", "phạt", "chửi", "la", "mắng",        
    "bệnh", "ốm", "sốt", "ho", "đen", "xui", "hạn",                 
    "đói", "khát", "nóng", "lạnh", "bẩn", "hôi", 
    "lừa đảo", "phí", "đắt", "thách", "phiền", "ồn", "đau đầu"
]

# 3. NHÓM TÍCH CỰC CƠ BẢN
KEYWORDS_POSITIVE = [
    "vui", "thích", "yêu", "tốt", "ngon", "đẹp", "xinh", "tuyệt", "hay", "đỉnh", 
    "phê", "sướng", "hạnh phúc", "happy", "ok", "ổn", "hài lòng", "thành công", 
    "đậu", "đỗ", "may mắn", "hên", "giỏi", "xuất sắc", "thư giãn", "chill", "sớm",
    "ưng", "sang trọng", "xịn", "dễ thương", "tự hào", "khỏe", "tuyệt vời"
]

# 2. XỬ LÝ DATABASE (SQLITE)

def init_db():
    """Khởi tạo bảng nếu chưa có"""
    conn = sqlite3.connect('sentiments.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sentiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_text TEXT, 
            clean_text TEXT,
            sentiment TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(original, clean, sentiment):
    """Lưu kết quả vào DB"""
    conn = sqlite3.connect('sentiments.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO sentiments (original_text, clean_text, sentiment, timestamp) VALUES (?, ?, ?, ?)", 
              (original, clean, sentiment, timestamp))
    conn.commit()
    conn.close()

def get_history(limit):
    """Lấy lịch sử có phân trang"""
    conn = sqlite3.connect('sentiments.db')
    c = conn.cursor()
    # Lấy tổng số dòng
    c.execute("SELECT COUNT(*) FROM sentiments")
    total_rows = c.fetchone()[0]
    # Lấy dữ liệu
    c.execute("SELECT original_text, clean_text, sentiment, timestamp FROM sentiments ORDER BY timestamp DESC LIMIT ?", (limit,))
    data = c.fetchall()
    conn.close()
    return data, total_rows

# 3. XỬ LÝ NLP & LOGIC HYBRID

@st.cache_resource
def load_sentiment_pipeline():
    """Load Model Transformer (PhoBERT Base)"""
    # Thêm num_labels=3 để model base hoạt động được trong pipeline classification
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return nlp

def preprocess_text(text):
    """Chuẩn hóa văn bản: Chữ thường + Thay thế viết tắt"""
    if not text: return ""
    text = text.lower()
    words = text.split()
    new_words = [ABBREVIATIONS.get(w, w) for w in words]
    text = " ".join(new_words)
    # Giới hạn ký tự
    if len(text) > 256: text = text[:256] 
    return text

def get_sentiment_hybrid(text, model_pipeline):
    """
    Logic lai ghép (Hybrid) Cập nhật:
    Sử dụng Regex (\b) để bắt chính xác từ, tránh lỗi 'ho' trong 'hoc'
    """
    
    # Hàm con để check từ khóa chính xác (Whole word check)
    def check_keyword(keyword_list, text_input):
        for phrase in keyword_list:
            # \b là ranh giới từ. re.escape để xử lý các ký tự đặc biệt nếu có
            # Cờ re.IGNORECASE để không phân biệt hoa thường
            pattern = r'\b' + re.escape(phrase) + r'\b'
            if re.search(pattern, text_input, re.IGNORECASE):
                return True
        return False

    # --- KIỂM TRA NHÓM ĐẶC BIỆT ---
    if check_keyword(KEYWORDS_SPECIAL_POSITIVE, text):
        return "POSITIVE", 0.99

    # --- KIỂM TRA TIÊU CỰC ---
    if check_keyword(KEYWORDS_NEGATIVE, text):
        return "NEGATIVE", 0.99
            
    # --- KIỂM TRA TÍCH CỰC ---
    if check_keyword(KEYWORDS_POSITIVE, text):
        return "POSITIVE", 0.99

    # --- MODEL BASE ---
    # Cắt ngắn text
    input_model = text[:50] 
    result = model_pipeline(input_model)[0]
    
    score = result['score']
    label_raw = result['label']
    
    if score < 0.5: return "NEUTRAL", score
    
    if "LABEL_2" in label_raw or "POS" in label_raw.upper(): return "POSITIVE", score
    if "LABEL_0" in label_raw or "NEG" in label_raw.upper(): return "NEGATIVE", score
    
    return "NEUTRAL", score

# 4. GIAO DIỆN NGƯỜI DÙNG (STREAMLIT UI)

def main():
    st.set_page_config(page_title="Đồ án NLP - Sentiment", page_icon="🤖")
    
    st.title("🤖 Trợ lý Phân loại Cảm xúc (Hybrid)")
    st.caption(f"Model: {MODEL_NAME} | Cơ chế: Từ điển + Transformer")
    st.markdown("---")

    with st.sidebar:
        st.header("Công cụ")
        if st.button("🗑️ Xóa dữ liệu cũ (Reset DB)"):
            if os.path.exists("sentiments.db"):
                os.remove("sentiments.db")
                st.toast("Đã xóa Database cũ!")
                st.rerun()
            else:
                st.info("Chưa có dữ liệu.")

    init_db()
    if 'history_limit' not in st.session_state:
        st.session_state.history_limit = INITIAL_LIMIT

    # --- LOAD MODEL ---
    try:
        with st.spinner('Đang tải mô hình ngôn ngữ...'):
            classifier = load_sentiment_pipeline()
    except Exception as e:
        st.error(f"Lỗi tải model: {e}")
        return

    # --- CHIA CỘT GIAO DIỆN ---
    col1, col2 = st.columns([1, 1])
    
    # === CỘT TRÁI: NHẬP LIỆU ===
    with col1:
        st.subheader("Nhập liệu")
        user_input = st.text_input("Nhập câu tiếng Việt:", "")
        
        if st.button("Phân loại cảm xúc", type="primary"):
            # Validate input
            if not user_input or len(user_input.strip()) < 2:
                st.warning("Vui lòng nhập câu dài hơn 2 ký tự!")
            else:
                # B1: Chuẩn hóa
                clean_text = preprocess_text(user_input)
                
                # B2: Phân loại Hybrid
                final_sentiment, final_score = get_sentiment_hybrid(clean_text, classifier)
                
                # B3: Hiển thị kết quả
                if final_sentiment == "POSITIVE":
                    st.success(f"Kết quả: **{final_sentiment}**")
                elif final_sentiment == "NEGATIVE":
                    st.error(f"Kết quả: **{final_sentiment}**")
                else:
                    st.info(f"Kết quả: **{final_sentiment}**")
                
                # B4: Hiển thị Dictionary JSON 
                output_dict = {
                    "text_original": user_input,
                    "text_cleaned": clean_text,
                    "sentiment": final_sentiment,
                    "confidence": f"{final_score:.2f}"
                }
                st.caption("Chi tiết kỹ thuật (JSON Output):")
                st.json(output_dict)
                
                # B5: Lưu Database
                save_to_db(user_input, clean_text, final_sentiment)
                st.toast("Đã lưu kết quả!")

    # === CỘT PHẢI: LỊCH SỬ ===
    with col2:
        st.subheader("Lịch sử phân loại")
        history, total_rows = get_history(st.session_state.history_limit)
        
        if history:
            st.caption(f"Đang hiển thị {len(history)} / {total_rows} bản ghi mới nhất")
            
            for item in history:
                orig, clean, sent, time = item
                # Icon theo cảm xúc
                icon = "😊" if sent == "POSITIVE" else "😡" if sent == "NEGATIVE" else "😐"
                
                with st.expander(f"{time} | {sent} {icon}"):
                    st.markdown(f"**📝 Gốc:** `{orig}`")
                    st.markdown(f"**⚙️ Xử lý:** `{clean}`")
            
            # Logic nút Xem Thêm
            if len(history) < total_rows:
                if st.button("Xem thêm 🔽"):
                    st.session_state.history_limit += STEP_LOAD
                    st.rerun()
            elif total_rows > INITIAL_LIMIT:
                if st.button("Thu gọn 🔼"):
                    st.session_state.history_limit = INITIAL_LIMIT
                    st.rerun()
        else:
            st.info("Chưa có dữ liệu lịch sử.")

if __name__ == "__main__":
    main()