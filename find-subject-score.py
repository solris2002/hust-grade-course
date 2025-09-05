import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, re
from pathlib import Path

# =========================
# UI / Layout
# =========================
st.set_page_config(page_title="Dự đoán điểm môn", layout="wide")
st.title("📘 DỰ ĐOÁN ĐIỂM SỐ HỌC PHẦN ")

template_path = Path("input-score.xlsx")
st.markdown("### 📥 Tải mẫu file nhập điểm có sẵn")
if template_path.exists():
    with open(template_path, "rb") as f:
        st.download_button(
            label="Tải xuống input-score.xlsx",
            data=f.read(),
            file_name="input-score.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.error(f"Không tìm thấy file mẫu tại {template_path}. Đặt `input-score.xlsx` vào thư mục chạy app.")

# =========================
# Artifacts
# =========================
SCALER_P   = Path("2/scaler.joblib")
SUBJECTS_P = Path("3/subjects.json")
GGM_PATH   = Path("models_streamlit_ggm/ggm.joblib")

# =========================
# Helpers
# =========================
@st.cache_resource
def load_subjects_means_stds():
    subjects = json.loads(SUBJECTS_P.read_text(encoding="utf-8"))
    scaler   = joblib.load(SCALER_P)
    means = pd.Series(scaler["means"])
    stds  = pd.Series(scaler["stds"]).replace(0, 1.0)
    return subjects, means, stds

@st.cache_resource
def load_ggm():
    if GGM_PATH.exists():
        return joblib.load(GGM_PATH)
    return None

LETTER_TO_GPA = {
    "A+": 4.0, "A": 4.0,
    "B+": 3.5, "B": 3.0,
    "C+": 2.5, "C": 2.0,
    "D+": 1.5, "D": 1.0,
}
def convert_letter_to_score(letter: str):
    if pd.isna(letter): return np.nan
    return LETTER_TO_GPA.get(str(letter).strip().upper(), np.nan)

def numeric_to_letter(score: float) -> str:
    if score >= 3.75: return "A / A+"
    if score >= 3.25: return "B+"
    if score >= 2.75: return "B"
    if score >= 2.25: return "C+"
    if score >= 1.75: return "C"
    if score >= 1.25: return "D+"
    return "D"

def predict_ggm_for_target(ggm, means, stds, user_numeric: dict, target: str, subjects: list):
    if ggm is None: return np.nan
    cov = ggm.get("cov", None)
    if cov is None: return np.nan

    idx = {s:i for i,s in enumerate(subjects)}

    # z-score user
    x = []
    O = []
    for s in subjects:
        v = user_numeric.get(s, np.nan)
        if pd.isna(v):
            x.append(np.nan)
        else:
            x.append((float(v) - means[s]) / stds[s])
            O.append(idx[s])

    if len(O) == 0 or target not in idx:
        return np.nan
    T = idx[target]
    O = np.array([o for o in O if o != T])
    if O.size == 0:
        return np.nan

    cov = np.asarray(cov)
    S_TO = cov[T, O].reshape(1, -1)
    S_OO = cov[np.ix_(O, O)]
    S_TT = cov[T, T]
    x_O  = np.array([x[o] for o in O])

    try:
        inv_S_OO = np.linalg.inv(S_OO)
    except np.linalg.LinAlgError:
        inv_S_OO = np.linalg.pinv(S_OO)

    y_std = (S_TO @ inv_S_OO @ (x_O - 0.0)).item()
    y = y_std * stds[target] + means[target]
    return float(y)

# =========================
# Sidebar & Input
# =========================
subjects, means, stds = load_subjects_means_stds()
ggm_art = load_ggm()

st.sidebar.header("1. Tải file điểm lên")
# uploaded = st.sidebar.file_uploader("Chọn file Excel đầu vào theo mẫu input-score.xlsx", type=["xlsx", "xls"])
uploaded = st.sidebar.file_uploader(
    "Tải file lên",
    type=["xlsx", "xls"]
)

# Ẩn dòng hướng dẫn mặc định (drag and drop...)
st.markdown("""
<style>
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.header("2. Chọn các môn cần dự đoán")
target_subjects = st.sidebar.multiselect("Chọn môn học muốn dự đoán", subjects)

do_predict = st.sidebar.button("Dự đoán")

st.sidebar.markdown("---")
st.sidebar.subheader("📖 Hướng dẫn sử dụng")
st.sidebar.markdown(
    """
    1. **Tải file Excel**: Sử dụng mẫu `input-score.xlsx` để nhập danh sách môn học và điểm chữ đã đạt.
    2. **Chọn một hoặc nhiều môn cần dự đoán**.
    3. Nhấn **Dự đoán** để xem kết quả.
    4. Nếu file thiếu dữ liệu nhiều môn, hệ thống sẽ báo lỗi.
    """
)

if uploaded is None:
    st.warning("Vui lòng tải lên file Excel chứa các môn và điểm chữ."); st.stop()

try:
    df_raw = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Không thể đọc file: {e}"); st.stop()

required_cols = ["Môn học", "Điểm chữ"]
if not all(col in df_raw.columns for col in required_cols):
    st.error(f"File đầu vào phải có ít nhất các cột: {required_cols}"); st.stop()

st.subheader("✅ Dữ liệu đã upload")
st.dataframe(df_raw[required_cols].head(50))

if not do_predict:
    st.info("Chọn môn và nhấn 'Chạy dự đoán' ở sidebar."); st.stop()

# =========================
# Chuẩn bị dữ liệu người dùng
# =========================
user_numeric = {}
for _, row in df_raw.iterrows():
    subj = str(row["Môn học"]).strip()
    sc   = convert_letter_to_score(row["Điểm chữ"])
    if subj in subjects:
        user_numeric[subj] = sc

if not target_subjects:
    st.error("Vui lòng chọn ít nhất một môn để dự đoán."); st.stop()

# =========================
# Predict GGM cho từng môn
# =========================
results = []
for target in target_subjects:
    pred = predict_ggm_for_target(ggm_art, means, stds, user_numeric, target, subjects)
    if not np.isfinite(pred):
        results.append({"Môn học": target, "Điểm chữ": "N/A", "Điểm số chuẩn": "N/A"})
    else:
        letter = numeric_to_letter(pred)
        results.append({
            "Môn học": target,
            "Điểm chữ": letter,
            "Điểm số chuẩn": round(LETTER_TO_GPA.get(letter.split()[0], np.nan), 1)
        })

# =========================
# In kết quả
# =========================
# df_result = pd.DataFrame(results)

# def format_score(x):
#     if pd.isna(x): 
#         return "N/A"
#     return f"{x:.1f}"

# styled = (
#     df_result.style
#         .hide(axis="index")
#         .set_properties(**{"text-align": "center"})   # căn giữa toàn bảng
#         .format({"Điểm số chuẩn": format_score})
# )

# st.subheader("🎯 Kết quả dự đoán")
# st.table(styled)

# st.markdown("""
# **Ghi chú:** Độ chính xác dự đoán sai lệch từ 0 đến 0.5  
# *(Điểm số còn dựa nhiều vào yếu tố ngoại cảnh)*
# """)

df_result = pd.DataFrame(results)

def format_score(x):
    if pd.isna(x): return "N/A"
    return f"{float(x):.1f}"

styled = (
    df_result.style
        .hide(axis="index")
        .format({"Điểm số chuẩn": format_score})
)

# ép center cho bảng của st.table
st.markdown("""
<style>
[data-testid="stTable"] table td, 
[data-testid="stTable"] table th { text-align:center !important; }
</style>
""", unsafe_allow_html=True)

st.subheader("🎯 Kết quả dự đoán")
st.table(styled)

st.markdown("""
**Ghi chú:** Độ chính xác dự đoán sai lệch từ 0 đến 0.5  
*(Điểm số còn dựa nhiều vào yếu tố ngoại cảnh)*
""")
