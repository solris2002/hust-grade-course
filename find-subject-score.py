import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
from pathlib import Path

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Dự đoán điểm môn",
    page_icon="imgs/favico.png",
    layout="wide",
)
st.title("📘 DỰ ĐOÁN ĐIỂM SỐ HỌC PHẦN")

# =========================
# Cấu hình ngành & ngưỡng dữ liệu
# =========================
BASE_DIR = Path(__file__).parent

MAJOR_CONFIG = {
    "ET1": {
        "label": "ET1: Điện tử và Viễn thông",
        "subjects": BASE_DIR / "ET1" / "ET1-subjects.json",
        "scaler":   BASE_DIR / "ET1" / "ET1-scaler.joblib",
        "ggm":      BASE_DIR / "ET1" / "ET1-ggm.joblib",
        "template": BASE_DIR / "ET1" / "ET1-input-score.xlsx",
    },
    "EE2": {
        "label": "EE2: Kỹ thuật Điều khiển -  Tự động hóa",
        "subjects": BASE_DIR / "EE2" / "EE2-subjects.json",
        "scaler":   BASE_DIR / "EE2" / "EE2-scaler.joblib",
        "ggm":      BASE_DIR / "EE2" / "EE2-ggm.joblib",
        "template": BASE_DIR / "EE2" / "EE2-input-score.xlsx",
    },
}

# Tối thiểu số môn đã có điểm (khác target) để cho phép dự đoán
MIN_OBS = 3
COVERAGE_RED_THRESHOLD = 0.60  # nghi nhầm template nếu < 60%

LETTER_TO_GPA = {
    "A+": 4.0, "A": 4.0,
    "B+": 3.5, "B": 3.0,
    "C+": 2.5, "C": 2.0,
    "D+": 1.5, "D": 1.0,
}

# =========================
# Sidebar: Back & chọn ngành
# =========================

st.sidebar.link_button("← Quay lại trang chủ", url="https://solris2002.github.io/home-seee-grade/", use_container_width=True)

major = st.sidebar.selectbox(
    "Bạn học ngành:",
    options=list(MAJOR_CONFIG.keys()),
    format_func=lambda k: MAJOR_CONFIG[k]["label"],
    index=0,
    key="major_select",
)
cfg = MAJOR_CONFIG[major]

# =========================
# Loaders (cache theo đường dẫn)
# =========================
@st.cache_resource
def load_subjects_means_stds(subjects_path: Path, scaler_path: Path):
    subjects = json.loads(subjects_path.read_text(encoding="utf-8"))
    scaler   = joblib.load(scaler_path)
    means = pd.Series(scaler["means"])
    stds  = pd.Series(scaler["stds"]).replace(0, 1.0)
    return subjects, means, stds

@st.cache_resource
def load_ggm(ggm_path: Path):
    if ggm_path.exists():
        return joblib.load(ggm_path)
    return None

subjects, means, stds = load_subjects_means_stds(cfg["subjects"], cfg["scaler"])
ggm_art = load_ggm(cfg["ggm"])

# =========================
# Tải file mẫu đúng ngành
# =========================
st.markdown("### 📥 Tải mẫu file nhập điểm có sẵn")
template_path = cfg["template"]
if template_path.exists():
    with open(template_path, "rb") as f:
        st.download_button(
            label=f"Tải xuống {template_path.name}",
            data=f.read(),
            file_name=template_path.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.error(f"Không tìm thấy file mẫu tại {template_path}")

# =========================
# Helpers
# =========================
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
st.sidebar.header("1. Tải file điểm lên")
uploaded = st.sidebar.file_uploader(
    "Tải file lên",
    type=["xlsx", "xls"],
    key=f"uploader_{major}",   # đổi ngành sẽ reset uploader
)

st.sidebar.header("2. Chọn các môn cần dự đoán")
target_subjects = st.sidebar.multiselect(
    "Chọn môn học muốn dự đoán", subjects, key=f"targets_{major}"
)

do_predict = st.sidebar.button("Dự đoán", key=f"predict_{major}")

st.sidebar.markdown("---")
st.sidebar.subheader("📖 Hướng dẫn sử dụng")
st.sidebar.markdown(
    """
    1. **Tải file Excel** theo mẫu ngành đã chọn (ET1/EE2) và nhập các môn + điểm chữ.
    2. **Chọn một hoặc nhiều môn cần dự đoán**.
    3. Nhấn **Dự đoán** để xem kết quả.
    """
)

# Ẩn dòng hướng dẫn mặc định (drag and drop...)
st.markdown("""
<style>
[data-testid="stFileUploaderDropzoneInstructions"] { display: none; }

/* Căn giữa tất cả st.table */
[data-testid="stTable"] table td, 
[data-testid="stTable"] table th { text-align:center !important; }

/* Tạo thanh cuộn dọc cho bảng (preview & kết quả) */
[data-testid="stTable"] { max-height: 360px; overflow-y: auto; }
[data-testid="stTable"] > div { max-height: 360px; overflow-y: auto; }
</style>
""", unsafe_allow_html=True)

# =========================
# Đọc & kiểm tra dữ liệu NGAY KHI UPLOAD
# =========================
if uploaded is None:
    st.warning("Vui lòng tải lên file Excel chứa các môn và điểm chữ."); st.stop()

try:
    df_raw = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Không thể đọc file: {e}"); st.stop()

required_cols = ["Môn học", "Điểm chữ"]
if not all(col in df_raw.columns for col in required_cols):
    st.error(f"File đầu vào phải có ít nhất các cột: {required_cols}"); st.stop()

# Tính trạng thái dòng + thống kê bao phủ ngành
subject_set = set(subjects)

def row_status(subj: str):
    s = (subj or "").strip()
    if not s: return ""
    return "✖ Không thuộc ngành" if s not in subject_set else ""

df_preview = df_raw.copy()
df_preview["Môn học"]  = df_preview["Môn học"].astype(str).str.strip()
# Điểm chữ: hiển thị None nếu trống
dc = df_preview["Điểm chữ"].astype(str)
df_preview["Điểm chữ"] = np.where(
    df_raw["Điểm chữ"].isna() | (dc.str.strip().str.lower().eq("nan")) | (dc.str.strip().eq("")),
    None,
    df_raw["Điểm chữ"]
)
df_preview["Trạng thái"] = df_preview["Môn học"].apply(row_status)

unknown_mask = df_preview["Trạng thái"].eq("✖ Không thuộc ngành")
match_mask   = ~unknown_mask & df_preview["Môn học"].astype(str).str.strip().ne("")

match_count   = int(match_mask.sum())
unknown_count = int(unknown_mask.sum())
total_nonempty = int((df_preview["Môn học"].astype(str).str.strip() != "").sum())
coverage = match_count / max(1, (match_count + unknown_count))

# Cảnh báo VÀNG: có môn không thuộc ngành
if unknown_count > 0:
    bads = df_preview.loc[unknown_mask, "Môn học"].astype(str).str.strip()
    bads = [b for b in bads if b and b.lower() != "nan"]
    preview_names = ", ".join(sorted(bads)[:10]) + (" ..." if len(bads) > 10 else "")
    st.warning(f"**Có {unknown_count} môn không thuộc ngành {cfg['label']}:** {preview_names}")

# Cảnh báo ĐỎ: nghi nhầm template
if total_nonempty > 0 and coverage < COVERAGE_RED_THRESHOLD:
    st.error(
        f"**Có vẻ bạn đang dùng file không khớp với ngành {cfg['label']}** "
        f"(tỷ lệ môn thuộc ngành chỉ ~{coverage:.0%}). "
        f"Hãy chọn đúng ngành hoặc tải lại mẫu phù hợp."
    )

# =========================
# Preview bảng có cuộn (giữ style đỏ cho môn sai ngành)
# =========================
st.subheader("✅ Dữ liệu đã upload")

def highlight_preview(row):
    if row.get("Trạng thái", "") == "✖ Không thuộc ngành":
        return ['color: #B00020; background-color: #FFE6E6;'] * len(row)
    return [''] * len(row)

preview_cols = ["Môn học", "Điểm chữ", "Trạng thái"]
df_preview.index = np.arange(1, len(df_preview) + 1)
styled_preview = (
    df_preview[preview_cols]
        .style.apply(highlight_preview, axis=1)
        .hide(axis="index")
)
st.table(styled_preview)

# =========================
# Nếu chưa bấm dự đoán thì dừng sau khi preview
# =========================
if not do_predict:
    st.info("Chọn môn và nhấn 'Dự đoán' ở sidebar."); st.stop()

# =========================
# Chuẩn bị dữ liệu người dùng cho dự đoán
# =========================
user_numeric = {}
for _, row in df_raw.iterrows():
    subj = str(row["Môn học"]).strip() if not pd.isna(row["Môn học"]) else ""
    if subj in subject_set:
        sc = convert_letter_to_score(row["Điểm chữ"])
        if np.isfinite(sc):
            user_numeric[subj] = sc

observed_subjects = sorted(user_numeric.keys())
observed_count = len(observed_subjects)

# Điều kiện dữ liệu tối thiểu (toàn cục)
if observed_count < MIN_OBS:
    names_preview = ", ".join(observed_subjects[:12]) + (" ..." if observed_count > 12 else "")
    st.error(
        f"**Không đủ dữ liệu dự đoán.** "
        f"Cần tối thiểu **{MIN_OBS}** môn đã có điểm. "
        f"Hiện có **{observed_count}**: {names_preview}"
    )
    st.stop()

# =========================
# Dự đoán GGM
# =========================
if not target_subjects:
    st.error("Vui lòng chọn ít nhất một môn để dự đoán."); st.stop()

results = []
for target in target_subjects:
    pred = predict_ggm_for_target(ggm_art, means, stds, user_numeric, target, subjects)
    if not np.isfinite(pred):
        results.append({"Môn học": target, "Điểm chữ": "Không đủ dữ liệu dự đoán", "Điểm số chuẩn": np.nan})
    else:
        letter = numeric_to_letter(pred)
        results.append({
            "Môn học": target,
            "Điểm chữ": letter,
            # "Điểm số ": LETTER_TO_GPA.get(letter.split()[0], np.nan)
            "Điểm số chuẩn": LETTER_TO_GPA.get(letter.split()[0], np.nan)
        })

# =========================
# In kết quả (căn giữa + tô đỏ nếu thiếu dữ liệu từng môn)
# =========================
df_result = pd.DataFrame(results)

def format_score(x):
    if pd.isna(x): return "N/A"
    return f"{float(x):.1f}"

def highlight_results(row):
    if row["Điểm chữ"] == "Không đủ dữ liệu dự đoán":
        styles = [''] * len(row)
        try:
            idx_col = list(row.index).index("Điểm chữ")
            styles[idx_col] = 'color: #B00020; font-weight: 700;'
        except Exception:
            pass
        return styles
    return [''] * len(row)

styled_result = (
    df_result.style
        .hide(axis="index")
        .format({"Điểm số chuẩn": format_score})
        .apply(highlight_results, axis=1)
)

st.subheader("🎯 Kết quả dự đoán")
st.table(styled_result)

st.markdown("""
**Ghi chú:** Độ chính xác dự đoán dựa trên lượng dữ liệu nghiên cứu có thể sai lệch ~0–0.5  
*(Điểm số còn phụ thuộc nhiều vào yếu tố ngoại cảnh.)*
""")
