import os
import io

import pymupdf as fitz
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ---------- Tesseract path (Windows) ----------
TESS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESS_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESS_PATH

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Bitext_Sample_Customer_Service_Testing_Dataset.csv",
)


# ---------- Model (trained once, cached) ----------
@st.cache_resource(show_spinner="Training model...")
def load_model():
    df = pd.read_csv(DATA_PATH)
    X = df["utterance"].astype(str)
    y = df["intent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=2000)),
    ])
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc


# ---------- Text extraction ----------
def ocr_image(img: Image.Image) -> str:
    return pytesseract.image_to_string(img.convert("RGB"))


def extract_from_pdf(data: bytes) -> str:
    text_parts = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            text = page.get_text().strip()
            if len(text) < 20:  # likely a scanned page -> OCR it
                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))  # ~216 dpi
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = ocr_image(img)
            text_parts.append(text)
    return "\n".join(text_parts).strip()


def extract_text(uploaded_file) -> str:
    data = uploaded_file.read()
    if uploaded_file.name.lower().endswith(".pdf"):
        return extract_from_pdf(data)
    return ocr_image(Image.open(io.BytesIO(data))).strip()


# ---------- UI ----------
st.set_page_config(page_title="Intent Classifier", page_icon="📄")
st.title("📄 Customer Message Intent Classifier")
st.write("Upload a PDF or image. The app extracts the text and predicts the intent.")

model, acc = load_model()
st.sidebar.metric("Model test accuracy", f"{acc * 100:.2f}%")

uploaded = st.file_uploader("Upload PDF or image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded is not None:
    if not uploaded.name.lower().endswith(".pdf"):
        st.image(uploaded, caption=uploaded.name, use_container_width=True)
        uploaded.seek(0)

    with st.spinner("Extracting text..."):
        text = extract_text(uploaded)

    if not text:
        st.error("No text could be extracted from this file.")
    else:
        text = st.text_area("Extracted text (you can edit it)", text, height=200)

        if st.button("Classify", type="primary"):
            probs = model.predict_proba([text])[0]
            top = probs.argsort()[::-1][:3]

            st.success(f"Predicted intent: **{model.classes_[top[0]]}**")
            st.subheader("Top 3 predictions")
            st.table(pd.DataFrame({
                "Intent": [model.classes_[i] for i in top],
                "Confidence": [f"{probs[i] * 100:.1f}%" for i in top],
            }))