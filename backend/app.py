import os
import re
import docx
import pdfplumber
import pytesseract
import nltk
import language_tool_python
from langdetect import detect
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from summa.summarizer import summarize as textrank_summarize

# Setup
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
nltk.download('punkt')
nltk.download('stopwords')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model
MODEL_PATH =r"C:\Users\Heera Prashanth\Desktop\Final year project\backend\models\fine_tuned_pegasus"
tokenizer = PegasusTokenizer.from_pretrained(MODEL_PATH)
model = PegasusForConditionalGeneration.from_pretrained(MODEL_PATH)
tool = language_tool_python.LanguageTool('en-US')

# Utils
def clean_text(text):
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    text = re.sub(r'[“”\"\'u2022·●▪]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', '', text)
    return text.strip()

def extract_text(file_path):
    ext = file_path.split('.')[-1].lower()
    text = ''
    if ext == 'pdf':
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    lines = page_text.split('\n')
                    filtered = [
                        l for l in lines if not re.search(r'[\w\.-]+@[\w\.-]+', l)
                        and not any(x in l.lower() for x in [
                            'doi', 'copyright', 'journal', 'author', 'volume',
                            'published', 'rights reserved', 'private limited']
                        )
                        and len(l.split()) > 5
                    ]
                    text += '\n'.join(filtered) + '\n'
    elif ext == 'docx':
        doc = docx.Document(file_path)
        try:
            paragraphs = doc.paragraphs
            filtered_paragraphs = [
                para.text for para in paragraphs
                if not re.match(r'^[A-Za-z0-9\s]+,\s*[A-Za-z\s]+$', para.text)
                and not re.match(r'\bAnalysis of.*\b', para.text)
            ]
            text = '\n'.join(filtered_paragraphs)
        finally:
            del doc
    elif ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        raise HTTPException(400, detail="Unsupported file format.")
    return clean_text(text)

def chunk_by_sentences(text, chunk_size=4):
    sentences = sent_tokenize(text)
    return [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

def correct_grammar(text):
    matches = tool.check(text)
    return language_tool_python.utils.correct(text, matches)

def clean_broken_words(text):
    patterns = [
        (r'\bconvolutionrization\b', 'convolutional summarization'),
        (r'\bnatural language pronouncing words\b', 'natural language processing'),
        (r'\bemploys\s+determine\b', 'aims to determine'),
        (r'\bdialogue\s+converse\b', 'dialogue generation'),
        (r'\bDIA\s+ton\b', 'dialogue context'),
        (r'\bthe topic flips\b', 'topic shifts occur'),
        (r'(\b[a-z]+-)+[a-z]+\b', ''),
        (r'\b(for|of|to|from|with)?\s*rouge-[nl12]\b', ''),
        (r'\b\d+\s+\d+\b', '')
    ]
    for pat, repl in patterns:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text

def is_valid_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

def is_low_quality_summary(text):
    words = word_tokenize(text.lower())
    stops = set(stopwords.words('english'))
    stop_ratio = sum(1 for w in words if w in stops) / (len(words) + 1e-5)
    avg_len = sum(len(s.split()) for s in sent_tokenize(text)) / (len(sent_tokenize(text)) + 1e-5)
    hallucinations = ['inform want', 'summer refined', 'DIA ton', 'topic flips']
    has_junk = any(h in text.lower() for h in hallucinations)
    return stop_ratio < 0.2 or avg_len < 6 or has_junk

def calculate_summary_length(text):
    num_tokens = len(tokenizer.tokenize(text))
    if num_tokens <= 300:
        return 80
    elif num_tokens <= 600:
        return 120
    elif num_tokens <= 1000:
        return 160
    elif num_tokens <= 1500:
        return 200
    else:
        return 250  # Max cap for performance


def extract_models_and_metrics(text):
    # More precise regex to capture models and accuracy
    model_candidates = re.findall(r'\b(SVM|CNN|LSTM|Random Forest|XGBoost|ResNet|BERT|Deep Learning|Neural Network|Decision Tree|Naive Bayes|Logistic Regression|Transformer|TextRank|Pegasus|GPT|ANN|RNN|GRU|ViT|ResNet|KNN|MLPs|VAE|Linear Regression|CLIP|GAN)\b', text)
    accuracy_vals = re.findall(r'(accuracy\s*(is|=|:)?\s*(\d{1,3})\s*(\.\d+)?%?)', text, re.IGNORECASE)
    dataset_names = re.findall(r'(CHB-MIT|Bonn|UCI|Epileptic Seizure Recognition)', text, re.IGNORECASE)

    accuracy_cleaned = [f"{acc[2]}%" for acc in accuracy_vals]
    return {
        "models": list(set(model_candidates)),
        "accuracy": accuracy_cleaned,
        "datasets": list(set(dataset_names)),
    }


def generate_final_summary(text):
    chunks = chunk_by_sentences(text, chunk_size=5)  # Slightly bigger chunks for context
    summaries = []
    summary_length = calculate_summary_length(text)

    for chunk in chunks[:3]:  # Use up to 3 chunks max for performance + relevance
        if not is_valid_english(chunk):
            continue
        chunk = clean_broken_words(chunk)
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding="longest", max_length=512)
        ids = model.generate(
            inputs.input_ids,
            max_length=summary_length,
            min_length=summary_length // 2,
            num_beams=5,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        raw = tokenizer.decode(ids[0], skip_special_tokens=True)
        sentences = sent_tokenize(raw)
        filtered_sentences = [
            s.strip().capitalize() for s in sentences
            if len(s.split()) > 4 and re.match(r'^[A-Z]', s.strip()) and s.strip().endswith(('.', '?', '!'))
        ]
        summaries.append(' '.join(filtered_sentences))

    full_summary = ' '.join(summaries[:5])
    full_summary = clean_broken_words(full_summary)

    # Extract key info AFTER full summary
    key_info = extract_models_and_metrics(text)
    key_points = ""
    if any(key_info.values()):
        key_points += "\n\n*Key Information Extracted:*"
        if key_info['models']:
            key_points += f"\n- Models Used: {', '.join(key_info['models'])}"
        if key_info['accuracy']:
            key_points += f"\n- Accuracy Mentioned: {', '.join(key_info['accuracy'])}"
        if key_info['datasets']:
            key_points += f"\n- Datasets: {', '.join(key_info['datasets'])}"

    final_summary = f"{full_summary}{key_points}"
    final_summary = correct_grammar(final_summary)
    return final_summary


@app.post("/summarize/")
async def summarize(file: UploadFile = File(...)):
    ext = file.filename.split('.')[-1].lower()
    if ext not in ['pdf', 'docx', 'txt']:
        raise HTTPException(400, detail="Unsupported file format.")

    os.makedirs("uploaded_files", exist_ok=True)
    path = f"uploaded_files/{file.filename}"

    try:
        with open(path, "wb") as f:
            content = await file.read()
            f.write(content)
        text = extract_text(path)
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to process the file: {str(e)}")
    finally:
        try:
            os.remove(path)
        except Exception as e:
            print(f"Warning: could not delete file {path}: {e}")

    if not text or len(text.split()) < 20:
        raise HTTPException(400, detail="Content too short after filtering.")

    summary = generate_final_summary(text)
    if is_low_quality_summary(summary):
        raise HTTPException(500, detail="Summary appears to be low quality. Try a different document.")

    return {
        "filename": file.filename,
        "word_count": len(text.split()),
        "final_summary": summary
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)