# 📝 Research Paper Summarizer (Pegasus + Frontend)

## 📌 Overview
This project is a **web application** that summarizes long research papers into concise, structured abstracts using the **Pegasus transformer model**.  
It combines a **Python backend** (for ML processing) with a **React frontend** (for user interaction), making research review faster and more efficient.  

---

## ⚙️ Features
- Upload research papers (PDF or text).  
- Automatic summarization using Pegasus.  
- Intuitive React-based user interface.  
- Reduces paper length by ~80% while preserving key insights.  

---

## 🚀 Tech Stack
- **Backend:** Python, FastAPI/Flask, Hugging Face Transformers, Pegasus Model  
- **Frontend:** React (JavaScript, App.js)  
- **Other:** Numpy, Scikit-learn, PyTorch/TensorFlow  

---

## ▶️ Usage

### 🔹 Backend (ML + API)
```bash
cd backend
pip install -r requirements.txt
python app.py
This will start the backend server on a local port (e.g., http://localhost:8000).

🔹 Frontend (React UI)
bash
Copy code
cd frontend
npm install
npm start
This will start the React app and open it in your browser.

📂 Project Structure
bash
Copy code
research-summarizer/
├─ backend/
│  ├─ app.py              # Python backend (API + Pegasus inference)
│  ├─ requirements.txt    # Python dependencies
│  └─ models/             # (ignored in GitHub) Pegasus weights go here
├─ frontend/
│  └─ src/App.js          # React frontend
├─ .gitignore
└─ README.md

## 📷 Demo

Here’s the Research Summarizer in action:

![Upload Demo](screenshots/demo1.png)
![Summary Output](screenshots/demo2.png)