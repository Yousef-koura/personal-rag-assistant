# YK Assistant 🤖

A RAG-powered personal AI chatbot that answers questions about Yousef Koura's skills, projects, and experience — built with LangChain, FastAPI, and ChromaDB.

---

## What is this?

Instead of reading through a CV, recruiters or visitors can just **ask**. The chatbot retrieves relevant information from a knowledge base (PDF) and generates accurate, grounded answers using an LLM.

```
User question
     │
     ▼
ChromaDB (vector search) → top 6 relevant chunks
     │
     ▼
Groq LLM (llama-4-scout) + context → answer
     │
     ▼
Response in the chat UI
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Backend | FastAPI + Uvicorn |
| RAG Framework | LangChain |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | ChromaDB (persisted locally) |
| LLM | Groq API (Llama 4 Scout) |
| Knowledge Base | PDF (compiled from CV + portfolio) |
| Frontend | Plain HTML/CSS/JS |

---

## Project Structure

```
yk-assistant/
├── api/
│   └── app.py                               # FastAPI backend
├── data/
│   └── Yousef_Koura_RAG_Knowledge_Base.pdf  # Knowledge base
├── chroma_db/                               # Auto-created on first run
├── index.html                               # Chat UI
├── requirements.txt
├── .env                                     # Your API keys (never commit)
├── .env.example                             # Safe template to commit
├── .gitignore
└── README.md
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Yousef-koura/yk-assistant.git
cd yk-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Activate
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Then open `.env` and add your keys:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at: https://console.groq.com

### 5. Run the backend

```bash
uvicorn api.app:app --reload
```

API will be live at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

### 6. Open the UI

Just open `index.html` directly in your browser — no server needed for the frontend.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Status check |
| GET | `/health` | Health + chain readiness |
| POST | `/chat` | Send a question, get an answer |

### Example request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What projects has Yousef built?"}'
```

### Example response

```json
{
  "answer": "Yousef has built several projects including Agri-Bot (96% accuracy crop disease detection robot deployed on NVIDIA Jetson Nano), an FMCG Data Engineering Pipeline on Databricks, a Ball Tracking Robot, a Steganography Detector, and a Breast Cancer AI classifier that won 3rd place in a competitive challenge."
}
```

---

## RAG Pipeline Details

| Step | Config |
|---|---|
| Chunk size | 1000 characters |
| Chunk overlap | 150 characters |
| Splitter | RecursiveCharacterTextSplitter |
| Embedding model | all-MiniLM-L6-v2 |
| Retrieved chunks (k) | 6 |
| Vector store | Chroma (persisted to `./chroma_db`) |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | From console.groq.com |
| `LANGCHAIN_API_KEY` | Optional | LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | Optional | Enable tracing (true/false) |
| `LANGCHAIN_PROJECT` | Optional | LangSmith project name |

---

## Author

**Yousef Koura** — Mechatronics Engineering Graduate & ML Engineer

- GitHub: https://github.com/Yousef-koura
- LinkedIn: https://linkedin.com/in/yousefkoura
- Portfolio: https://yousefkoura.github.io
