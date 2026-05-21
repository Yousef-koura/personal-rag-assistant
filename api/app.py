from fastapi import FastAPI, HTTPException
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ── Global RAG chain (built once on startup) ──────────────────────────────────
rag_chain = None


def build_rag_chain():
    # 1. Load
    loader = PyPDFLoader(r"Data\Yousef_Koura_RAG_Knowledge_Base_v2.pdf")
    docs = loader.load()

    # 2. Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(docs)

    # 3. Embed + store (persisted to disk so it survives restarts)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory="./chroma_db",
    )

    # 4. Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # 5. Prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are YK Assistant — a personal AI assistant for Yousef Koura's portfolio. "
            "Answer questions accurately using the provided context about Yousef. "
            "If the answer requires inferring from dates or numbers, do so and explain briefly. "
            "If the context truly doesn't contain enough information, say: "
            "'I don't have that information in my knowledge base.' "
            "Keep responses clear, friendly, and concise."
        ),
        HumanMessagePromptTemplate.from_template(
            "Context:\n{context}\n\nQuestion:\n{question}"
        ),
    ])

    # 6. LLM  ← fix the model name here
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",          # change to your preferred Groq model
        api_key=os.getenv("GROQ_API_KEY"),
    )

    # 7. Chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain
    print("🔧 Building RAG chain...")
    rag_chain = build_rag_chain()
    print("✅ RAG chain ready.")
    yield
    print("👋 Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="YK Assistant API",
    description="RAG-powered chatbot for Yousef Koura's portfolio",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow your GitHub Pages portfolio to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yousef-koura.github.io",   # your portfolio URL
        "http://localhost:3000",             # local dev
        "http://127.0.0.1:5500",            # VS Code Live Server
        "*",                                 # remove this in production
    ],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "YK Assistant is running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok", "chain_ready": rag_chain is not None}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG chain not ready yet.")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer = rag_chain.invoke(request.question)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)