from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load LLM ===
llm = Llama(model_path="D:/Projects/Deepseek-agent/llama.cpp/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", n_ctx=2048, n_threads=8)


# === Load RAG components ===
INDEX_PATH = "company_docs/index.faiss"
CHUNKS_PATH = "company_docs/chunks.pkl"
EMBED_MODEL = 'all-MiniLM-L6-v2'

embedding_model = SentenceTransformer(EMBED_MODEL)
faiss_index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "rb") as f:
    text_chunks = pickle.load(f)

# === Request Model ===
class PromptRequest(BaseModel):
    prompt: str

# === RAG Retriever ===
def retrieve_context(query: str, top_k: int = 5):
    query_vec = embedding_model.encode([query])
    D, I = faiss_index.search(np.array(query_vec), top_k)
    return [text_chunks[i] for i in I[0]]

# === Route ===
@app.post("/ask")
async def ask_model(request: PromptRequest):
    query = request.prompt
    context_chunks = retrieve_context(query)
    context_block = "\n".join(context_chunks)

    full_prompt = f"""
<|im_start|>system
You are a helpful assistant with access to company knowledge. Use the context below to help answer the user's question.

Context:
{context_block}
<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""

    output = llm(full_prompt, max_tokens=500, stop=["<|im_end|>"])
    response = output["choices"][0]["text"].strip()
    return {"response": response}
