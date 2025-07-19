from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
from typing import List, Dict, Optional
import uuid
import threading
import time
import queue
import datetime
import json
import sqlite3
from contextlib import contextmanager
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DATABASE_FILE = "conversations.db"

# Model for the conversation history
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = None
    response_type: str = "regular"  # "regular" or "rag"

class Conversation(BaseModel):
    id: str
    title: str = "New conversation"
    messages: List[Message] = []
    created_at: str = None
    updated_at: str = None

class PromptRequest(BaseModel):
    prompt: str
    conversation_id: Optional[str] = None
    title: Optional[str] = None
    use_rag: bool = False  # New field to specify RAG or regular response

class ConversationResponse(BaseModel):
    response: str
    conversation_id: str
    title: str
    response_type: str  # "regular" or "rag"

class ConversationListItem(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    preview: str  # First few characters of the first message

# Global model instances
model = None
rag_model = None
embedding_model = None
faiss_index = None
text_chunks = None

# Request queue for handling concurrent requests
request_queue = queue.Queue()

# Response dictionary to store results
response_dict = {}
response_lock = threading.Lock()

# Maximum number of previous exchanges to remember
MAX_HISTORY = 5

# RAG configuration
INDEX_PATH = "company_docs/index.faiss"
CHUNKS_PATH = "company_docs/chunks.pkl"
EMBED_MODEL = 'all-MiniLM-L6-v2'

# Database setup functions
def init_db():
    """Initialize the database with required tables"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Create conversations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        ''')
        
        # Create messages table with response_type column
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            response_type TEXT DEFAULT 'regular',
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        )
        ''')
        
        # Add response_type column if it doesn't exist (for existing databases)
        try:
            cursor.execute('ALTER TABLE messages ADD COLUMN response_type TEXT DEFAULT "regular"')
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        conn.commit()

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def get_conversation_from_db(conversation_id: str) -> Optional[Conversation]:
    """Retrieve a conversation by ID from the database"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get conversation info
        cursor.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?", 
            (conversation_id,)
        )
        
        conversation_row = cursor.fetchone()
        if not conversation_row:
            return None
            
        # Get messages for this conversation
        cursor.execute(
            "SELECT role, content, timestamp, response_type FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC", 
            (conversation_id,)
        )
        
        message_rows = cursor.fetchall()
        messages = [
            Message(
                role=row['role'], 
                content=row['content'], 
                timestamp=row['timestamp'],
                response_type=row['response_type'] if 'response_type' in row.keys() else 'regular'
            ) for row in message_rows
        ]
        
        return Conversation(
            id=conversation_row['id'],
            title=conversation_row['title'],
            messages=messages,
            created_at=conversation_row['created_at'],
            updated_at=conversation_row['updated_at']
        )

def save_conversation_to_db(conversation: Conversation):
    """Save a conversation to the database"""
    now = datetime.datetime.utcnow().isoformat()
    
    if not conversation.created_at:
        conversation.created_at = now
    conversation.updated_at = now
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Insert or update conversation
        cursor.execute(
            """
            INSERT OR REPLACE INTO conversations (id, title, created_at, updated_at) 
            VALUES (?, ?, ?, ?)
            """,
            (conversation.id, conversation.title, conversation.created_at, conversation.updated_at)
        )
        
        # Check which messages need to be saved
        for message in conversation.messages:
            if not message.timestamp:
                message.timestamp = now
            
            cursor.execute(
                """
                SELECT id FROM messages 
                WHERE conversation_id = ? AND role = ? AND content = ? AND timestamp = ?
                """,
                (conversation.id, message.role, message.content, message.timestamp)
            )
            
            existing_message = cursor.fetchone()
            
            if not existing_message:
                cursor.execute(
                    """
                    INSERT INTO messages (id, conversation_id, role, content, timestamp, response_type)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (str(uuid.uuid4()), conversation.id, message.role, 
                     message.content, message.timestamp, message.response_type)
                )
        
        conn.commit()

def get_all_conversations():
    """Get a list of all conversations with preview"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT c.id, c.title, c.created_at, c.updated_at, 
                   (SELECT content FROM messages 
                    WHERE conversation_id = c.id AND role = 'user'
                    ORDER BY timestamp ASC LIMIT 1) as preview
            FROM conversations c
            ORDER BY c.updated_at DESC
        """)
        
        rows = cursor.fetchall()
        
        result = []
        for row in rows:
            preview = row['preview']
            if preview and len(preview) > 60:
                preview = preview[:60] + "..."
                
            result.append(ConversationListItem(
                id=row['id'],
                title=row['title'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                preview=preview or ""
            ))
            
        return result

def delete_conversation(conversation_id: str):
    """Delete a conversation and its messages from the database"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        
        conn.commit()
        return cursor.rowcount > 0
def initialize_model():
    """Initialize the regular LLM model"""
    global model
    
    model_path = os.path.join(os.path.dirname(__file__), "llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    print(f"Loading Deepseek model from {model_path}...")
    
    model = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_batch=512,
        n_threads=4,
        n_gpu_layers=0,
        verbose=False
    )
    
    print("Deepseek model loaded successfully!")

def initialize_rag_system():
    """Initialize the RAG system components"""
    global rag_model, embedding_model, faiss_index, text_chunks
    
    try:
        # Use the same model for RAG
        rag_model_path = os.path.join(os.path.dirname(__file__), "llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        
        if os.path.exists(rag_model_path):
            print(f"Loading RAG model from {rag_model_path}...")
            rag_model = Llama(model_path=rag_model_path, n_ctx=2048, n_threads=8)
            print("RAG model loaded successfully!")
        else:
            print(f"RAG model not found at {rag_model_path}, RAG features will be disabled")
            return
        
        # Load embedding model and FAISS index as before
        if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
            print("Loading RAG components...")
            embedding_model = SentenceTransformer(EMBED_MODEL)
            faiss_index = faiss.read_index(INDEX_PATH)
            with open(CHUNKS_PATH, "rb") as f:
                text_chunks = pickle.load(f)
            print("RAG components loaded successfully!")
        else:
            print(f"RAG index files not found at {INDEX_PATH} or {CHUNKS_PATH}, RAG features will be disabled")
            
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        rag_model = None
        embedding_model = None
        faiss_index = None
        text_chunks = None

def retrieve_context(query: str, top_k: int = 5):
    """Retrieve relevant context for RAG"""
    if not all([embedding_model, faiss_index, text_chunks]):
        return []
    
    try:
        query_vec = embedding_model.encode([query])
        D, I = faiss_index.search(np.array(query_vec), top_k)
        return [text_chunks[i] for i in I[0]]
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return []

def inference_worker():
    """Worker thread that processes requests from the queue"""
    while True:
        try:
            request_id, prompt, max_tokens, conversation_id, title, use_rag = request_queue.get()
            
            if use_rag and rag_model:
                # Use RAG model
                output = rag_model(
                    prompt,
                    max_tokens=max_tokens,
                    stop=["<|im_end|>"],
                    echo=False,
                    temperature=0.7
                )
                response_type = "rag"
            else:
                # Use regular model
                output = model(
                    prompt,
                    max_tokens=max_tokens,
                    stop=["User:", "\n\nUser:", "<|im_end|>", "<|endoftext|>"],
                    echo=False,
                    temperature=0.7,
                    repeat_penalty=1.15
                )
                response_type = "regular"
            
            response_text = output["choices"][0]["text"].strip()
            
            with response_lock:
                response_dict[request_id] = (response_text, conversation_id, title, response_type)
            
            request_queue.task_done()
            
        except Exception as e:
            print(f"Error in inference worker: {str(e)}")
            with response_lock:
                response_dict[request_id] = (f"Error: {str(e)}", conversation_id, None, "error")
            request_queue.task_done()

@app.on_event("startup")
async def startup_event():
    """Initialize models, database and start worker threads on application startup"""
    try:
        init_db()
        print("Database initialized successfully!")
        
        initialize_model()
        initialize_rag_system()
        
        # Start worker thread
        worker_thread = threading.Thread(target=inference_worker, daemon=True)
        worker_thread.start()
        
    except Exception as e:
        print(f"Error during startup: {str(e)}")

@app.post("/ask", response_model=ConversationResponse)
async def ask(request: PromptRequest, background_tasks: BackgroundTasks):
    print(f"Received request: {request}")
    
    # Ensure at least the regular model is loaded
    global model
    if model is None:
        try:
            initialize_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize model: {str(e)}")
    
    # Check if RAG is requested but not available
    if request.use_rag and not rag_model:
        raise HTTPException(status_code=400, detail="RAG model not available")
    
    # Get or create conversation
    conversation_id = request.conversation_id
    conversation = None
    
    if conversation_id:
        conversation = get_conversation_from_db(conversation_id)
    
    if not conversation:
        conversation_id = str(uuid.uuid4())
        conversation = Conversation(
            id=conversation_id,
            title=request.title or "New conversation",
            messages=[]
        )
    else:
        if request.title:
            conversation.title = request.title
    
    # Add user message to history
    now = datetime.datetime.utcnow().isoformat()
    new_message = Message(
        role="user", 
        content=request.prompt, 
        timestamp=now,
        response_type="regular"  # User messages are always regular
    )
    conversation.messages.append(new_message)
    
    save_conversation_to_db(conversation)
    
    # Format prompt based on whether RAG is used
    if request.use_rag and rag_model:
        # RAG prompt formatting
        context_chunks = retrieve_context(request.prompt)
        context_block = "\n".join(context_chunks)
        
        full_prompt = f"""
<|im_start|>system
You are a helpful assistant with access to company knowledge. Use the context below to help answer the user's question.

Context:
{context_block}
<|im_end|>
<|im_start|>user
{request.prompt}<|im_end|>
<|im_start|>assistant
"""
    else:
        # Regular prompt formatting
        system_prompt = "<|im_start|>system\nYou are a helpful AI assistant. You provide accurate, helpful and detailed responses.<|im_end|>"
        
        conversation_parts = [system_prompt]
        
        relevant_history = conversation.messages[-MAX_HISTORY*2:] if len(conversation.messages) > 1 else []
        
        if relevant_history:
            for msg in relevant_history:
                if msg.role == "user":
                    conversation_parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
                else:
                    conversation_parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")
        else:
            conversation_parts.append(f"<|im_start|>user\n{request.prompt}<|im_end|>")
            
        conversation_parts.append("<|im_start|>assistant")
        full_prompt = "\n".join(conversation_parts)
    
    try:
        request_id = str(uuid.uuid4())
        
        # Submit request to queue
        request_queue.put((request_id, full_prompt, 1024, conversation_id, conversation.title, request.use_rag))
        
        # Poll for result
        max_retries = 60
        for _ in range(max_retries):
            with response_lock:
                if request_id in response_dict:
                    response_text, conv_id, title, response_type = response_dict.pop(request_id)
                    
                    conversation = get_conversation_from_db(conv_id)
                    if not conversation:
                        conversation = Conversation(
                            id=conv_id,
                            title=title or "New conversation",
                            messages=[]
                        )
                    
                    # Auto-generate title if needed
                    if conversation.title == "New conversation" and len(conversation.messages) == 1:
                        first_msg = conversation.messages[0].content
                        title_words = first_msg.split()[:6]
                        auto_title = " ".join(title_words)
                        if len(auto_title) > 30:
                            auto_title = auto_title[:30] + "..."
                        conversation.title = auto_title
                    
                    # Add assistant response to conversation history
                    now = datetime.datetime.utcnow().isoformat()
                    new_response = Message(
                        role="assistant", 
                        content=response_text, 
                        timestamp=now,
                        response_type=response_type
                    )
                    conversation.messages.append(new_response)
                    
                    save_conversation_to_db(conversation)
                    
                    return {
                        "response": response_text,
                        "conversation_id": conversation_id,
                        "title": conversation.title,
                        "response_type": response_type
                    }
            
            time.sleep(0.5)
        
        # Timeout
        return {
            "response": "⏱️ Request timed out. The model may be busy processing other requests.",
            "conversation_id": conversation_id,
            "title": conversation.title if conversation else "New conversation",
            "response_type": "error"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_message = f"Error: {str(e)}"
        return {
            "response": f"❌ {error_message}",
            "conversation_id": conversation_id,
            "title": conversation.title if conversation else "New conversation",
            "response_type": "error"
        }

# All other endpoints remain the same...
@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    conversation = get_conversation_from_db(conversation_id)
    if conversation:
        return conversation
    raise HTTPException(status_code=404, detail="Conversation not found")

@app.get("/conversations")
async def list_conversations():
    return get_all_conversations()

@app.post("/conversation/{conversation_id}/title")
async def update_conversation_title(conversation_id: str, title: str):
    conversation = get_conversation_from_db(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation.title = title
    save_conversation_to_db(conversation)
    return {"status": "Title updated"}

@app.post("/conversation/{conversation_id}/reset")
async def reset_conversation(conversation_id: str):
    conversation = get_conversation_from_db(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
        
    title = conversation.title
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.commit()
    
    new_conversation = Conversation(
        id=conversation_id,
        title=title,
        messages=[]
    )
    save_conversation_to_db(new_conversation)
    
    return {"status": "Conversation reset"}

@app.delete("/conversation/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str):
    success = delete_conversation(conversation_id)
    if success:
        return {"status": "Conversation deleted"}
    raise HTTPException(status_code=404, detail="Conversation not found")

@app.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "regular_model_loaded": model is not None,
        "rag_model_loaded": rag_model is not None,
        "rag_available": all([rag_model, embedding_model, faiss_index, text_chunks]),
        "database": os.path.exists(DATABASE_FILE)
    }

@app.get("/status")
async def status():
    return {
        "queue_size": request_queue.qsize(),
        "active_conversations": len(get_all_conversations()),
        "regular_model": "TinyLlama-1.1B-Chat" if model else "Not loaded",
        "rag_model": "Mistral-7B-Instruct" if rag_model else "Not loaded",
        "rag_available": all([rag_model, embedding_model, faiss_index, text_chunks]),
        "database_size": os.path.getsize(DATABASE_FILE) if os.path.exists(DATABASE_FILE) else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)