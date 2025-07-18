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

class ConversationResponse(BaseModel):
    response: str
    conversation_id: str
    title: str

class ConversationListItem(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    preview: str  # First few characters of the first message

# Global model instance - load once and reuse
model = None

# Request queue for handling concurrent requests
request_queue = queue.Queue()

# Response dictionary to store results
response_dict = {}
response_lock = threading.Lock()

# Maximum number of previous exchanges to remember
MAX_HISTORY = 5  # This means 5 pairs of user/assistant messages

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
        
        # Create messages table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        )
        ''')
        
        conn.commit()

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row  # This enables column access by name
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
            "SELECT role, content, timestamp FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC", 
            (conversation_id,)
        )
        
        message_rows = cursor.fetchall()
        messages = [Message(role=row['role'], content=row['content'], timestamp=row['timestamp']) for row in message_rows]
        
        # Construct and return the conversation object
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
    
    # Set timestamps if not already set
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
            
            # Check if this message already exists
            cursor.execute(
                """
                SELECT id FROM messages 
                WHERE conversation_id = ? AND role = ? AND content = ? AND timestamp = ?
                """,
                (conversation.id, message.role, message.content, message.timestamp)
            )
            
            existing_message = cursor.fetchone()
            
            if not existing_message:
                # Insert the new message
                cursor.execute(
                    """
                    INSERT INTO messages (id, conversation_id, role, content, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (str(uuid.uuid4()), conversation.id, message.role, message.content, message.timestamp)
                )
        
        conn.commit()

def get_all_conversations():
    """Get a list of all conversations with preview"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get all conversations
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
        
        # Delete messages first (foreign key constraint)
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        
        # Delete the conversation
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        
        conn.commit()
        
        return cursor.rowcount > 0  # Return True if conversation was deleted

def initialize_model():
    """Initialize the LLM model once at startup"""
    global model
    
    # Update to use TinyLlama model path - update this to your actual path
<<<<<<< HEAD
    model_path = "E:/Deepseek-agent/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
=======
    model_path = "D:/Projects/Deepseek-agent/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
>>>>>>> 4f07cb9 (added rag.py)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    print(f"Loading TinyLlama model from {model_path}...")
    
    # Load the model with parameters optimized for TinyLlama
    model = Llama(
        model_path=model_path,
        n_ctx=4096,        # TinyLlama can handle decent context
        n_batch=512,       # Batch size for prompt processing
        n_threads=4,       # CPU threads, adjust based on your system
        n_gpu_layers=0,    # Set to higher value if you have a compatible GPU 
        verbose=False      # Set to True for debugging
    )
    
    print("TinyLlama model loaded successfully!")

def inference_worker():
    """Worker thread that processes requests from the queue"""
    while True:
        try:
            # Get request from queue
            request_id, prompt, max_tokens, conversation_id, title = request_queue.get()
            
            # Process with model
            output = model(
                prompt,
                max_tokens=max_tokens,
                stop=["User:", "\n\nUser:", "<|im_end|>", "<|endoftext|>"],  # Stop tokens for TinyLlama
                echo=False,
                temperature=0.7,  # Good balance for chat
                repeat_penalty=1.15  # Slightly increased for TinyLlama which can be repetitive
            )
            
            # Extract the response
            response_text = output["choices"][0]["text"].strip()
            
            # Store the result
            with response_lock:
                response_dict[request_id] = (response_text, conversation_id, title)
            
            # Mark task as done
            request_queue.task_done()
            
        except Exception as e:
            print(f"Error in inference worker: {str(e)}")
            with response_lock:
                response_dict[request_id] = (f"Error: {str(e)}", conversation_id, None)
            request_queue.task_done()

@app.on_event("startup")
async def startup_event():
    """Initialize model, database and start worker threads on application startup"""
    try:
        # Initialize the database
        init_db()
        print("Database initialized successfully!")
        
        # Initialize the model
        initialize_model()
        
        # Start worker thread
        worker_thread = threading.Thread(target=inference_worker, daemon=True)
        worker_thread.start()
        
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        # Don't raise exception here, let the app start and handle errors in endpoints

@app.post("/ask", response_model=ConversationResponse)
async def ask(request: PromptRequest, background_tasks: BackgroundTasks):
    # Print request for debugging
    print(f"Received request: {request}")
    
    # Ensure model is loaded
    global model
    if model is None:
        try:
            initialize_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize model: {str(e)}")
    
    # Get or create conversation
    conversation_id = request.conversation_id
    conversation = None
    
    if conversation_id:
        conversation = get_conversation_from_db(conversation_id)
        print(f"Retrieved conversation: {conversation}")
    
    if not conversation:
        # Create new conversation
        conversation_id = str(uuid.uuid4())
        conversation = Conversation(
            id=conversation_id,
            title=request.title or "New conversation",
            messages=[]
        )
        print(f"Created new conversation with ID: {conversation_id}")
    else:
        print(f"Using existing conversation with ID: {conversation_id}")
        # Update title if provided
        if request.title:
            conversation.title = request.title
    
    # Add user message to history
    now = datetime.datetime.utcnow().isoformat()
    new_message = Message(role="user", content=request.prompt, timestamp=now)
    conversation.messages.append(new_message)
    
    # Save conversation with the new user message
    save_conversation_to_db(conversation)
    
    # Format prompt for TinyLlama Chat
    # TinyLlama chat versions were fine-tuned on the Alpaca-LoRA format
    system_prompt = "<|im_start|>system\nYou are a helpful AI assistant. You provide accurate, helpful and detailed responses.<|im_end|>"
    
    # Build conversation history in the format TinyLlama expects
    conversation_parts = [system_prompt]
    
    # Add relevant history - use the full history
    relevant_history = conversation.messages[-MAX_HISTORY*2:] if len(conversation.messages) > 1 else []
    
    if relevant_history:
        for msg in relevant_history:
            if msg.role == "user":
                conversation_parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
            else:
                conversation_parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")
    else:
        # Just add the current user message if no history
        conversation_parts.append(f"<|im_start|>user\n{request.prompt}<|im_end|>")
        
    # Add the assistant prompt to generate the response
    conversation_parts.append("<|im_start|>assistant")
    
    # Join all parts to create the full prompt
    full_prompt = "\n".join(conversation_parts)
    
    print(f"Full prompt length: {len(full_prompt)} characters")
    
    try:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Submit request to queue
        request_queue.put((request_id, full_prompt, 1024, conversation_id, conversation.title))
        
        # Poll for result
        max_retries = 60  # 30 seconds with 0.5s sleep
        for _ in range(max_retries):
            with response_lock:
                if request_id in response_dict:
                    response_text, conv_id, title = response_dict.pop(request_id)
                    
                    # Get updated conversation from DB (it might have changed)
                    conversation = get_conversation_from_db(conv_id)
                    if not conversation:
                        # If conversation got deleted during processing, create a new one
                        conversation = Conversation(
                            id=conv_id,
                            title=title or "New conversation",
                            messages=[]
                        )
                    
                    # Auto-generate title from first user message if not already set
                    if conversation.title == "New conversation" and len(conversation.messages) == 1:
                        first_msg = conversation.messages[0].content
                        # Create a title from the first 5-6 words
                        title_words = first_msg.split()[:6]
                        auto_title = " ".join(title_words)
                        if len(auto_title) > 30:
                            auto_title = auto_title[:30] + "..."
                        conversation.title = auto_title
                    
                    # Add assistant response to conversation history
                    now = datetime.datetime.utcnow().isoformat()
                    new_response = Message(role="assistant", content=response_text, timestamp=now)
                    conversation.messages.append(new_response)
                    
                    # Save updated conversation to DB
                    save_conversation_to_db(conversation)
                    
                    return {
                        "response": response_text,
                        "conversation_id": conversation_id,
                        "title": conversation.title
                    }
            
            # Wait a bit before checking again
            time.sleep(0.5)
        
        # If we reach here, the request timed out
        return {
            "response": "⏱️ Request timed out. The model may be busy processing other requests.",
            "conversation_id": conversation_id,
            "title": conversation.title if conversation else "New conversation"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_message = f"Error: {str(e)}"
        print(error_message)
        return {
            "response": f"❌ {error_message}",
            "conversation_id": conversation_id,
            "title": conversation.title if conversation else "New conversation"
        }

# Endpoint to get conversation history
@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    conversation = get_conversation_from_db(conversation_id)
    if conversation:
        print(f"Retrieved conversation: {conversation}")
        return conversation
    raise HTTPException(status_code=404, detail="Conversation not found")

# Endpoint to get all conversations (for sidebar)
@app.get("/conversations")
async def list_conversations():
    conversations = get_all_conversations()
    print(f"Retrieved {len(conversations)} conversations")
    return conversations

# Endpoint to update conversation title
@app.post("/conversation/{conversation_id}/title")
async def update_conversation_title(conversation_id: str, title: str):
    conversation = get_conversation_from_db(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation.title = title
    save_conversation_to_db(conversation)
    return {"status": "Title updated"}

# Endpoint to reset/clear a conversation
@app.post("/conversation/{conversation_id}/reset")
async def reset_conversation(conversation_id: str):
    # Delete the old conversation and create a new one with the same ID
    conversation = get_conversation_from_db(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
        
    # Keep the title but clear messages
    title = conversation.title
    
    # Delete all messages for this conversation
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.commit()
    
    # Create a new conversation with the same ID
    new_conversation = Conversation(
        id=conversation_id,
        title=title,
        messages=[]
    )
    save_conversation_to_db(new_conversation)
    
    return {"status": "Conversation reset"}

# Endpoint to delete a conversation
@app.delete("/conversation/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str):
    success = delete_conversation(conversation_id)
    if success:
        return {"status": "Conversation deleted"}
    raise HTTPException(status_code=404, detail="Conversation not found")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "model_loaded": model is not None,
        "model_name": "TinyLlama Chat",
        "database": os.path.exists(DATABASE_FILE)
    }

# Debug: Get raw prompt endpoint
@app.get("/debug/prompt/{conversation_id}")
async def get_raw_prompt(conversation_id: str):
    """Debug endpoint to see how prompts are being constructed"""
    conversation = get_conversation_from_db(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
        
    # Build the prompt the same way as in /ask
    system_prompt = "<|im_start|>system\nYou are a helpful AI assistant. You provide accurate, helpful and detailed responses.<|im_end|>"
    
    conversation_parts = [system_prompt]
    relevant_history = conversation.messages[-MAX_HISTORY*2:] if len(conversation.messages) else []
    
    if relevant_history:
        for msg in relevant_history:
            if msg.role == "user":
                conversation_parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
            else:
                conversation_parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")
    
    conversation_parts.append("<|im_start|>assistant")
    full_prompt = "\n".join(conversation_parts)
    
    return {
        "raw_prompt": full_prompt,
        "token_estimate": len(full_prompt) // 4,  # Rough token estimate
        "message_count": len(conversation.messages)
    }

# Add an endpoint to see queue status
@app.get("/status")
async def status():
    return {
        "queue_size": request_queue.qsize(),
        "active_conversations": len(get_all_conversations()),
        "model": "TinyLlama-1.1B-Chat",
        "database_size": os.path.getsize(DATABASE_FILE) if os.path.exists(DATABASE_FILE) else 0
    }

# Debug database endpoint
@app.get("/debug/database")
async def debug_database():
    """Debug endpoint to check database contents"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get conversation count
        cursor.execute("SELECT COUNT(*) as count FROM conversations")
        conversation_count = cursor.fetchone()['count']
        
        # Get message count
        cursor.execute("SELECT COUNT(*) as count FROM messages")
        message_count = cursor.fetchone()['count']
        
        # Get sample conversations
        cursor.execute("SELECT id, title FROM conversations LIMIT 5")
        sample_conversations = [dict(row) for row in cursor.fetchall()]
        
        return {
            "conversation_count": conversation_count,
            "message_count": message_count,
            "sample_conversations": sample_conversations
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)