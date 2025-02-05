# main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from requests import Session
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.models.document import ChatHistory
from app.services.document_service import DocumentService
from .auth.auth_handler import AuthHandler
from .services.document_processor import DocumentProcessor
from .services.llm_service import LLMService
from .auth.routes import router as auth_router
from .models import user
from .database import engine, get_db
from fastapi import Request


app = FastAPI()
auth_handler = AuthHandler()
document_processor = DocumentProcessor()
llm_service = LLMService()
user.Base.metadata.create_all(bind=engine)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth", tags=["auth"])


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: user.User = Depends(auth_handler.get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Read content once
        content = await file.read()
        chunks = document_processor.process_pdf(content)

        # Reset file position
        await file.seek(0)

        result = await llm_service.analyze_document(chunks)
        document_service = DocumentService(db)
        document = await document_service.save_document(file, current_user.id, title=result["title"])

        document_id = f"user_{current_user.id}_{document.id}"
        document_service.vector_store.store_chunks(chunks, document_id)

        return {
            "document_id": document.id,
            "title": result["title"],
            "analysis": result["analysis"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_question(
    question: str,
    document_id: str,
    user_id: str = Depends(auth_handler.verify_token)
):
    # Here you'd retrieve the relevant chunks from your database
    # For now, using a placeholder
    context = "document_context"
    answer = await llm_service.answer_question(context, question)
    return {"answer": answer}

# New endpoints in main.py


@app.get("/documents/{document_id}")
async def get_document(
    document_id: int,
    current_user: user.User = Depends(auth_handler.get_current_user),
    db: Session = Depends(get_db)
):
    document = db.query(user.Document)\
        .filter(
            user.Document.id == document_id,
            user.Document.user_id == current_user.id
    )\
        .first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "id": document.id,
        "filename": document.filename,
        "created_at": document.created_at
    }


@app.get("/documents/{document_id}/chat")
@limiter.limit("5/minute")
async def get_chat_history(
    request: Request,  # Add this parameter
    document_id: int,
    current_user: user.User = Depends(auth_handler.get_current_user),
    db: Session = Depends(get_db)
):
    # Verify document belongs to user
    document = db.query(user.Document)\
        .filter(
            user.Document.id == document_id,
            user.Document.user_id == current_user.id
    )\
        .first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Get chat history
    chats = db.query(ChatHistory)\
        .filter(ChatHistory.document_id == document_id)\
        .order_by(ChatHistory.created_at.asc())\
        .all()

    return [
        {
            "id": chat.id,
            "question": chat.question,
            "answer": chat.answer,
            "created_at": chat.created_at
        }
        for chat in chats
    ]


@app.post("/documents/{document_id}/chat")
async def add_chat(
    document_id: int,
    question: str,
    current_user: user.User = Depends(auth_handler.get_current_user),
    db: Session = Depends(get_db)
):
    # Verify document belongs to user
    document = db.query(user.Document)\
        .filter(
            user.Document.id == document_id,
            user.Document.user_id == current_user.id
    )\
        .first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Get answer using LLM
    answer = await llm_service.answer_question(question, str(document_id))

    # Store in chat history
    chat = ChatHistory(
        document_id=document_id,
        question=question,
        answer=answer
    )
    db.add(chat)
    db.commit()
    db.refresh(chat)

    return {
        "id": chat.id,
        "question": chat.question,
        "answer": chat.answer,
        "created_at": chat.created_at
    }


@app.get("/documents/history")
async def get_document_history(
    current_user: user.User = Depends(auth_handler.get_current_user),
    db: Session = Depends(get_db)
):
    documents = db.query(user.Document)\
        .filter(user.Document.user_id == current_user.id)\
        .order_by(user.Document.created_at.desc())\
        .all()

    history = []
    for doc in documents:
        latest_chat = db.query(ChatHistory)\
            .filter(ChatHistory.document_id == doc.id)\
            .order_by(ChatHistory.created_at.desc())\
            .first()

        history.append({
            "document_id": doc.id,
            "filename": doc.filename,
            "title": doc.title,
            "created_at": doc.created_at,
            "last_chat": {
                "question": latest_chat.question if latest_chat else None,
                "answer": latest_chat.answer if latest_chat else None,
                "timestamp": latest_chat.created_at if latest_chat else None
            }
        })

    return history
