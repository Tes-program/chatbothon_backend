import os
from app.models.document import DocumentAnalysis
from app.models.user import Document
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStore
from sqlalchemy.orm import Session
from fastapi import UploadFile


class DocumentService:
    def __init__(self, db: Session):
        self.db = db
        self.upload_dir = "uploads"
        os.makedirs(self.upload_dir, exist_ok=True)  # Add this line
        self.vector_store = VectorStore()

    async def process_and_store_document(self, file: UploadFile, user_id: int):
        # Save file
        document = await self.save_document(file, user_id)

        # Process content
        content = await file.read()
        processor = DocumentProcessor()
        chunks = processor.process_pdf(content)

        # Store in vector database
        document_id = f"user_{user_id}_{document.id}"
        self.vector_store.store_chunks(chunks, document_id)

        return document, chunks

    async def save_document(
        self, file: UploadFile, user_id: int, title: str
    ) -> Document:
        file_path = os.path.join(self.upload_dir, f"{user_id}_{file.filename}")
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        db_document = Document(
            user_id=user_id,
            filename=file.filename,
            title=title,
            content_path=file_path
        )

        self.db.add(db_document)
        self.db.commit()
        self.db.refresh(db_document)

        return db_document

    async def store_document_analysis(self, document_id: int, analysis: str):
        db_analysis = DocumentAnalysis(
            document_id=document_id,
            analysis=analysis
        )
        self.db.add(db_analysis)
        self.db.commit()
        self.db.refresh(db_analysis)
        return db_analysis
