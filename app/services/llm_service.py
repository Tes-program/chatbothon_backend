# app/services/llm_service.py
from groq import Groq
from typing import List
from .vector_store import VectorStore
# from config import settings
import os
from dotenv import load_dotenv


class LLMService:
    def __init__(self):
        load_dotenv()
        # Replace with env variable
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.client = Groq(api_key=api_key)
        self.vector_store = VectorStore()

    async def _analyze_content(self, chunks: List[str]) -> str:
        # Analyze first chunk for initial insights
        prompt = f"""Analyze this legal document and provide:
        1. Document type and purpose
        2. Key points
        3. Important terms

        Content: {chunks[0]}"""

        response = self.client.chat.completions.create(
            model="llama-3.2-3b-preview",
            messages=[
                {"role": "system",
                 "content": "You are a legal document analyzer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_completion_tokens=1024,
            top_p=1
        )

        return response.choices[0].message.content

    async def analyze_document(self, chunks: List[str]) -> dict:
        # First chunk for analysis
        prompt = f"""Analyze this document and provide:
1. Document type and purpose
2. Key points
3. Important terms

Content: {chunks[0]}

Provide a clear analysis of the above points based on the content."""

        response = self.client.chat.completions.create(
            model="llama-3.2-3b-preview",
            messages=[
                {"role": "system",
                 "content": "You are a document analyzer. "
                 "Analyze the provided content directly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_completion_tokens=1024
        )

        # Generate title
        title_response = await self.generate_title(chunks[0])

        return {
            "title": title_response,
            "analysis": response.choices[0].message.content
        }

    async def answer_question(self, question: str, document_id: str) -> str:
        relevant_chunks = self.vector_store.get_relevant_chunks(
            question,
            document_id=document_id  # Pass to vector store
        )
        answers = []

        for chunk in relevant_chunks:
            completion = self.client.chat.completions.create(
                model="llama-3.2-3b-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a legal assistant. "
                                "Provide clear and concise answers."
                    },
                    {
                        "role": "user",
                        "content": (f"Using this context: {chunk}\n\n"
                                    f"Answer this question: {question}")
                    }
                ],
                temperature=0.6,
                max_completion_tokens=1024,
                top_p=1
            )
            answers.append(completion.choices[0].message.content)

        return " ".join(answers)

    async def generate_title(self, content: str) -> str:
        prompt = (
            f"""Generate a concise and descriptive title for this legal
            document.
        The title should be clear and indicate the document's main purpose.

        Content: {content}

        Return only the title, nothing else."""
        )

        response = self.client.chat.completions.create(
            model="llama-3.2-3b-preview",
            messages=[
                {"role": "system",
                 "content": "You are a legal document analyzer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_completion_tokens=50
        )

        return response.choices[0].message.content.strip()
