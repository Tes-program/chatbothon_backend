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
        try:
            # Debug logging
            print(f"Getting chunks for document: {document_id}")

            relevant_chunks = self.vector_store.get_relevant_chunks(
                question=question,
                document_id=document_id
            )

            # Debug logging
            print(f"Retrieved chunks: {relevant_chunks}")

            if not relevant_chunks:
                return (
                    "I couldn't find relevant information to answer your question.")

            # Single completion instead of multiple
            completion = self.client.chat.completions.create(
                model="llama-3.2-3b-preview",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant analyzing documents. "
                            "Provide clear and concise answers based on the given context."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"""Using this context: """
                            f"""{' '.join(relevant_chunks)}

Question: {question}

Provide a clear and direct answer based on the context."""
                        )
                    }
                ],
                temperature=0.7,
                max_completion_tokens=1024
            )

            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error in answer_question: {str(e)}")
            return f"Error processing question: {str(e)}"

    async def generate_title(self, content: str) -> str:
        prompt = f"""Generate a concise title (4-6 words max) that captures the document's core purpose.
    
Content: {content}

Return ONLY the title."""

        response = self.client.chat.completions.create(
            model="llama-3.2-3b-preview",
            messages=[
                {"role": "system", "content": "Generate brief, focused document titles"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Reduced for more focused output
            max_completion_tokens=20  # Reduced token limit
        )

        return response.choices[0].message.content.strip()
