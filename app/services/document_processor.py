# app/services/document_processor.py
import os
from typing import List
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdfminer.high_level import extract_text
from io import BytesIO
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self):
        nltk.data.path.append(os.path.expanduser("~/nltk_data"))
        self.stopwords_set = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=5
        )

    def process_pdf(self, content: bytes) -> List[str]:
        try:
            logger.debug("Starting PDF text extraction")
            document = extract_text(BytesIO(content))
            logger.debug("PDF text extracted successfully")       
            logger.debug("Starting text preprocessing")
            cleaned_text = self.preprocess_text(document)
            logger.debug("Text preprocessing completed")

            logger.debug("Creating chunks")
            chunks = self.text_splitter.split_text(cleaned_text)
            logger.debug(f"Created {len(chunks)} chunks")

            return chunks
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise

    def preprocess_text(self, text: str) -> str:
        # Convert to lowercase
        text = text.lower()
        # Remove special characters
        text = re.sub(r'[^\w\s]', "", text)
        # Tokenize
        tokens = word_tokenize(text)
        # Lemmatize and remove stopwords
        tokens = [self.lemmatizer.lemmatize(token)
                  for token in tokens
                  if token not in self.stopwords_set]
        return ' '.join(tokens)
