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
        document = extract_text(BytesIO(content))
        cleaned_text = self.preprocess_text(document)
        chunks = self.text_splitter.split_text(cleaned_text)
        return chunks

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
