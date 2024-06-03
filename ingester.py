import uuid
import time
import logging
import chromadb
from ollama import Client
from tokenizers import Tokenizer
from typing import Set, Dict, Any
from semantic_text_splitter import TextSplitter
from sklearn.feature_extraction.text import CountVectorizer


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.2f} seconds!")
        return result
    return wrapper


class Ingester:
    def __init__(self, ollama_client: Client, chroma_client: chromadb.PersistentClient, ollama_embed_model: str, tokenizer: str, max_tokens: int, threshold: float = 0.1) -> None:
        self.client = ollama_client
        self.chroma_client = chroma_client
        self.embed_model = ollama_embed_model
        self.threshold = threshold
        self.tokenizer = Tokenizer.from_pretrained(tokenizer)
        self.splitter: TextSplitter = TextSplitter.from_huggingface_tokenizer(self.tokenizer, capacity=max_tokens, overlap=50, trim=True)
    
    def chunk_it(self, text: str) -> list[str]:
        chunks = self.splitter.chunks(text)
        return chunks
    
    def embed_and_store(self, collection: chromadb.Collection, text: str, filename: str, block_type: str, description: str) -> None:
        chunks = self.chunk_it(text)
        for chunk in chunks:
            uid = uuid.uuid4().hex
            logging.info(id)
            embed = self.client.embeddings(self.embed_model, chunk)["embedding"]
            collection.add([uid], [embed], documents=[chunk], metadatas=[{"source": filename, "block_type": block_type, "description": description}])

    def ingest(
        self,
        text: str,
        filename: str,
        block_type: str,
        description: str,
        chroma_collection_name: str,
    ) -> None:
        collection = self.chroma_client.get_or_create_collection(chroma_collection_name)
        self.embed_and_store(collection, text, filename, block_type, description)

    @staticmethod
    def extract_keywords(text: str, max_features: int = 10):
        vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
        vectorizer.fit_transform([text])
        keywords = vectorizer.get_feature_names_out()
        return set(keywords)

    @staticmethod
    def keyword_similarity(query_keywords: Set[str], doc_keywords: Set[str]) -> float:
        common_keywords = query_keywords.intersection(doc_keywords)
        similarity = len(common_keywords) / len(query_keywords.union(doc_keywords))
        return similarity

    @timeit
    def retrieve_specific(self, collection: str, query: str, block_type: str) -> Dict[str, Any]:
        collection: chromadb.Collection = self.chroma_client.get_or_create_collection(collection)
        documents = collection.get(include=["documents", "metadatas"])
        query_keywords = self.extract_keywords(query)
        filtered_docs = []
        for doc in documents["metadatas"]:
            doc_keywords = self.extract_keywords(doc['description'])
            similarity = self.keyword_similarity(query_keywords, doc_keywords)
            if similarity >= self.threshold:
                filtered_docs.append(doc)

        if len(filtered_docs) == 0:
            returns = collection.get(include=["documents", "metadatas"], where={"$and": [{"block_type": {"$eq": block_type}}, {"source": {"$eq": "default.py"}}]})
            return dict(returns["metadatas"][0])

        return filtered_docs[0]
