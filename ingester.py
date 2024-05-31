import uuid
import logging
import chromadb
from ollama import Client
from tokenizers import Tokenizer
from semantic_text_splitter import TextSplitter


class Ingester:
    def __init__(self, ollama_client: Client, ollama_embed_model: str, tokenizer: str, max_tokens: int) -> None:
        self.client = ollama_client
        self.embed_model = ollama_embed_model
        self.tokenizer = Tokenizer.from_pretrained(tokenizer)
        self.splitter: TextSplitter = TextSplitter.from_huggingface_tokenizer(self.tokenizer, capacity=max_tokens, overlap=50, trim=True)
    
    def chunk_it(self, text: str) -> list[str]:
        chunks = self.splitter.chunks(text)
        return chunks
    
    def embed_and_store(self, collection: chromadb.Collection, text: str, filename: str, block_type: str) -> None:
        chunks = self.chunk_it(text)
        for chunk in chunks:
            uid = uuid.uuid4().hex
            logging.info(id)
            embed = self.client.embeddings(self.embed_model, chunk)["embedding"]
            collection.add([uid], [embed], documents=[chunk], metadatas=[{"source": filename, "block_type": block_type}])

    def ingest(
        self,
        text: str,
        filename: str,
        block_type: str,
        chroma_client: chromadb.PersistentClient,
        chroma_collection_name: str,
    ) -> None:
        collection = chroma_client.get_or_create_collection(chroma_collection_name)
        self.embed_and_store(collection, text, filename, block_type)
        
