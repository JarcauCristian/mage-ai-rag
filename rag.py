from embed import Embed
from ollama import Client
from typing import Dict, Any
from chromadb import PersistentClient
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA


class RAGPipeline:
    def __init__(
        self,
        ollama_url: str, 
        ollama_model: str,
        ollama_client: Client,
        ollama_embed_model: str,
        chroma_client: PersistentClient,
        chroma_collection_name: str,
    ) -> None:
        self.ollama_llm = Ollama(base_url=ollama_url, model=ollama_model, temperature=0.2)
        self.embeddings = Embed(ollama_client, ollama_embed_model)
        self.db_retriever = Chroma(
            client=chroma_client,
            collection_name=chroma_collection_name,
            embedding_function=self.embeddings,
        )
        self.rag = RetrievalQA.from_chain_type(
            self.ollama_llm,
            retriever=self.db_retriever.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10}),
            return_source_documents=True,
        )

    @staticmethod
    def build_prompt(data):
        return '''
            You are an expert at build Mage AI ETL pipelines, by interconnecting Mage AI formatted loader, transformer, exporter blocks to create an ideal pipeline based on the description of the user.
            You must return the output as an YAML object, exactly in the format provided inside **Example output** section, without any other information beside it.
            All the python code should strictly adhere to the Mage AI block templates based on block type, and inside the decorated function add all the necessary addition to the code, including imports.
            **Example Output**
            ```
            random_name: |
                <Python Code>
            random_name: |
                <Python Code>
            random_name: |
                <Python Code>
            ...
            random_name: |
                <Python Code>
            ```
            
            Here is the description for the pipeline:
            {}  
        '''.format(
            data
        )

    def invoke(self, description: str, query_filter: Dict[str, Any]) -> Dict[str, Any]:
        self.rag = RetrievalQA.from_chain_type(
            self.ollama_llm,
            retriever=self.db_retriever.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'filter': query_filter}),
            return_source_documents=True,
        )
        return self.rag.invoke({"query": self.build_prompt(description)})
