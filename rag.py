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
            retriever=self.db_retriever.as_retriever(),
            return_source_documents=True,
        )

    def build_prompt(self, data):
        return '''
            You are an expert in building MageAI ETL pipelines by interconnecting loader, transformer, exporter, and sensor blocks to create the ideal pipeline based on the description received from the user.

            **Instructions:**

            1. **Output Format**: Return the Python code for each block in the exact MageAI format as a dictionary. The dictionary should have random block names as keys and the Python code for each block as values. Return only the dictionary and no additional text.

            2. **Block Templates**: 
            - Use the default MageAI templates for loader, transformer, exporter, and sensor blocks.
            - If the block type requested is already available in the templates, adjust it according to the user's requirements.
            - Otherwise, use the default.py template and build upon it.

            3. **Code Format**: Output all the code inside the retrived templates or the default.py file only add additional functionallity inside the functions with the decorators:
            - @data_loader
            - @transformer
            - @data_exporter
            - @sensor

            **Example Output**:
            {{
                "loader": "from mage_ai.data_preparation.decorators import data_loader\\nimport pandas as pd\\n\\n@data_loader\\ndef load_data(*args, **kwargs):\\n    return pd.read_csv('path_to_csv_file.csv')\\n",
                "remove_null_columns": "from mage_ai.data_preparation.decorators import transformer\\n\\n@transformer\\ndef remove_null_columns(df, *args, **kwargs):\\n    return df.dropna(axis=1, how='all')\\n",
                "exporter": "from mage_ai.data_preparation.decorators import data_exporter\\n\\n@data_exporter\\ndef export_data(df, *args, **kwargs):\\n    df.to_csv('exported_file.csv', index=False)\\n"
            }}
            You must not add anything besides the dictionary, not a single word only the dictionary as presented in the Example Output section. Also, you must be sure that what is returned from a block is only a pandas DataFrame, and you must adhere to the rules presented above very strictly, any deviation from them will be punished. 
            Here is the description for the pipeline:
            {}  

            You will find each indivudual block from the pipeline listed below:
        '''.format(
            data
        )

    def invoke(self, description: str) -> Dict[str, Any]:
        return self.rag.invoke({"query": self.build_prompt(description)})