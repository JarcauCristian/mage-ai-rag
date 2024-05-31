from embed import Embed
from ollama import Client
from typing import Dict, Any
from utils import return_filter
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
            retriever=self.db_retriever.as_retriever(search_type="mmr", search_kwargs={"k": 5}),
            return_source_documents=True,
        )

    @staticmethod
    def build_prompt(data):
        return '''
            You are an expert at build Mage AI ETL pipelines, by interconnecting Mage AI formated loader, transformer, exporter and sensor blocks to create an ideal pipeline based on the description of the user.
            You must return the output as a YAML kind object where the keys are the names of blocks and the values is the Python Code for that specific block, based on the template with additional modifications if needed.
            You must and are obligated to provide only the YAML kind object as the output without any other words beside the YAML kind object, like in the **Example Output** section.
            **Example Output**
            loader: "from mage_ai.io.file import FileIO\nif 'data_loader' not in globals():\n    from mage_ai.data_preparation.decorators import data_loader\n\n@data_loader\ndef load_data_from_file(*args, **kwargs):\n    """\n    Template for loading data from filesystem.\n    Load data from 1 file or multiple file directories.\n\n    For multiple directories, use the following:\n        FileIO().load(file_directories=['dir_1', 'dir_2'])\n\n    Docs: https://docs.mage.ai/design/data-loading#fileio\n    """\n    filepath = 'path/to/your/file.csv'\n\n    return FileIO().load(filepath)",
            transformer: "if 'transformer' not in globals():\n    from mage_ai.data_preparation.decorators import transformer\nif 'test' not in globals():\n    from mage_ai.data_preparation.decorators import test\n\n@transformer\ndef remove_columns_with_missing_values(data, *args, **kwargs):\n    threshold = 0.5 if kwargs.get('threshold') is not None else kwargs.get('threshold')\n    return data.loc[:, data.isnull().mean() < threshold]\n\n@test\ndef test_output(output, *args) -> None:\n    assert output is not None, 'The output is undefined'",
            exporter: "from mage_ai.io.file import FileIO\nfrom pandas import DataFrame\n\nif 'data_exporter' not in globals():\n    from mage_ai.data_preparation.decorators import data_exporter\n\n@data_exporter\ndef export_data_to_file(df: DataFrame, **kwargs) -> None:\n    """\n    Template for exporting data to filesystem.\n\n    Docs: https://docs.mage.ai/design/data-loading#fileio\n    """\n    filepath = 'path/to/write/dataframe/to.csv'\n    FileIO().export(df, filepath)"
            
            Here is the description for the pipeline:
            {}  
        '''.format(
            data
        )

    def invoke(self, description: str) -> Dict[str, Any]:
        return self.rag.invoke({"query": self.build_prompt(description)})
