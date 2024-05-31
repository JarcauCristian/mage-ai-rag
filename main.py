import os
import re
import ast
import json
import logging
import chromadb
from pathlib import Path
from ollama import Client
from rag import RAGPipeline
from ingester import Ingester
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG, filemode='w', format='%(name)s - %(levelname)s - %(message)s', filename="rag.log")

load_dotenv("./.env")
ollama_client = Client(os.getenv("OLLAMA_URL"))
db_path_exists = os.path.exists("./db")
chroma_client = chromadb.PersistentClient(path="./db")
ingester = Ingester(ollama_client, os.getenv("OLLAMA_EMBED_MODEL"), os.getenv("TOKENIZER"), int(os.getenv("MAX_TOKENS")))
rag = RAGPipeline(os.getenv("OLLAMA_URL"), os.getenv("OLLAMA_MODEL"), ollama_client, os.getenv("OLLAMA_EMBED_MODEL"), chroma_client, os.getenv("CHROMA_COLLECTION"))


def load_and_store(directory: str, block_type: str) -> None:
    for path in Path(directory).rglob("*.py"):
        print("Creating embeddings for %s" % path.name)
        logging.info("Creating embeddings for %s" % path.name)
        with open(path, "r") as f:
            file_content = f.read()
            ingester.ingest(file_content, path.name, block_type, chroma_client, os.getenv("CHROMA_COLLECTION"))


if __name__ == "__main__":
    if not db_path_exists:
        for p in Path("./").glob("*"):
            if p.is_dir() and p.name in ["loaders", "transforms", "exporters", "sensors"]:
                load_and_store(p.__str__(), p.name)

    result = rag.invoke("Can you build me a pipeline that will get a dataset from the following MySQL database: host -> 62.72.21.79, port -> 5432, database -> postgres, table -> iris, username -> postgres, password -> postgres. For this table remove each row that has a number of empty columns greater than 75 percent. Exported it back as a CSV with name iris.csv.")
    logging.critical(result["source_documents"])
    try:
        string: str = result["result"]
        string = string.replace('\\', '\\\\').replace('\"\"\"', '\\\"\\\"\\\"')
        parsed_dict = ast.literal_eval(string)

        for key, value in parsed_dict.items():
            print(key)
            with open(f"./outputs/{key}.py", 'w') as output:
                output.write(value.replace('\\n', '\n'))
    except json.decoder.JSONDecodeError as e:
        logging.error(e)
        print("Error:", e)
 
