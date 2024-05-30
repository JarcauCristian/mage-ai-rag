import os
import json
import logging
import chromadb
from pprint import pprint
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
ingester = Ingester(ollama_client, os.getenv("OLLAMA_EMBED_MODEL"), os.getenv("TOKENIZER"), 1024)
rag = RAGPipeline(os.getenv("OLLAMA_URL"), os.getenv("OLLAMA_MODEL"), ollama_client, os.getenv("OLLAMA_EMBED_MODEL"), chroma_client, os.getenv("CHROMA_COLLECTION"))

def load_and_store(dir: str, block_type: str):
    for path in Path(dir).rglob("*.py"):
        print("Creating embeddings for %s" % path.name)
        logging.info("Creating embeddings for %s" % path.name)
        with open(path, "r") as f:
            file_content = f.read()
            ingester.ingest(file_content, path.name, block_type, chroma_client, os.getenv("CHROMA_COLLECTION"))


def clean_files(dir: str):
    for path in Path(dir).rglob("*.py"):
        print(f"Cleaning {path.name}")
        with open(path, 'r') as f:
            res = ""
            lines = f.readlines()
            for line in lines:
                if "{%" in line or '{{' in line:
                    continue

                res += line

            with open(path, 'w') as w:
                w.write(res)




if __name__ == "__main__":
    if not db_path_exists:
        for path in Path("./").glob("*"):
            if path.is_dir() and path.name in ["loaders", "transforms", "exporters", "sensors"]:
                load_and_store(path, path.name)

    result = rag.invoke("Can you build me a pipeline that will get a dataset from the following MySQL database: host -> 62.72.21.79, port -> 5432, database -> postgres, table -> iris, username -> postgres, password -> postgres. For this table remove each row that has a number of empty columns greater than 75 percent. Exported it back as a CSV with name iris.csv.")
    print(result["source_documents"])
    try:
        for key, value in json.loads(result["result"]).items():
            print(key)
            with open(f"./outputs/{key}.py", 'w') as f:
                f.write(value)
    except json.decoder.JSONDecodeError:
        print(result["result"], result["source_documents"])
 
