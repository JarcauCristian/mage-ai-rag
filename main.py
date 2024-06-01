import os
import yaml
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


def preprocess_yaml_string(yaml_string):
    # Ensure consistent indentation and format issues are resolved
    lines = yaml_string.split('\n')
    processed_lines = []
    found = False
    count = 0
    for i, line in enumerate(lines):
        if count == 2 and i < len(lines):
            break
        if "`" in line:
            count += 1
            continue
        if "|" in line:
            found = True
            processed_lines.append('  ' + line.strip())
        elif not found:
            continue
        else:
            processed_lines.append('    ' + line)

    return '\n'.join(processed_lines)


if __name__ == "__main__":
    if not db_path_exists:
        for p in Path("./").glob("*"):
            if p.is_dir() and p.name in ["loaders", "transforms", "exporters", "sensors"]:
                load_and_store(p.__str__(), p.name)

    result = rag.invoke("Can you build me a pipeline that will get a dataset from the following MySQL database: host -> 62.72.21.79, port -> 5432, database -> postgres, table -> iris, username -> postgres, password -> postgres. For this table remove each row that has a number of empty columns greater than 75 percent. After that impute the remaining missing values using KNNImputer. At the end exported it back as a CSV with name iris.csv.")
    logging.critical(result["source_documents"])
    print(result["result"])
    string = preprocess_yaml_string(result["result"])
    print(string)
    parsed_data = yaml.safe_load(string)
    for block_name, code in parsed_data.items():
        file_path = os.path.join("output", f"{block_name}.py")
        with open(file_path, 'w') as file:
            file.write(code.strip())
