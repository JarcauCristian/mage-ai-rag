import os
import yaml
import utils
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
chroma_client = chromadb.PersistentClient("./db")
ingester = Ingester(ollama_client, chroma_client, os.getenv("OLLAMA_EMBED_MODEL"), os.getenv("TOKENIZER"), int(os.getenv("MAX_TOKENS")))
rag = RAGPipeline(os.getenv("OLLAMA_URL"), os.getenv("OLLAMA_MODEL"), ollama_client, os.getenv("OLLAMA_EMBED_MODEL"), chroma_client, os.getenv("CHROMA_COLLECTION"))


def load_and_store(directory: str, block_type: str) -> None:
    if block_type == "loaders":
        utils.add_loaders(directory, ingester)
    elif block_type == "transformers":
        utils.add_transformers(directory, ingester)
    elif block_type == "exporters":
        utils.add_exporters(directory, ingester)


if __name__ == "__main__":
    if not db_path_exists:
        for p in Path("./blocks").glob("*"):
            print(p.name)
            if p.is_dir() and p.name in ["loaders", "transformers", "exporters"]:
                load_and_store(p.__str__(), p.name)

    query = "Can you build me a pipeline that will get a dataset from the following MySQL database: host -> 62.72.21.79, port -> 5432, database -> postgres, table -> iris, username -> postgres, password -> postgres :: For this table remove each row that has a number of empty columns greater than 75 percent :: After that impute the remaining missing values using KNNImputer :: At the end exported it back as a CSV file with name iris.csv."
    entries = query.split("::")
    flt = {"$and": [{"block_type": {"$in": []}}, {"source": {"$in": []}}]}
    for i, entry in enumerate(entries):
        if i == 0:
            bt = "loader"
        elif i == len(entries) - 1:
            bt = "exporter"
        else:
            bt = "transformer"

        output = ingester.retrieve_specific(os.getenv("CHROMA_COLLECTION"), entry, bt)

        flt["$and"][0]["block_type"]["$in"].append(output["block_type"])
        flt["$and"][1]["source"]["$in"].append(output["source"])

    result = rag.invoke(query, flt)
    logging.critical(result["source_documents"])
    print(result["result"])
    string = utils.preprocess_yaml_string(result["result"])
    print(string)
    parsed_data = yaml.safe_load(string)
    for block_name, code in parsed_data.items():
        file_path = os.path.join("output", f"{block_name}.py")
        with open(file_path, 'w') as file:
            file.write(code.strip())
