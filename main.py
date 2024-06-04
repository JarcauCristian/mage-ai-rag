import os
import yaml
import utils
import uvicorn
import logging
import chromadb
from pathlib import Path
from schemas import Query
from ollama import Client
from rag import RAGPipeline
from typing import Dict, Any
from ingester import Ingester
from dotenv import load_dotenv
from pydantic import ValidationError
from contextlib import asynccontextmanager
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

load_dotenv("./.env")
logging.basicConfig(level=logging.DEBUG, filemode='w', format='%(name)s - %(levelname)s - %(message)s', filename="rag.log")


@asynccontextmanager
async def lifespan(_):
    global ing, rag
    ollama_client = Client(os.getenv("OLLAMA_URL"))
    chroma_client = chromadb.PersistentClient("./db")
    ing = Ingester(ollama_client, chroma_client, os.getenv("OLLAMA_EMBED_MODEL"), os.getenv("TOKENIZER"), int(os.getenv("MAX_TOKENS")))

    if not os.path.exists("./db"):
        for p in Path("./blocks").glob("*"):
            if p.is_dir() and p.name in ["loaders", "transformers", "exporters"]:
                if p.name == "loaders":
                    utils.add_loaders(p.name, ing)
                elif p.name == "transformers":
                    utils.add_transformers(p.name, ing)
                elif p.name == "exporters":
                    utils.add_exporters(p.name, ing)

    rag = RAGPipeline(os.getenv("OLLAMA_URL"), os.getenv("OLLAMA_MODEL"), ollama_client,
                      os.getenv("OLLAMA_EMBED_MODEL"), chroma_client, os.getenv("CHROMA_COLLECTION"))

    yield
    del rag, ing


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
async def entry():
    return JSONResponse("Server is running on 0.0.0.0!", status_code=200)


@app.websocket("/ws")
async def socket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            validated_data = Query(**data)

            if validated_data.block_type not in ["loader", "transformer", "exporter"]:
                await websocket.send_json({"detail": "Invalid block_type. Only loader, transformer and exporter are allowed!"})
            else:
                result_block = await get_model_response(validated_data)
                await websocket.send_bytes(list(result_block.values())[0].encode("utf-8"))
    except WebSocketDisconnect:
        await websocket.send_json({"detail": "Websocket disconnect successfully!"})
    except ValidationError:
        await websocket.send_json({"detail": "JSON validation error!"})


async def get_model_response(query: Query) -> Dict[str, Any]:
    flt = {"$and": [{"block_type": {"$in": []}}, {"source": {"$in": []}}]}

    output = ing.retrieve_specific(os.getenv("CHROMA_COLLECTION"), query.description, query.block_type)

    flt["$and"][0]["block_type"]["$in"].append(output["block_type"])
    flt["$and"][1]["source"]["$in"].append(output["source"])

    print("Sending request to model!")
    result = rag.invoke(query.description, flt)

    string = utils.preprocess_yaml_string(result["result"])

    print("Parsing data!")
    parsed_data = yaml.safe_load(string)

    return parsed_data


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0')

    # query = "Can you give me a block that will load a dataset from the following MySQL database: host -> 62.72.21.79, port -> 5432, database -> postgres, table -> iris, username -> postgres, password -> postgres."
    # entries = query.split("::")
    # print(entries)

    #
    # for i in range(5):
    #     result = rag.invoke(query, flt)
    #     logging.critical(result["source_documents"])
    #     print(result["result"])
    #     string = utils.preprocess_yaml_string(result["result"])
    #     print(string)
    #     time.sleep(30)
    # parsed_data = yaml.safe_load(string)
    # for block_name, code in parsed_data.items():
    #     file_path = os.path.join("output", f"{block_name}.py")
    #     with open(file_path, 'w') as file:
    #         file.write(code.strip())
