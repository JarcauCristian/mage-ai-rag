from pydantic import BaseModel


class Query(BaseModel):
    block_type: str
    description: str
