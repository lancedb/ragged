from cmath import inf
from pydantic import BaseModel
from .search_utils import QueryType

class RetriverResult(BaseModel):
    # Float for each query type in QueryType
    vector: float = -inf
    fts: float = -inf
    rerank_vector: float = -inf
    rerank_fts: float = -inf
    hybrid: float = -inf
    