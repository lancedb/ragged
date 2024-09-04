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

class RAGResult(BaseModel):
    faithfulness: float = -inf
    answer_relevancy: float = -inf
    context_precision: float = -inf
    context_recall: float = -inf
    harmfulness: float = -inf

class RerankerResult(BaseModel):
    latency: float = -inf
    delta_accuracy: float = -inf
