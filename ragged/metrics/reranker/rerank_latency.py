import copy
from ..retriever.base import Metric
import lancedb
import logging
import pandas as pd
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
import time

from ...search_utils import QueryType, deduce_query_type, search_table, QueryConfigError
from ..retriever.hit_rate import Metric
from ...results import RerankerResult

# Set logging level to INFO
logger = logging.getLogger("lancedb")
logger.setLevel(logging.INFO)

class RerankerLatency:
    def __init__(self, metric: Metric) -> None:
        if metric.reranker is None:
            raise ValueError("Reranker must be provided to evaluate reranker latency")
        self.metric = metric
    
    def evaluate(self, query_type:str, top_k: int = 5) -> RerankerResult:
        if query_type == QueryType.HYBRID:
            raise ValueError("HYBRID query type is not supported for reranker latency evaluation")
        elif query_type == QueryType.VECTOR:
            qt, reranked_qt = QueryType.VECTOR, QueryType.RERANK_VECTOR
        elif query_type == QueryType.FTS:
            qt, reranked_qt = QueryType.FTS, QueryType.RERANK_FTS
        else:
            raise ValueError(f"Invalid query type: {query_type}. Supported query types are VECTOR and FTS")
        
        t1 = time.time()
        eval_reranker = self.metric.evaluate_query_type(reranked_qt, top_k)
        t2 = time.time()
        reranker_latency = t2 - t1

        t3 = time.time()
        eval_wo_reranker = self.metric.evaluate_query_type(qt, top_k)
        t4 = time.time()
        wo_reranker_latency = t4 - t3

        delta_accuracy = eval_reranker - eval_wo_reranker
        data_size = len(self.metric.dataset.to_pandas())
        latency = (reranker_latency - wo_reranker_latency) / data_size

        return RerankerResult(latency=latency, delta_accuracy=delta_accuracy)


