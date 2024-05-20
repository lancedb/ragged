import lancedb
from typing import Optional
from lancedb.rerankers import Reranker

class QueryType:
    VECTOR = "vector"
    FTS = "fts"
    RERANK_VECTOR = "rerank_vector"
    RERANK_FTS = "rerank_fts"
    HYBRID = "hybrid"
    AUTO = "auto"

    ALL = "all"

class QueryConfigError(Exception):
    # Exception for query config errors
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def deduce_query_type(query_type: str, reranker: Optional[Reranker]):
    if query_type == QueryType.AUTO:
        if reranker is None:
        # This basically discourages the use of FTS with reranking which
        # isn't intended
            return QueryType.VECTOR
        else:
            return QueryType.HYBRID
    return query_type

def search_table(table: lancedb.table, reranker:Reranker, query_type: str, query_string, top_k:int=5, overfetch_factor:int=2):
    if query_type in [QueryType.RERANK_VECTOR, QueryType.RERANK_FTS, QueryType.HYBRID] and reranker is None:
        raise QueryConfigError(f"Reranker must be provided for query type: {query_type}")
    if query_type in [QueryType.VECTOR, QueryType.FTS]:
        rs = table.search(query_string, query_type=query_type).limit(top_k).to_pandas()
    elif query_type == QueryType.RERANK_VECTOR:
        rs = table.search(query_string, query_type=QueryType.VECTOR).rerank(reranker=reranker).limit(overfetch_factor*top_k).to_pandas()
    elif query_type == QueryType.RERANK_FTS:
        rs = table.search(query_string, query_type=QueryType.FTS).rerank(reranker=reranker).limit(overfetch_factor*top_k).to_pandas()
    elif query_type == QueryType.HYBRID:
        rs = table.search(query_string, query_type=QueryType.HYBRID).rerank(reranker=reranker).limit(top_k).to_pandas()
    else:
        raise ValueError(f"Invalid query type: {query_type}")
    return rs