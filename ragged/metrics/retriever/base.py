from abc import ABC, abstractmethod
from ...dataset.base import Dataset
from ...search_utils import QueryType, deduce_query_type
from ...results import RetriverResult
from lancedb.embeddings import get_registry
from lancedb.rerankers import Reranker
from tqdm import tqdm   
import logging
import sys

logger = logging.getLogger("lancedb")
logger.setLevel(logging.INFO)

class Metric(ABC):
    def __init__(self, 
                dataset: Dataset,
                embedding_registry_id: str = "huggingface",
                embed_model_kwarg: dict = {},
                reranker: Reranker = None,
                uri: str = None,
     ) -> None:
        super().__init__()
        self.dataset = dataset
        self.embedding_registry_id = embedding_registry_id
        self.embed_model_kwarg = embed_model_kwarg
        self.embedding_func = get_registry().get(self.embedding_registry_id).create(**self.embed_model_kwarg)
        self.reranker = reranker
        self.uri = uri
        if self.uri is None:
            self.uri = f"/tmp/ragged/"

        # Table for storing documents
        self.table = None
    
    @abstractmethod
    def ingest_docs(self, batched: bool = False, use_existing_table: bool = False):
        """
        Ingest documents into the database and initialize the table
        """
        pass

    @abstractmethod
    def evaluate_query_type(self,query_type:str, top_k:5) -> float:
        """
        Evaluate a single query type

        params:
        query_type: str
            Type of query to run
        top_k: int
            Number of top results to consider
        """
        pass

    def evaluate(self,
                 top_k: int, 
                 create_index: bool = False, 
                 query_type=QueryType.VECTOR, 
                 batched: bool = False,
                 use_existing_table: bool = False) -> RetriverResult:
        """
        Run evaluaion

        params:
        top_k: int
            Number of top results to consider
        create_index: bool
            Whether to create index
        query_type: str
            Type of query to run. Default is QueryType.VECTOR. 
            If "all" is passed, all query types will be evaluated
        """
        self.ingest_docs(batched, use_existing_table)
        if create_index:
            self.table.create_index(metric="L2", num_partitions=256, num_sub_vectors=96)

        self.table.create_fts_index("text", replace=True)

        results = {}
        if query_type == "all":
            # Evaluate all query types with progress
            for qt in [QueryType.VECTOR, QueryType.FTS, QueryType.RERANK_VECTOR, QueryType.RERANK_FTS, QueryType.HYBRID]:
                logger.info(f"Evaluating query type: {qt}")
                if self.reranker is None and qt in [QueryType.RERANK_VECTOR, QueryType.RERANK_FTS, QueryType.HYBRID]:
                    logger.warning(f"Reranker is not provided. Skipping query type: {qt}")
                    continue
                results[qt] = self.evaluate_query_type(top_k=top_k, query_type=qt)
                logger.info(f"Hit rate for {qt}: {results[qt]}")
            return RetriverResult(**results)
            
        if query_type == "auto":
            query_type = deduce_query_type(query_type, self.reranker)
        
        logger.info(f"Evaluating query type: {query_type}")
        results[query_type] = self.evaluate_query_type(top_k=top_k, query_type=query_type)

        return RetriverResult(**results)
