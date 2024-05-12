import pytest
import pandas as pd

from ragged.dataset import LlamaIndexDataset
from ragged.metrics.retriever import HitRate
from ragged.search_utils import QueryType, QueryConfigError
from lancedb.rerankers import CrossEncoderReranker

from math import inf


def test_llamadataset():
    dataset = LlamaIndexDataset(dataset_name="PaulGrahamEssayDataset")
    assert dataset is not None
    assert dataset.to_pandas() is not None

def test_hitrate():
    dataset = LlamaIndexDataset(dataset_name="PaulGrahamEssayDataset")
    dataset.documents = dataset.documents[:5]
    # use subset of pandas df
    hit_rate = HitRate(
            dataset,
            embedding_registry_id="sentence-transformers",
            reranker=CrossEncoderReranker(),
            )

    res = hit_rate.evaluate(top_k=5, query_type=QueryType.ALL)
    assert res.fts is not -inf
    assert res.rerank_vector is not -inf
    assert res.rerank_fts is not -inf
    assert res.hybrid is not -inf
    assert res.vector is not -inf


    hit_rate = HitRate(
            dataset,
            embedding_registry_id="sentence-transformers",
            )
    
    assert hit_rate.evaluate(top_k=5, query_type=QueryType.VECTOR).vector is not -inf
    assert hit_rate.evaluate(top_k=5, query_type=QueryType.FTS).fts is not -inf

    with pytest.raises(QueryConfigError):
        hit_rate.evaluate(top_k=5, query_type=QueryType.RERANK_VECTOR)
        hit_rate.evaluate(top_k=5, query_type=QueryType.RERANK_FTS)
