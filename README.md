# Ragged

Simple utilities for piece-wise evaluation of LLM based chat and retrieval system.

### Setup
Build from source
```
pip install -e .
```

## GUI quickstart 
### VectorDB retrieval eval
```
ragged --quickstart vectordb
```

## API Usage
### VectorDB retrieval eval
```python
from ragged.dataset import LlamaIndexDataset
from ragged.metrics.retriever import HitRate
from ragged.search_utils import QueryType
from lancedb.rerankers import CrossEncoderReranker

# 1. Select dataset
# Automatically download the dataset from llama-hub or pass existing path="/path/to/dataset"
dataset = LlamaIndexDataset("Uber10KDataset2021")

# 2. Select eval metrics
hit_rate = HitRate(
            dataset,
            embedding_registry_id="sentence-transformers",
            embed_model_kwarg={"name":"BAAI/bge-small-en-v1.5"},
            reranker=CohereReranker(),
            )

# 3. Evaluate on desired query types

#print(hit_rate.evaluate(top_k=5, query_type=QueryType.VECTOR)) # Evaluate vector search
print(hit_rate.evaluate(top_k=5, query_type="all")) # Evaliate all possible query types
```

## Create custom Dataset, Metrics, Reranking connectors
# TODO