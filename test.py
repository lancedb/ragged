from ragged.metrics.retriever import HitRate, QueryType
from ragged.dataset import LlamaIndexDataset
from ragged.search_utils import QueryType
from lancedb.rerankers import CohereReranker

# Automatically download the dataset from llama-hub or pass existing path="/path/to/dataset"
dataset = LlamaIndexDataset(dataset_name="PaulGrahamEssayDataset") 
hit_rate = HitRate(
            dataset,
            embedding_registry_id="openai",
            #embed_model_kwarg={"name":"BAAI/bge-small-en-v1.5"},
            reranker=CohereReranker(),
            )

#print(hit_rate.evaluate(top_k=5, query_type=QueryType.VECTOR)) # Evaluate vector search
print(hit_rate.evaluate(top_k=5, query_type="all")) # Evaliate all possible query types

