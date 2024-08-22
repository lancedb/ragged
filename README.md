# Ragged

Simple utilities for piece-wise evaluation of LLM based chat and retrieval systems

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
<details open>
  <summary>Demo</summary>
  <img src="https://github.com/lancedb/ragged/assets/15766192/ab1313ef-04f5-461e-8429-28a6d0bdc13c" width=550 height=600 />

</details>

### Dataset Quality eval [Coming soon]

### End-to-End RAG eval [Coming soon]


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
### Evaluate across various query types and Rerankers
```
from ragged.dataset import CSVDataset, SquadDataset
from ragged.rag import llamaIndexRAG
from ragged.metrics.retriever.hit_rate import HitRate
from lancedb.rerankers import LinearCombinationReranker
from ragged.search_utils import QueryType
import wandb

dataset = SquadDataset()
reranker = LinearCombinationReranker()
hit_rate = HitRate(dataset, embedding_registry_id="sentence-transformers", embed_model_kwarg={"name": "tuned_model_4", "device": "cuda"})

query_types = [QueryType.VECTOR]
use_existing_table = False
for query_type in query_types:
    run = wandb.init(project="ragged_bench", name=f"Base_4")
    hr = hit_rate.evaluate(5, query_type=query_type, use_existing_table=use_existing_table)
    run.log({f"{query_type}": hr.model_dump()[f"{query_type}"]})
    use_existing_table = True

wandb.finish()
```

### Generate a custom semantic search dataset
Most of popular toy datasets are not semantically challenging enough to evaluate the performance of LLM based retrieval systems. Most of them work well with simple BM25 based retrieval systems. To generate a custom dataset, that is semantically challenging, you can use the following code snippet.
NOTE: `directory` can contain pdfs, txt files or any other file format that can be handled by Llama-index directory reader.
```python
from ragged.dataset.gen.gen_retrieval_data import gen_query_context_dataset
fragged.dataset.gen.llm_calls import OpenAIInferenceClient

clinet = OpenAIInferenceClient()
df = gen_query_context_dataset(directory="data/source_files", inference_client=clinet)

print(df.head())
# save the dataframe
df.to_csv("data.csv")
```

Now, you can evaluate this dataset using the `ragged --quickstart vectordb` GUI or via the API:
```python
from ragged.dataset.csv import CSVDataset
from ragged.metrics.retriever import HitRate
from lancedb.rerankers import CohereReranker

data = CSVDataset(path="data.csv")
reranker = CohereReranker()

hit_rate = HitRate(data, reranker=reranker, embedding_registry_id="openai", embed_model_kwarg={"model":"text-embedding-3-small"})
res = hit_rate.evaluate(top_k=5, query_type="all")
print(res)
```

### Dataset Quality eval [Coming soon]

### End-to-End RAG eval [Coming soon]

