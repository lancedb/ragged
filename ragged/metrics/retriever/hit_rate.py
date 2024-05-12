from .base import Metric
import lancedb
import logging
import pandas as pd
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
import tqdm
import sys

from ...search_utils import QueryType, deduce_query_type, search_table, QueryConfigError
from ...results import RetriverResult

# Set logging level to INFO
#logging.basicConfig(level=logging.INFO)

class HitRate(Metric):
    def evaluate_query_type(self,query_type:str, top_k:5) -> float:
        eval_results = []
        ds = self.dataset.to_pandas()
        for idx in tqdm.tqdm(range(len(ds))):
            query = ds['query'][idx]
            reference_context = ds['reference_contexts'][idx]
            if not reference_context:
                logging.warning("reference_context is None for query: {idx}. \
                                Skipping this query. Please check your dataset.")
                continue
            try:
                rs = search_table(self.table, self.reranker, query_type, query, top_k)
            except Exception as e:
                if isinstance(e, QueryConfigError):
                    raise e
                continue
            retrieved_texts = rs['text'].tolist()[:top_k]
            expected_text = reference_context[0]
            is_hit = False
            # HACK: to handle new line characters added my llamaindex doc reader
            if expected_text in retrieved_texts or expected_text+'\n' in retrieved_texts:
                is_hit = True
            eval_result = {
                'is_hit': is_hit,
                'retrieved': retrieved_texts,
                'expected': expected_text,
                'query': query,
            }
            eval_results.append(eval_result)
        
        result = pd.DataFrame(eval_results)
        hit_rate = result['is_hit'].mean()
        return hit_rate


    def ingest_docs(self):
        db = lancedb.connect(self.uri)
        embed_model = get_registry().get(self.embedding_registry_id).create(**self.embed_model_kwarg)

        class Schema(LanceModel):
            id: str
            text: str = embed_model.SourceField()
            vector: Vector(embed_model.ndims()) = embed_model.VectorField(default=None)
        
        tbl = db.create_table("documents", schema=Schema, mode="overwrite")
        batch_size = 1000
        num_batches = (len(self.dataset.documents) // batch_size) + 1 if len(self.dataset.documents) % batch_size != 0 else 0
        # tqdm
        logging.info(f"Adding {len(self.dataset.documents)} documents to LanceDB, in {num_batches} batches of size {batch_size}")
        for i in tqdm.tqdm(range(num_batches), desc="Adding documents to LanceDB"):
            batch = self.dataset.documents[i:i+batch_size]
            pydantic_batch = []
            for doc in tqdm.tqdm(batch, desc="Adding batch to LanceDB"):
                pydantic_batch.append(Schema(id=str(doc.id_), text=doc.text))
            tbl.add(pydantic_batch)
        self.table = tbl

    

        
        

