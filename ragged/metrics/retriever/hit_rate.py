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
logger = logging.getLogger("lancedb")
logger.setLevel(logging.INFO)

class HitRate(Metric):
    def evaluate_query_type(self,query_type:str, top_k:int = 5) -> float:
        if not self.table:
            self.ingest_docs()
            self.table.create_fts_index("text", replace=True)

        eval_results = []
        ds = self.dataset.to_pandas()
        for idx in tqdm.tqdm(range(len(ds))):
            query = ds[self.dataset.query_column_name][idx]
            reference_context = ds[self.dataset.context_column_name][idx]
            if not reference_context:
                logger.warning("reference_context is None for query: {idx}. \
                                Skipping this query. Please check your dataset.")
                continue
            try:
                rs = search_table(self.table, self.reranker, query_type, query, top_k)
            except Exception as e:
                if isinstance(e, QueryConfigError):
                    raise e
                logger.warn(f'Error with query: {idx} {e}')
                eval_results.append({
                    'is_hit': False,
                    'retrieved': [],
                    'expected': reference_context,
                    'query': query,
                })
                continue
            retrieved_texts = rs['text'].tolist()[:top_k]
            expected_text = reference_context[0] if isinstance(reference_context, list) else reference_context
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


    def ingest_docs(self, batched: bool = False, use_existing_table: bool = False):
        db = lancedb.connect(self.uri)
        embed_model = get_registry().get(self.embedding_registry_id).create(**self.embed_model_kwarg)

        class Schema(LanceModel):
            id: str
            text: str = embed_model.SourceField()
            vector: Vector(embed_model.ndims()) = embed_model.VectorField(default=None)
            
        if use_existing_table and "documents" in db.table_names():
            logger.info("Using existing table")
            self.table = db["documents"]
            return
        
        tbl = db.create_table("documents", schema=Schema, mode="overwrite")
        contexts = self.dataset.get_contexts()
        batch_size = len(contexts) if not batched else 1000
        num_batches = 1
        if batched:
            num_batches = (len(contexts) // batch_size) + 1 if len(contexts) % batch_size != 0 else 0
        
        logger.info(f"Adding {len(contexts)} documents to LanceDB, in {num_batches} batches of size {batch_size}")
        for i in range(num_batches):
            batch = contexts[i:i+batch_size]
            pydantic_batch = []
            for doc in tqdm.tqdm(batch, desc="Adding batch to LanceDB"):
                pydantic_batch.append(Schema(id=str(doc.id), text=doc.text))
            logger.info(f"Adding batch {i} to LanceDB")
            tbl.add(pydantic_batch)
        logger.info(f"created table with length {len(tbl)}")
        self.table = tbl

    

        
        

