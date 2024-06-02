import os
import llama_index
from ragged.dataset.base import Dataset
from ..base import BaseRAG
from lancedb.embeddings import get_registry
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.schema import Document
from .lancedb_vector_store import LanceDBVectorStore
from typing import List, Optional
from lancedb.embeddings import TextEmbeddingFunction


try:
    from llama_index.llms.openai import OpenAI
except ImportError:
    raise ImportError("Please install the llama_index package by running `pip install llama-index-llms-openai`")


LLMS = {
    "openai": OpenAI
}

class LlamaIndexEmbeddings(BaseEmbedding):

    model: Optional[TextEmbeddingFunction] = None

    def __init__(self, embed_model) -> None:
        super().__init__()
        self.model = embed_model
        

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self.model.generate_embeddings([query])[0]
        return embeddings

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self.model.generate_embeddings([text])[0]
        return embeddings

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.generate_embeddings(texts)
        return embeddings
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
    
    
class llamaIndexRAG(BaseRAG):
    def __init__(
                self, 
                dataset: Dataset,
                embedding_registry_id: str = "openai",
                llm: str = "openai", 
                embed_model_kwargs: dict = {},
                llm_kwargs: dict = {},
                uri: str = "~/ragged/rag/llama-index",
                mode: str = "overwrite",
                lancedb_args: dict = {}
                ) -> None:
        embed_model = get_registry().get(embedding_registry_id).create(**embed_model_kwargs)
        self.lancedb_embed_model = embed_model
        self.embed_model = LlamaIndexEmbeddings(embed_model)
        self.dataset = dataset
        self.llm = LLMS[llm](**llm_kwargs)
        self.uri = uri
        self.mode = mode
        self.lancedb_args = lancedb_args

        self.rag = self.init_rag()


    def init_rag(self):
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        docs = []
        for idx, context in enumerate(self.dataset.get_contexts()):
            docs.append(Document(id_=str(idx), text=context.text))
        vector_store = LanceDBVectorStore(uri=self.uri, mode="overwrite", reranker=self.lancedb_args.get("reranker", None), query_type=self.lancedb_args.get("query_type", "vector"))
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
        query_engine = index.as_query_engine()

        return query_engine

    def query(self, query):
        response = self.rag.query(query)
        return str(response) + "\n"
    
    def get_contexts(self, query):
        return self.rag.vector

    def get_rag(self):
        return self.rag