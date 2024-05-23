from abc import ABC, abstractmethod
from ragged.dataset.base import Dataset

class BaseRAG(ABC):
    def __init__(self, dataset:Dataset, embedding_registry_id: str, embed_model_kwargs: str) -> None:
        pass

    @abstractmethod
    def init_rag(self):
        pass

    @abstractmethod
    def query(self, query:str):
        pass
    
    @abstractmethod
    def get_rag(self):
        pass