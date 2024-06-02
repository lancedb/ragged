from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List
import pandas as pd

class TextNode(BaseModel):
    id: str
    text: str

class Dataset(ABC):
    @abstractmethod
    def to_pandas(self)->pd.DataFrame:
        pass
    
    @abstractmethod
    def get_contexts(self)->List[TextNode]:
        pass
    
    @abstractmethod
    def get_queries(self)->List[str]:
        pass
    
    def get_ground_truths(self)->List[str]:
        """
        Optional to implement
        """
        return None
    
    @staticmethod
    def available_datasets():
        """
        List of available datasets that can be loaded
        """
        return []
    
    @property
    @abstractmethod
    def context_column_name(self):
        pass

    @property
    @abstractmethod
    def query_column_name(self):
        pass

    @property
    @abstractmethod
    def answer_column_name(self):
        pass