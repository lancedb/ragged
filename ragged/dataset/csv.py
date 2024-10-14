from .base import Dataset, TextNode
from typing import List
from datasets import load_dataset
import pandas as pd


class CSVDataset(Dataset):
    def __init__(self, path: str, context_column: str = "context", query_column: str = "query", answer_column: str = None):
        self.dataset = pd.read_csv(path)
        # get unique contexts from the train dataframe
        contexts = self.dataset[context_column].unique()

        docs = []
        for i, context in enumerate(contexts):
            if isinstance(context, list):
                docs.extend([TextNode(id=f"{i}_{j}", text=c) for j, c in enumerate(context)])
            else:
                docs.extend([TextNode(id=str(i), text=context)])
        self.documents = docs
            
        self.context_column = context_column
        self.query_column = query_column
        self.answer_column = answer_column
    
    def to_pandas(self):
        return self.dataset

    def get_contexts(self)->List[TextNode]:
        return self.documents
    
    def get_queries(self) -> List[str]:
        return self.dataset[self.query_column_name].tolist()
    
    def get_answers(self) -> List[str]:
        return self.dataset[self.answer_column_name].tolist()
    
    @property
    def context_column_name(self):
        return self.context_column
    
    @property
    def query_column_name(self):
        return self.query_column
    
    @property
    def answer_column_name(self):
        return self.answer_column

    @staticmethod
    def available_datasets():
        return []
