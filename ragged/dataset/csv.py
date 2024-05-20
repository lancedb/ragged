from .base import Dataset, TextNode
from typing import List
from datasets import load_dataset
import pandas as pd


class CSVDataset(Dataset):
    def __init__(self, path: str, context_column: str = "context", query_column: str = "query"):
        self.dataset = pd.read_csv(path)
        # get unique contexts from the train dataframe
        contexts = self.dataset[context_column].unique()
        self.documents = [TextNode(id=str(i), text=context) for i, context in enumerate(contexts)]


    def to_pandas(self):
        return self.dataset

    
    def get_contexts(self)->List[TextNode]:
        return self.documents
    
    @property
    def context_column_name(self):
        return "context"
    
    @property
    def query_column_name(self):
        return "query"

    @staticmethod
    def available_datasets():
        return []