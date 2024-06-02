from .base import Dataset, TextNode
from typing import List
from datasets import load_dataset


class SquadDataset(Dataset):
    def __init__(self, dataset_name: str = "rajpurkar/squad"):
        self.dataset = load_dataset(dataset_name)
        # get unique contexts from the train dataframe
        contexts = self.dataset["train"].to_pandas()["context"].unique()
        self.documents = [TextNode(id=str(i), text=context) for i, context in enumerate(contexts)]


    def to_pandas(self):
        return self.dataset["train"].to_pandas()

    
    def get_contexts(self)->List[TextNode]:
        return self.documents
    
    def get_queries(self) -> List[str]:
        return self.dataset[self.query_column_name].tolist()
    
    @property
    def context_column_name(self):
        return "context"
    
    @property
    def query_column_name(self):
        return "question"
    
    @property
    def answer_column_name(self):
        return None

    @staticmethod
    def available_datasets():
        return ["rajpurkar/squad"]