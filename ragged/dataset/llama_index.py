from typing import List, Optional

from ragged.dataset.base import TextNode
from .base import Dataset
import logging
import os
from pathlib import Path

try:
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.llama_dataset import LabelledRagDataset
    from llama_index.core.node_parser import SentenceSplitter

except ImportError:
    raise ImportError("Please install the llama_index package by running `pip install llama_index`")

class LlamaIndexDataset(Dataset):
    def __init__(self, dataset_name: Optional[str] = None, path: Optional[str] = None, context_column_name="reference_contexts", query_column_name="query" ):
        if path is None and dataset_name is None:
            raise ValueError("Either path or dataset_name must be provided")
        if path is not None and dataset_name is not None:
            raise ValueError("Only one of path or dataset_name must be provided")
        if dataset_name is not None:
            # download the dataset to home directory
            path = str(Path.home() / "ragged_datasets" / dataset_name)

            if os.path.exists(path):
                logging.info(f"Dataset already exists at {path}. Reusing")
            else:
                os.system(f"llamaindex-cli download-llamadataset {dataset_name} --download-dir {path}")
        if path is not None:
            try:
                rag_dataset = LabelledRagDataset.from_json(f"{path}/rag_dataset.json")
                documents = SimpleDirectoryReader(input_dir=f"{path}/source_files").load_data()
            except FileNotFoundError:
                raise FileNotFoundError("Please provide a valid path to the dataset or a valid\
                                         dataset name to download the dataset")
        
        self.dataset = rag_dataset

        parser = SentenceSplitter()
        nodes = parser.get_nodes_from_documents(documents)
        self.nodes = nodes
        self.documents = [TextNode(id=node.id_, text=node.text) for node in nodes]
        self.context_column = context_column_name
        self.query_column = query_column_name

    def to_pandas(self):
        return self.dataset.to_pandas()
    
    def get_contexts(self) -> List[TextNode]:
        return self.documents

    def get_queries(self) -> List[str]:
        return self.to_pandas()[self.query_column_name].tolist()
    
    @property
    def context_column_name(self):
        return self.context_column
    
    @property
    def query_column_name(self):
        return self.query_column
    
    @property
    def answer_column_name(self):
        return None
    
    @staticmethod
    def available_datasets():
        return [
                "Uber10KDataset2021",
                "MiniEsgBenchDataset",
                "OriginOfCovid19Dataset",
                "MiniTruthfulQADataset",
                "Llama2PaperDataset",
                "OriginOfCovid19Dataset",
                ]
    
