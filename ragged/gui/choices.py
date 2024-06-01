from ragged.dataset import LlamaIndexDataset, SquadDataset, CSVDataset
from lancedb.rerankers import CohereReranker, ColbertReranker, CrossEncoderReranker


def dataset_provider_options():
    return {
        "Llama-Index": LlamaIndexDataset,
        "Squad": SquadDataset,
        "CSV": CSVDataset
    }

def datasets_options():
    return {
        "Llama-Index": LlamaIndexDataset.available_datasets(),
        "Squad": SquadDataset.available_datasets(),
        "CSV": CSVDataset.available_datasets()
    }


def reranker_options():
    return {
        "None": None,
        "CohereReranker": CohereReranker,
        "ColbertReranker": ColbertReranker,
        "CrossEncoderReranker": CrossEncoderReranker
    }

def embedding_provider_options():
    return {
        "openai": ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"],
        "huggingface": ["BAAI/bge-small-en-v1.5", "BAAI/bge-large-en-v1.5"],
        "sentence-transformers": ["all-MiniLM-L12-v2", "all-MiniLM-L6-v2", "all-MiniLM-L12-v1", "BAAI/bge-small-en-v1.5", "BAAI/bge-large-en-v1.5"],
    }
