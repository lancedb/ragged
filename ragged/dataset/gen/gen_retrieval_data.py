try:
    import llama_index
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.node_parser import SentenceSplitter
except ImportError:
    raise ImportError("Please install the llama_index package by running `pip install llama_index`")

from tqdm import tqdm
import pandas as pd
from ragged.dataset.base import TextNode
from .prompts import Q_FROM_CONTEXT_DEFAULT, QA_FROM_CONTEXT_DEFAULT
from .llm_calls import HFInferenceClient, BaseInferenceClient

def gen_query_context_dataset(directory: str, 
                       inference_client: BaseInferenceClient,
                       num_questions_per_context: int = 2,
                       query_column: str = "query", 
                       context_column: str = "context"):
    """
    Generate query and contexts from a pandas dataframe
    """
    docs = SimpleDirectoryReader(input_dir=directory).load_data()
    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs)
    nodes = [TextNode(id=node.id_, text=node.text) for node in nodes]

    pylist = []
    for node in nodes:
        context = node.text
        queries = inference_client(Q_FROM_CONTEXT_DEFAULT.format(context=context, num_questions=num_questions_per_context))
        for query in queries:
            pylist.append({
                query_column: query,
                context_column: context
            })

    # create a dataframe
    df = pd.DataFrame(pylist)
    return df

def gen_QA_dataset(
    directory: str, 
    inference_client: BaseInferenceClient,
    num_questions_per_context: int = 2,
    query_column: str = "query", 
    context_column: str = "context",
    answer_column: str = "answer"
    ):
    """
    Generate QA dataset from a pandas dataframe
    """
    docs = SimpleDirectoryReader(input_dir=directory).load_data()
    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs)
    nodes = [TextNode(id=node.id_, text=node.text) for node in nodes]

    pylist = []
    for node in tqdm(nodes):
        context = node.text
        queries = inference_client(QA_FROM_CONTEXT_DEFAULT.format(context=context, num_questions=num_questions_per_context, question=query_column, answer=answer_column))
        for query in queries:
            pylist.append({
                query_column: query['query'],
                context_column: context,
                answer_column: query['answer'] 
            })

    df = pd.DataFrame(pylist)
    return df

