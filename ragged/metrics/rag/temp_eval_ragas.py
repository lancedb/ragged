from ragged.rag import llamaIndexRAG
from ragged.dataset.base import Dataset
from ragged.results import RAGResult
from ragas import evaluate
from llama_index.core.async_utils import run_async_tasks
from datasets import Dataset as HuggingFaceDataset

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.metrics.critique import harmfulness

metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall, # Needs ground truth answers
    harmfulness,
]

def eval_rag(
            dataset: Dataset, 
            rag = None,
            embedding_registry_id: str="openai",
            llm: str="openai",
            llm_kwargs: dict = {},
            embed_model_kwargs: dict = {}
            ):
    if rag is None:
        rag = llamaIndexRAG(
                            dataset=dataset,
                            embedding_registry_id=embedding_registry_id,
                            llm=llm,
                            llm_kwargs=llm_kwargs,
                            embed_model_kwargs=embed_model_kwargs
                            )
    questions = dataset.get_queries()
   

    responses = run_async_tasks([rag.get_rag().aquery(q) for q in questions])

    answers = []
    contexts = []
    for r in responses:
        answers.append(r.response)
        contexts.append([c.node.get_content() for c in r.source_nodes])
    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }

    if dataset.answer_column_name is not None:
        dataset_dict["ground_truths"] = [[a] for a in dataset.get_answers()]

    ds = HuggingFaceDataset.from_dict(dataset_dict)
    result = evaluate(ds, metrics)

    return RAGResult(**result)
