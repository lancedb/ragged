import json
import streamlit as st
import streamlit.components.v1 as components
from ragged.dataset import LlamaIndexDataset, SquadDataset, CSVDataset
from ragged.metrics.retriever import HitRate, QueryType
from ragged.results import RetriverResult
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

def metric_options():
    return {
        "HitRate": HitRate
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

def safe_import_wandb():
    try:
        import wandb
        from wandb import __version__
        return wandb
    except ImportError:
        return None
    
def init_wandb(dataset: str, embed_model: str):
    wandb = safe_import_wandb()
    if wandb is None:
        st.error("Please install wandb to log metrics using `pip install wandb`")
        return
    run = wandb.init(project=f"ragged-vectordb", name=f"{dataset}-{embed_model}") if wandb.run is None else None

def eval_retrieval():
    st.title("Retrieval Evaluator Quickstart")
    st.write("For custom dataset and retriever evaluation, use the API")
    col1, col2 = st.columns(2)
    with col1:
        provider = st.selectbox("Select a provider", datasets_options().keys(), placeholder="Choose a provider")
    with col2:
        if provider == "CSV":
            # choose a csv file
            dataset = st.file_uploader("Upload a CSV file", type=["csv"])
        else:
            dataset = st.selectbox("Select a dataset", datasets_options()[provider], placeholder="Choose a dataset", disabled=provider is None)

    col1, col2 = st.columns(2)
    with col1:
        metric = st.selectbox("Select a metric", metric_options().keys(), placeholder="Choose a metric")
    with col2:
        top_k = st.number_input("Top K", value=5)
    
    col1, col2 = st.columns(2)
    with col1:
        embed_provider = st.selectbox("Select an embedding provider", embedding_provider_options().keys(), placeholder="Choose an embedding provider")
    with col2:
        embed_model = st.selectbox("Select an embedding model", embedding_provider_options()[embed_provider], placeholder="Choose an embedding model", disabled=embed_provider is None)

    col1, col2 = st.columns(2)
    with col1:
        reranker = st.selectbox("Select a reranker", reranker_options(), placeholder="Choose a reranker")
    with col2:
        kwargs = st.text_input("Reranker kwargs", value="{}")
    
    col1, col2 = st.columns(2)
    with col1:
        query_type = st.selectbox("Select a query type", [qt for qt in QueryType.__dict__.keys() if not qt.startswith("__")], placeholder="Choose a query type")
    with col2:
        log_wandb = st.checkbox("Log to WandB and plot in real-time", value=False)
        use_existing_table = st.checkbox("Use existing table", value=False)
        create_index = st.checkbox("Create index", value=False)

    
    eval_button = st.button("Evaluate")
    results = RetriverResult()
    if eval_button:
        dataset = dataset_provider_options()[provider](dataset)
        reranker_kwargs = json.loads(kwargs)
        reranker = reranker_options()[reranker](**reranker_kwargs) if reranker != "None" else None
        query_type = QueryType.__dict__[query_type]
        metric = metric_options()[metric](
            dataset,
            embedding_registry_id=embed_provider,
            embed_model_kwarg={"name": embed_model},
            reranker=reranker
        )

        results = metric.evaluate(top_k=top_k,
                                  query_type=query_type,
                                  create_index=create_index,
                                  use_existing_table=use_existing_table) 
        total_metrics = len(results.model_dump())
        cols = st.columns(total_metrics)
        for idx, (k,v) in enumerate(results.model_dump().items()):
            with cols[idx]:
                st.metric(label=k, value=v)
        
        if log_wandb:
            wandb = safe_import_wandb()
            if wandb is None:
                st.error("Please install wandb to log metrics using `pip install wandb`")
                return
            init_wandb(dataset, embed_model)
            wandb.log(results.model_dump())

    
    if log_wandb:
        st.title("Wandb Project Page")
        wandb = safe_import_wandb()
        if wandb is None:
            st.error("Please install wandb to log metrics using `pip install wandb`")
            return
        init_wandb(dataset, embed_model)
        project_url = wandb.run.get_project_url()
        st.markdown("""
        Visit the WandB project page to view the metrics in real-time.
        [WandB Project Page]({project_url})
        """)


if __name__ == "__main__":
    eval_retrieval()