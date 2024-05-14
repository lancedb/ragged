import os
import argparse
from pathlib import Path


def cli():
    parser = argparse.ArgumentParser(description="CLI for running VectorDB quickstart")
    parser.add_argument("--quickstart", type=str, help="Name of the app")
    args = parser.parse_args()

    if args.quickstart == "vectordb":
        run_vectordb_quickstart_gui()
    else:
        raise ValueError(f"App {args.name} not found. Available apps: vectordb")


def run_vectordb_quickstart_gui():
    # get path of the parent directory
    parent_dir = Path(__file__).parent.parent
    # get path of the executable
    executable = os.path.join(parent_dir, "gui/vectordb.py")
    # run the executable
    os.system(f"streamlit run {executable}")