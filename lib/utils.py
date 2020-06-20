import json
import pandas as pd
from pathlib  import Path

def file_exists(x):
    return Path(x).is_file()

def read_results_json(results_path):
    if file_exists(results_path):
        return pd.read_json(results_path)
    else:
        raise FileNotFoundError("Results file not found.")
