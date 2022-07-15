"""Module implementing the document inspection utility functions.
"""

import json
import os
import string
from typing import Set, List, Dict

import nltk


def get_not_graphed() -> Set[str]:
    """Generates the default set of tokens not to be graphed."""
    not_graphed = set(nltk.corpus.stopwords.words('english'))
    not_graphed.update(string.punctuation, string.whitespace)
    return not_graphed


def get_files(directory: str, ext: str = '.txt') -> List[str]:
    """Gets all the files in `directory` having extension `ext`.

    Args:
        directory: The directory from which to extract the files.
        ext: The extension of the files to be included.

    Returns:
        The files in `directory` having extension `ext`.
    """
    files = os.listdir(directory)
    return [f'{directory}\\{file}' for file in files if file.endswith(ext)]


def save_json(dictionary: Dict[str, int], save_path: str):
    """Saves a `dictionary` as a json file at `save_path`."""
    with open(save_path, 'w') as fp:
        json.dump(dictionary, fp)
        fp.flush()


def parse_dataset(dataset_path: str) -> str:
    """Parses the dataset from single to multiple files."""
    parsed_dataset_path = f"{dataset_path}\\parsed"
    os.mkdir(parsed_dataset_path)

    for file in os.listdir(dataset_path):
        if file.endswith('.txt'):
            subdir = file.split('.')[0]
            subdir = f"{parsed_dataset_path}\\{subdir}"
            os.mkdir(subdir)
            with open(f"{dataset_path}\\{file}", 'r') as f:
                for line in f:
                    doc = json.loads(line)
                    file_name = doc['article_id']
                    with open(f"{subdir}\\{file_name}.txt", 'w') as outfile:
                        json.dump(doc, outfile)
    return parsed_dataset_path
