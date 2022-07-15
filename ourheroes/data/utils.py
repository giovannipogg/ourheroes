"""Module implementing utility functions used by Parsers and Datasets."""

import json
from typing import List

from ourheroes.data.types import Data, Digest, FilePath, Section


def load_json(filepath: FilePath, encoding: str = 'utf-8') -> Data:
    """Loads a `json` file and returns a `Data` object.
    
    Args:
        filepath: The location of the file.
        encoding: The encoding of the file.
        
    Returns:
        The content of the file as a dictionary with `str` keys.
    """
    with open(filepath, 'r', encoding=encoding) as fp:
        data = json.load(fp)
    return data


def select(document: List[Section], digest: Digest) -> List[Section]:
    """Filters the literal document to obtain the literal digest.
    
    Args:
        document: The document literal, a `List[Section]=List[List[str]]`.
        digest: The digest in terms of selected sections' and sentences' numerals.
        
    Returns:
        The digest literal, of the same type as document.
    """
    output = []
    for section, sentences in digest.items():
        sub = []
        for sentence in sentences:
            sub.append(document[section][sentence])
        output.append(sub)
    return output
