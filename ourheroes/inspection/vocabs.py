"""Module implementing the vocabulary retrieval routines.
"""

import asyncio
import json
import os
from collections import Counter
from typing import List, Dict

import nltk
from tqdm import tqdm

from ourheroes.data.types import Tokenizer
from ourheroes.inspection.utils import save_json


async def get_file_vocab(vocab: Counter, file: str, tq: tqdm, encoding: str = 'utf-8',
                         sections: str = 'sections', summary: str = 'abstract_text',
                         tokenize: Tokenizer = nltk.tokenize.word_tokenize):
    """Updates the dataset vocabulary with file tokens.

    Args:
        vocab: The dataset vocabulary to be updated.
        file: The file used to update the vocabulary.
        tq: The tqdm progress bar manager.
        encoding: The encoding of the file.
        sections: The sections key in the file data.
        summary: The summary key in the file data.
        tokenize: The callable used for tokenizing.
    """
    with open(file, 'r', encoding=encoding) as fp:
        document = json.load(fp)
    vocab.update([token for section in document[sections]
                  for sentence in section
                  for token in tokenize(sentence)])
    vocab.update([token for sentence in document[summary]
                  for token in tokenize(sentence)])
    tq.update(1)


def load_saved_vocab(save_path: str) -> Dict[str, int]:
    """Loads a precomputed vocabulary.

    Args:
        save_path: The path at which the vocabulary is saved.

    Returns:
        The saved vocabulary.
    """
    with open(save_path, 'r') as fp:
        vocab = json.load(fp)
    return vocab


async def get_files_vocab(files: List[str], save_path: str = 'files\\dataset.vocab', encoding: str = 'utf-8',
                          sections: str = 'sections', summary: str = 'abstract_text',
                          tokenize: Tokenizer = nltk.tokenize.word_tokenize) -> Dict[str, int]:
    """Generates the dataset vocabulary.

    Args:
        files: The files in the dataset.
        save_path: The path to which the vocabulary is saved.
        encoding: The encoding of the dataset files.
        sections: The sections key in the file data.
        summary: The summary key in the file data.
        tokenize: The callable used for tokenizing.

    Returns:
        The dateset vocabulary.
    """
    if os.path.isfile(save_path):
        return load_saved_vocab(save_path)
    return await _get_files_vocab(files, save_path, encoding, sections, summary, tokenize)


async def _get_files_vocab(files: List[str], vocab_path: str = 'files\\dataset.vocab', encoding: str = 'utf-8',
                           sections: str = 'sections', summary: str = 'abstract_text',
                           tokenize: Tokenizer = nltk.tokenize.word_tokenize) -> Dict[str, int]:
    tq = tqdm(total=len(files))
    tq.set_description('computing vocab')
    vocab = Counter()
    await asyncio.gather(*[get_file_vocab(vocab, file, tq, encoding, sections, summary, tokenize) for file in files])
    save_json(vocab, vocab_path)
    tq.close()
    return vocab


def get_representable(glove_path: str = ".vector_cache\\glove.840B.300d.txt",
                      save_path: str = "files\\glove.vocab") -> Dict[str, int]:
    """Returns the vocabulary of tokens representable by GloVe.

    Args:
        glove_path: The GloVe cached vectors.
        save_path: The path to which the vocabulary is saved.

    Returns:
        The GloVe-representable vocabulary.
    """
    if os.path.isfile(save_path):
        return load_saved_vocab(save_path)
    glove = _get_representable(glove_path)
    save_json(glove, save_path)
    return glove


def _get_representable(glove_path: str = ".vector_cache\\glove.840B.300d.txt") -> Dict[str, int]:
    with open(glove_path, 'r', encoding='utf-8') as fp:
        representable = fp.readlines()
    representable = {line.split(' ')[0]: i for i, line in enumerate(representable)}
    return representable
