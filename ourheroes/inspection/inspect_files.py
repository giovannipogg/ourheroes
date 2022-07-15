"""Module implementing the document inspection routines.
"""

import asyncio
import json
import os.path
from collections import Counter
from typing import Set, Dict, List, Any, Tuple

import nltk.corpus
from tqdm import tqdm

from ourheroes.data.types import Tokenizer
from ourheroes.inspection.utils import get_files, get_not_graphed, save_json, parse_dataset
from ourheroes.inspection.vocabs import get_files_vocab, get_representable


def vocab_representability(vocab: Dict[str, int], glove: Dict[str, int]):
    """Prints information about the dataset vocabulary `vocab` and the GloVe-representable vocabulary `glove`."""
    print(f"Total GloVe Representable Tokens            : {len(glove)}\n")
    print(f"Distinct Dataset Tokens (before lowercasing): {len(vocab)}")
    absolutely_representable = set(vocab.keys()).intersection(set(glove.keys()))
    print(f"GloVe Representable Tokens (absolute)       : {len(absolutely_representable)}")
    total_dataset_tokens = sum([val for val in vocab.values()])
    relatively_representable = sum([val for key, val in vocab.items() if key in glove])
    relatively_representable /= total_dataset_tokens
    print(f"GloVe Representable Tokens (relative)       : {relatively_representable * 100:.2f}%\n")

    lower_vocab = Counter({key.lower(): value for key, value in vocab.items()})
    print(f"Distinct Dataset Tokens (after  lowercasing): {len(lower_vocab)}")
    absolutely_representable = set(lower_vocab.keys()).intersection(set(glove.keys()))
    print(f"GloVe Representable Tokens (absolute)       : {len(absolutely_representable)}")
    total_dataset_tokens = sum([val for val in lower_vocab.values()])
    relatively_representable = sum([val for key, val in lower_vocab.items() if key in glove])
    relatively_representable /= total_dataset_tokens
    print(f"GloVe Representable Tokens (relative)       : {relatively_representable * 100:.2f}%")


async def inspect_file(file: str, representable: Set[str], not_graphed: Set[str],
                       tq: tqdm, encoding: str = 'utf-8',
                       sections: str = 'sections', summary: str = 'abstract_text',
                       tokenize: Tokenizer = nltk.tokenize.word_tokenize) -> Tuple[str, Dict[str, Any]]:
    """Generates statistics about `file`."""
    with open(file, 'r', encoding=encoding) as fp:
        document = json.load(fp)
    output = {
        'body': inspect_sections(document[sections], representable, not_graphed, tokenize),
        'summary': inspect_summary(document[summary], representable, not_graphed, tokenize)
    }
    tq.update(1)
    return file, output


def inspect_sentence(sentence: str, representable: Set[str], not_graphed: Set[str],
                     tokenize: Tokenizer = nltk.tokenize.word_tokenize) -> Tuple[int, int, int]:
    """Generates statistics about `sentence`."""
    sentence = tokenize(sentence)
    total = len(sentence)
    sentence = [token for token in sentence if token in representable]
    representable = len(sentence)
    sentence = [token for token in sentence if token not in not_graphed]
    graphable = len(sentence)
    return total, representable, graphable


def inspect_sections(sections: List[List[str]], representable: Set[str], not_graphed: Set[str],
                     tokenize: Tokenizer = nltk.tokenize.word_tokenize) -> List[List[Tuple[int, int, int]]]:
    """Generates statistics about `sections`."""
    output = [[inspect_sentence(sentence, representable, not_graphed, tokenize) for sentence in section]
              for section in sections]
    return output


def inspect_summary(summary: List[str], representable: Set[str], not_graphed: Set[str],
                    tokenize: Tokenizer = nltk.tokenize.word_tokenize) -> List[Tuple[int, int, int]]:
    """Generates statistics about `summary`."""
    output = [inspect_sentence(sentence, representable, not_graphed, tokenize)
              for sentence in summary]
    return output


async def inspect_files(files: List[str], representable: Set[str], not_graphed: Set[str],
                        save_path: str = 'files\\documents.stats',
                        encoding: str = 'utf-8', sections: str = 'sections', summary: str = 'abstract_text',
                        tokenize: Tokenizer = nltk.tokenize.word_tokenize):
    """Generates statistics about dataset `files`."""
    if os.path.isfile(save_path):
        return
    tq = tqdm(total=len(files))
    tq.set_description('inspecting files')
    statistics = await asyncio.gather(*[
        inspect_file(file, representable, not_graphed, tq, encoding, sections, summary, tokenize) for file in files
    ])
    tq.close()
    statistics = {file: stats for file, stats in statistics}
    save_json(statistics, save_path)


async def main(root: str = "pubmed-dataset"):
    """Performs the datasets inspection."""

    # Train
    if not os.path.isdir(f"{root}\\parsed"):
        parse_dataset(root)
    files = f"{root}\\parsed\\train"
    files = get_files(files)

    vocab = await get_files_vocab(files)
    glove = get_representable()
    vocab_representability(vocab, glove)

    representable = set(glove.keys())
    not_graphed = get_not_graphed()

    await inspect_files(files, representable, not_graphed)

    # Eval
    files = f"{root}\\parsed\\val"
    files = get_files(files)
    await inspect_files(files, representable, not_graphed, save_path='files\\val.stats')


if __name__ == '__main__':
    asyncio.run(main())
