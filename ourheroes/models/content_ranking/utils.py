"""Module implementing utility functions for content ranking.
"""

from typing import List, Tuple

import torch

from ourheroes.data.types import DocumentTensors, TitleTensor, SentenceTensor


def extract(documents: List[DocumentTensors]) -> Tuple[List[TitleTensor], List[SentenceTensor], List[int]]:
    """Extracts flat lists for title and sentence tensors as well as the section lengths
    
    Args:
        documents: The documents for which to extract the features.

    Returns:
        The title and the sentence tensors in flat lists, and the section lengths.
    """
    titles = [title for document in documents for title, _ in document]
    sections = [section for document in documents for _, section in document]
    sentences = [sentence for section in sections for sentence in section]
    section_lengths = list(map(len, sections))
    return titles, sentences, section_lengths


def resection(sentences: List[torch.Tensor], lengths: List[int]) -> List[torch.Tensor]:
    """Regroups sentences by section.
    
    Args:
        sentences: Rearranges the sentence embeddings grouping them by section.
        lengths: The section lengths.

    Returns:
        The sections as variable-length tensors obtained by stacking the encoded sentences.
    """
    sections, previous = [], 0
    for length in lengths:
        current = previous + length
        section = torch.stack(sentences[previous:current])
        sections.append(section)
        previous = current
    return sections
