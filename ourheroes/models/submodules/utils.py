"""Module implementing utility functions used by the submodules.
"""

from collections import defaultdict
from typing import List, Tuple, Dict

import torch

from ourheroes.data.types import SentenceTensor


def pack(sentences: List[SentenceTensor]) -> Tuple[torch.nn.utils.rnn.PackedSequence, List[int]]:
    """Performs packing of variable length sequences.

    Args:
        sentences: The variable length tensors to pack.
    Returns:
        A pytorch PackedSequence along with the lengths of the tensors,
        useful for reversing the operation. 
    """
    dims = [sentence.size(0) for sentence in sentences]
    return torch.nn.utils.rnn.pack_sequence(sentences, enforce_sorted=False), dims


def unpack(packed: torch.nn.utils.rnn.PackedSequence, dims: List[int]) -> List[torch.Tensor]:
    """Performs the unpacking of a pytorch PackedSequence.

    Args:
        packed: The PackedSequence to unpack.
        dims: The original tensors' dimensions.
    Returns:
        The unpacked sequence. 
    """
    output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
    return [sub[:dim] for sub, dim in zip(output, dims)]


def get_sections_lists(sentences: Dict[Tuple[int, int], torch.Tensor]) -> Dict[int, List[torch.Tensor]]:
    """Maps sentence tensors to section tensors.

    Args:
        sentences: The dictionary of sentence tensors index at sentence level.
    Returns:
        A dictionary index by section with List[torch.Tensor] values.
    """
    output = defaultdict(lambda: list())
    for key in sorted(sentences.keys()):
        sentence = sentences[key]
        output[key[:-1]].append(sentence)
    return output


def get_section_tensors(sentences: Dict[Tuple[int, int], torch.Tensor]) -> Dict[int, torch.Tensor]:
    """Maps sentence tensors to section tensors.

    Args:
        sentences: The dictionary of sentence tensors index at sentence level.
    Returns:
        A dictionary index by section with torch.Tensor values.
    """
    output = {}
    temp = get_sections_lists(sentences)
    for section, values in temp.items():
        output[section] = torch.stack(values)
    return output
