from typing import List, Tuple

import torch

from models.submodules.attention import Attention
from ourtypes import SentenceTensor


def pack(sentences: List[SentenceTensor]) -> Tuple[torch.nn.utils.rnn.PackedSequence, List[int]]:
    dims = [sentence.size(0) for sentence in sentences]
    return torch.nn.utils.rnn.pack_sequence(sentences, enforce_sorted=False), dims


def unpack(packed: torch.nn.utils.rnn.PackedSequence, dims: List[int]) -> List[torch.Tensor]:
    output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
    return [sub[:dim] for sub, dim in zip(output, dims)]


class Encoder(torch.nn.Module):
    BiLSTM: torch.nn.LSTM
    attention: Attention

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.BiLSTM = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True)
        self.attention = Attention(2 * hidden_size)

    def forward(self, sentences: List[SentenceTensor]) -> List[torch.Tensor]:
        output, dims = pack(sentences)
        output, _ = self.BiLSTM(output)
        output = unpack(output, dims)
        return [self.attention(sub) for sub in output]

    def __call__(self, sentences: List[SentenceTensor]) -> List[torch.Tensor]:
        return self.forward(sentences)