"""Module implementing the SentenceEncoder class.

This submodule is implemented by the GraphSummarizer class
in order to project variable length sentence representations
to fixed-size ones.
"""

from typing import Sequence, List, Dict, Tuple

import torch
import torch.nn.utils.rnn as trnn

from ourheroes.models.submodules.utils import pack, unpack


class SentenceEncoder(torch.nn.Module):
    convs: torch.nn.ModuleList
    inter_size: int
    BiLSTM: torch.nn.LSTM
    output_size: int

    """Encoder for the representation of sentences as used by the graph summarization module.
    
    Attributes:
        convs: The convolution modules list.
        inter_size: The size of the output of the convolution modules,
            thus the input of the BiLSTM module.
        BiLSTM: The Bidirectional LSTM.
        output_size: The dimension of the output embeddings.
    """

    def __init__(self, input_size: int = 300, kernel_output: int = 50,
                 kernel_sizes: Sequence[int] = tuple(range(2, 8)),
                 hidden_size: int = 256, num_layers: int = 2):
        """Initializes the module.
        
        Args:
            input_size: The size of the word embeddings.
            kernel_sizes: The sizes of the convolutions' kernels.
            hidden_size: The hidden size of the BiLSTM.
            num_layers: The number of layers of the BiLSTM.
        """
        super().__init__()
        self.inter_size = len(kernel_sizes) * kernel_output
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(
            input_size, kernel_output, kernel_size=(n,), padding=(n // 2)) for n in kernel_sizes])
        self.BiLSTM = torch.nn.LSTM(input_size=self.inter_size, hidden_size=hidden_size,
                                    num_layers=num_layers, bidirectional=True, batch_first=True)
        self.output_size = hidden_size * 2

    def convolve(self, values: List[torch.Tensor]) -> List[torch.Tensor]:
        """Performs the convolutions on the input tensors.
        
        Args:
            values: The sentences variable length representations.
        Returns:
            The result of the convolutions.
        """
        values = [torch.swapaxes(value, 0, 1).unsqueeze(0) for value in values]
        values = [[conv(value) for conv in self.convs] for value in values]
        values = [[torch.swapaxes(sub.squeeze(), 0, 1) for sub in value] for value in values]
        values = [trnn.pad_sequence(value) for value in values]
        values = [value.reshape(value.size(0), self.inter_size) for value in values]
        return values

    def forward(self, sentences: Dict[Tuple[int, int], torch.Tensor]) -> Dict[Tuple[int, int], torch.Tensor]:
        """Encodes the sentences.
        
        Encoding is performed lest projection as described in the reference paper.
        
        Args:
            sentences: The sentences to be encoded.
        Returns:
            The encoded sentences.
        """
        values = [sentences[i] for i in sorted(sentences.keys())]
        values = self.convolve(values)
        packed, dims = pack(values)
        packed, _ = self.BiLSTM(packed)
        output = unpack(packed, dims)
        return {key: sub for key, sub in zip(sorted(sentences.keys()), output)}

    def __call__(self, sentences: Dict[Tuple[int, int], torch.Tensor]) -> Dict[Tuple[int, int], torch.Tensor]:
        """Encodes the sentences, same as self.forward(sentences)
        
        Encoding is performed lest projection as described in the reference paper.
        
        Args:
            sentences: The sentences to be encoded.
        Returns:
            The encoded sentences.
        """
        return self.forward(sentences)
