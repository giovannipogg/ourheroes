"""Module implementing the base Encoder submodule.

The Encoder submodule is used to encode variable length
sentence and section representations into fixed size ones.
"""

from typing import List

import torch

from ourheroes.models.submodules.attention import Attention
from ourheroes.models.submodules.utils import pack, unpack
from ourheroes.data.types import SentenceTensor


class Encoder(torch.nn.Module):
    BiLSTM: torch.nn.LSTM
    attention: Attention

    """Encoder used by the content ranking module.
    
    Attributes:
        BiLSTM: The Bidirectional LSTM module.
        Attention: The attention module.
    """

    def __init__(self, input_size: int, hidden_size: int):
        """Initializes the module.
        
        Args:
            input_size: The expected input_size.
            hidden_size: The size of the output and of the hidden state of
                the BiLSTM.
        """
        super().__init__()
        self.BiLSTM = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True)
        self.attention = Attention(2 * hidden_size)

    def forward(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Performs encoding.
        
        Args:
            tensors: The variable length tensors representing sentences or sections.
        Returns:
            The fixed-length representation of encoded sentences or sections.
        """
        output, dims = pack(tensors)
        output, _ = self.BiLSTM(output)
        output = unpack(output, dims)
        return [self.attention(sub) for sub in output]

    def __call__(self, tensors: List[SentenceTensor]) -> List[torch.Tensor]:
        """Performs encoding, equivalent to self.forward(sentences).
        
        Args:
            tensors: The variable length tensors representing sentences or sections.
        Returns:
            The fixed-length representation of encoded sentences or sections.
        """
        return self.forward(tensors)
