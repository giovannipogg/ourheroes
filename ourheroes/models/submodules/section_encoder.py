"""Module implementing the SectionEncoder class.

This submodule is implemented by the GraphSummarizer class
in order to project variable length section representations
to fixed-size ones.
"""

from typing import Dict, Tuple

import torch

from ourheroes.models.submodules.attention import Attention
from ourheroes.models.submodules.utils import pack, unpack, get_section_tensors


class SectionEncoder(torch.nn.Module):
    attention: Attention
    BiLSTM: torch.nn.LSTM
    output_size: int

    """Performs section encoding for purposes of graph summarization.
    
    Attributes:
        attention: The Attention submodule.
        BiLSTM: The Bidirectional LSTM module.
        output_size: The dimension of the output embeddings.
    """

    def __init__(self, input_size: int = 512, hidden_size: int = 256, num_layers: int = 2):
        """Initializes the encoder.
        
        Args:
            input_size: The size of the expected input.
            hidden_size: The size of the hidden_state of the BiLSTM.
            num_layers: The number of layers of the BiLSTM module.
        """
        super().__init__()
        self.attention = Attention(input_size)
        self.BiLSTM = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                    num_layers=num_layers, bidirectional=True, batch_first=True)
        self.output_size = hidden_size * 2

    def forward(self, sentences: Dict[Tuple[int, int], torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Performs section encoding.
        
        Args:
            sentences: The dictionary of sentence tensors.
        Returns:
            The dictionary of encoded sentence tensors.
        """
        # x = self.attention(x)
        sentences = {key: self.attention(sentence) for key, sentence in sentences.items()}
        sections = get_section_tensors(sentences)
        values = [sections[key] for key in sorted(sections.keys())]
        packed, dims = pack(values)
        packed, _ = self.BiLSTM(packed)
        unpacked = unpack(packed, dims)
        return {section: sub for section, sub in zip(sorted(sections.keys()), unpacked)}

    def __call__(self, sentences: Dict[Tuple[int, int], torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Performs section encoding, equivalent to self.forward(sentences).
        
        Args:
            sentences: The dictionary of sentence tensors.
        Returns:
            The dictionary of encoded sentence tensors.
        """
        return self.forward(sentences)
