"""Module implementing the word level GAT pipeline.

The WordNet class is a submodule implemented by the GraphSummarizer class
in order to update word-level features.
"""

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from ourheroes.models.submodules.ffn import FFN


class WordNet(torch.nn.Module):
    gat: GATConv
    activation: Callable[[torch.Tensor], torch.Tensor]
    ffn: FFN
    dropout_p: Optional[float]

    """The Pipeline for updating word-level features.
    
    Attributes:
        gat: The Graph Attention Network.
        activation: The activation function down stream of the GAT.
        ffn: The feed forward layer.
        dropout_p: The dropout probability between GAT and FFN.
    """

    def __init__(self, word_size: int = 300, sentence_size: int = 640, n_heads: int = 6,
                 activation: Callable[[torch.Tensor], torch.Tensor] = F.elu, dropout_p: Optional[float] = .1):
        """Initializes the module.
        
        Args:
            word_size: The size of the word embeddings.
            sentence_size: The size of the sentence embeddings.
            n_heads: The number of attention heads of the GAT.
            activation: The activation function down-stream of the GAT.
            dropout_p: The dropout probability.
        """
        super().__init__()
        self.gat = GATConv((sentence_size, word_size), word_size, n_heads)
        self.ffn = FFN(n_heads * word_size, word_size, dropout_p=dropout_p)
        self.activation = activation
        self.dropout_p = dropout_p

    def forward(self, Hw: torch.Tensor, Hs: torch.Tensor, s2w: torch.Tensor) -> torch.Tensor:
        """Performs the word-level features update.
        
        Args:
            Hw: The word embeddings.
            Hs: The sentence embeddings.
            s2w: The sentence-to-word edges.
        Returns:
            The updated features.
        """
        Uw = self.gat((Hs, Hw), s2w)
        Uw = self.activation(Uw)
        if self.dropout_p is not None:
            Uw = F.dropout(Uw, self.dropout_p)
        Hw = self.ffn(Uw, Hw)
        return Hw

    def __call__(self, Hw: torch.Tensor, Hs: torch.Tensor, s2w: torch.Tensor) -> torch.Tensor:
        """Performs the word-level features update, equivalent to self.forward(Hw, Hs, s2w)
        
        Args:
            Hw: The word embeddings.
            Hs: The sentence embeddings.
            s2w: The sentence-to-word edges.
        Returns:
            The updated features.
        """
        return self.forward(Hw, Hs, s2w)
