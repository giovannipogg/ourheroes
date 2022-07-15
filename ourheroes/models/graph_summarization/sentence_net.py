"""Module implementing the sentence-level GAT pipeline.

The WordNet class is a submodule implemented by the GraphSummarizer class
in order to update sentence-level features.
"""

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from ourheroes.models.submodules.ffn import FFN
from ourheroes.models.submodules.fusion import Fusion


class SentenceNet(torch.nn.Module):
    gat_w: GATConv
    gat_s: GATConv
    gat_S: GATConv
    activation: Callable[[torch.Tensor], torch.Tensor]
    fusion1: Fusion
    fusion2: Fusion
    ffn: FFN
    dropout_p: Optional[float]

    """The Pipeline for updating sentence-level features.
    
    Attributes:
        gat_w: The word-to-sentence Graph Attention Network.
        gat_s: The sentence-to-sentence GAT.
        gat_S: The section-to-sentence GAT.
        activation: The activation function down stream of the GATs.
        fusion1: The first features fusion layer.
        fusion2: The second features fusion layer.
        ffn: The feed forward layer.
        dropout_p: The dropout probability between the last Fusion and the FFN.
    """

    def __init__(self, word_size: int = 300, sentence_size: int = 640, section_size: int = 512, n_heads: int = 8,
                 activation: Callable[[torch.Tensor], torch.Tensor] = F.elu, dropout_p: Optional[float] = .1):
        """Initializes the module.
        
        Args:
            word_size: The size of the word embeddings.
            sentence_size: The size of the sentence embeddings.
            section_size: The size of the section embeddings.
            n_heads: The number of attention heads of the GAT.
            activation: The activation function downstream of the GAT.
            dropout_p: The dropout probability.
        """
        super().__init__()
        self.gat_w = GATConv((word_size, sentence_size), sentence_size, n_heads)
        self.gat_s = GATConv(sentence_size, sentence_size, n_heads)
        self.gat_S = GATConv((section_size, sentence_size), sentence_size, n_heads)
        self.fusion1 = Fusion(n_heads * sentence_size)
        self.fusion2 = Fusion(n_heads * sentence_size)
        self.ffn = FFN(n_heads * sentence_size, sentence_size, dropout_p=dropout_p)
        self.activation = activation
        self.dropout_p = dropout_p

    def forward(self, Hs: torch.Tensor, Hw: torch.Tensor, HS: torch.Tensor,
                w2s: torch.Tensor, s2s: torch.Tensor, S2s: torch.Tensor) -> torch.Tensor:
        """Performs the sentence-level features update.
        
        Args:
            Hs: The sentence embeddings.
            Hw: The word embeddings.
            HS: The section embeddings.
            w2s: The word-to-sentence edges.
            s2s: The sentence-to-sentence edges.
            S2s: The section-to-sentence edges.
        Returns:
            The updated features.
        """
        Uw = self.activation(self.gat_w((Hw, Hs), w2s))
        Us = self.activation(self.gat_s(Hs, s2s))
        US = self.activation(self.gat_S((HS, Hs), S2s))
        U1 = self.fusion1(Uw, Us)
        U2 = self.fusion2(U1, US)
        if self.dropout_p is not None:
            U2 = F.dropout(U2, self.dropout_p)
        Hs = self.ffn(U2, Hs)
        return Hs

    def __call__(self, Hs: torch.Tensor, Hw: torch.Tensor, HS: torch.Tensor,
                 w2s: torch.Tensor, s2s: torch.Tensor, S2s: torch.Tensor) -> torch.Tensor:
        """Performs the sentence-level features update,
        equivalent to self.forward(Hs, Hw, HS, w2s, s2s, S2s)
        
        Args:
            Hs: The sentence embeddings.
            Hw: The word embeddings.
            HS: The section embeddings.
            w2s: The word-to-sentence edges.
            s2s: The sentence-to-sentence edges.
            S2s: The section-to-sentence edges.
        Returns:
            The updated features.
        """
        return self.forward(Hs, Hw, HS, w2s, s2s, S2s)
