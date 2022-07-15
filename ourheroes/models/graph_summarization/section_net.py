"""Module implementing the section-level GAT pipeline.

The WordNet class is a submodule implemented by the GraphSummarizer class
in order to update section-level features.
"""

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from ourheroes.models.submodules.ffn import FFN
from ourheroes.models.submodules.fusion import Fusion


class SectionNet(torch.nn.Module):
    gat_s: GATConv
    gat_S: GATConv
    activation: Callable[[torch.Tensor], torch.Tensor]
    fusion: Fusion
    ffn: FFN
    dropout_p: Optional[float]

    """The Pipeline for updating section-level features.
    
    Attributes:
        gat_s: The sentence-to-section GAT.
        gat_S: The section-to-section GAT.
        activation: The activation function down stream of the GATs.
        fusion: The features fusion layer.
        ffn: The feed forward layer.
        dropout_p: The dropout probability between the last Fusion and the FFN.
    """

    def __init__(self, sentence_size: int = 640, section_size: int = 512, n_heads: int = 8,
                 activation: Callable[[torch.Tensor], torch.Tensor] = F.elu, dropout_p: Optional[float] = .1):
        """Initializes the module.

        Args:
            sentence_size: The size of the sentence embeddings.
            section_size: The size of the section embeddings.
            n_heads: The number of attention heads of the GAT.
            activation: The activation function downstream of the GAT.
            dropout_p: The dropout probability.
        """
        super().__init__()
        self.gat_s = GATConv((sentence_size, section_size), section_size, n_heads)
        self.gat_S = GATConv(section_size, section_size, n_heads)
        self.fusion = Fusion(n_heads * section_size)
        self.ffn = FFN(n_heads * section_size, section_size, dropout_p=dropout_p)
        self.activation = activation
        self.dropout_p = dropout_p

    def forward(self, HS: torch.Tensor, Hs: torch.Tensor, s2S: torch.Tensor, S2S: torch.Tensor) -> torch.Tensor:
        """Performs the section-level features update.

        Args:
            HS: The section embeddings.
            Hs: The sentence embeddings.
            s2S: The sentence-to-section edges.
            S2S: The section-to-section edges.
        Returns:
            The updated features.
        """
        Us = self.activation(self.gat_s((Hs, HS), s2S))
        US = self.activation(self.gat_S(HS, S2S))
        US = self.fusion(US, Us)
        if self.dropout_p is not None:
            US = F.dropout(US, self.dropout_p)
        HS = self.ffn(US, HS)
        return HS

    def __call__(self, HS: torch.Tensor, Hs: torch.Tensor, s2S: torch.Tensor, S2S: torch.Tensor) -> torch.Tensor:
        """Performs the section-level features update, equivalent to self.forward(HS, Hs, s2S, S2S).

        Args:
            HS: The section embeddings.
            Hs: The sentence embeddings.
            s2S: The sentence-to-section edges.
            S2S: The section-to-section edges.
        Returns:
            The updated features.
        """
        return self.forward(HS, Hs, s2S, S2S)
