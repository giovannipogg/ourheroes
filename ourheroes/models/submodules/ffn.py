"""Module implementing the FFN submodule.

The FFN submodule is as simple two-layer feed forward network
implemented by the graph summarization GAT networks.
"""

from typing import Optional

import torch
import torch.nn.functional as F


class FFN(torch.nn.Module):
    l1: torch.nn.Linear
    l2: torch.nn.Linear
    dropout_p: Optional[float]

    """A two layer feed forward network.
    
    Attributes:
        l1: The first linear layer.
        l2: The second linear layer.
        dropout: The probability of dropout between the two layers.
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 2048, dropout_p: Optional[float] = .1):
        """Initializes the module.
        
        Args:
            input_size: The expected input size.
            output_size: The size of the output.
            hidden_size: The size of the first (resp. second) layer output (resp. input).
            dropout_p: The probability of performing dropout between the two layers.
        """
        super().__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, output_size)
        self.dropout_p = dropout_p

    def forward(self, U: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Performs the operations described in the reference paper.
        
        Args:
            U: The features processed by a GAT.
            H: The features before processing by the GAT.
        Returns:
            The result of feed forward and addition.
        """
        U = self.l1(U)
        if self.dropout_p is not None:
            U = F.dropout(U, self.dropout_p)
        U = self.l2(U)
        return U + H

    def __call__(self, U: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Performs the operations described in the reference paper,
        equivalent to self.forward(U, H).
        
        Args:
            U: The features processed by a GAT.
            H: The features before processing by the GAT.
        Returns:
            The result of feed forward and addition.
        """
        return self.forward(U, H)
