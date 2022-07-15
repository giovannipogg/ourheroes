"""Module implementing the Fusion submodule.

The Fusion submodule is implemented by the graph summarization GAT networks
in order to update features from multiple sources.
"""

from collections import Callable

import torch


class Fusion(torch.nn.Module):
    linear: torch.nn.Linear
    activation: Callable[[torch.Tensor], torch.Tensor]

    """Performs features fusion as described in the reference paper.
    
    Attributes:
        linear: The linear layer.
        activation: The activation function.
    """

    def __init__(self, input_size: int):
        """Initializes the module.
        
        Args:
            input_size: The size of the expected input.
        """
        super().__init__()
        self.linear = torch.nn.Linear(2 * input_size, input_size)
        self.activation = torch.sigmoid

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Performs features fusion.
        
        Args:
            X: The (left) tensor of features.
            Y: The (right) tensor of features.
        Returns:
            The result of feature fusion.
        """
        XY = torch.cat([X, Y], dim=-1)
        Z = self.activation(self.linear(XY))
        return Z * X + (1 - Z) * Y

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Performs features fusion, equivalent to self.forward(X, Y).
        
        Args:
            X: The (left) tensor of features.
            Y: The (right) tensor of features.
        Returns:
            The result of feature fusion.
        """
        return self.forward(X, Y)
