"""Module implementing the Attention submodule.

The module reproduces the attention mechanism described
in the reference paper.
"""

import torch


class Attention(torch.nn.Module):
    linear: torch.nn.Linear
    u_att: torch.nn.Parameter
    softmax: torch.nn.Softmax

    """Class implementing the attention mechanism
    as describe in the reference paper.
    
    Attributes:
        linear: The linear layer.
        u_att: The learned attention query.
        softmax: The softmax layer.
    """

    def __init__(self, input_size: int):
        """Initializes the module.
        
        Args:
            input_size: The expected size of the input.
        """
        super().__init__()
        self.linear = torch.nn.Linear(in_features=input_size, out_features=input_size)
        self.u_att = torch.nn.Parameter(torch.zeros(input_size))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the attention mechanism.
        
        Args:
            x: The input.
        Returns:
            The fixed-length combination of the input with attention weighting.
        """
        u = self.linear(x)  # (N, input_size) -> (N, input_size)
        u = torch.tanh(u)
        u = torch.matmul(u, self.u_att)  # (N, input_size)  @ (input_size) -> (N,)
        alpha = self.softmax(u)
        output = alpha.unsqueeze(-1) * x  # (N, 1) * (N, input_size) -> (N, input_size)
        output = torch.sum(output, dim=0)  # (N, input_size) -> (input_size,)
        return output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the attention mechanism, equivalent to self.forward(x)
        
        Args:
            x: The input.
        Returns:
            The fixed-length combination of the input with attention weighting.
        """
        return self.forward(x)
