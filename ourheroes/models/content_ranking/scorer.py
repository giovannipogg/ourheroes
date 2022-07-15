""""Module implementing the Scorer class.

The scorer class performs relevance scoring within the context of
the content ranking module.
"""

from typing import List

import torch


class Scorer(torch.nn.Module):
    linear: torch.nn.Linear

    """Module performing sentence or section relevance scoring.
    
    Attributes:
        linear: The linear layer performing scoring.
    """

    def __init__(self, input_size: int):
        """Initializes the module.

        Args:
            input_size: The expected embedding dimension.
        """
        super().__init__()
        self.linear = torch.nn.Linear(in_features=input_size, out_features=1)

    def forward(self, sequence: List[torch.Tensor]) -> torch.Tensor:
        """Assigns relevance scores to the sentence or section sequence.

        Args:
            sequence: The sentence or section embeddings to be scored.
        Returns:
            The computed relevance scores.
        """
        sequence = torch.stack(sequence)
        output = self.linear(sequence)
        return output.squeeze()

    def __call__(self, sequence: List[torch.Tensor]) -> torch.Tensor:
        """Assigns relevance scores to the sentence or section sequence,
        equivalent to self.forward(sequence).

        Args:
            sequence: The sentence or section embeddings to be scored.
        Returns:
            The computed relevance scores.
        """
        return self.forward(sequence)
