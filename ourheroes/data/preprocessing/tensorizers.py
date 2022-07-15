"""Module implementing the tensorizer classes.

This classes are used to map tokens and sequence of tokens
to their GloVe vector embeddings.
""" 

from dataclasses import dataclass
from typing import Optional

import torch
import torchtext

from ourheroes.data.types import Tokenized, Token


@dataclass
class TokenTensorizer:
    vectors: torchtext.vocab.GloVe

    """Callable wrapper class for token embedding.
    
    Attributes:
        vectors: The vectors from which the token embeddings
            are retrieved (although torchtext.vocab.GloVe is
            expected, any Mapping[Token, torch.Tensor] satisfies
            the implementation).
    """

    def tensorize(self, token: Token) -> torch.Tensor:
        """Returns the embedding of token.
        
        Args:
            token: The token to embed.
        Returns:
            The embedding of the token.
        """        
        return self.vectors[token]

    def __call__(self, token: Token) -> torch.Tensor:
        """Returns the embedding of token,
        equivalent to self.tensorize(token).
        
        Args:
            token: The token to embed.
        Returns:
            The embedding of the token.
        """        
        return self.tensorize(token)


@dataclass
class SentenceTensorizer:
    tensorizer: TokenTensorizer
    start_of_sentence: Optional[str] = None
    end_of_sentence: Optional[str] = None

    """Class for mapping a sentence from `Tokenized` to Tensor.
    
    Attributes:
        tensorizer: The implemented TokenTensorizer.
        start_of_sentence: the start-of-sentence token or None to disable.
        end_of_sentence: the end-of-sentence token or None to disable.
    """

    def tensorize(self, tokenized: Tokenized) -> torch.Tensor:
        """Returns the embedding of the tokenized sentence.
        
        Args:
            tokenized: The tokenized sentence to embed.
        Returns:
            The sequence of tokens as a pytorch Tensor
                with shape
                (number_of_tokens, embedding_dimension).
        """    
        tensors = []
        if self.start_of_sentence is not None:
            tensors += [self.tensorizer(self.start_of_sentence)]
        tensors += [self.tensorizer(token) for token in tokenized]
        if self.end_of_sentence is not None:
            tensors += [self.tensorizer(self.end_of_sentence)]
        return torch.stack(tensors)

    def __call__(self, tokenized: Tokenized) -> torch.Tensor:
        """Returns the embedding of the tokenized sentence,
        equivalent to self.tensorize(tokenized).
        
        Args:
            tokenized: The tokenized sentence to embed.
        Returns:
            The sequence of tokens as a pytorch Tensor
                with shape
                (number_of_tokens, embedding_dimension).
        """    
        return self.tensorize(tokenized)
