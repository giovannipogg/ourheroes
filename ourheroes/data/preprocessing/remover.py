"""Module implementing the Remover class.

The Remover object removes unwanted n-grams (passed at time of
initialization) from a string.
"""

from dataclasses import dataclass
from typing import Sequence


@dataclass
class Remover:
    removes: Sequence[str]

    """A callable remover.
    
    Arguments:
        removes: the n-grams to be removed when called.
    """

    def remove(self, sentence: str):
        """Removes unwanted n-grams from sentence.
        
        Args:
            sentence: The sentence to remove from.
        Returns:
            The cleaned version of sentence.
        """
        for remove in self.removes:
            sentence = sentence.replace(remove, ' ')
        return sentence

    def __call__(self, sentence: str):
        """Removes unwanted n-grams from sentence,
        equivalent to self.remove(sentence).
        
        Args:
            sentence: The sentence to remove from.
        Returns:
            The cleaned version of sentence.
        """
        return self.remove(sentence)
