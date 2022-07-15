"""Module implementing the Recursor class.

The Recursor class applies the preprocessing routine given
at initialization and applies it recursively preserving the
structure of the Sequence to preprocess.
"""

from dataclasses import dataclass
from typing import Union, Sequence

from ourheroes.data.preprocessing.pipeline import PreprocessingPipeline
from ourheroes.data.types import Stage, ProcessedString


@dataclass
class Recursor:
    process: Union[Stage, PreprocessingPipeline]

    """A utility class for preprocessing nested sequences.
    
    This class is useful for preprocessing nested sequences
    as it applies the process it is initialized with to the
    (inner most) string levels while reconstructing the
    structure retaining the types.
    
    Attributes:
        process: The routine applied to strings.
    """

    def recurse(self, sequence: Union[Sequence, str]) -> Union[Sequence, ProcessedString]:
        """Returns the sequence with processed string.
        
        Args:
            sequence: The sequence or string to process.
        Returns:
            The same type(s) and nesting structure with
                process strings.
        """
        if isinstance(sequence, str):
            return self.process(sequence)
        return type(sequence)([self.recurse(sub) for sub in sequence])

    def __call__(self, sequence: Union[Sequence, str]) -> Union[Sequence, ProcessedString]:
        """Returns the sequence with processed string,
        equivalent to self.recurse(sequence).
        
        Args:
            sequence: The sequence or string to process.
        Returns:
            The same type(s) and nesting structure with
                process strings.
        """
        return self.recurse(sequence)
