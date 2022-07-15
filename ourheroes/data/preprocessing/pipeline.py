"""Module implementing the PreprocessingPipeline class.

The PreprocessingPipeline pipeline is our custom class for
string preprocessing.
"""

from dataclasses import dataclass
from typing import Sequence

from ourheroes.data.types import ProcessedString


@dataclass
class PreprocessingPipeline:
    stages: Sequence

    """A preprocessing pipeline.
    
    Attributes:
        stages: The ordered sequence of stages to apply.
    """

    def preprocess(self, sentence: str) -> ProcessedString:
        """Orderly applies all stages to sentence.
        
        Args:
            sentence: the string to process.
        Returns:
            The processed sentence.
        """
        output = sentence
        for stage in self.stages:
            output = stage(output)
        return output

    def __call__(self, sentence: str) -> ProcessedString:
        """Orderly applies all stages to sentence,
        equivalent to self.preprocess(sentence).
        
        Args:
            sentence: the string to process.
        Returns:
            The processed sentence.
        """
        return self.preprocess(sentence)
