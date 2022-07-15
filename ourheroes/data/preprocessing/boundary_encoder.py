"""Module implementing the BoundaryEncoder class.

The callable BoundaryEncoder object generates positional encoding
for sentences within sections given a document.
"""

from dataclasses import dataclass
from typing import List

import torch
import math as m

from ourheroes.data.types import Section, Digest, Boundaries


@dataclass
class BoundaryEncoder:
    dimension: int = 64

    """Class for encoding the boundary distances of sentences
    within a section.
    
    The encoding is under all aspects analogous to positional
    encoding of tokens within sentences, although in this
    case it is encoding the position of sentences within sections.
    
    Attributes:
        dimension: The dimension of the encoding.
            Since the distances are computed both from the
            start and from the end of a section, the output
            dimension will be twice this value.
    """

    def boundary(self, sentence: int, length: int) -> torch.Tensor:
        """Computes the positional encoding of a sentence within a section.
        
        Args:
            sentence: The sentence position in the section.
            length: The total section length.
        Returns:
            The positional encoding of the sentence within the section.
        """
        from_start, from_end = [], []
        dist_end = length - sentence
        sentence += 1
        for k in range(1, self.dimension // 2 + 1):
            from_start.append(m.sin(sentence / (10_000 ** (2 * k / self.dimension))))
            from_start.append(m.cos(sentence / (10_000 ** (2 * k / self.dimension))))
            from_end.append(m.sin(dist_end / (10_000 ** (2 * k / self.dimension))))
            from_end.append(m.cos(dist_end / (10_000 ** (2 * k / self.dimension))))
        output = from_start + from_end
        return torch.tensor(output)

    def boundaries(self, sections: List[Section], digest: Digest) -> Boundaries:
        """Computes the positional encoding of sentences included in the digest.
        
        Args:
            sections: The literal of the document sections.
            digest: The digest produce by the trained content ranking module.
        Returns:
            The positional encodings of the sentences included in the digest
            within their respective sections.
        """
        lengths = {section: len(sections[section]) for section in digest.keys()}

        boundaries = {}
        for i, (section, sentences) in enumerate(digest.items()):
            for j, sentence in enumerate(sentences):
                boundaries[(i, j)] = self.boundary(sentence, lengths[section])
        return boundaries

    def __call__(self, sections: List[Section], digest: Digest) -> Boundaries:
        """Computes the positional encoding of sentences included in the digest,
        equivalent to self.boundaries(sections, digest).
        
        Args:
            sections: The literal of the document sections.
            digest: The digest produce by the trained content ranking module.
        Returns:
            The positional encodings of the sentences included in the digest
            within their respective sections.
        """
        return self.boundaries(sections, digest)
