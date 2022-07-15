"""Module implementing utility functions for the scores module.
"""

from typing import List

import numpy as np

from ourheroes.data.types import Section


def resection(scores: np.ndarray, sections: List[Section]) -> List[List[float]]:
    """Rearranges the scores to reflect the original document structure.
    
    Args:
        scores: The sentences' scores.
        sections: The document as originally structured.
    Returns:
        The scores rearranged as to reflect the original document structure.
    """
    output, cnt = [], 0
    for section in sections:
        inter = []
        for _ in section:
            inter.append(scores[cnt])
            cnt += 1
        output.append(inter)
    return output
