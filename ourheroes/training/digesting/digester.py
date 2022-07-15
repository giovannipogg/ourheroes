"""Module implementing the Digester class.

The digester class is used to generate document digests
given a trained ContentRanker relevance scores.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from ourheroes.data.types import CRSentenceScores, CRSectionScores, Data


@dataclass
class Digester:
    n_sections: int
    n_sentences: int
    sections: str

    """Class that creates digest numerals given the content ranking relevance scores.    
    
    Attributes:
        n_sections: The maximum number of sections to be selected.
        n_sentences: The maximum number of sentences to be selected
            in each of the selected sections.
        sections: The sections key in data.
    """

    def get_indexes(self, data: Data) -> Tuple[List[int], int]:
        """Retrieves from a document the sentence indexes and the number of sections.

        Args:
            data: The document to retrieve the indexes from.

        Returns:
            The sentence indexes and the number of sections.
        """
        cum_lengths, previous = [], 0
        for section in data[self.sections]:
            length = len(section)
            cum_lengths.append(previous + length)
            previous += length
        return [0] + cum_lengths, len(data[self.sections])

    def digest(self, sentence_scores: CRSentenceScores, section_scores: CRSectionScores, data: Data, section_start: int,
               sentence_start: int) -> Tuple[Dict[int, List[int]], int, int]:
        """Creates the digest for a document given the content ranking relevance scores.

        Args:
            sentence_scores: The relevance scores of the sentences.
            section_scores: The relevance scores of the sections.
            data: The Data associated with the document to be digested.
            section_start: The index of the first document section in the flat section scores.
            sentence_start: The index of the first document sentence in the flat section scores.

        Returns:
            The digest of the input document, and the scores' indexes for the next document.
        """
        indexes, section_stop = self.get_indexes(data)
        section_stop += section_start
        selected_sections, _ = torch.sort(torch.argsort(section_scores[section_start:section_stop])[:self.n_sections])

        digest = {}
        for section in selected_sections:
            start, stop = indexes[section], indexes[section + 1]
            start += sentence_start
            stop += sentence_start
            selected_sentences, _ = torch.sort(torch.argsort(sentence_scores[start:stop])[:self.n_sentences])
            digest[int(section)] = list(map(int, selected_sentences))
        return digest, section_stop, sentence_start + indexes[-1]

    def digests(self, sentence_scores: CRSentenceScores, section_scores: CRSectionScores, data: List[Data]
                ) -> List[Dict[int, List[int]]]:
        """Creates the digests for documents given the content ranking relevance scores.

        Args:
            sentence_scores: The relevance scores of the sentences.
            section_scores: The relevance scores of the sections.
            data: The Data associated with the documents to be digested.

        Returns:
            The digests of the input documents.
        """
        section_start = sentence_start = 0
        output = []
        for sub in data:
            digest, section_start, sentence_start = self.digest(
                sentence_scores, section_scores, sub, section_start, sentence_start)
            output.append(digest)
        return output

    def __call__(self, sentence_scores: CRSentenceScores, section_scores: CRSectionScores, data: List[Data]
                 ) -> List[Dict[int, List[int]]]:
        """Creates the digests for documents given the content ranking relevance scores,
        equivalent to self.digests(sentence_scores, section_scores, data).

        Args:
            sentence_scores: The relevance scores of the sentences.
            section_scores: The relevance scores of the sections.
            data: The Data associated with the documents to be digested.

        Returns:
            The digests of the input documents.
        """
        return self.digests(sentence_scores, section_scores, data)
