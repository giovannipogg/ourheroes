from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from ourtypes import CRSentenceScores, CRSectionScores, Data


@dataclass
class Digester:
    m_sections: int = 4
    n_sentences: int = 30
    sections: str = 'sections'

    def get_indexes(self, data: Data) -> Tuple[List[int], int]:
        cum_lengths, previous = [], 0
        for section in data[self.sections]:
            length = len(section)
            cum_lengths.append(previous + length)
            previous += length
        return [0] + cum_lengths, len(data[self.sections])

    def digest(self, sentence_scores: CRSentenceScores, section_scores: CRSectionScores, data: Data, section_start: int,
               sentence_start: int) -> Tuple[Dict[int, List[int]], int, int]:
        indexes, section_stop = self.get_indexes(data)
        section_stop += section_start
        selected_sections, _ = torch.sort(torch.argsort(section_scores[section_start:section_stop])[:self.m_sections])

        digest = {}
        for section in selected_sections:
            start, stop = indexes[section], indexes[section + 1]
            start += sentence_start
            stop += sentence_start
            selected_sentences, _ = torch.sort(torch.argsort(sentence_scores[start:stop])[:self.n_sentences])
            digest[int(section)] = list(map(int, selected_sentences))
        return digest, section_stop, sentence_start + indexes[-1]

    def digests(self, sentence_scores: CRSentenceScores, section_scores: CRSectionScores, data: List[Data]) -> List[Dict[
        int, List[int]]]:
        section_start = sentence_start = 0
        output = []
        for sub in data:
            digest, section_start, sentence_start = self.digest(
                sentence_scores, section_scores, sub, section_start, sentence_start)
            output.append(digest)
        return output

    def __call__(self, sentence_scores: CRSentenceScores, section_scores: CRSectionScores, data: List[Data]) -> List[Dict[
        int, List[int]]]:
        return self.digests(sentence_scores, section_scores, data)
