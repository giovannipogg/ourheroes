import statistics as st
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
from rouge_score.rouge_scorer import RougeScorer

from ourtypes import BaseScorer, Document, Summary, CRSentenceScores, CRSectionScores, Section


@dataclass
class MeanRougeScorer:
    rouge_types: List[str]
    metric: str
    scorer: RougeScorer

    def score(self, target: str, prediction: str) -> float:
        scores = self.scorer.score(target, prediction)
        scores = [scores[rouge_type] for rouge_type in self.rouge_types]
        scores = [getattr(score, self.metric) for score in scores]
        return st.mean(scores)

    def __call__(self, target: str, prediction: str) -> float:
        return self.score(target, prediction)

def get_mean_rouge_scorer(rouge_types: List[str], metric: str, use_stemmer: bool = False) -> MeanRougeScorer:
    scorer = RougeScorer(rouge_types, use_stemmer)
    return MeanRougeScorer(rouge_types, metric, scorer)


def default_sentence_scorer() -> MeanRougeScorer:
    return get_mean_rouge_scorer(['rouge1', 'rouge2', 'rougeL'], 'fmeasure')


def default_section_scorer() -> MeanRougeScorer:
    return get_mean_rouge_scorer(['rouge2'], 'recall')


@dataclass
class CRScorer:
    sentence_scorer: BaseScorer = default_sentence_scorer()
    section_scorer: BaseScorer = default_section_scorer()

    def score(self, document: Document, summary: Summary) -> Tuple[CRSentenceScores, CRSectionScores]:
        sections = [section for _, section in document]
        sentences = [sentence for section in sections for sentence in section]
        sections = [' '.join(section) for section in sections]
        summary = ' '.join(summary)
        sentence_scores = [self.sentence_scorer(summary, sentence) for sentence in sentences]
        sentence_scores = torch.tensor(sentence_scores)
        section_scores = [self.section_scorer(summary, section) for section in sections]
        section_scores = torch.tensor(section_scores)
        return sentence_scores, section_scores

    def __call__(self, document: Document, summary: Summary) -> Tuple[CRSentenceScores, CRSectionScores]:
        return self.score(document, summary)


@dataclass
class CRMultiScorer:
    scorer: CRScorer = CRScorer()

    def score(self, documents: List[Document], summaries: List[Summary]) -> Tuple[CRSentenceScores, CRSectionScores]:
        scores = [self.scorer(document, summary) for document, summary in zip(documents, summaries)]
        sentences_scores = [score[0] for score in scores]
        section_scores = [score[1] for score in scores]
        return torch.cat(sentences_scores), torch.cat(section_scores)

    def __call__(self, documents: List[Document], summaries: List[Summary]) -> Tuple[CRSentenceScores, CRSectionScores]:
        return self.score(documents, summaries)

def resection(scores: np.ndarray, sections: List[Section]) -> List[List[float]]:
    output, cnt = [], 0
    for section in sections:
        inter = []
        for _ in section:
            inter.append(scores[cnt])
            cnt += 1
        output.append(inter)
    return output

@dataclass
class GSScorer:
    scorer: BaseScorer = default_sentence_scorer()
    select_n: int = 2

    def score(self, sections: List[Section], summary: Summary) -> List[List[float]]:
        final_scores = np.zeros(sum(map(len, sections)))
        for reference in summary:
            scores = [self.scorer(reference, sentence) for section in sections for sentence in section]
            arg_sorted = np.argsort(scores)[::-1]
            final_scores[arg_sorted[:self.select_n]] = 1.
        final_scores = resection(final_scores, sections)
        return final_scores

    def __call__(self, document: Document, summary: Summary) -> List[List[float]]:
        return self.score(document, summary)


# def default_gs_scorer(device: torch.device) -> GSScorer:
#     return GSScorer(default_sentence_scorer(), 2)
