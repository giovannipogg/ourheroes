"""Callable objects generating rouge evaluations and training targets.

The classes are designed to be callable for generalization purposes,
and are mainly implemented by datasets for generating the training targets.
"""

import statistics as st
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from rouge_score.rouge_scorer import RougeScorer

from ourheroes.data.scoring.utils import resection
from ourheroes.data.types import BaseScorer, Document, Summary, CRSentenceScores, CRSectionScores, Section


@dataclass
class MeanRougeScorer:
    rouge_types: List[str]
    metric: str
    scorer: RougeScorer

    """A utility class used for rouge evaluations.
    
    The class serves the purpose of generating targets used while training,
    which, as described in the reference paper, the mean of a given metric
    of different rouge flavors.-
    
    Attributes:
        rouge_types: The rouge types from which the average is taken.
        metric: The metric on which the average is taken.
        scorer: The rouge evaluator.
    """

    def score(self, target: str, prediction: str) -> float:
        """Evaluates the average rouge similarity between target and prediction.
        
        Args:
            target: The reference phrase or summary.
            prediction: The evaluated phrase or summary.
        Returns:
            The mean of the metric of all the rouge flavors as
            passed at the moment of initialization.
        """
        scores = self.scorer.score(target, prediction)
        scores = [scores[rouge_type] for rouge_type in self.rouge_types]
        scores = [getattr(score, self.metric) for score in scores]
        return st.mean(scores)

    def __call__(self, target: str, prediction: str) -> float:
        """Evaluates the average rouge similarity between target and prediction,
        equivalent to self.score(target, prediction).
        
        Args:
            target: The reference phrase or summary.
            prediction: The evaluated phrase or summary.
        Returns:
            The mean of the metric of all the rouge flavors as
            passed at the moment of initialization.
        """
        return self.score(target, prediction)


@dataclass
class CRScorer:
    sentence_scorer: BaseScorer
    section_scorer: BaseScorer

    """The class generating labels for training the content ranking module.
    
    Attributes:
        sentence_scorer: The sentence level evaluator.
        section_scorer: The section level evaluator.
    """

    def score(self, summary: Summary, document: Document) -> Tuple[CRSentenceScores, CRSectionScores]:
        """Generates labels for the content ranking module given the document and the golden summary.
        
        Args:
            summary: The document's golden summary.
            document: The original document.
        Returns:
            The target labels for training the content ranking module.
        """
        sections = [section for _, section in document]
        sentences = [sentence for section in sections for sentence in section]
        sections = [' '.join(section) for section in sections]
        summary = ' '.join(summary)
        sentence_scores = [self.sentence_scorer(summary, sentence) for sentence in sentences]
        sentence_scores = torch.tensor(sentence_scores)
        section_scores = [self.section_scorer(summary, section) for section in sections]
        section_scores = torch.tensor(section_scores)
        return sentence_scores, section_scores

    def __call__(self, summary: Summary, document: Document) -> Tuple[CRSentenceScores, CRSectionScores]:
        """Generates labels for the content ranking module given the document and the golden summary,
        equivalent to self.score(summary, document)
        
        Args:
            summary: The document's golden summary.
            document: The original document.
        Returns:
            The target labels for training the content ranking module.
        """
        return self.score(summary, document)


@dataclass
class GSScorer:
    scorer: BaseScorer
    select_n: int

    """The class generating labels for training the graph summarization module.
    
    The targets are binary labels which evaluate to 1 the top n document
    sentences most similar the each sentence in the golden summary. 
    
    Attributes:
        scorer: The base scorer performing the sentence level evaluations.
        select_n: The number of document sentences to label as positive (1)
            for each sentence in the summary.
    """

    def score(self, summary: Summary, sections: List[Section]) -> List[List[float]]:
        """Generates labels for the graph summarization module given the document's
        section and the golden summary.
        
        Args:
            summary: The document's golden summary.
            sections: The original document sections.
        Returns:
            The target labels for training the graph summarization module.
        """
        final_scores = np.zeros(sum(map(len, sections)))
        for reference in summary:
            scores = [self.scorer(reference, sentence) for section in sections for sentence in section]
            arg_sorted = np.argsort(scores)[::-1]
            final_scores[arg_sorted[:self.select_n]] = 1.
        final_scores = resection(final_scores, sections)
        return final_scores

    def __call__(self, summary: Summary, sections: List[Section]) -> List[List[float]]:
        """Generates labels for the graph summarization module given the document's
        section and the golden summary, equivalent to self.score(summary, sections).
        
        Args:
            summary: The document's golden summary.
            sections: The original document sections.
        Returns:
            The target labels for training the graph summarization module.
        """
        return self.score(summary, sections)
