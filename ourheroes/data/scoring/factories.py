"""Module implementing factory methods of scorers.

The functions in this module return the default instance of
the scorer objects used for target label generation.
"""

from typing import List, Union

from rouge_score.rouge_scorer import RougeScorer

from ourheroes.data.scoring.scorers import MeanRougeScorer, CRScorer, GSScorer


def get_mean_rouge_scorer(
        rouge_types: Union[str, List[str]], metric: str, use_stemmer: bool = False) -> MeanRougeScorer:
    """Factory method for generating a MeanRougeScorer.
    
    Args:
        rouge_types: The types of rouge to be evaluated.
        metric: The metric of which the average is taken by the MeanRougeScorer.
        use_stemmer: Whether or not the implemented rouge scorer should use a stemmer.
    Returns:
        The corresponding MeanRougeScorer.
    """
    if isinstance(rouge_types, str):
        rouge_types = [rouge_types]
    scorer = RougeScorer(rouge_types, use_stemmer)
    return MeanRougeScorer(rouge_types, metric, scorer)


def default_sentence_scorer() -> MeanRougeScorer:
    """Factory method returning the default MeanRougeScorer for evaluating sentences."""
    return get_mean_rouge_scorer(['rouge1', 'rouge2', 'rougeL'], 'fmeasure')


def default_section_scorer() -> MeanRougeScorer:
    """Factory method returning the default MeanRougeScorer for evaluating sections."""
    return get_mean_rouge_scorer(['rouge2'], 'recall')


def default_cr_scorer() -> CRScorer:
    """Factory method returning the default CRScorer."""
    sentence_scorer = default_sentence_scorer()
    section_scorer = default_section_scorer()
    return CRScorer(sentence_scorer, section_scorer)


def default_gs_scorer() -> GSScorer:
    """Factory method returning the default GSScorer."""
    scorer = default_sentence_scorer()
    return GSScorer(scorer, 2)
