"""Module implementing the evaluator classes.

The evaluator classes are used at time of evaluation
in order to compute the metrics given document(s) and target(s).
"""

from dataclasses import dataclass
from typing import Dict, Callable, Tuple, Sequence, List, Union

import torch

from ourheroes.data.scoring.factories import get_mean_rouge_scorer
from ourheroes.data.scoring.scorers import MeanRougeScorer
from ourheroes.data.types import Tokenized

BaseEvaluator = Callable[[List[str], torch.Tensor, List[List[Tokenized]]], Union[float, Dict[str, float]]]


@dataclass
class SingleEvaluator:
    score: MeanRougeScorer
    prepare_pred: Callable[[torch.Tensor, List[List[Tokenized]]], str]
    prepare_target: Callable[[List[str]], str]

    """Class computing a single evaluation metric.
    
    The class first processes the prediction and target to make
    them apt for scoring, then computes and return the evaluation result.    
    
    Attributes:
        score: The callable scorer used for computing the method.
        prepare_pred: The prediction preparation routine.
        prepare_target: The target preparation routine.
    """

    def evaluate(self, target: List[str], prediction: torch.Tensor, digest: List[List[Tokenized]]) -> float:
        """Evaluates prediction against target.

        Args:
            target: The target.
            prediction: The prediction.
            digest: The document digest literal.

        Returns:
            The evaluation score.
        """
        prediction = self.prepare_pred(prediction, digest)
        target = self.prepare_target(target)
        return self.score(target, prediction)

    def __call__(self, target: List[str], prediction: torch.Tensor, digest: List[List[Tokenized]]) -> float:
        return self.evaluate(target, prediction, digest)


@dataclass
class MultiEvaluator:
    evaluators: Sequence[Tuple[str, SingleEvaluator]]

    """Class computing a multiple evaluation metrics.   
    
    Attributes:
        evaluators: The single evaluators to be used, a sequence of tuples
            of evaluation type (<str>) and the corresponding SingleEvaluator.
    """

    def evaluate(self, target: List[str], prediction: torch.Tensor, digest: List[List[Tokenized]]) -> Dict[str, float]:
        """Evaluates prediction against target.

        Args:
            target: The target.
            prediction: The prediction.
            digest: The document digest literal.

        Returns:
            The evaluation scores as a dictionary with the evaluation types as keys.
        """
        output = {}
        for name, evaluator in self.evaluators:
            output[name] = evaluator(target, prediction, digest)
        return output

    def __call__(self, target: List[str], prediction: torch.Tensor, digest: List[List[Tokenized]]) -> Dict[str, float]:
        return self.evaluate(target, prediction, digest)


@dataclass
class MultiDocumentEvaluator:
    evaluator: BaseEvaluator

    def evaluate(self, targets: List[List[str]], prediction: torch.Tensor, digests: List[List[List[Tokenized]]]
                 ) -> List[Dict]:
        output = []
        end = start = 0
        for target, digest in zip(targets, digests):
            delta = sum([len(section) for section in digest])
            end += delta
            pred = prediction[start:end]
            output.append(self.evaluator(target, pred, digest))
            start = end
        return output

    def __call__(self, targets: List[List[str]], prediction: torch.Tensor, digests: List[List[List[Tokenized]]]
                 ) -> List[Dict]:
        return self.evaluate(targets, prediction, digests)


@dataclass
class Selector:
    max_words: int

    def select(self, prediction: torch.Tensor, digest: List[List[Tokenized]]) -> List[str]:
        document = [sentence for section in digest for sentence in section]
        total_words, n_sentences = 0, 0
        selected = torch.argsort(prediction, descending=True)
        for sentence in selected:
            total_words += len(document[sentence])
            n_sentences += 1
            if total_words >= self.max_words:
                break
        selected = torch.sort(selected[:n_sentences]).values
        return [' '.join(document[sentence]) for sentence in selected]

    def __call__(self, prediction: torch.Tensor, digest: List[List[Tokenized]]) -> List[str]:
        return self.select(prediction, digest)


def flat_join(target: List[str]) -> str:
    return ' '.join(target)


def newline_join(target: List[str]) -> str:
    return '\n'.join(target)


@dataclass
class Preparer:
    selector: Selector
    joiner: Callable[[List[str]], str]

    def prepare(self, prediction: torch.Tensor, digest: List[List[Tokenized]]):
        prediction = self.selector(prediction, digest)
        return self.joiner(prediction)

    def __call__(self, prediction: torch.Tensor, digest: List[List[Tokenized]]):
        return self.prepare(prediction, digest)


def get_base_evaluator(rouge: str, metric: str = 'fmeasure', max_words: int = 200) -> SingleEvaluator:
    strategy = {
        'rouge1': flat_join,
        'rouge2': flat_join,
        'rougeLsum': newline_join
    }
    strategy = strategy[rouge]
    scorer = get_mean_rouge_scorer(rouge, metric)
    return SingleEvaluator(scorer, Preparer(Selector(max_words), strategy), strategy)


def default_gs_evaluator() -> MultiDocumentEvaluator:
    return MultiDocumentEvaluator(MultiEvaluator(
        [('rouge1', get_base_evaluator('rouge1')),
         ('rouge2', get_base_evaluator('rouge2')),
         ('rougeL', get_base_evaluator('rougeLsum'))]
    ))
