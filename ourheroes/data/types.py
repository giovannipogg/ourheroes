"""Module implementing type aliases for enhanced readability.

The type aliases in this module are used for typing purposes
as well as for a gain in readability of the code.
"""

from typing import Dict, Any, Callable, List, Tuple, Union, TypeVar, Set

import os
import torch

# scorer types
Target = Prediction = str
BaseScorer = Callable[[Target, Prediction], float]

CRSentenceScores = torch.Tensor  # shape=(total_n_sentences,)
CRSectionScores = torch.Tensor  # shape=(n_sections,)

# dataset types
Data = Dict[str, Any]
FilePath = Union[str, bytes, os.PathLike]
BaseLoader = Callable[[FilePath], Data]

Title = Sentence = str
Summary = Section = List[Sentence]
Document = List[Tuple[Title, Section]]
Digest = Dict[int, List[int]]

Parsed = TypeVar('Parsed')
BaseParser = Callable[[Data], Parsed]
BaseSectionParser = Callable[[Data], List[Section]]
BaseTitleParser = Callable[[Data], List[Title]]
BaseDocumentParser = Callable[[Data], Document]
BaseSummaryParser = Callable[[Data], Summary]
BaseDigestParser = Callable[[Data], Digest]

BaseCRParser = Callable[[Data], Tuple[Document, Summary]]
BaseGSParser = Callable[[Data], Tuple[List[Section], Summary, Digest]]

TitleTensor = SentenceTensor = torch.Tensor  # shape=(n_words,)
SectionTensors = List[SentenceTensor]  # len=n_sentences
DocumentTensors = List[Tuple[TitleTensor, SectionTensors]]  # len=n_sections
CRScoreTensors = Tuple[CRSentenceScores, CRSectionScores]

# preprocessing types
Vocab = Union[Set[str], Dict[str, int]]
StringPreprocessor = Callable[[str], str]

Token = Union[int, str]
Tokenized = List[Token]
Tokenizer = Callable[[str], Tokenized]

TokenTensorizer = Callable[[Token], torch.Tensor]
SentenceTensorizer = Callable[[Tokenized], torch.Tensor]
Tensorizer = Union[TokenTensorizer, SentenceTensorizer]

Stage = Union[StringPreprocessor, Tokenizer, Tensorizer]
ProcessedString = Union[str, Tokenized, torch.Tensor]

Boundaries = Dict[Tuple[int, int], torch.Tensor]

# ourheroes types
TargetTensor = PredictionTensor = Union[torch.Tensor, Any]
Loss = torch.Tensor

Kwargs = Any
Criterion = Callable[[PredictionTensor, TargetTensor, Kwargs], Loss]
Devicer = Callable[[Any, torch.device], None]
Scheduler = Callable[[torch.optim.Optimizer, int, Kwargs], float]
