from typing import Dict, Any, Callable, List, Tuple, Union

import torch

# scorer types
Target = Prediction = str
BaseScorer = Callable[[Target, Prediction], float]

CRSentenceScores = torch.Tensor  # shape=(total_n_sentences,)
CRSectionScores = torch.Tensor  # shape=(n_sections,)

# dataset types
Data = Dict[str, Any]
FilePath = str
BaseLoader = Callable[[FilePath], Data]

Title = Sentence = str
Summary = Section = List[Sentence]
Document = List[Tuple[Title, Section]]
Digest = Dict[int, List[int]]

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
StringPreprocessor = Callable[[str], str]

Token = Union[int, str]
Tokenized = List[Token]
Tokenizer = Callable[[str], Tokenized]

Tensorizer = Callable[[Tokenized], torch.Tensor]

Stage = Union[StringPreprocessor, Tokenizer, Tensorizer]
ProcessedString = Union[str, Tokenized, torch.Tensor]

Boundaries = Dict[Tuple[int, int], torch.Tensor]

# ourheroes types
Loss = Target = Prediction = torch.Tensor
Criterion = Callable[[Prediction, Target], Loss]
