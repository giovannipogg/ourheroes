from typing import Tuple, List

import torch.nn

from models.submodules.encoder import Encoder
from ourtypes import DocumentTensors, TitleTensor, SentenceTensor
from models.submodules.attention import Attention


class Scorer(torch.nn.Module):
    linear: torch.nn.Linear

    def __init__(self, input_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=input_size, out_features=1)

    def forward(self, sentences: List[torch.Tensor]) -> torch.Tensor:
        sentences = torch.stack(sentences)
        output = self.linear(sentences)
        return output.squeeze()

    def __call__(self, sentences: List[SentenceTensor]) -> torch.Tensor:
        return self.forward(sentences)


def extract(documents: List[DocumentTensors]) -> Tuple[List[TitleTensor], List[int], List[SentenceTensor]]:
    titles = [title for document in documents for title, _ in document]
    sections = [section for document in documents for _, section in document]
    sentences = [sentence for section in sections for sentence in section]
    section_lengths = list(map(len, sections))
    return titles, section_lengths, sentences


def resection(sentences: List[torch.Tensor], lengths: List[int]) -> List[torch.Tensor]:
    output, previous = [], 0
    for length in lengths:
        current = previous + length
        section = torch.stack(sentences[previous:current])
        output.append(section)
        previous = current
    return output


class ContentRanker(torch.nn.Module):
    sentence_encoder: Encoder
    sentence_scorer: Scorer
    title_encoder: Encoder
    section_encoder: Encoder
    titled_attention: Attention
    section_scorer: Scorer

    def __init__(self, input_size: int = 300, hidden_size: int = 512):
        super().__init__()
        self.sentence_encoder = Encoder(input_size, hidden_size)
        self.title_encoder = Encoder(input_size, hidden_size)
        self.section_encoder = Encoder(2 * hidden_size, hidden_size)
        self.sentence_scorer = Scorer(2 * hidden_size)
        self.section_scorer = Scorer(2 * hidden_size)
        self.titled_attention = Attention(2 * hidden_size)

    def sentence_scoring(self, sentences: List[SentenceTensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        encoded_sentences = self.sentence_encoder(sentences)
        sentence_scores = self.sentence_scorer(encoded_sentences)
        return encoded_sentences, sentence_scores

    def section_scoring(self, titles: List[TitleTensor], sections: List[torch.Tensor]) -> torch.Tensor:
        encoded_titles = self.title_encoder(titles)
        encoded_sections = self.section_encoder(sections)
        encoded_sections = [torch.stack([title, section]) for title, section in zip(encoded_titles, encoded_sections)]
        encoded_sections = [self.titled_attention(section) for section in encoded_sections]
        return self.section_scorer(encoded_sections)

    def forward(self, documents: List[DocumentTensors]) -> Tuple[torch.Tensor, torch.Tensor]:
        titles, section_lengths, sentences = extract(documents)
        encoded_sentences, sentence_scores = self.sentence_scoring(sentences)
        sections = resection(encoded_sentences, section_lengths)
        section_scores = self.section_scoring(titles, sections)
        return sentence_scores, section_scores

    def __call__(self, documents: List[DocumentTensors]) -> Tuple:
        return self.forward(documents)
