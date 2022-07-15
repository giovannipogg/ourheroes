"""Module implementing the content ranking network.

The network is used to produce shorter document digests
which are used to contain the size of the graphs used
in later stages of the model, in particular during the
graph summarization phase.
"""

from typing import Tuple, List

import torch.nn

from ourheroes.models.content_ranking.scorer import Scorer
from ourheroes.models.content_ranking.utils import extract, resection
from ourheroes.models.submodules.encoder import Encoder
from ourheroes.data.types import DocumentTensors, TitleTensor, SentenceTensor
from ourheroes.models.submodules.attention import Attention


class ContentRanker(torch.nn.Module):
    sentence_encoder: Encoder
    sentence_scorer: Scorer
    title_encoder: Encoder
    section_encoder: Encoder
    titled_attention: Attention
    section_scorer: Scorer

    """The class performing content ranking.
    
    Following the description given in the reference paper,
    this module assigns a relevance score to both sections
    and sentences. This is used to perform a first selection
    in order to produce shorter documents called digests
    which are then fed to the graph summarization module
    thanks to the reduced size of the associated graph. 
    
    Attributes:
        sentence_encoder: The sentence encoding module.
        sentence_scorer: The sentence scoring module.
        title_encoder: The title encoding module.
        section_encoder: The section encoding module.
        titled_attention: The attention module processing
            the coupled title and section embeddings.
        section_scorer: The section scoring module.
    """

    def __init__(self, input_size: int = 300, hidden_size: int = 512):
        """Initializes the network.

        Args:
            input_size: The size of the word embeddings.
            hidden_size: The size of the hidden state of the
                BiLSTM.
        """
        super().__init__()
        self.sentence_encoder = Encoder(input_size, hidden_size)
        self.title_encoder = Encoder(input_size, hidden_size)
        self.section_encoder = Encoder(2 * hidden_size, hidden_size)
        self.sentence_scorer = Scorer(2 * hidden_size)
        self.section_scorer = Scorer(2 * hidden_size)
        self.titled_attention = Attention(2 * hidden_size)

    def sentence_scoring(self, sentences: List[SentenceTensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Computes the relevance scores of the sentences.

        Args:
            sentences: The sentences to score.

        Returns:
            The relevance scores of each sentence.
        """
        encoded_sentences = self.sentence_encoder(sentences)
        sentence_scores = self.sentence_scorer(encoded_sentences)
        return encoded_sentences, sentence_scores

    def section_scoring(self, titles: List[TitleTensor], sections: List[torch.Tensor]) -> torch.Tensor:
        """Computes the relevance scores of the sections.

        Args:
            titles: The titles of the sections.
            sections: The sections to be scored.

        Returns:
            The relevance scores of each section.
        """
        encoded_titles = self.title_encoder(titles)
        encoded_sections = self.section_encoder(sections)
        encoded_sections = [torch.stack([title, section]) for title, section in zip(encoded_titles, encoded_sections)]
        encoded_sections = [self.titled_attention(section) for section in encoded_sections]
        return self.section_scorer(encoded_sections)

    def forward(self, documents: List[DocumentTensors]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the relevance scores of the sentences and sections in the documents.

        Args:
            documents: The documents to score.

        Returns:
            The relevance scores of the documents' sentences and sections.
        """
        titles, sentences, section_lengths = extract(documents)
        encoded_sentences, sentence_scores = self.sentence_scoring(sentences)
        sections = resection(encoded_sentences, section_lengths)
        section_scores = self.section_scoring(titles, sections)
        return sentence_scores, section_scores

    def __call__(self, documents: List[DocumentTensors]) -> Tuple:
        """Computes the relevance scores of the sentences and sections in the documents,
        equivalent to self.forward(documents).

        Args:
            documents: The documents to score.

        Returns:
            The relevance scores of the documents' sentences and sections.
        """
        return self.forward(documents)
