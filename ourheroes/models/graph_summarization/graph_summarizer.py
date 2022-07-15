"""Module implementing the graph summarization network.

The GraphSummarizer class is used to perform graph based
extractive summarization of documents given their digests.
"""

from typing import Tuple, Dict, Optional

import networkx as nx
import torch
import torch.nn.functional as F
import torch.types

from ourheroes.data.graphing.utils import get_sentence_nodes
from ourheroes.models.graph_summarization.section_net import SectionNet
from ourheroes.models.graph_summarization.sentence_net import SentenceNet
from ourheroes.models.graph_summarization.utils import prepare_words, data_to_tensor, get_edges
from ourheroes.models.graph_summarization.word_net import WordNet
from ourheroes.models.submodules.attention import Attention
from ourheroes.models.submodules.section_encoder import SectionEncoder
from ourheroes.models.submodules.sentence_encoder import SentenceEncoder


class GraphSummarizer(torch.nn.Module):
    sentence_encoder: SentenceEncoder
    section_encoder: SectionEncoder
    sentence_projector: Attention
    section_projector: Attention
    word_net: torch.nn.ModuleList
    sentence_net: torch.nn.ModuleList
    section_net: torch.nn.ModuleList
    n_iters: int
    recursive: bool
    dropout_p: Optional[float]
    linear: torch.nn.Linear

    """The model performing the graph summarization routine.
    
    This is the network producing the final summary, trained with
    the abstract of the papers as target. 
    Since in the reference paper not better defined
    projection matrices are said to be exploited,
    and finding ourselves at difficulty understanding how
    a fixed-size matrix was employed to project variable-
    length sequences, we opted to employ an Attention layers
    for performing such reduction for both sentences and sections.
        

    Attributes:
        sentence_encoder: The sentence encoding module.
        section_encoder: The section encoding module.
        sentence_projector: The sentence projecting module.
            The same problem as for projecting sentences was encountered.
        word_net: The word-nodes feature update module.
        sentence_net: sentence-nodes feature update module.
        section_net: section-nodes feature update module.
        n_iters: The number of iterative updates to perform.
        recursive: Whether or not the updating modules are shared
            between iterations (focus of our extension).
        dropout_p: The dropout probability before the final prediction.
        linear: The layer computing the logits of each phrase being included
            in the predicted summary.
    """

    def __init__(self, n_iters: int = 2, dropout_p: Optional[float] = .1, recursive: bool = True):
        """Initializes the network.

        Args:
            n_iters: The number of iterative updates to perform.
            recursive: Whether or not the updating modules are shared
                between iterations (focus of our extension).
            dropout_p: The dropout probability before the final prediction.
        """
        super().__init__()
        self.sentence_encoder = SentenceEncoder()
        self.section_encoder = SectionEncoder()
        self.sentence_projector = Attention(self.sentence_encoder.output_size)
        self.section_projector = Attention(self.section_encoder.output_size)
        if recursive:
            self.word_net = torch.nn.ModuleList([WordNet()] * (n_iters - 1))
            self.sentence_net = torch.nn.ModuleList([SentenceNet()] * n_iters)
            self.section_net = torch.nn.ModuleList([SectionNet()] * (n_iters - 1))
        else:
            self.word_net = torch.nn.ModuleList([WordNet() for _ in range(n_iters - 1)])
            self.sentence_net = torch.nn.ModuleList([SentenceNet() for _ in range(n_iters)])
            self.section_net = torch.nn.ModuleList([SectionNet() for _ in range(n_iters - 1)])
        self.n_iters = n_iters
        self.dropout_p = dropout_p
        self.linear = torch.nn.Linear(640, 1)

    def prepare_sentences(self, G: nx.DiGraph) -> Dict[Tuple[int, int], torch.Tensor]:
        """Encodes the sentences.

        Args:
            G: The graph from which the features are retrieved.

        Returns:
            The sentences' projected features as a dictionary with tuples (section_number, sentence_number)
            as keys and their respective fixed-size embeddings as values.
        """
        sentences = get_sentence_nodes(G, data=True)
        output = {}
        for sentence, data in sentences.items():
            output[sentence] = data['value']
        return self.sentence_encoder(output)

    def project_sentences(self, sentences: Dict[Tuple[int, int], torch.Tensor],
                          G: nx.DiGraph) -> Dict[Tuple[int, int], torch.Tensor]:
        """Projects the sentences' variable-length embeddings to fixed-size ones.

        Args:
            sentences: The sentences to be projected.
            G: The graph from which the features are retrieved.

        Returns:
            The sentences' projected features as a dictionary with tuples (section_number, sentence_number)
            as keys and their respective fixed-size embeddings as values.
        """
        sentences = {key: self.sentence_projector(sentence) for key, sentence in sentences.items()}
        data = get_sentence_nodes(G, data=True)
        sentences = {key: torch.cat([sentence, data[key]['boundary']]) for key, sentence in sentences.items()}
        return sentences

    def project_sections(self, sections: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Projects the sections' variable-length embeddings to fixed-size ones.

        Args:
            sections: The sections to be projected.

        Returns:
            The sections' projected features as a dictionary with the section numbers
            as keys and their respective fixed-size embeddings as values.
        """
        sections = {key: self.section_projector(sentence) for key, sentence in sections.items()}
        return sections

    def get_matrices(self, G: nx.DiGraph) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Creates the features matrices given the graph.

        Args:
            G: The graph from which the features are retrieved.

        Returns:
            The word, sentence and section features matrices in this order.
        """
        words = prepare_words(G)
        sentences = self.prepare_sentences(G)
        sections = self.section_encoder(sentences)

        sections = self.project_sections(sections)
        sentences = self.project_sentences(sentences, G)

        Hw = data_to_tensor(words)
        Hs = data_to_tensor(sentences)
        HS = data_to_tensor(sections)

        return Hw, Hs, HS

    def forward(self, G: nx.DiGraph):
        """Performs graph summarization with iterative features update exploiting Graph Attention Networks.

        Args:
            G: The graph representing the document(s) to summarize.
        Returns:
            The logits for each sentence of being included in the summary (or summaries).
        """
        Hw, Hs, HS = self.get_matrices(G)
        s2w, w2s, s2s, S2s, s2S, S2S = get_edges(G, Hw.device)

        for t in range(self.n_iters):
            Hs_ = self.sentence_net[t](Hs, Hw, HS, w2s, s2s, S2s)
            if t == self.n_iters - 1:
                Hs = Hs_
                break
            Hw_ = self.word_net[t](Hw, Hs, s2w)
            HS_ = self.section_net[t](HS, Hs, s2S, S2S)
            Hs, Hw, HS = Hs_, Hw_, HS_

        if self.dropout_p is not None:
            Hs = F.dropout(Hs, self.dropout_p)
        return self.linear(Hs).squeeze()

    def __call__(self, G: nx.DiGraph):
        """Performs graph summarization with iterative features update exploiting Graph Attention Networks,
        equivalent to self.forward(G)

        Args:
            G: The graph representing the document(s) to summarize.
        Returns:
            The logits for each sentence of being included in the summary (or summaries).
        """
        return self.forward(G)
