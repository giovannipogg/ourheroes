"""Module implementing the GSGrapher class for document graphs creation.
"""

from dataclasses import dataclass
from typing import List

import networkx as nx
import torch

from ourheroes.data.graphing.utils import w2s, s2w, s2s, S2s, s2S, S2S
from ourheroes.data.types import Tokenized, Boundaries, Vocab


@dataclass
class GSGrapher:
    not_graphed: Vocab

    """Utility class for generating the graph summarization source graph.
    
    Attributes:
        punctuation: The tokens to be considered punctuation.
        stop_words: The tokens considered stopwords.
    """

    def nodes(self, G: nx.DiGraph, sections: List[List[Tokenized]],
              tensors: List[List[List[torch.Tensor]]], boundaries: Boundaries):
        """Generates and adds nodes to G.
        
        Section, sentence and word level nodes are generated as described
        in the reference paper; projection of the sentence and section nodes
        features is performed by the model and punctuation and stop-word tokens
        are not included.
        
        Args:
            G: The graph representing the document.
            sections: The tokenized sections of the document.
            tensors: The embedding of the document's tokens.
            boundaries: The positional encoding of sentences within sections.
        """
        nodes = []
        for i, (section, section_tensors) in enumerate(zip(sections, tensors)):
            nodes.append((i, {'type': 'S'}))
            for j, (sentence, sentence_tensors) in enumerate(zip(section, section_tensors)):
                tokens = []
                for token, tensor in zip(sentence, sentence_tensors):
                    if token in self.not_graphed:
                        continue
                    nodes.append((token, {'type': 'w', 'value': tensor}))
                    tokens.append(token)
                nodes.append(
                    (
                        (i, j), {
                            'type': 's', 'graphed_tokens': tokens,
                            'value': torch.stack(sentence_tensors),
                            'boundary': boundaries[(i, j)]
                        }
                    )
                )
        G.add_nodes_from(nodes)

    def graph(self, document: List[List[Tokenized]], tensors: List[List[List[torch.Tensor]]],
              boundaries: Boundaries) -> nx.DiGraph:
        """Generates the graph for the summarization module given the document features.
        
        Args:
            document: The tokenized sections.
            tensors: The tokens' tensors.
            boundaries: The positional encoding of sentences within sections.
        Returns:
            The directed graph representing the document.
        """
        G = nx.DiGraph()
        self.nodes(G, document, tensors, boundaries)
        w2s(G)
        s2w(G)
        s2s(G)
        S2s(G)
        s2S(G)
        S2S(G)
        return G

    def __call__(self, document: List[List[Tokenized]], tensors: List[List[List[torch.Tensor]]],
                 boundaries: Boundaries) -> nx.DiGraph:
        """Generates the graph for the summarization module given the document features,
        equivalent to self.graph(document, tensor, boundaries).
        
        Args:
            document: The tokenized sections.
            tensors: The tokens' tensors.
            boundaries: The positional encoding of sentences within sections.
        Returns:
            The directed graph representing the document.
        """
        return self.graph(document, tensors, boundaries)
