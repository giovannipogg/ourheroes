import string
from dataclasses import dataclass
from typing import List

import itertools as it
import networkx as nx
import math as m
import torch

from ourtypes import Digest, Boundaries, Tokenized, Section
from nltk.corpus import stopwords

@dataclass
class BoundaryEncoder:
    dimension: int = 64

    def boundary(self, sentence: int, length: int) -> torch.Tensor:
        from_start, from_end = [], []
        dist_end = length - sentence
        sentence += 1
        for k in range(1, self.dimension // 2 + 1):
            from_start.append(m.sin(sentence / (10_000 ** (2 * k / self.dimension))))
            from_start.append(m.cos(sentence / (10_000 ** (2 * k / self.dimension))))
            from_end.append(m.sin(dist_end / (10_000 ** (2 * k / self.dimension))))
            from_end.append(m.cos(dist_end / (10_000 ** (2 * k / self.dimension))))
        output = from_start + from_end
        return torch.tensor(output)

    def boundaries(self, sections: List[Section], digest: Digest) -> Boundaries:
        lengths = {section: len(sections[section]) for section in digest.keys()}

        boundaries = {}
        for i, (section, sentences) in enumerate(digest.items()):
            for j, sentence in enumerate(sentences):
                boundaries[(i, j)] = self.boundary(sentence, lengths[section])
        return boundaries

    def __call__(self, sections: List[Section], digest: Digest) -> Boundaries:
        return self.boundaries(sections, digest)


def get_typed_nodes(G: nx.DiGraph, type: str, data: bool = False):
    if data:
        return {node: data for node, data in G.nodes(data=True) if data['type'] == type}
    return [node for node, data in G.nodes(data=True) if data['type'] == type]


def get_word_nodes(G: nx.DiGraph, data: bool = False):
    return get_typed_nodes(G, 'w', data)


def get_sentence_nodes(G: nx.DiGraph, data: bool = False):
    return get_typed_nodes(G, 's', data)


def get_section_nodes(G: nx.DiGraph, data: bool = False):
    return get_typed_nodes(G, 'S', data)


def w2s(G: nx.DiGraph):
    sentence_nodes = get_sentence_nodes(G, data=True)
    edges = []
    for sentence, data in sentence_nodes.items():
        for word in data['tokens']:
            edges.append((word, sentence, {'type': 'w2s'}))
    G.add_edges_from(edges)


def s2w(G: nx.DiGraph):
    sentence_nodes = get_sentence_nodes(G, data=True)
    edges = []
    for sentence, data in sentence_nodes.items():
        for word in data['tokens']:
            edges.append((sentence, word, {'type': 's2w'}))
    G.add_edges_from(edges)


def s2s(G: nx.DiGraph):
    sentence_nodes = get_sentence_nodes(G)
    edges = []
    for s1, s2 in it.product(sentence_nodes, sentence_nodes):
        if s1[0] == s2[0]:
            edges.append((s1, s2, {'type': 's2s'}))
    G.add_edges_from(edges)


def S2s(G: nx.DiGraph):
    sentence_nodes = get_sentence_nodes(G)
    section_nodes = get_section_nodes(G)
    edges = [(S, s, {'type': 'S2s'}) for S, s in it.product(section_nodes, sentence_nodes)]
    G.add_edges_from(edges)


def s2S(G: nx.DiGraph):
    sentence_nodes = get_sentence_nodes(G)
    edges = [(s, s[0], {'type': 's2S'}) for s in sentence_nodes]
    G.add_edges_from(edges)


def S2S(G: nx.DiGraph):
    section_nodes = get_section_nodes(G)
    edges = [(S1, S2, {'type': 'S2S'}) for S1, S2 in it.product(section_nodes, section_nodes)]
    G.add_edges_from(edges)


@dataclass
class GSGrapher:
    punct = string.punctuation
    stop_words = stopwords.words('english')
        
    def nodes(self, G: nx.DiGraph, sections: List[List[Tokenized]],
          tensors: List[List[List[torch.Tensor]]], boundaries: Boundaries):
        nodes = []
        for i, (section, section_tensors) in enumerate(zip(sections, tensors)):
            nodes.append((i, {'type': 'S'}))
            for j, (sentence, sentence_tensors) in enumerate(zip(section, section_tensors)):
                tmp = []
                for token, tensor in zip(sentence, sentence_tensors):
                    if token in self.punct or token in self.stop_words or\
                            torch.allclose(tensor, torch.zeros((1,))):
                        continue
                    nodes.append((token, {'type': 'w', 'value': tensor}))
                    tmp.append(token)
                nodes.append(((i, j), {'type': 's', 'tokens': tmp, 'boundary': boundaries[(i, j)]}))
        G.add_nodes_from(nodes)

    def graph(self, document: List[List[Tokenized]], tensors: List[List[List[torch.Tensor]]],
              boundaries: Boundaries) -> nx.DiGraph:
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
        return self.graph(document, tensors, boundaries)
