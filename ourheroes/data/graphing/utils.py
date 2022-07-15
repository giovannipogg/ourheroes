"""Module implementing utility functions for document graphs creation.
"""

import itertools as it
from typing import Union, Dict, List

import networkx as nx


def get_typed_nodes(G: nx.DiGraph, nodes_type: str, data: bool = False) -> Union[Dict, List]:
    """Returns the nodes of graph `G` that are of type `nodes_type`.

    Args:
        G: The graph from which to retrieve the nodes.
        nodes_type: The type of nodes to retrieve (`type` attribute of the nodes' data).
        data: Whether or not to also return the  selected nodes' data.

    Returns:
        A dictionary (if `data=True`) with nodes as keys and their respective data as values,
        or a list (if `data=False`) of nodes, where the nodes have the requested type.
    """
    if data:
        return {node: data for node, data in G.nodes(data=True) if data['type'] == nodes_type}
    return [node for node, data in G.nodes(data=True) if data['type'] == nodes_type]


def get_word_nodes(G: nx.DiGraph, data: bool = False) -> Union[Dict, List]:
    """Returns the word nodes of graph `G`.

    Args:
        G: The graph from which to retrieve the nodes.
        data: Whether or not to also return the  selected nodes' data.

    Returns:
        A dictionary (if `data=True`) with nodes as keys and their respective data as values,
        or a list (if `data=False`) of nodes, where the nodes are of type 'w'.
    """
    return get_typed_nodes(G, 'w', data)


def get_sentence_nodes(G: nx.DiGraph, data: bool = False) -> Union[Dict, List]:
    """Returns the sentence nodes of graph `G`.

    Args:
        G: The graph from which to retrieve the nodes.
        data: Whether or not to also return the  selected nodes' data.

    Returns:
        A dictionary (if `data=True`) with nodes as keys and their respective data as values,
        or a list (if `data=False`) of nodes, where the nodes are of type 's'.
    """
    return get_typed_nodes(G, 's', data)


def get_section_nodes(G: nx.DiGraph, data: bool = False) -> Union[Dict, List]:
    """Returns the section nodes of graph `G`.

    Args:
        G: The graph from which to retrieve the nodes.
        data: Whether or not to also return the  selected nodes' data.

    Returns:
        A dictionary (if `data=True`) with nodes as keys and their respective data as values,
        or a list (if `data=False`) of nodes, where the nodes are of type 'S'.
    """
    return get_typed_nodes(G, 'S', data)


def w2s(G: nx.DiGraph):
    """Adds to graph `G` word-to-sentence edges.

    As described in the reference paper, these kind of edge is present
    between a word and all the sentences in which it appears.

    Args:
        G: The graph for which the edges are to be added.
    """
    sentence_nodes = get_sentence_nodes(G, data=True)
    edges = []
    for sentence, data in sentence_nodes.items():
        for word in data['graphed_tokens']:
            edges.append((word, sentence, {'type': 'w2s'}))
    G.add_edges_from(edges)


def s2w(G: nx.DiGraph):
    """Adds to graph `G` sentence-to-word edges.

    As described in the reference paper, these kind of edge is present
    between a sentence and all the words which appear in it.

    Args:
        G: The graph for which the edges are to be added.
    """
    sentence_nodes = get_sentence_nodes(G, data=True)
    edges = []
    for sentence, data in sentence_nodes.items():
        for word in data['graphed_tokens']:
            edges.append((sentence, word, {'type': 's2w'}))
    G.add_edges_from(edges)


def s2s(G: nx.DiGraph):
    """Adds to graph `G` sentence-to-sentence edges.

    As described in the reference paper, these kind of edge is present
    between all sentences belonging to the same section.

    Args:
        G: The graph for which the edges are to be added.
    """
    sentence_nodes = get_sentence_nodes(G)
    edges = []
    for s1, s2 in it.product(sentence_nodes, sentence_nodes):
        if s1[0] == s2[0]:
            edges.append((s1, s2, {'type': 's2s'}))
    G.add_edges_from(edges)


def S2s(G: nx.DiGraph):
    """Adds to graph `G` section-to-sentence edges.

    As described in the reference paper, these kind of edge is present
    between all sentences and all sections.

    Args:
        G: The graph for which the edges are to be added.
    """
    sentence_nodes = get_sentence_nodes(G)
    section_nodes = get_section_nodes(G)
    edges = [(S, s, {'type': 'S2s'}) for S, s in it.product(section_nodes, sentence_nodes)]
    G.add_edges_from(edges)


def s2S(G: nx.DiGraph):
    """Adds to graph `G` sentence-to-sentence edges.

    As described in the reference paper, these kind of edge is present
    between all sentences and the section to which they belong.

    Args:
        G: The graph for which the edges are to be added.
    """
    sentence_nodes = get_sentence_nodes(G)
    edges = [(s, s[0], {'type': 's2S'}) for s in sentence_nodes]
    G.add_edges_from(edges)


def S2S(G: nx.DiGraph):
    """Adds to graph `G` section-to-section edges.

    As described in the reference paper, these kind of edge is present
    between all sections.

    Args:
        G: The graph for which the edges are to be added.
    """
    section_nodes = get_section_nodes(G)
    edges = [(S1, S2, {'type': 'S2S'}) for S1, S2 in it.product(section_nodes, section_nodes)]
    G.add_edges_from(edges)
