"""Module implementing utility functions used by the `graph_summarization` package.
"""

from typing import Dict, Any, Tuple

import networkx as nx
import torch

from ourheroes.data.graphing.utils import get_typed_nodes, get_word_nodes


def data_to_tensor(data: Dict[Any, torch.Tensor]) -> torch.Tensor:
    """Creates the features matrix from a dictionary of features.

    Compatibility with other methods is ensured by sorting keys at insertion.

    Args:
        data: The dictionary to convert to matrix.

    Returns:
        The features matrix as a torch.Tensor.
    """
    output = []
    for key in sorted(data.keys()):
        output.append(data[key])
    return torch.stack(output)


def get_typed_edges(G: nx.DiGraph, from_: str, to: str, device: torch.device) -> torch.Tensor:
    """Creates the typed edges index for pytorch-geometric implementation of the GAT.

    Compatibility with other methods is ensured by sorting keys at insertion.

    Args:
        G: The graph for which to create the index.
        from_: The edges' tail-nodes' type.
        to: The edges' head-nodes' type.
        device: The device on which the index is created.

    Returns:
        The included typed edges index.
    """
    froms = get_typed_nodes(G, from_)
    froms = {key: i for i, key in enumerate(sorted(froms))}

    tos = get_typed_nodes(G, to)
    tos = {key: i for i, key in enumerate(sorted(tos))}

    edges = [(u, v) for u, v, data in G.edges(data=True) if data['type'] == f'{from_}2{to}']

    output = []
    for u, v in edges:
        output.append([froms[u], tos[v]])

    return torch.tensor(output, device=device).T


def prepare_words(G: nx.DiGraph) -> Dict[str, torch.Tensor]:
    """Creates a dictionary of tokens and the respective embeddings.

    Args:
        G: The graph from which to extract the word-nodes embeddings.

    Returns:
        A dictionary with tokens as key and their respective embeddings as values.
    """
    words = get_word_nodes(G, data=True)
    return {key: data['value'] for key, data in words.items()}


def get_edges(G: nx.DiGraph, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                            torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns all of the edges indexes.

    Args:
        G: The graph for which to create the indexes.
        device: The device on which the indexes are created.

    Returns:
        Being w words, s sentences, and S sections, it returns
        s2w, w2s, s2s, S2s, s2S, and S2S indexes in this order.
    """
    s2w = get_typed_edges(G, 's', 'w', device)
    w2s = get_typed_edges(G, 'w', 's', device)
    s2s = get_typed_edges(G, 's', 's', device)
    S2s = get_typed_edges(G, 'S', 's', device)
    s2S = get_typed_edges(G, 's', 'S', device)
    S2S = get_typed_edges(G, 'S', 'S', device)
    return s2w, w2s, s2s, S2s, s2S, S2S
