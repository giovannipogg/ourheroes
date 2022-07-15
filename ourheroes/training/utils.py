"""Module implementing the utility functions used for training.
"""

import os
from datetime import datetime
from typing import Any, Union, Sequence, Tuple, List, Dict, Optional

import networkx as nx
import torch
import torch.nn.functional as F
from torch.cuda import amp

from ourheroes.data.types import Loss, FilePath


def tensor_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Wrapper function for pytorch tensor to device.

    Args:
        tensor: The tensor to send to device.
        device: The device to which to send the tensor.

    Returns:
        The tensor on device.
    """
    return tensor.to(device)


def recursive_to_device(data: Union[torch.Tensor, Sequence], device: torch.device) -> Union[Sequence, torch.Tensor]:
    """Recursive function for sending (possibly nested) sequence(s) of pytorch tensor to device.

    Args:
        data: The sequence of tensors (possibly nested) to send to device.
        device: The device to which to send the tensors.

    Returns:
        The sequence(s) of tensors (of the same types and hierarchy) on device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return type(data)([recursive_to_device(sub, device) for sub in data])


def log(info: Any, log_path: str):
    """Logs info to file.

    Args:
        info: The info to log.
        log_path: The file path to log to.
    """
    with open(log_path, 'a') as fp:
        print(datetime.now(), info, file=fp)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, directory: str):
    """Saves a model and optimizer states checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        epoch: The current epoch.
        directory: The directory in which to save
    """
    torch.save({
        'epoch': epoch,
        'state': model.state_dict(),
        'optim': optimizer.state_dict()
    }, f"{directory}\\{epoch}.pth")


def get_checkpoints(directory: str) -> Any:
    """Retrieves the checkpoints in directory (if any).

    Args:
        directory: The directory from which to retrieve the checkpoints.

    Returns:
        All the checkpoints in the directory.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)
    files = os.listdir(directory)
    return [f'{directory}\\{file}' for file in files]


def load_checkpoint(checkpoint_path: FilePath, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None
                    ) -> int:
    """Loads a model and optimizer checkpoint.

    Args:
        checkpoint_path: The path of the checkpoint to be loaded.
        model: The model to initialize with the saved checkpoint.
        optimizer: The model to initialize with the saved checkpoint.

    Returns:
        The last epoch.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optim'])
    epoch = checkpoint['epoch']
    del checkpoint
    torch.cuda.empty_cache()
    return epoch


def load_last(directory: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> int:
    """Loads the most recent epoch checkpoint (if any).

    Args:
        directory: The directory in which the checkpoints are saved.
        model: The model to initialize with the last checkpoint.
        optimizer: The optimizer to initialize with the last checkpoint.

    Returns:
        The next epoch to be performed.
    """
    checkpoints = get_checkpoints(directory)
    if len(checkpoints) == 0:
        return -1
    return load_checkpoint(sorted(checkpoints)[-1], model, optimizer)


def zero_grad(model: torch.nn.Module):
    """Zeroes the gradient in the model.

    This method is equivalent to optimizer.zero_grad() but suggested by
    pytorch for speeding up the training process.

    Args:
        model: The model of which the gradient is to be zeroed.
    """
    for param in model.parameters():
        param.grad = None


def clip_gradient(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: amp.GradScaler, **kwargs):
    """Performs gradient clipping.

    Args:
        model: The model for which gradient clipping is performed.
        optimizer: The optimizer used to train the model.
        scaler: The scaler used while training the model.
        **kwargs: The keyword arguments to be used for pytorch's clip_grad_norm_ function.
    """
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), **kwargs)


def seed_torch(seed: int):
    """Seeds pytorch rngs.

    Args:
        seed: The seed to be used.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cr_loss(prediction: Tuple[torch.Tensor, torch.Tensor], target: Tuple[torch.Tensor, torch.Tensor], **kwargs
            ) -> Loss:
    """Computes the BCE with logits loss for training the content ranking module.

    Args:
        prediction: The predicted sentence and section relevance scores.
        target: The target sentence and section relevance scores.
        **kwargs: The keyword arguments used in pytorch's binary_cross_entropy_with_logits function.

    Returns:
        The total section and sentence loss.
    """
    sentence_scores, section_scores = prediction
    sentence_target, section_target = target
    sentence_loss = F.binary_cross_entropy_with_logits(sentence_scores, sentence_target, **kwargs)
    section_loss = F.binary_cross_entropy_with_logits(section_scores, section_target, **kwargs)
    return sentence_loss + section_loss


def cr_collate(batch: Sequence):
    """Collates a batch for training the content ranking module.

    Args:
        batch: The batch to collate.

    Returns:
        The collated batch.
    """
    data = [sub[0] for sub in batch]
    targets = [sub[1] for sub in batch]
    sentence_target = [t[0] for t in targets]
    sentence_target = torch.cat(sentence_target)
    section_target = [t[1] for t in targets]
    section_target = torch.cat(section_target)
    return data, (sentence_target, section_target)


def relabel_node(node: Union[int, Tuple[int, int], str], prefix: int
                 ) -> Union[Tuple[int, int], Tuple[int, int, int], Tuple[int, str]]:
    """Creates the new node identifier for creating a batched graph.

    Args:
        node: The node to relabel.
        prefix: The prefix for the node.

    Returns:
        A tuple with as first element the prefix and subsequent element(s)
        the original node identifier(s).
    """
    if isinstance(node, tuple):
        node = (prefix,) + node
    else:
        node = (prefix, node)
    return node


def relabel_nodes(nodes: Sequence[Tuple], prefix: int) -> List:
    """Relabels all nodes for creating a batched graph.

    Args:
        nodes: The nodes to relabel.
        prefix: The prefix for the nodes.

    Returns:
        The list of relabelled nodes and their data.
    """
    output = []
    for node, data in nodes:
        if 'tokens' in data:
            data['tokens'] = [(prefix, token) for token in data['tokens']]
        output.append((relabel_node(node, prefix), data))
    return output


def relabel_edges(edges: Dict, prefix: int) -> List:
    """Relabels all edges for creating a batched graph.

    Args:
        edges: The edges to relabel.
        prefix: The prefix for the edges' nodes.

    Returns:
        The relabelled edges and their data.
    """
    output = []
    for u, v, data in edges:
        output.append((relabel_node(u, prefix), relabel_node(v, prefix), data))
    return output


def join_graphs(graphs: List[nx.DiGraph]) -> nx.DiGraph:
    """Creates a single graph representing the batch.

    Args:
        graphs: The graphs to be included.

    Returns:
        A single graph representing a whole batch with as isolated components
        as there are input graphs.
    """
    out_graph = nx.DiGraph()
    for i, graph in enumerate(graphs):
        nodes = relabel_nodes(graph.nodes(data=True), i)
        edges = relabel_edges(graph.edges(data=True), i)
        out_graph.add_nodes_from(nodes)
        out_graph.add_edges_from(edges)
    return out_graph


def _gs_collate_train(batch: Sequence):
    """Collates a batch for training the graph summarization module.

    Args:
        batch: The batch to collate.

    Returns:
        The collated batch.
    """
    graphs = [graph for graph, _ in batch]
    targets = [target for _, target in batch]
    # summaries = [summary for _, _, summary, _ in batch]
    # digests = [digest for _, _, _, digest in batch]
    target = torch.cat(targets)
    graph = join_graphs(graphs)
    return graph, target  # , summaries, digests


def _gs_collate_eval(batch: Sequence):
    """Collates a batch for evaluating the graph summarization module.

    Args:
        batch: The batch to collate.

    Returns:
        The collated batch.
    """
    graphs = [graph for graph, _, _ in batch]
    targets = [target for _, target, _ in batch]
    data = [data for _, _, data in batch]
    graph = join_graphs(graphs)
    return graph, targets, data  # , summaries, digests


def gs_collate(batch: Sequence):
    """Collates a batch for the graph summarization module.

    Args:
        batch: The batch to collate.

    Returns:
        The collated batch.
    """
    if len(batch[0]) == 3:
        return _gs_collate_eval(batch)
    return _gs_collate_train(batch)


def attribute_to_device(G: nx.DiGraph, attribute: str, device: torch.device):
    """Sends nodes data to device.

    Args:
        G: The graph for which the data is to be sent to device.
        attribute: The attribute to send to device.
        device: The device to which the data is sent.
    """
    attributes = {}
    for node, data in G.nodes(data=True):
        if attribute not in data:
            continue
        attributes[node] = data[attribute].to(device)
    nx.set_node_attributes(G, attributes, name=attribute)


def graph_to_device(G: nx.DiGraph, device: torch.device):
    """Sends the data of the graph to device.

    Args:
        G: The graph for which to send data to device.
        device: The device to which the data is sent.

    Returns:
        The graph with data located on device.
    """
    attribute_to_device(G, 'value', device)
    attribute_to_device(G, 'boundary', device)
    return G
