from collections import defaultdict
from typing import Sequence, Tuple, List, Dict, Optional, Any, Callable

import networkx as nx
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as trnn
import torch.types
from torch_geometric.nn import GATConv

from grapher import get_sentence_nodes, get_typed_nodes, get_word_nodes, get_section_nodes
from models.submodules.attention import Attention
from models.submodules.encoder import pack, unpack


def default_kernel_sizes() -> Tuple[int]:
    return tuple(range(2, 8))


class SentenceEncoder(torch.nn.Module):

    def __init__(self, input_size: int = 300, kernel_output: int = 50,
                 kernel_sizes: Sequence[int] = default_kernel_sizes(),
                 hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.inter_size = len(kernel_sizes) * kernel_output
        self.output_size = 2 * hidden_size
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(
            input_size, kernel_output, kernel_size=(n,), padding=(n // 2)) for n in kernel_sizes])
        self.BiLSTM = torch.nn.LSTM(input_size=self.inter_size, hidden_size=hidden_size,
                                    num_layers=num_layers, bidirectional=True, batch_first=True)

    def convolve(self, values: List[torch.Tensor]):
        values = [torch.swapaxes(value, 0, 1).unsqueeze(0) for value in values]
        values = [[conv(value) for conv in self.convs] for value in values]
        values = [[torch.swapaxes(sub.squeeze(), 0, 1) for sub in value] for value in values]
        values = [trnn.pad_sequence(value) for value in values]
        values = [value.reshape(value.size(0), self.inter_size) for value in values]
        return values

    def forward(self, sentences: Dict[Tuple[int, int], torch.Tensor]) -> Dict[Tuple[int, int], torch.Tensor]:
        values = [sentences[i] for i in sorted(sentences.keys())]
        values = self.convolve(values)
        packed, dims = pack(values)
        packed, _ = self.BiLSTM(packed)
        output = unpack(packed, dims)
        return {key: sub for key, sub in zip(sorted(sentences.keys()), output)}

    def __call__(self, sentences: Dict[Tuple[int, int], torch.Tensor]) -> Dict[Tuple[int, int], torch.Tensor]:
        return self.forward(sentences)


def section_tensors(sentences: Dict[Tuple[int, int], torch.Tensor]) -> Dict[int, torch.Tensor]:
    output = defaultdict(lambda: list())
    for key in sorted(sentences.keys()):
        sentence = sentences[key]
        output[key[:-1]].append(sentence)
    for section, values in output.items():
        output[section] = torch.stack(values)
    return output


class SectionEncoder(torch.nn.Module):
    def __init__(self, input_size: int = 512, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.output_size = 2 * hidden_size
        self.attention = Attention(input_size)
        self.BiLSTM = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                    num_layers=num_layers, bidirectional=True, batch_first=True)
        # self.attention2 = Attention(512)

    def forward(self, sentences: Dict[Tuple[int, int], torch.Tensor]) -> Dict[int, torch.Tensor]:
        # x = self.attention(x)
        sentences = {key: self.attention(sentence) for key, sentence in sentences.items()}
        sections = section_tensors(sentences)
        values = [sections[key] for key in sorted(sections.keys())]
        packed, dims = pack(values)
        packed, _ = self.BiLSTM(packed)
        unpacked = unpack(packed, dims)
        return {section: sub for section, sub in zip(sorted(sections.keys()), unpacked)}

    def __call__(self, sentences: Dict[Tuple[int, int], torch.Tensor]) -> Dict[int, torch.Tensor]:
        return self.forward(sentences)


class Fusion(torch.nn.Module):
    linear: torch.nn.Linear
    activation: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self, input_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(2 * input_size, input_size)
        self.activation = torch.sigmoid

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        XY = torch.cat([X, Y], dim=-1)
        Z = self.activation(self.linear(XY))
        return Z * X + (1 - Z) * Y

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return self.forward(X, Y)


class FFN(torch.nn.Module):
    l1: torch.nn.Linear
    l2: torch.nn.Linear
    dropout_p: Optional[float]

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 2048, dropout_p: Optional[float] = .1):
        super().__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, output_size)
        self.dropout_p = dropout_p

    def forward(self, U: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        U = self.l1(U)
        if self.dropout_p is not None:
            U = F.dropout(U, self.dropout_p)
        U = self.l2(U)
        return U + H

    def __call__(self, U: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        return self.forward(U, H)


class WordNet(torch.nn.Module):
    gat: GATConv
    activation: Callable[[torch.Tensor], torch.Tensor]
    ffn: FFN
    dropout_p: Optional[float]

    def __init__(self, word_size: int = 300, sentence_size: int = 640, n_heads: int = 6,
                 activation: Callable[[torch.Tensor], torch.Tensor] = F.elu, dropout_p: Optional[float] = .1):
        super().__init__()
        self.gat = GATConv((sentence_size, word_size), word_size, n_heads)
        self.ffn = FFN(n_heads * word_size, word_size, dropout_p=dropout_p)
        self.activation = activation
        self.dropout_p = dropout_p

    def forward(self, Hw: torch.Tensor, Hs: torch.Tensor, s2w: torch.Tensor) -> torch.Tensor:
        Uw = self.gat((Hs, Hw), s2w)
        Uw = self.activation(Uw)
        if self.dropout_p is not None:
            Uw = F.dropout(Uw, self.dropout_p)
        Hw = self.ffn(Uw, Hw)
        return Hw

    def __call__(self, Hw: torch.Tensor, Hs: torch.Tensor, s2w: torch.Tensor) -> torch.Tensor:
        return self.forward(Hw, Hs, s2w)


class SentenceNet(torch.nn.Module):
    gat_w: GATConv
    gat_s: GATConv
    gat_S: GATConv
    activation: Callable[[torch.Tensor], torch.Tensor]
    fusion1: Fusion
    fusion2: Fusion
    ffn: FFN
    dropout_p: Optional[float]

    def __init__(self, word_size: int = 300, sentence_size: int = 640, section_size: int = 512, n_heads: int = 8,
                 activation: Callable[[torch.Tensor], torch.Tensor] = F.elu, dropout_p: Optional[float] = .1):
        super().__init__()
        self.gat_w = GATConv((word_size, sentence_size), sentence_size, n_heads)
        self.gat_s = GATConv(sentence_size, sentence_size, n_heads)
        self.gat_S = GATConv((section_size, sentence_size), sentence_size, n_heads)
        self.fusion1 = Fusion(n_heads * sentence_size)
        self.fusion2 = Fusion(n_heads * sentence_size)
        self.ffn = FFN(n_heads * sentence_size, sentence_size, dropout_p=dropout_p)
        self.activation = activation
        self.dropout_p = dropout_p

    def forward(self, Hs: torch.Tensor, Hw: torch.Tensor, HS: torch.Tensor,
                w2s: torch.Tensor, s2s: torch.Tensor, S2s: torch.Tensor) -> torch.Tensor:
        Uw = self.activation(self.gat_w((Hw, Hs), w2s))
        Us = self.activation(self.gat_s(Hs, s2s))
        US = self.activation(self.gat_S((HS, Hs), S2s))
        U1 = self.fusion1(Uw, Us)
        U2 = self.fusion2(U1, US)
        if self.dropout_p is not None:
            U2 = F.dropout(U2, self.dropout_p)
        Hs = self.ffn(U2, Hs)
        return Hs

    def __call__(self, Hs: torch.Tensor, Hw: torch.Tensor, HS: torch.Tensor,
                 w2s: torch.Tensor, s2s: torch.Tensor, S2s: torch.Tensor) -> torch.Tensor:
        return self.forward(Hs, Hw, HS, w2s, s2s, S2s)


class SectionNet(torch.nn.Module):
    gat_s: GATConv
    gat_S: GATConv
    activation: Callable[[torch.Tensor], torch.Tensor]
    fusion: Fusion
    ffn: FFN
    dropout_p: Optional[float]

    def __init__(self, sentence_size: int = 640, section_size: int = 512, n_heads: int = 8,
                 activation: Callable[[torch.Tensor], torch.Tensor] = F.elu, dropout_p: Optional[float] = .1):
        super().__init__()
        self.gat_s = GATConv((sentence_size, section_size), section_size, n_heads)
        self.gat_S = GATConv(section_size, section_size, n_heads)
        self.fusion = Fusion(n_heads * section_size)
        self.ffn = FFN(n_heads * section_size, section_size, dropout_p=dropout_p)
        self.activation = activation
        self.dropout_p = dropout_p

    def forward(self, HS: torch.Tensor, Hs: torch.Tensor, s2S: torch.Tensor, S2S: torch.Tensor) -> torch.Tensor:
        Us = self.activation(self.gat_s((Hs, HS), s2S))
        US = self.activation(self.gat_S(HS, S2S))
        US = self.fusion(US, Us)
        if self.dropout_p is not None:
            US = F.dropout(US, self.dropout_p)
        HS = self.ffn(US, HS)
        return HS

    def __call__(self, HS: torch.Tensor, Hs: torch.Tensor, s2S: torch.Tensor, S2S: torch.Tensor) -> torch.Tensor:
        return self.forward(HS, Hs, s2S, S2S)


def data_to_tensor(data: Dict[Any, torch.Tensor]):
    output = []
    for key in sorted(data.keys()):
        output.append(data[key])
    return torch.stack(output)


def get_typed_edges(G: nx.DiGraph, from_: str, to: str, device: torch.types.Device):
    froms = get_typed_nodes(G, from_)
    froms = {key:i for i, key in enumerate(sorted(froms))}

    tos = get_typed_nodes(G, to)
    tos = {key:i for i, key in enumerate(sorted(tos))}

    edges = [(u, v) for u, v, data in G.edges(data=True) if data['type'] == f'{from_}2{to}']

    output = []
    for u, v in edges:
        output.append([froms[u], tos[v]])

    return torch.tensor(output, device=device).T

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb

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

    def __init__(self, n_iters: int = 2, dropout_p: Optional[float] = .1, recursive: bool = True):
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
        print()

    def project_sentences(self, sentences: Dict[Tuple[int, int], torch.Tensor],
                          G: nx.DiGraph) -> Dict[Tuple[int, int], torch.Tensor]:
        sentences = {key: self.sentence_projector(sentence) for key, sentence in sentences.items()}
        data = get_sentence_nodes(G, data=True)
        sentences = {key: torch.cat([sentence, data[key]['boundary']]) for key, sentence in sentences.items()}
        return sentences

    def project_sections(self, sections: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        sections = {key: self.section_projector(sentence) for key, sentence in sections.items()}
        return sections

    def prepare_words(self, G: nx.DiGraph) -> Dict[str, torch.Tensor]:
        words = get_word_nodes(G, data=True)
        return {key:data['value'] for key, data in words.items()}

    def prepare_sentences(self, words: Dict[str, torch.Tensor], G: nx.DiGraph) -> Dict[Tuple[int, int], torch.Tensor]:
        sentences = get_sentence_nodes(G, data=True)
        output = {}
        for sentence, data in sentences.items():
            output[sentence] = torch.stack([words[token] for token in data['tokens']])
        return self.sentence_encoder(output)

    def get_matrices(self, G: nx.DiGraph) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        words = self.prepare_words(G)
        sentences = self.prepare_sentences(words, G)
        sections = self.section_encoder(sentences)

        sections = self.project_sections(sections)
        sentences = self.project_sentences(sentences, G)

        Hw = data_to_tensor(words)
        Hs = data_to_tensor(sentences)
        HS = data_to_tensor(sections)

        return Hw, Hs, HS

    def get_edges(self, G: nx.DiGraph, device: torch.types.Device):
        s2w = get_typed_edges(G, 's', 'w', device)
        w2s = get_typed_edges(G, 'w', 's', device)
        s2s = get_typed_edges(G, 's', 's', device)
        S2s = get_typed_edges(G, 'S', 's', device)
        s2S = get_typed_edges(G, 's', 'S', device)
        S2S = get_typed_edges(G, 'S', 'S', device)
        return s2w, w2s, s2s, S2s, s2S, S2S


    def forward(self, G: nx.DiGraph):
        Hw, Hs, HS = self.get_matrices(G)
        s2w, w2s, s2s, S2s, s2S, S2S = self.get_edges(G, Hw.device)

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
        return self.forward(G)
