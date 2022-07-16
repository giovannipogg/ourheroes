import os

import torch
from torch.utils.data import DataLoader

from models.content_ranker import SentenceRanker, SectionRanker
from dataset.dataset_filter import PercentageFilter
from dataset.long_doc_dataset2 import LongDocDataset
from dataset.utils import parse_dataset, pad_collate2
from dataset.vocab import GloVeWrapper
from labelers import default_summary_scorer
from train_content_ranker import VECTORS
from utils import data_to_device
from collections import Counter
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch_sparse import SparseTensor

SEED = 42
DATASET = 'pubmed-dataset'
INPUT_SIZE = 300
HIDDEN_SIZE = 512
PRETRAINED = 'content_ranker\\checkpoints\\Unzipped'
M, N = 2, 2 # 4, 30
T = 2
EXTENSION = True
EPOCHS = 10

if EXTENSION:
    GRAPH_DIR = 'graph\\checkpoints\\EXTENSION'

    for epoch in range(EPOCHS):
        if not os.path.isdir(f"{GRAPH_DIR}\\{epoch}"):
            os.makedirs(f"{GRAPH_DIR}\\{epoch}")
else:
    GRAPH_DIR = 'graph\\checkpoints\\PAPER'

    for epoch in range(EPOCHS):
        if not os.path.isdir(f"{GRAPH_DIR}\\{epoch}"):
            os.makedirs(f"{GRAPH_DIR}\\{epoch}")

import networkx as nx
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import GATConv
from torch import optim
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# models
class Model_for_senteces(torch.nn.Module):
    def __init__(self):
        super(Model_for_senteces, self).__init__()
        self.list_of_conv = torch.nn.ModuleList([torch.nn.Conv1d(300, 50, kernel_size=num, padding=(num//2)).to(device) for num in np.arange(2, 8)])
        self.BiLSTM = torch.nn.LSTM(input_size=364, hidden_size=256, num_layers=2, bidirectional=True,
                                    batch_first=True)
        self.attention = Attention(512)

    def forward(self, x, sentence_id, sentence_len):
        x = pad_sequence([
            torch.swapaxes(conv(torch.swapaxes(x, 0, 1).unsqueeze(0)).squeeze(0), 0, 1)
            for conv in self.list_of_conv])
        x = torch.reshape(x, (x.size(0), 300))

        positional_embeddings = boundary_distance(sentence_id, sentence_len)
        x = torch.cat([x, positional_embeddings.repeat(x.size(0), 1).to(device)], dim=-1)
        x = x.unsqueeze(0)
        x, _ = self.BiLSTM(x)
        x = self.attention(x)
        # x = torch.cat([x, positional_embeddings.to(device)], dim=-1)

        return x


class Attention(torch.nn.Module):
    linear: torch.nn.Linear
    u_att: torch.nn.Parameter
    softmax: torch.nn.Softmax

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.u_att = torch.nn.Parameter(torch.zeros(hidden_size))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input: torch.Tensor):
        u = self.linear(input)
        u = torch.tanh(u)
        u = torch.matmul(u, self.u_att)
        alpha = self.softmax(u)
        output = alpha[:, :, None] * input
        output = torch.sum(output, dim=1)
        return output















def digest(section_outputs, sentence_outputs, sections, doc_literal):

    selected_sections_init = torch.argsort(section_outputs, descending=True).long()

    # non_empty_paragraphs = [i for i, doc in enumerate(doc_literal[0]) if len(doc[1])>1 and [word for d in doc[1] for word in d.tokens if word.isalnum()]]
    # literal = [(i, [[word for word in d.tokens if word.isalnum()]
    #                 for d in doc[1]])
    #            for i, doc in enumerate(doc_literal[0])]
    literal = [(i, [[word for word in d.tokens]
                    for d in doc[1]])
               for i, doc in enumerate(doc_literal[0])]
                            # if len(doc[1]) > 1]

    # non_empty_paragraphs = [i for i, sec in literal if any(len(s) > 1 for s in sec)]
    # sentence_mask = [torch.arange(len(sec)) for i, sec in literal if len(sec) > 1]
    non_empty_paragraphs = [i for i, sec in literal if sum([len(s)  for s in sec])>1]
    non_empty_sentence = [[len(s) for s in sec] for i, sec in literal]
    # non_empty_sentence = [[len(s)  for s in sec] for i, sec in literal]
    # for
    selected_sections_list = []
    for s in selected_sections_init:
        if int(s) in non_empty_paragraphs:
            selected_sections_list.append(int(s))
    selected_sections = torch.tensor(selected_sections_list)[:M].long()
    selected_sentences = []
    selected_sentences_from_digest = []

    if selected_sections_list:

        sentence_mask = torch.isclose(sections[selected_sections], torch.zeros(size=(1,)).to(device))
        sentence_mask = torch.logical_not(torch.all(sentence_mask, dim=-1))

        section_lenghts = np.array([len(section) for _, section in doc_literal[0]])
        section_lenghts_cumulative = [np.sum(section_lenghts[:index]) if index > 0 else 0 for index in
                                      range(len(section_lenghts))]
        for i, (sentence, mask) in enumerate(zip(sentence_outputs[selected_sections], sentence_mask)):
            sentence = sentence[mask]

            ordered_sentences = torch.argsort(sentence, descending=True)
            indeces_of_sentences = []
            for index_of_sentence in ordered_sentences:
                if non_empty_sentence[selected_sections[i]][index_of_sentence]>1:
                    indeces_of_sentences.append(index_of_sentence)
            selected_sentences.append(torch.tensor(indeces_of_sentences)[:N].long())
            selected_sentences_from_digest += [section_lenghts_cumulative[selected_sections[i]] + index for index in
                                               selected_sentences[-1]]
        selected_sentences_from_digest = torch.hstack(selected_sentences_from_digest).long()
    non_empty_sentence = [non_empty_sentence[index] for index in selected_sections]
    return selected_sections, selected_sentences, selected_sentences_from_digest, selected_sections_list, non_empty_sentence

class Create_Features(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.sentence_model = Model_for_senteces()
        self.section_model = Model_for_sections()

    def forward(self, selected_sections, selected_sentences, doc_literal, data, non_empty_sentence, T):



        dict_of_features = {key: {} for key in range(T + 1)}
        edges = []
        n_sections = torch.max(selected_sections)
        n_sections = n_sections if n_sections > 0 else 1

        for section_id, sentences, len_sen in zip(selected_sections, selected_sentences, non_empty_sentence):
            section_id = int(section_id)

            # edges from the others sections to the selected section
            edges += [(int(other_section_id), section_id,
                       {"level": "S2S", "pos": torch.tensor([
                            1 - torch.abs(torch.tensor(section_id - int(other_section_id)))/n_sections,
                            1.,
                            1.,
                            1.])}) for
                      other_section_id in selected_sections]

            sentence_modeled_list = []
            n_sentences = torch.max(sentences)
            n_sentences = n_sentences if n_sentences > 0 else 1
            for sentence_id, sentence_len in zip(sentences, len_sen):
                sentence_id = int(sentence_id)

                sentence_modeled = self.sentence_model.forward(data[1][section_id][sentence_id], sentence_id, sentence_len)
                dict_of_features[0][f"{section_id}_{sentence_id}"] = sentence_modeled.squeeze()
                sentence_modeled_list.append(sentence_modeled)

                # edges from the others sections to the selected sentence
                edges += [(int(other_section_id), f"{section_id}_{sentence_id}",
                           {"level": "S2s", "pos": torch.tensor([
                            1 - torch.abs(torch.tensor(section_id - int(other_section_id)))/n_sections,
                            1. / n_sentences,
                            1.,
                            1.])}) for
                          other_section_id in selected_sections]

                # edges from the the selected sentence to the belonging section
                edges += [(f"{section_id}_{sentence_id}", section_id,
                           {"level": "s2S", "pos":  torch.tensor([
                            1.,
                            1./n_sentences,
                            1.,
                            1.])})]

                # edges from the others sentences in the section to the selected sentence
                edges += [(f"{section_id}_{int(other_sentence_id)}", f"{section_id}_{sentence_id}",
                           {"level": "s2s", "pos": torch.tensor([
                            1.,
                            1.,
                            1 - torch.abs(torch.tensor(sentence_id - int(other_sentence_id)))/n_sentences,
                            1.])}) for
                          other_sentence_id in sentences]

                n_words = len(data[1][0][0])
                words_in_sentence = doc_literal[0][section_id][1][sentence_id].tokens
                counter_of_words_in_sentence = Counter(words_in_sentence)
                for word_id, word in enumerate(data[1][section_id][sentence_id]):

                    if torch.allclose(word, torch.Tensor([0.]).to(device)):
                        continue
                    word_literal = words_in_sentence[word_id]
                    dict_of_features[0][f"{word_literal}"] = word
                    word_weight = counter_of_words_in_sentence[words_in_sentence[word_id]] / n_words
                    # edges from the the selected sentence to the selcted word
                    edges += [(f"{section_id}_{sentence_id}", word_literal,
                               {"level": "s2w", "pos":  torch.tensor([
                                1.,
                                1.,
                                1.,
                                word_weight])})]

                    # edges from the the selected word to the belonging sentence
                    edges += [(word_literal, f"{section_id}_{sentence_id}",
                               {"level": "w2s", "pos":  torch.tensor([
                                1.,
                                1.,
                                1.,
                                word_weight])})]

            dict_of_features[0][section_id] = self.section_model.forward(torch.cat(sentence_modeled_list, dim=0))[1]

        return dict_of_features, edges

class Extractive_summarization(torch.nn.Module):

    def __init__(self, input_channels_dict, fusion_sizes, heads_dict, T, EXTENSION, key_to_skip_at_last_iteration = ["S2S", "s2S", "s2w"]):
        super().__init__()
        self.EXTENSION = EXTENSION
        self.T = T
        self.models_at_iteration = torch.nn.ModuleDict({})
        self.input_channels_dict = input_channels_dict
        self.key_to_skip_at_last_iteration = key_to_skip_at_last_iteration
        for key, input_channel in self.input_channels_dict.items():
            self.models_at_iteration[f"GAT_dict_{key}"] = GAT_network(input_channel, heads_dict[key], self.T, key, self.key_to_skip_at_last_iteration, self.EXTENSION).to(device)

        self.models_at_iteration["fusion_wS2s"] = Fusion(input_size=(fusion_sizes['S2s'] + fusion_sizes['w2s']), T=self.T).to(
            device)

        self.models_at_iteration["fusion_s"] = Fusion(input_size=(fusion_sizes['S2s'] + fusion_sizes['s2s']), T=self.T).to(device)

        self.models_at_iteration["fusion_S"] = Fusion(input_size=(fusion_sizes['s2S'] + fusion_sizes['S2S']), T=self.T-1).to(device)

        self.models_at_iteration["FFNw"] = FFN(fusion_sizes['s2w'], self.input_channels_dict["s2w"][1], self.T-1).to(device)

        self.models_at_iteration["FFNs"] = FFN(fusion_sizes['S2s'], self.input_channels_dict["S2s"], self.T).to(device)

        self.models_at_iteration["FFNS"] = FFN(fusion_sizes['S2S'], self.input_channels_dict["S2S"], self.T-1).to(device)

        self.Prob_extractor = PE(self.input_channels_dict["S2s"]).to(device)

        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, dict_of_features, edges):

        node_indexes = {}
        for node_index, node_id_key in enumerate(dict_of_features[0].keys()):
            node_indexes[node_id_key] = node_index
        index_nodes = {v: k for k, v in node_indexes.items()}

        G = nx.DiGraph()
        G.add_nodes_from(list(dict_of_features[0].keys()))
        G.add_edges_from(edges)

        sub_graphs_edges = {"S2S": [], "S2s": [], "s2S": [], "s2s": [], "s2w": [], "w2s": []}
        sub_graphs_weigth = {"S2S": [], "S2s": [], "s2S": [], "s2s": [], "s2w": [], "w2s": []}

        for node in G.nodes():
            # get in-edges of a given node:
            for u, v, data in G.in_edges(node, data=True):
                sub_graphs_edges[data["level"]].append((u, v))
                sub_graphs_weigth[data["level"]].append(data["pos"])
                # print(u, v, data)

        for key, sub_grap_weight in sub_graphs_weigth.items():
            try:
                sub_graphs_weigth[key] = torch.stack(sub_grap_weight)
            except Exception:
                print()

        edges_in_each_subgraph = {}
        for key, sub_graph_edges in sub_graphs_edges.items():
            # try:
                edges_in_each_subgraph[key] = torch.vstack(
                    [torch.Tensor([node_indexes[u], node_indexes[v]]).long() for u, v in sub_graph_edges]).T
            # except Exception:
            #     print()

        # node_tensor = torch.zeros((len(node_indexes), dict_of_features[t][sub_graph_edges[0][0]].size(dim=0)))
        U = {t: {} for t in range(self.T)}
        Hw = Us = Hs = US = HS = {t: {} for t in range(self.T)}
        sub_graphs_nodes = {t: {} for t in range(self.T)}
        nodes_in_each_subgraph = {t: {} for t in range(self.T)}
        node_tensor = node_tensor_1 = node_tensor_2 = {t: {} for t in range(self.T)}
        node_tensor_index = node_tensor_index_1 = node_tensor_index_2 = {t: {} for t in range(self.T)}
        t = 0

        while t < self.T:

            for key, sub_graph_edges in sub_graphs_edges.items():

                if t == self.T - 1 and key in self.key_to_skip_at_last_iteration:
                    continue

                if key[0] == key[-1]:
                    node_tensor[t][f"{key}_{key[0]}"] = torch.zeros(
                        (len(node_indexes), dict_of_features[t][sub_graph_edges[0][0]].size(dim=0)))
                    node_tensor_index[t][f"{key}_{key[0]}"] = torch.zeros((len(node_indexes)))
                    for u, v in sub_graph_edges:
                        node_tensor[t][f"{key}_{key[0]}"][node_indexes[u]] = dict_of_features[t][u].clone()
                        node_tensor[t][f"{key}_{key[0]}"][node_indexes[v]] = dict_of_features[t][v].clone()
                        node_tensor_index[t][f"{key}_{key[0]}"][node_indexes[u]] = 1
                        node_tensor_index[t][f"{key}_{key[0]}"][node_indexes[v]] = 1

                    sub_graphs_nodes[t][key] = node_tensor[t][f"{key}_{key[0]}"]
                    nodes_in_each_subgraph[t][key] = node_tensor_index[t][f"{key}_{key[0]}"]
                else:
                    node_tensor_1[t][f"{key}_{key[0]}"] = torch.zeros(
                        (len(node_indexes), dict_of_features[t][sub_graph_edges[0][0]].size(dim=0)))
                    node_tensor_2[t][f"{key}_{key[-1]}"] = torch.zeros(
                        (len(node_indexes), dict_of_features[t][sub_graph_edges[0][1]].size(dim=0)))
                    node_tensor_index_1[t][f"{key}_{key[0]}"] = torch.zeros((len(node_indexes)))
                    node_tensor_index_2[t][f"{key}_{key[-1]}"] = torch.zeros((len(node_indexes)))

                    for u, v in sub_graph_edges:
                        node_tensor_1[t][f"{key}_{key[0]}"][node_indexes[u]] = dict_of_features[t][u].clone()
                        node_tensor_2[t][f"{key}_{key[-1]}"][node_indexes[v]] = dict_of_features[t][v].clone()
                        node_tensor_index_1[t][f"{key}_{key[0]}"][node_indexes[u]] = 1
                        node_tensor_index_2[t][f"{key}_{key[-1]}"][node_indexes[v]] = 1

                    sub_graphs_nodes[t][key] = (
                    node_tensor_1[t][f"{key}_{key[0]}"], node_tensor_2[t][f"{key}_{key[-1]}"])
                    nodes_in_each_subgraph[t][key] = node_tensor_index_2[t][f"{key}_{key[-1]}"]

            for key in self.input_channels_dict.keys():

                if t == self.T - 1 and key in self.key_to_skip_at_last_iteration:
                    continue

                edges_in_each_subgraph[key] = edges_in_each_subgraph[key].to(device)
                sub_graphs_weigth[key] = sub_graphs_weigth[key].type(torch.float).to(device)
                if isinstance(sub_graphs_nodes[t][key], tuple):
                    sub_graphs_nodes[t][key] = (
                    sub_graphs_nodes[t][key][0].to(device), sub_graphs_nodes[t][key][1].to(device))
                else:
                    sub_graphs_nodes[t][key] = sub_graphs_nodes[t][key].to(device)
                U[t][key] = self.models_at_iteration[f"GAT_dict_{key}"].forward(sub_graphs_nodes[t][key],
                                                                           edges_in_each_subgraph[key],
                                                                           sub_graphs_weigth[key], t)

            Us[t] = self.dropout(
                self.models_at_iteration["fusion_s"].forward(
                self.models_at_iteration["fusion_wS2s"].forward(U[t]['w2s'], U[t]['S2s'], t), U[t]['s2s'], t)
            )
            Hs[t] = self.models_at_iteration["FFNs"].forward(Us[t], sub_graphs_nodes[t]['s2s'], t)

            nodes = torch.nonzero(nodes_in_each_subgraph[t]['s2s'])
            for node_index in nodes:
                node_key = index_nodes[int(node_index)]
                dict_of_features[t + 1][node_key] = Hs[t][node_index].view(Hs[t][node_index].size(1)).clone()

            if t == self.T-1:
                t += 1
                continue

            Hw[t] = self.models_at_iteration["FFNw"].forward(self.dropout(U[t]['s2w']), sub_graphs_nodes[t]['s2w'][1], t)

            nodes = torch.nonzero(nodes_in_each_subgraph[t]['s2w'])
            for node_index in nodes:
                node_key = index_nodes[int(node_index)]
                dict_of_features[t + 1][node_key] = Hw[t][node_index].view(Hw[t][node_index].size(1)).clone()

            US[t] = self.dropout(
                self.models_at_iteration["fusion_S"].forward(U[t]['s2S'], U[t]['S2S'], t)
            )
            HS[t] = self.models_at_iteration["FFNS"].forward(US[t], sub_graphs_nodes[t]['S2S'], t)

            nodes = torch.nonzero(nodes_in_each_subgraph[t]['S2S'])
            for node_index in nodes:
                node_key = index_nodes[int(node_index)]
                dict_of_features[t + 1][node_key] = HS[t][node_index].view(HS[t][node_index].size(1)).clone()
            t += 1

        node_tensor_final = torch.zeros(
            (len(node_indexes), dict_of_features[t][sub_graphs_edges["s2s"][0][0]].size(dim=0)))
        node_tensor_index_final = torch.zeros((len(node_indexes)))
        for u, v in sub_graphs_edges["s2s"]:
            node_tensor_final[node_indexes[u]] = dict_of_features[t][u].clone()
            node_tensor_final[node_indexes[v]] = dict_of_features[t][v].clone()
            node_tensor_index_final[node_indexes[u]] = 1
            node_tensor_index_final[node_indexes[v]] = 1

        p = self.Prob_extractor(node_tensor_final.to(device)).squeeze()
        node_tensor_index_final = node_tensor_index_final.long()
        p = p[node_tensor_index_final == 1]

        return p

def main():
    torch.manual_seed(SEED)

    if not os.path.isdir(f"{DATASET}\\parsed"):
        parse_dataset(DATASET)

    vectors = VECTORS[INPUT_SIZE]
    vectors = vectors()
    vocab = GloVeWrapper(vectors, f"{DATASET}\\vocab")

    filter_ = PercentageFilter(f"{DATASET}\\parsed\\train", percentage=0.6) # !!!

    dataset = LongDocDataset(f"{DATASET}\\parsed", mode='train', tok2tensor=vocab, filter=filter_)
    dataloader = DataLoader(dataset=dataset, batch_size=1, collate_fn=pad_collate2, shuffle=True, prefetch_factor=2)

    sentence_ranker = SentenceRanker(INPUT_SIZE, HIDDEN_SIZE).to(device)
    section_ranker = SectionRanker(INPUT_SIZE, HIDDEN_SIZE).to(device)

    if os.path.isdir(PRETRAINED):
        last = int(sorted(os.listdir(PRETRAINED), reverse=True)[0])
        checkpoint = torch.load(f"{PRETRAINED}\\{last}")
        sentence_ranker.load_state_dict(checkpoint['sentence_state_dict'])
        section_ranker.load_state_dict(checkpoint['section_state_dict'])

    sentence_ranker.eval()
    section_ranker.eval()


    input_channels_dict = {"S2S": 512, "S2s": 512, "s2S": 512, "s2s": 512, "s2w": (512, 300), "w2s": (300, 512)}
    heads_dict = {"S2S": 8, "S2s": 8, "s2S": 8, "s2s": 8, "s2w": 6, "w2s": 8}

    fusion_sizes = {}
    for key, value in input_channels_dict.items():
        if isinstance(value, tuple):
            value = value[1]
        fusion_sizes[key] = heads_dict[key] * value


    features_creator = Create_Features().to(device)
    all_params = list(features_creator.parameters())

    extractive_model = Extractive_summarization(input_channels_dict, fusion_sizes, heads_dict, T, EXTENSION).to(device)
    # all_params += list({"name": n, "params": p} for n, p in extractive_model.named_parameters())
    all_params += list(extractive_model.parameters())

    scorer = default_summary_scorer()
    learning_rate = 1e-4 # args.learning_rate
    optimizer = optim.Adam(all_params, lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    graph_scaler = GradScaler()
    iters_to_accumulate = 32

    for epoch in range(EPOCHS):

        tq = tqdm(total=len(dataloader)) # * args.batch_size)
        tq.set_description(f'train: epoch {epoch}, lr {learning_rate}')

        graph_loss = []
        for doc_counter, (data, _, doc_literal, abstract_listeral) in enumerate(dataloader):
            data = data_to_device(data, device)

            titles, sections = data

            with torch.no_grad():
                sections, sentence_outputs = sentence_ranker(sections)
                section_outputs = section_ranker(titles, sections)

            if section_outputs.dim() == 0:
                continue
            optimizer.zero_grad()

            selected_sections, selected_sentences, selected_sentences_from_digest, selected_sections_list, non_empty_sentence = digest(
                section_outputs, sentence_outputs, sections, doc_literal)
            if not selected_sections_list:
                continue
            target = scorer(*doc_literal, *abstract_listeral).float().to(device)
            target_selected = target[selected_sentences_from_digest]

            with autocast():

                dict_of_features, edges = features_creator.forward(selected_sections, selected_sentences, doc_literal, data, non_empty_sentence, T)
                p = extractive_model.forward(dict_of_features, edges)
                # print("Almost done...")
                loss = criterion(p, target_selected)


            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(all_params, 2.0)
            # optimizer.step()
            # print("Stepped")
            # graph_loss = loss.item()
            # print(doc_counter, ") Number of parameters with grad = None: ", len([p for p in optimizer.param_groups[0]["params"] if p.grad==None]))

            graph_scaler.scale(loss).backward()
            tq.update(1)  # args.batch_size)
            # graph_loss.append(loss.clone().item())
            #
            # if (doc_counter + 1) % iters_to_accumulate == 0:
            graph_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, 2.0)

            graph_scaler.step(optimizer)
            graph_scaler.update()
            torch.cuda.empty_cache()

            tq.set_postfix({'Graph_losses': sum(graph_loss)/iters_to_accumulate, "Number of parameters with grad = None": len([p for p in optimizer.param_groups[0]["params"] if p.grad==None])})
            # tq.update(iters_to_accumulate) #args.batch_size)
            # graph_loss = []

            if (doc_counter + 1) % 250 == 0:
                torch.save({
                    'epoch': epoch,
                    'features_creator_state_dict': features_creator.state_dict(),
                    'extractive_model_state_dict': extractive_model.state_dict(),
                    'graph_optim_state_dict': optimizer.state_dict()
                }, f"{GRAPH_DIR}\\{epoch}\\{doc_counter + 1}")



        torch.save({
            'epoch': epoch,
            'features_creator_state_dict': features_creator.state_dict(),
            'extractive_model_state_dict': extractive_model.state_dict(),
            'graph_optim_state_dict': optimizer.state_dict()
        }, f"{GRAPH_DIR}\\{epoch}\\{doc_counter + 1}")
        tq.close()

if __name__ == '__main__':
    main()
