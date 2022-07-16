import asyncio
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Type, Sequence, List, Union, Tuple, Optional, Callable

import networkx as nx
import torch
import torch.nn.functional as F
import torch.types
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DocumentDataset, CRDataset, DigestableDataset, GSDataset
from digester import Digester
from grapher import get_word_nodes, get_sentence_nodes
from models.content_ranker import ContentRanker
from models.graph_summarizer import GraphSummarizer
from ourtypes import Criterion, Digest, Tokenized
from scorer import get_mean_rouge_scorer


def log(to_log: Any, log_path: str):
    with open(log_path, 'a') as fp:
        print(datetime.now(), to_log, file=fp)


def poly_lr_scheduler(optimizer: torch.optim.Optimizer, iter: int, init_lr: float = 1e-4,
                      max_iter: int = 10, power: float = 0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    lr = init_lr * (1 - iter / max_iter) ** power
    optimizer.param_groups[0]['lr'] = lr
    return lr


@dataclass
class Config:
    device: torch.types.Device
    criterion: Criterion
    optimizer: Type[torch.optim.Optimizer]
    num_epochs: int
    save_epochs: int
    save_path: str
    log_path: str
    eval_path: str = '.\\last_eval'
    dataloader_kwargs: Optional[Dict[str, Any]] = None
    criterion_kwargs: Optional[Dict[str, Any]] = None
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    scheduler: Optional[Callable[[torch.optim.Optimizer, int, ...], float]] = None
    scheduler_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.dataloader_kwargs is None:
            self.dataloader_kwargs = {}
        if self.criterion_kwargs is None:
            self.criterion_kwargs = {}
        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {}
        if self.scheduler_kwargs is None:
            self.scheduler_kwargs = {}


def default_cr_config() -> Config:
    return Config(
        dataloader_kwargs={'batch_size': 32, 'num_workers': 2, 'shuffle': True, 'prefetch_factor': 2},
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        criterion=F.binary_cross_entropy_with_logits,
        optimizer=torch.optim.SGD,
        optimizer_kwargs={'lr': .0001},
        num_epochs=5,
        save_epochs=1,
        save_path='cr_checkpoints',
        log_path='cr_log'
    )


def default_gs_config() -> Config:
    return Config(
        dataloader_kwargs={'batch_size': 8, 'num_workers': 1},  # 'shuffle': True},
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        criterion=F.binary_cross_entropy_with_logits,
        optimizer=torch.optim.Adam,
        optimizer_kwargs={'lr': .0001},
        num_epochs=10,
        save_epochs=1,
        save_path='D:\\gs_checkpoints_official',
        log_path='gs_log_official'
    )


def cr_collate(batch: Sequence):
    data = [sub[0] for sub in batch]
    targets = [sub[1] for sub in batch]
    sentence_target = [t[0] for t in targets]
    sentence_target = torch.cat(sentence_target)
    section_target = [t[1] for t in targets]
    section_target = torch.cat(section_target)
    return data, sentence_target, section_target


def dig_collate(batch: Sequence):
    fnames = [sub[0] for sub in batch]
    data = [sub[1] for sub in batch]
    tensors = [sub[2] for sub in batch]
    return fnames, data, tensors


def relabel_node(node: Union[int, Tuple[int, int], str], prefix: int) -> Union[
    Tuple[int, int], Tuple[int, int, int], Tuple[int, str]]:
    if isinstance(node, tuple):
        node = (prefix,) + node
    else:
        node = (prefix, node)
    return node


def relabel_nodes(nodes: Sequence[Tuple], prefix: int) -> List:
    output = []
    for node, data in nodes:
        if 'tokens' in data:
            data['tokens'] = [(prefix, token) for token in data['tokens']]
        output.append((relabel_node(node, prefix), data))
    return output


def relabel_edges(edges: Dict, prefix: int) -> List:
    output = []
    for u, v, data in edges:
        output.append((relabel_node(u, prefix), relabel_node(v, prefix), data))
    return output


def join_graphs(graphs: List[nx.DiGraph]) -> nx.DiGraph:
    out_graph = nx.DiGraph()
    for i, graph in enumerate(graphs):
        nodes = relabel_nodes(graph.nodes(data=True), i)
        edges = relabel_edges(graph.edges(data=True), i)
        out_graph.add_nodes_from(nodes)
        out_graph.add_edges_from(edges)
    return out_graph


def gs_collate(batch: Sequence):
    graphs = [graph for graph, _, _, _ in batch]
    targets = [target for _, target, _, _ in batch]
    summaries = [summary for _, _, summary, _ in batch]
    digests = [digest for _, _, _, digest in batch]
    target = torch.cat(targets)
    graph = join_graphs(graphs)
    return graph, target, summaries, digests


def to_device_r(data: Sequence, device: torch.types.Device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return type(data)([to_device_r(sub, device) for sub in data])


def graph_to_device(G: nx.DiGraph, device: torch.types.Device):
    attributes = {}
    words = get_word_nodes(G, data=True)
    for word, data in words.items():
        attributes[word] = data['value'].to(device)
    nx.set_node_attributes(G, attributes, name='value')

    sentences = get_sentence_nodes(G, data=True)
    attributes = {}
    for sentence, data in sentences.items():
        attributes[sentence] = data['boundary'].to(device)
    nx.set_node_attributes(G, attributes, name='boundary')


# async def digest(fname: str, data: Dict[str, Any], tensors: torch.Tensor, content_ranker: ContentRanker, digester: Digester, tq: tqdm):
#     with torch.no_grad():
#         sentence_scores, section_scores = content_ranker(tensors)
#     data['digest'] = digester(sentence_scores, section_scores, data)
#     with open(fname, 'w') as fp:
#         json.dump(data, fp)
#         fp.flush()
#     tq.update(1)


async def write_digest(fname: str, digest: Digest, tq: tqdm):
    with open(fname, 'r') as fp:
        doc = json.load(fp)
    doc['digest'] = digest
    with open(fname, 'w') as fp:
        json.dump(doc, fp)
        fp.flush()
    tq.update(1)


async def write_digests(output: Dict[str, Digest]):
    tq = tqdm(total=len(output))
    await asyncio.gather(*[write_digest(fname, digest, tq) for fname, digest in output.items()])
    tq.close()


@dataclass
class OurHeroes:
    content_ranker: ContentRanker = ContentRanker()
    digester: Digester = Digester()
    graph_summarizer: GraphSummarizer = GraphSummarizer()

    def fit_cr(self, dataset: DocumentDataset, config: Config):
        log('fit_cr', config.log_path)
        log(config, config.log_path)

        self.content_ranker = self.content_ranker.to(config.device)
        self.content_ranker.train()

        dataset = CRDataset(dataset.files)
        dataloader = DataLoader(dataset, collate_fn=cr_collate, **config.dataloader_kwargs)

        optimizer = config.optimizer(self.content_ranker.parameters(), **config.optimizer_kwargs)
        scaler = amp.GradScaler()

        for epoch in range(config.num_epochs):

            lr = config.optimizer_kwargs['lr']
            if config.scheduler is not None:
                lr = config.scheduler(optimizer, epoch, **config.scheduler_kwargs)

            log(f'epoch {epoch}, lr {lr:.2e}', config.log_path)
            tq = tqdm(total=len(dataloader))
            tq.set_description(f'train: epoch {epoch}, lr {lr:.2e}')

            for data, sentence_target, section_target in dataloader:
                # optimizer.zero_grad()
                for param in self.content_ranker.parameters():
                    param.grad = None

                data, sentence_target, section_target = to_device_r(
                    (data, sentence_target, section_target), config.device)

                with amp.autocast():
                    sentence_scores, section_scores = self.content_ranker(data)
                    sentence_loss = config.criterion(sentence_scores, sentence_target)
                    section_loss = config.criterion(section_scores, section_target)
                    loss = sentence_loss + section_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.empty_cache()

                to_log = {
                    'sentence_loss': sentence_loss.item(),
                    'section_loss': section_loss.item()
                }
                log(to_log, config.log_path)
                tq.set_postfix(to_log)
                tq.update(1)

            if epoch % config.save_epochs == 0:
                if not os.path.isdir(config.save_path):
                    os.makedirs(config.save_path)
                torch.save({
                    'epoch': epoch,
                    'state': self.content_ranker.state_dict(),
                    'optim': optimizer.state_dict()
                }, f"{config.save_path}\\{epoch}.pth")

            tq.close()

    def digest(self, dataset: DocumentDataset):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = DigestableDataset(dataset.files)
        dataloader = DataLoader(dataset, batch_size=32, collate_fn=dig_collate, num_workers=0)
        self.content_ranker = self.content_ranker.to(device)
        self.content_ranker.eval()
        output = {}

        tq = tqdm(total=len(dataloader))
        tq.set_description(f'digest ')
        for fnames, data, tensors in dataloader:
            tensors = to_device_r(tensors, device)
            with torch.no_grad():
                sentence_scores, section_scores = self.content_ranker(tensors)
            digests = self.digester(sentence_scores, section_scores, data)
            output.update({fname: digest for fname, digest in zip(fnames, digests)})
            tq.update(1)
        with open('digests.json', 'w') as fp:
            json.dump(output, fp)
            fp.flush()

    def gs_select(self, document: List[List[Tokenized]], prediction: torch.Tensor):
        document = [sentence for section in document for sentence in section]
        total_words, n_sentences = 0, 0
        selected = torch.argsort(prediction, descending=True)
        for sentence in selected:
            total_words += len(document[sentence])
            n_sentences += 1
            if total_words >= 200:
                break
        selected = torch.sort(selected[:n_sentences]).values
        return [' '.join(document[sentence]) for sentence in selected]

    def eval_gs(self, dataset: DocumentDataset, config: Config):
        dataset = GSDataset(dataset.files)
        dataloader = DataLoader(dataset, collate_fn=gs_collate)

        rouge1 = get_mean_rouge_scorer(['rouge1'], 'fmeasure')  # , True)
        rouge2 = get_mean_rouge_scorer(['rouge2'], 'fmeasure')  # , True)
        rougeL = get_mean_rouge_scorer(['rougeL'], 'fmeasure')  # , True)
        rougeLsum = get_mean_rouge_scorer(['rougeLsum'], 'fmeasure')  # , True)

        rouges = {}

        checkpoints = list(sorted(os.listdir(config.save_path)))
        for i, checkpoint in enumerate(checkpoints):

            rouges[i] = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': []}

            checkpoint = f"{config.save_path}\\{checkpoint}"
            checkpoint = torch.load(checkpoint)
            self.graph_summarizer.load_state_dict(checkpoint['state'])

            self.graph_summarizer = self.graph_summarizer.to(config.device)
            self.graph_summarizer.eval()
            tq = tqdm(total=len(dataloader))
            tq.set_description(f'eval: {i}')

            for graph, _, summary, digest in dataloader:
                try:
                    summary, digest = summary[0], digest[0]
                    summaryLsum = '\n'.join(summary)
                    summary = ' '.join(summary)

                    graph_to_device(graph, config.device)

                    with amp.autocast():
                        prediction = self.graph_summarizer(graph)

                    digest = self.gs_select(digest, prediction)
                    digestLsum = '\n'.join(digest)
                    digest = ' '.join(digest)

                    rouges[i]['rouge1'].append(rouge1(summary, digest))
                    rouges[i]['rouge2'].append(rouge2(summary, digest))
                    rouges[i]['rougeL'].append(rougeL(summary, digest))
                    rouges[i]['rougeLsum'].append(rougeLsum(summaryLsum, digestLsum))

                    torch.cuda.empty_cache()
                    # tq.set_postfix({'rouge1': rouge1_score, 'rouge2': rouge2_score, 'rougeL': rougeL_score})
                except Exception as e:
                    continue
                finally:
                    tq.update(1)

            tq.close()

        with open(config.eval_path, 'w') as fp:
            json.dump(rouges, fp)
            fp.flush()

    def fit_gs(self, dataset: DocumentDataset, config: Config):
        log('fit_gs', config.log_path)
        log(config, config.log_path)

        self.graph_summarizer = self.graph_summarizer.to(config.device)
        dataset = GSDataset(dataset.files)
        dataloader = DataLoader(dataset, collate_fn=gs_collate, **config.dataloader_kwargs)
        optimizer = config.optimizer(self.graph_summarizer.parameters(), **config.optimizer_kwargs)
        scaler = amp.GradScaler()

        if not os.path.isdir(config.save_path):
            os.makedirs(config.save_path)
        if len(os.listdir(config.save_path)) == 0:
            last_epoch = 0
        else:
            checkpoints = os.listdir(config.save_path)
            last = sorted(checkpoints)[-1]
            checkpoint = torch.load(f'{config.save_path}\\{last}')
            last_epoch = checkpoint['epoch'] + 1
            self.graph_summarizer.load_state_dict(checkpoint['state'])
            optimizer.load_state_dict(checkpoint['optim'])
            del checkpoint
            torch.cuda.empty_cache()

        for epoch in range(last_epoch, config.num_epochs):

            # Early stopping
            if epoch == 5: break

            self.graph_summarizer.train()

            lr = config.optimizer_kwargs['lr']
            if config.scheduler is not None:
                lr = config.scheduler(optimizer, epoch, **config.scheduler_kwargs)

            log(f'epoch {epoch}, lr {lr:.2e}', config.log_path)
            tq = tqdm(total=len(dataloader))
            tq.set_description(f'train: epoch {epoch}, lr {lr:.2e}')

            for graph, target, _, _ in dataloader:

                # optimizer.zero_grad()
                for param in self.graph_summarizer.parameters():
                    param.grad = None

                target = target.to(config.device)
                graph_to_device(graph, config.device)

                try:
                    with amp.autocast():
                        prediction = self.graph_summarizer(graph)
                        loss = config.criterion(prediction, target)
                    scaler.scale(loss).backward()

                    loss = loss.item()
                    log({
                        'epoch': epoch,
                        'loss': loss,
                    }, config.log_path)
                    tq.set_postfix({'loss': loss})

                    del loss
                    torch.cuda.empty_cache()

                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.graph_summarizer.parameters(), 2.0)
                    scaler.step(optimizer)
                    scaler.update()
                except Exception as e:
                    log({
                        'epoch': epoch,
                        'exception': e,
                    }, config.log_path)
                    if repr(e) == "IndexError('Dimension out of range (expected to be in range of [-1, 0], but got 1)')" \
                            or repr(e) == "RuntimeError('stack expects a non-empty TensorList')":
                        continue
                    print(repr(e))
                    exit()
                finally:
                    torch.cuda.empty_cache()
                    tq.update(1)

            if epoch % config.save_epochs == 0:
                torch.save({
                    'epoch': epoch,
                    'state': self.graph_summarizer.state_dict(),
                    'optim': optimizer.state_dict()
                }, f"{config.save_path}\\gs_{epoch}.pth")

            tq.close()

    def fit(self, dataset: DocumentDataset,
            cr_config: Config = default_cr_config(),
            gs_config: Config = default_gs_config()):
        self.fit_cr(dataset, cr_config)
        self.digest(dataset)
        self.fit_gs(dataset, gs_config)


def our_heroes_default() -> OurHeroes:
    content_ranker = ContentRanker()
    return OurHeroes(content_ranker, Digester())


import json


def filter_files_for_gs_test(files):
    output = []
    tq = tqdm(total=len(files))
    for file in files:
        with open(file, 'r') as fp:
            doc = json.load(fp)
            if 'digest' in doc:
                output.append(file)
        tq.update(1)
    with open('digested.json', 'w') as fp:
        json.dump(output, fp)
    return output


def train_cr():
    with open('retained.json', 'r') as fp:
        files = json.load(fp)
    doc_dataset = DocumentDataset(files)
    model = OurHeroes()
    config = default_cr_config()
    model.fit_cr(doc_dataset, config)


def digest():
    with open('retained.json', 'r') as fp:
        files = json.load(fp)
    doc_dataset = DocumentDataset(files)
    content_ranker = ContentRanker()
    checkpoint = torch.load(f"cr_checkpoints\\4.pth")
    content_ranker.load_state_dict(checkpoint['state'])
    model = OurHeroes(content_ranker)
    model.digest(doc_dataset)


def output_digests():
    with open('digests.json', 'r') as fp:
        digests = json.load(fp)
    asyncio.run(write_digests(digests))


async def check_digested(file: str, tq: tqdm):
    with open(file, 'r') as fp:
        doc = json.load(fp)
    tq.update(1)
    if 'digest' in doc:
        return file, True
    return file, False


async def check_retained():
    with open('retained.json', 'r') as fp:
        files = json.load(fp)
    tq = tqdm(total=len(files))
    files = await asyncio.gather(*[check_digested(file, tq) for file in files])
    files = [file for file, retain in files if retain]
    with open('retained.json', 'w') as fp:
        json.dump(files, fp)
        fp.flush()


def seed_torch(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_gs(*, seed: Optional[int] = None):
    if seed is not None:
        seed_torch(seed)
    with open('retained.json', 'r') as fp:
        files = json.load(fp)
    # files = files[:len(files)//10]
    doc_dataset = DocumentDataset(files)
    model = OurHeroes()
    config = default_gs_config()
    model.fit_gs(doc_dataset, config)


async def is_digested(path: str, tq: tqdm):
    with open(path, 'r') as fp:
        file = json.load(fp)
    tq.update(1)
    if 'digest' in file:
        return path, True
    return path, False


async def write_digested_val_json(digested_path: str, val_folder: str):
    files = [f"{val_folder}\\{file}" for file in os.listdir(val_folder) if file.endswith('.txt')]
    tq = tqdm(total=len(files))
    files = await asyncio.gather(*[is_digested(file, tq) for file in files])
    tq.close()
    files = [file for file, digested in files if digested]
    with open(digested_path, 'w') as fp:
        json.dump(files, fp)
        fp.flush()


def eval_gs():
    # with open('retained_val.json', 'r') as fp:
    #     files = json.load(fp)
    # files = files[:len(files)//10]
    val_folder = 'C:\\Users\\giova\\Documents\\Poli\\S3\\NLP\\Project\\pubmed-dataset\\parsed\\val'
    root = "."
    digested_path = f'{root}\\val_digested.json'
    if digested_path not in os.listdir(root):
        asyncio.run(write_digested_val_json(digested_path, val_folder))
    with open(digested_path, 'r') as fp:
        files = json.load(fp)
    with open('retained_val.json', 'r') as fp:
        files2 = json.load(fp)
    # files = files[:len(files)//10]
    files = list(set(files).intersection(set(files2)))
    doc_dataset = DocumentDataset(files)
    model = OurHeroes()
    config = default_gs_config()
    model.eval_gs(doc_dataset, config)


def val_digest():
    root = 'C:\\Users\\giova\\Documents\\Poli\\S3\\NLP\\Project\\pubmed-dataset\\parsed\\val'
    files = [f'{root}\\{file}' for file in os.listdir(root) if file.endswith('.txt')]
    doc_dataset = DocumentDataset(files)
    content_ranker = ContentRanker()
    checkpoint = torch.load(f"cr_checkpoints\\4.pth")
    content_ranker.load_state_dict(checkpoint['state'])
    model = OurHeroes(content_ranker)
    model.digest(doc_dataset)


def output_val_digests():
    with open('digests.json', 'r') as fp:
        digests = json.load(fp)
    digests = {key: val for key, val in digests.items() if key.split('\\')[-2] == 'val'}
    asyncio.run(write_digests(digests))


def train_gs_w_sched(*, seed: Optional[int] = None):
    if seed is not None:
        seed_torch(seed)
    with open('retained.json', 'r') as fp:
        files = json.load(fp)
    # files = files[:len(files)//10]
    doc_dataset = DocumentDataset(files)
    model = OurHeroes()
    config = default_gs_config()
    config.save_path = 'D:\\gs_checkpoints_w_sched_full'
    config.log_path = 'gs_log_w_sched_full'
    config.scheduler = poly_lr_scheduler
    model.fit_gs(doc_dataset, config)


def eval_gs_w_sched():
    # with open('retained_val.json', 'r') as fp:
    #     files = json.load(fp)
    # files = files[:len(files)//10]
    val_folder = 'C:\\Users\\giova\\Documents\\Poli\\S3\\NLP\\Project\\pubmed-dataset\\parsed\\val'
    root = "."
    digested_path = f'{root}\\val_digested.json'
    if digested_path not in os.listdir(root):
        asyncio.run(write_digested_val_json(digested_path, val_folder))
    with open(digested_path, 'r') as fp:
        files = json.load(fp)
    with open('retained_val.json', 'r') as fp:
        files2 = json.load(fp)
    # files = files[:len(files)//10]
    files = list(set(files).intersection(set(files2)))
    doc_dataset = DocumentDataset(files)
    model = OurHeroes()
    config = default_gs_config()
    config.save_path = 'D:\\gs_checkpoints_w_sched_full'
    config.log_path = 'gs_log_w_sched_full'
    config.scheduler = poly_lr_scheduler
    config.eval_path = '.\\eval_w_sched_full.json'
    model.eval_gs(doc_dataset, config)


def train_gs_non_recurse(*, seed: Optional[int] = None):
    if seed is not None:
        seed_torch(seed)
    with open('retained.json', 'r') as fp:
        files = json.load(fp)
    # files = files[:len(files)//10]
    doc_dataset = DocumentDataset(files)
    model = OurHeroes(graph_summarizer=GraphSummarizer(recursive=False))
    config = default_gs_config()
    config.save_path = 'D:\\gs_checkpoints_non_recurse_full'
    config.log_path = 'gs_log_non_recurse_full'
    model.fit_gs(doc_dataset, config)


def eval_gs_non_recurse():
    # with open('retained_val.json', 'r') as fp:
    #     files = json.load(fp)
    # files = files[:len(files)//10]
    val_folder = 'C:\\Users\\giova\\Documents\\Poli\\S3\\NLP\\Project\\pubmed-dataset\\parsed\\val'
    root = "."
    digested_path = f'{root}\\val_digested.json'
    if digested_path not in os.listdir(root):
        asyncio.run(write_digested_val_json(digested_path, val_folder))
    with open(digested_path, 'r') as fp:
        files = json.load(fp)
    with open('retained_val.json', 'r') as fp:
        files2 = json.load(fp)
    # files = files[:len(files)//10]
    files = list(set(files).intersection(set(files2)))
    doc_dataset = DocumentDataset(files)
    model = OurHeroes(graph_summarizer=GraphSummarizer(recursive=False))
    config = default_gs_config()
    config.save_path = 'D:\\gs_checkpoints_non_recurse_full'
    config.log_path = 'gs_log_non_recurse_full'
    config.eval_path = '.\\eval_non_recurse_full.json'
    model.eval_gs(doc_dataset, config)


def train_gs_both(*, seed: Optional[int] = None):
    if seed is not None:
        seed_torch(seed)
    with open('retained.json', 'r') as fp:
        files = json.load(fp)
    # files = files[:len(files)//10]
    doc_dataset = DocumentDataset(files)
    model = OurHeroes(graph_summarizer=GraphSummarizer(recursive=False))
    config = default_gs_config()
    config.save_path = 'D:\\gs_checkpoints_both_full'
    config.log_path = 'gs_log_both_full'
    config.scheduler = poly_lr_scheduler
    model.fit_gs(doc_dataset, config)


def eval_gs_both():
    # with open('retained_val.json', 'r') as fp:
    #     files = json.load(fp)
    # files = files[:len(files)//10]
    val_folder = 'C:\\Users\\giova\\Documents\\Poli\\S3\\NLP\\Project\\pubmed-dataset\\parsed\\val'
    root = "."
    digested_path = f'{root}\\val_digested.json'
    if digested_path not in os.listdir(root):
        asyncio.run(write_digested_val_json(digested_path, val_folder))
    with open(digested_path, 'r') as fp:
        files = json.load(fp)
    with open('retained_val.json', 'r') as fp:
        files2 = json.load(fp)
    # files = files[:len(files)//10]
    files = list(set(files).intersection(set(files2)))
    doc_dataset = DocumentDataset(files)
    model = OurHeroes(graph_summarizer=GraphSummarizer(recursive=False))
    config = default_gs_config()
    config.save_path = 'D:\\gs_checkpoints_both_full'
    config.log_path = 'gs_log_both_full'
    config.scheduler = poly_lr_scheduler
    config.eval_path = '.\\eval_both_full.json'
    model.eval_gs(doc_dataset, config)


def train_gs_small(*, seed: Optional[int] = None):
    if seed is not None:
        seed_torch(seed)
    with open('retained.json', 'r') as fp:
        files = json.load(fp)
    files = files[:len(files) // 10]
    doc_dataset = DocumentDataset(files)
    model = OurHeroes()
    config = default_gs_config()
    config.save_path = 'D:\\gs_checkpoints_small2'
    config.log_path = 'gs_log_small2'
    model.fit_gs(doc_dataset, config)


def eval_gs_small():
    # with open('retained_val.json', 'r') as fp:
    #     files = json.load(fp)
    # files = files[:len(files)//10]
    val_folder = 'C:\\Users\\giova\\Documents\\Poli\\S3\\NLP\\Project\\pubmed-dataset\\parsed\\val'
    root = "."
    digested_path = f'{root}\\val_digested.json'
    if digested_path not in os.listdir(root):
        asyncio.run(write_digested_val_json(digested_path, val_folder))
    with open(digested_path, 'r') as fp:
        files = json.load(fp)
    with open('retained_val.json', 'r') as fp:
        files2 = json.load(fp)
    files = files[:len(files) // 10]
    files = list(set(files).intersection(set(files2)))
    doc_dataset = DocumentDataset(files)
    model = OurHeroes()
    config = default_gs_config()
    config.save_path = 'D:\\gs_checkpoints_small2'
    config.log_path = 'gs_log_small2'
    config.eval_path = '.\\eval_small2.json'
    model.eval_gs(doc_dataset, config)

def no_collate(batch):
    return batch

def prep_gs(*, seed: Optional[int] = None):
    if seed is not None:
        seed_torch(seed)
    with open('retained.json', 'r') as fp:
        files = json.load(fp)
    # files = files[:len(files)//10]
    doc_dataset = DocumentDataset(files)

    dataset = GSDataset(doc_dataset.files, return_fname=True)
    dataloader = DataLoader(dataset, collate_fn=no_collate, num_workers=1, batch_size=1)

    tq = tqdm(total=len(dataloader))
    tq.set_description(f'preprocess: ')

    for batch in dataloader:
        for fname, graph, target, summary, digest in batch:
            root = fname.split('\\')[:-1]
            root = '\\'.join(root)
            root = f"{root}_prepped\\"
            name = fname.split('\\')[-1]

            with open(f"{root}\\graph_{name}", "wb") as fp:  # Pickling
                pickle.dump(graph, fp, protocol=pickle.HIGHEST_PROTOCOL)
                fp.flush()

            with open(f"{root}\\target_{name}", "wb") as fp:  # Pickling
                pickle.dump(target, fp, protocol=pickle.HIGHEST_PROTOCOL)
                fp.flush()

            with open(f"{root}\\summary_{name}", "wb") as fp:  # Pickling
                pickle.dump(summary, fp, protocol=pickle.HIGHEST_PROTOCOL)
                fp.flush()

            with open(f"{root}\\digest_{name}", "wb") as fp:  # Pickling
                pickle.dump(digest, fp, protocol=pickle.HIGHEST_PROTOCOL)
                fp.flush()

            with open(f"{root}\\graph_{name}", "rb") as fp:  # Unpickling
                graph = pickle.load(fp)

            with open(f"{root}\\target_{name}", "rb") as fp:  # Unpickling
                target = pickle.load(fp)

            with open(f"{root}\\summary_{name}", "rb") as fp:  # Unpickling
                summary = pickle.load(fp)

            with open(f"{root}\\digest_{name}", "rb") as fp:  # Unpickling
                digest = pickle.load(fp)

    tq.close()


if __name__ == '__main__':
    # train_cr()
    # digest()
    # output_digests()
    # asyncio.run(check_retained())
    # train_gs()
    # val_digest()
    # output_val_digests()
    # eval_gs()
    # ---extension----
    # train_gs_w_sched(seed=42)
    # train_gs_non_recurse(seed=42)
    # train_gs_both(seed=42)
    # train_gs_small()
    # eval_gs_w_sched()
    # eval_gs_non_recurse()
    # eval_gs_both()
    # eval_gs_small()
    # prep_gs(seed=42)
    # train_gs(seed=42)
    train_gs_non_recurse(seed=42)
    eval_gs()
    eval_gs_non_recurse()
