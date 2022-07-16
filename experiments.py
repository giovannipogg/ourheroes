import json
from typing import Optional

from ourheroes.data.factories import default_cr_dataset, default_gs_dataset
from ourheroes.models.content_ranking.content_ranker import ContentRanker
from ourheroes.models.graph_summarization.graph_summarizer import GraphSummarizer
from ourheroes.training import digesting
from ourheroes.training.configuration.factories import default_cr_config, default_gs_config
from ourheroes.training.digesting.factories import default_digester
from ourheroes.training.eval.eval import evaluate
from ourheroes.training.eval.evaluator import default_gs_evaluator
from ourheroes.training.schedulers import EpochPolyScheduler
from ourheroes.training.train import fit
from ourheroes.training.utils import load_checkpoint


def train_cr(with_scheduler: bool = False):
    with open('files/retained.json', 'r') as fp:
        files = json.load(fp)
    dataset = default_cr_dataset(files)
    model = ContentRanker()
    config = default_cr_config()
    config.num_epochs = 10
    if with_scheduler:
        config.scheduler = EpochPolyScheduler(
            config.optimizer_kwargs['lr'],
            config.num_epochs, len(dataset) // config.dataloader_kwargs['batch_size']
        )
        config.save_path += '_w_sched'
        config.log_path += '_w_sched'
    fit(model, dataset, config)


def digest(files: str = 'files/retained.json', checkpoint_path: str = 'D:\\cr_checkpoints\\4.pth'):
    print(f'Digesting {files} with {checkpoint_path}')
    with open(files, 'r') as fp:
        files = json.load(fp)
    dataset = default_cr_dataset(files)
    model = ContentRanker()
    load_checkpoint(checkpoint_path, model)
    digester = default_digester()
    digesting.app.digest(model, digester, dataset)


def train_gs(recursive: bool = True, save_path: Optional[str] = None, max_epochs: Optional[int] = None):
    with open('files/retained.json', 'r') as fp:
        files = json.load(fp)
    dataset = default_gs_dataset(files)
    model = GraphSummarizer(recursive=recursive)
    config = default_gs_config()
    if save_path is not None:
        config.save_path = save_path
    if max_epochs is not None:
        config.num_epochs = max_epochs
    fit(model, dataset, config)


def eval_gs(save_path: Optional[str] = None, eval_path: Optional[str] = None):
    with open('files/retained_val.json', 'r') as fp:
        files = json.load(fp)
    dataset = default_gs_dataset(files, val=True)
    model = GraphSummarizer()
    config = default_gs_config()
    if save_path is not None:
        config.save_path = save_path
    if eval_path is not None:
        config.eval_path = eval_path
    evaluator = default_gs_evaluator()
    evaluate(model, dataset, evaluator, config)


def main():
    # Content Ranking
    train_cr()
    train_cr(with_scheduler=True)

    # Graph Summarization
    # Experiments with cr 5
    digest(checkpoint_path='D:\\cr_checkpoints\\4.pth')

    # Recursive
    train_gs(save_path='D:\\cr_5_gs', max_epochs=5)
    eval_gs(save_path='D:\\cr_5_gs', eval_path='files\\eval_cr_5_gs')
    exit()

    # Non Recursive
    train_gs(recursive=False, save_path='D:\\cr_5_gs', max_epochs=5)
    eval_gs(save_path='D:\\cr_5_gs_nr', eval_path='files\\eval_cr_5_gs_nr')

    # Experiments with cr 10
    digest(checkpoint_path='D:\\cr_checkpoints\\9.pth')

    # Recursive
    train_gs(save_path='D:\\cr_10_gs', max_epochs=5)
    eval_gs(save_path='D:\\cr_10_gs', eval_path='files\\eval_cr_10_gs')

    # Experiments with cr 10 w/ sched
    digest(checkpoint_path='D:\\cr_checkpoints_w_sched\\9.pth')

    # Recursive
    train_gs(save_path='D:\\cr_10_w_sched_gs', max_epochs=5)
    eval_gs(save_path='D:\\cr_10_w_sched_gs', eval_path='files\\eval_cr_10_w_sched_gs')


if __name__ == '__main__':
    main()
