"""Module implementing the evaluation routine.
"""

import json
import os
from typing import Dict, Any, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ourheroes.training.configuration.config import Config
from ourheroes.training.eval.evaluator import BaseEvaluator
from ourheroes.training.utils import get_checkpoints, load_checkpoint


def no_collate(batch: Sequence) -> Tuple:
    return tuple(*batch)


def evaluate(model: torch.nn.Module, dataset: torch.utils.data.Dataset, evaluator: BaseEvaluator, config: Config):
    """Performs the evaluation of a model.

    Args:
        model: The model to evaluate.
        dataset: The dataset on which to evaluate the model.
        evaluator: The evaluator callable assigning evaluation metrics.
        config: The configuration used for evaluating.
    """
    if not os.path.isdir(config.eval_path):
        os.makedirs(config.eval_path)
    dataloader = DataLoader(dataset, **config.dataloader_kwargs)
    model = model.to(config.device)
    checkpoints = get_checkpoints(config.save_path)
    for checkpoint in checkpoints:
        epoch = load_checkpoint(checkpoint, model)
        model.eval()
        evaluate_epoch(model, dataloader, evaluator, config, epoch)


def evaluate_epoch(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                   evaluator: BaseEvaluator, config: Config, epoch: int):
    """Evaluates the model at a given epoch.

    Args:
        model: The model to evaluate.
        dataloader: The dataloader used to evaluate the model.
        evaluator: The evaluator callable assigning evaluation metrics.
        config: The configuration used for evaluating.
        epoch: The currently evaluated epoch.
    """
    tq = tqdm(total=len(dataloader))
    tq.set_description(f'eval : epoch {epoch}')
    for source, target, data in dataloader:
        source = config.source_to_device(source, config.device)
        with torch.no_grad():
            prediction = model(source)
        scores = evaluator(target, prediction, data)
        save_scores(scores, config.eval_path, epoch)
        tq.update(1)


def save_scores(scores: Dict[str, Any], eval_dir: str, epoch: int):
    """Saves a the current iteration evaluation scores.

    Args:
        scores: The scores to be saved.
        eval_dir: The directory to which the scores are saved.
        epoch: The epoch of which the evaluation was performed.
    """
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)
    with open(f'{eval_dir}\\{epoch}.val', 'a') as fp:
        json.dump(scores, fp)
        fp.write('\n')
        fp.flush()
