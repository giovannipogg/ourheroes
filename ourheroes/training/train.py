"""Module implementing the general training routine.

The fit function can be used to train both the ContentRanker and
the GraphSummarizer networks.
"""

import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

from ourheroes.training.configuration.config import Config
from ourheroes.training.utils import log, load_last, seed_torch, save_checkpoint, zero_grad, clip_gradient


def fit(model: torch.nn.Module, dataset: torch.utils.data.Dataset, config: Config):
    """Fits a model to a given dataset.

    Args:
        model: The model to be fitted.
        dataset: The dataset to fit the model to.
        config: The configuration for training the model.
    """
    log(f'fitting {model}', config.log_path)
    log(config, config.log_path)

    dataloader = DataLoader(dataset, **config.dataloader_kwargs)
    model = model.to(config.device)
    optimizer = config.optimizer(model.parameters(), **config.optimizer_kwargs)
    scaler = amp.GradScaler()
    last_epoch = load_last(config.save_path, model, optimizer)

    model.train()
    for epoch in range(last_epoch + 1, config.num_epochs):  # <<< + 1 for TESTING ONLY
        if config.seed is not None:
            seed_torch(config.seed + epoch)
        tq = tqdm(total=len(dataloader))
        log(f'epoch {epoch}', config.log_path)
        iterate(model, dataloader, optimizer, scaler, epoch, tq, config)
        if epoch % config.save_epochs == 0:
            save_checkpoint(model, optimizer, epoch, config.save_path)
        tq.close()


def iterate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
            scaler: amp.GradScaler, epoch: int, tq: tqdm, config: Config):
    """Carries out an epoch for training a model with a given dataset.

    Args:
        model: The model to be fitted.
        dataloader: The dataloader from which the batches are retrieved.
        optimizer: The optimizer used for training the model.
        scaler: The scaler used while training.
        epoch: The epoch to perform.
        tq: The progress bar manager.
        config: The configuration with which to train the model.
    """
    past_iterations = len(dataloader) * epoch
    for iteration, (source, target) in enumerate(dataloader):
        lr = config.scheduler(optimizer, iteration + past_iterations, **config.scheduler_kwargs)
        tq.set_description(f'train: epoch {epoch}: lr: {lr:.2e}')
        zero_grad(model)

        source = config.source_to_device(source, config.device)
        target = config.target_to_device(target, config.device)

        with amp.autocast():
            prediction = model(source)
            del source
            torch.cuda.empty_cache()
            loss = config.criterion(prediction, target, **config.criterion_kwargs)
        scaler.scale(loss).backward()
        loss_ = loss.item()
        del prediction, loss
        torch.cuda.empty_cache()
        if config.clip_gradient:
            clip_gradient(model, optimizer, scaler, **config.gradient_clipping_kwargs)
        scaler.step(optimizer)
        scaler.update()
        log({'epoch': epoch, 'loss': loss_}, config.log_path)
        tq.set_postfix({'loss': loss_})
        tq.update(1)
