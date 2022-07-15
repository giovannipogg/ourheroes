"""Module implementing default configuration factories.

The functions in this module return the default configuration
objects used by the training and evaluation routines. The
dataloader kwargs are optimized for hardware used by the authors.
"""

import torch
import torch.nn.functional as F

from ourheroes.training.configuration.config import Config
from ourheroes.training.utils import cr_loss, cr_collate, graph_to_device, gs_collate


def default_cr_config() -> Config:
    """Creates the default configuration for training the content ranking network.

    Returns:
        The default configuration for training the content ranking network.
    """
    return Config(
        num_epochs=5,
        criterion=cr_loss,
        optimizer=torch.optim.SGD,
        save_epochs=1,
        save_path='D:\\cr_checkpoints',
        log_path='files/cr_log',
        clip_gradient=False,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        seed=42,
        dataloader_kwargs={
            'batch_size': 32,
            'num_workers': 4,
            'prefetch_factor': 4,
            'shuffle': True,
            'collate_fn': cr_collate
        },
        optimizer_kwargs={'lr': .0001}
    )


def default_gs_config() -> Config:
    """Creates the default configuration for training the graph summarization network.

    Returns:
        The default configuration for training the graph summarization network.
    """
    return Config(
        num_epochs=5,
        criterion=F.binary_cross_entropy_with_logits,
        optimizer=torch.optim.Adam,
        save_epochs=1,
        save_path='D:\\gs_checkpoints',
        log_path='files/gs_log_official',
        clip_gradient=True,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        seed=42,
        source_to_device=graph_to_device,
        dataloader_kwargs={
            'batch_size': 8,
            'num_workers': 4,
            'prefetch_factor': 4,
            'shuffle': True,
            'collate_fn': gs_collate
        },
        optimizer_kwargs={'lr': .0001},
        gradient_clipping_kwargs={'max_norm': 2.0}
    )
