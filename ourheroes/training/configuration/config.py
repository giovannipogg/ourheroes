"""Module implementing the utility class Config.

The Config object is used at the time of training and evaluation
of the model to inform the training and evaluation routines.
"""

from dataclasses import dataclass
from typing import Type, Optional, Dict, Any

import torch

from ourheroes.data.types import Criterion, Scheduler, Devicer
from ourheroes.training.schedulers import no_scheduler
from ourheroes.training.utils import recursive_to_device


@dataclass
class Config:
    num_epochs: int
    criterion: Criterion
    optimizer: Type[torch.optim.Optimizer]
    save_epochs: int
    save_path: str
    log_path: str
    clip_gradient: bool
    device: torch.device
    seed: Optional[int] = None
    source_to_device: Optional[Devicer] = recursive_to_device
    target_to_device: Optional[Devicer] = recursive_to_device
    scheduler: Optional[Scheduler] = no_scheduler
    eval_path: str = 'files/last_eval_dir/'
    dataloader_kwargs: Optional[Dict[str, Any]] = None
    criterion_kwargs: Optional[Dict[str, Any]] = None
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    scheduler_kwargs: Optional[Dict[str, Any]] = None
    gradient_clipping_kwargs: Optional[Dict[str, Any]] = None

    """Utility class for training our networks.
    
    Attributes:
        num_epochs: The number of epochs for which to train.
        criterion: The loss criterion.
        optimizer: The optimizer to use.
        save_epochs: The number of epochs between each checkpoint saved
            (if epoch=save_epochs in modulo a checkpoint is created).
        save_path: The directory in which checkpoints are saved.
        log_path: The file used to log training information.
        clip_gradient: Whether or not to perform gradient clipping.
        device: The device to be used for training.
        seed: The seed for reproducibility.
        source_to_device: The method to transfer the source to device.
        target_to_device: The method to transfer the target to device.
        scheduler: The learning rate scheduler. 
        eval_path: The directory to save evaluation info.
        dataloader_kwargs: Keyword arguments for the dataloader factory.
        criterion_kwargs: Keyword arguments for the loss function.
        optimizer_kwargs: Keyword arguments for the optimizer factory.
        scheduler_kwargs: Keyword arguments for the scheduler callable.
        gradient_clipping_kwargs: Keyword arguments for gradient clipping callable.
    """

    def __post_init__(self):
        """Performs post initialization.

        Substitutes non initialized attributes which are used as
        keyword arguments in training with empty dictionaries.
        """
        if self.dataloader_kwargs is None:
            self.dataloader_kwargs = {}
        if self.criterion_kwargs is None:
            self.criterion_kwargs = {}
        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {}
        if self.scheduler_kwargs is None:
            self.scheduler_kwargs = {}
        if self.gradient_clipping_kwargs is None:
            self.gradient_clipping_kwargs = {}
