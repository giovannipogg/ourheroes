"""Module implementing learning rate schedulers.
"""

from dataclasses import dataclass

import torch


@dataclass
class EpochPolyScheduler:
    initial_lr: float
    max_epochs: int
    iterations_per_epoch: int
    power: float = .9

    """Polynomial epoch-wise learning rate scheduler.
    
    Attributes:
        initial_lr: The initial learning rate.
        max_epochs: The maximum number of epochs for which the model is trained.
        iterations_per_epoch: The number of iterations per epoch.
        power: The exponent of the polynomial learning rate schedule.
    """

    def schedule(self, optimizer: torch.optim.Optimizer, iteration: int) -> float:
        """Computes and sets the optimizer learning rate.

        Args:
            optimizer: The optimizer for which to set the new scheduled learning rate.
            iteration: The iteration within the schedule.

        Returns:
            The newly scheduled learning rate.
        """
        epoch = iteration // self.iterations_per_epoch
        lr = self.initial_lr * (1 - epoch / self.max_epochs) ** self.power
        optimizer.param_groups[0]['lr'] = lr
        return lr

    def __call__(self, optimizer: torch.optim.Optimizer, iteration: int) -> float:
        """Computes and sets the optimizer learning rate, equivalent to self.schedule(optimizer, iteration).

        Args:
            optimizer: The optimizer for which to set the new scheduled learning rate.
            iteration: The iteration within the schedule.

        Returns:
            The newly scheduled learning rate.
        """
        return self.schedule(optimizer, iteration)


@dataclass
class IterationPolyScheduler:
    initial_lr: float
    max_iterations: int
    power: float

    """Polynomial batch-wise learning rate scheduler.
    
    Attributes:
        initial_lr: The initial learning rate.
        max_iterations: The maximum number of batch iterations for which the model is trained.
        power: The exponent of the polynomial learning rate schedule.
    """

    def schedule(self, optimizer: torch.optim.Optimizer, iteration: int) -> float:
        """Computes and sets the optimizer learning rate.

        Args:
            optimizer: The optimizer for which to set the new scheduled learning rate.
            iteration: The iteration within the schedule.

        Returns:
            The newly scheduled learning rate.
        """
        lr = self.initial_lr * (1 - iteration / self.max_iterations) ** self.power
        optimizer.param_groups[0]['lr'] = lr
        return lr

    def __call__(self, optimizer: torch.optim.Optimizer, iteration: int) -> float:
        """Computes and sets the optimizer learning rate, equivalent to self.schedule(optimizer, iteration).

        Args:
            optimizer: The optimizer for which to set the new scheduled learning rate.
            iteration: The iteration within the schedule.

        Returns:
            The newly scheduled learning rate.
        """
        return self.schedule(optimizer, iteration)


def no_scheduler(optimizer: torch.optim.Optimizer, _: int) -> float:
    """Utility function for no learning rate scheduling.

    Args:
        optimizer: The optimizer training the model.
        _: The batch iteration.

    Returns:
        The unchanged learning rate of the optimizer.
    """
    return optimizer.param_groups[0]['lr']
