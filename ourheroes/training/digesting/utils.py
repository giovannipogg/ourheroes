"""Module implementing utility functions for the `digesting` module.
"""

from typing import Sequence


def digest_collate(batch: Sequence):
    """Collate function for the digesting app.

    Args:
        batch: The data loaded from a DigestableDataset.

    Returns:
        The collated batch.
    """
    file_names = [sub[0] for sub in batch]
    data = [sub[1] for sub in batch]
    tensors = [sub[2] for sub in batch]
    return file_names, data, tensors
