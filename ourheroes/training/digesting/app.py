"""Module implementing the routine to produce and save document digests.
"""

import asyncio
import json

import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm

from ourheroes.data.factories import default_digestable_dataset
from ourheroes.data.types import Digest
from ourheroes.training.digesting.digester import Digester
from ourheroes.training.digesting.utils import digest_collate
from ourheroes.training.utils import recursive_to_device


async def write_digest(fname: str, digest_: Digest):
    """Writes the document digest in the original file.

    Args:
        fname: The location of the document file.
        digest_: The document digest.
    """
    with open(fname, 'r') as fp:
        doc = json.load(fp)
    doc['digest'] = digest_
    with open(fname, 'w') as fp:
        json.dump(doc, fp)
        fp.flush()


async def _digest(model: torch.nn.Module, digester: Digester, dataset: torch.utils.data.Dataset):
    """Produces and saves the document digests.

    Args:
        model: The content ranking module.
        digester: The digester object.
        dataset: The dataset to digest.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = default_digestable_dataset(dataset.files)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=digest_collate, num_workers=0)
    model = model.to(device)
    model.eval()

    tq = tqdm(total=len(dataloader))
    tq.set_description(f'digest ')
    for file_names, data, tensors in dataloader:
        tensors = recursive_to_device(tensors, device)
        with torch.no_grad():
            sentence_scores, section_scores = model(tensors)
        digests = digester(sentence_scores, section_scores, data)
        await asyncio.gather(*[write_digest(fname, digest_) for fname, digest_ in zip(file_names, digests)])
        del tensors, sentence_scores, section_scores
        torch.cuda.empty_cache()
        tq.update(1)


def digest(model: torch.nn.Module, digester: Digester, dataset: torch.utils.data.Dataset):
    """Runs the digesting routine.

    Args:
        model: The content ranking module.
        digester: The digester object.
        dataset: The dataset to digest.
    """
    asyncio.run(_digest(model, digester, dataset))
