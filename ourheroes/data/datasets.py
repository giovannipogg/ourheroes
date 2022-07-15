"""Module implementing datasets used for training and evaluating our models.

The datasets in this module implement preprocessing and label generation routines
used for training and evaluating the model.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch.utils.data

from ourheroes.data.graphing.grapher import GSGrapher
from ourheroes.data.types import Data, BaseLoader, CRScoreTensors, DocumentTensors, BaseDocumentParser, BaseCRParser, \
    BaseGSParser, FilePath
from ourheroes.data.preprocessing.boundary_encoder import BoundaryEncoder
from ourheroes.data.preprocessing.recursor import Recursor
from ourheroes.data.scoring.scorers import CRScorer, GSScorer
from ourheroes.data.utils import select


@dataclass
class DocumentDataset(torch.utils.data.Dataset):
    files: List[FilePath]
    loader: BaseLoader

    """A Dataset for accessing documents as `Document` type.    
    
    Attributes:
        files: The files included in the dataset.
        loader: The `BaseLoader` returning a `Data` object.
    """

    def __getitem__(self, index) -> Data:
        """Loads and return the file at `index`.
        
        Args:
            index: The index of the file to retrieve.
        Returns:
            The document as a `Data` object`.
        """
        return self.loader(self.files[index])

    def __len__(self):
        """Returns the number of files in the dataset.

        Returns:
            The number of files in the dataset.
        """
        return len(self.files)


@dataclass
class CRDataset(torch.utils.data.Dataset):
    files: List[FilePath]
    loader: BaseLoader
    parser: BaseCRParser
    scorer: CRScorer
    tensorizer: Recursor
    preprocess: Optional[Recursor] = None

    """A Dataset for training the content ranking module.    
    
    Attributes:
        files: The files included in the dataset.
        loader: The `BaseLoader` returning a `Data` object.
        parser: The parser for retrieving Document and Summary from Data.
        preprocess: The shared Document and Summary normalization recursor.
        scorer: The scorer for generating the content ranking target.
        tensorizer: The recursor generating the DocumentTensors.
    """

    def __getitem__(self, index: int) -> Tuple[DocumentTensors, CRScoreTensors]:
        """Retrieves source and target for training the content ranking module.
        
        Args:
            index: The index of the document to retrieve.
        Returns:
            The source and target for training the content ranking module.
        """
        data = self.loader(self.files[index])
        document, summary = self.parser(data)
        if self.preprocess is not None:
            document, summary = self.preprocess((document, summary))
        scores = self.scorer(summary, document)
        document = self.tensorizer(document)
        return document, scores

    def __len__(self):
        """Returns the number of files in the dataset.

        Returns:
            The number of files in the dataset.
        """
        return len(self.files)


@dataclass
class DigestableDataset(torch.utils.data.Dataset):
    files: List[str]
    loader: BaseLoader
    parser: BaseDocumentParser
    tensorizer: Recursor

    """A Dataset for retrieving documents' data to produce digests.    
    
    Attributes:
        files: The files included in the dataset.
        loader: The `BaseLoader` returning a `Data` object.
        parser: The parser retrieving the Document
        preprocess: The parser in charge of normalizing and
            generating the DocumentTensors.
    """

    def __getitem__(self, index) -> Tuple[str, Data, DocumentTensors]:
        """Retrieves the filename, data and tensors for generating documents' digests.
        
        Args:
            index: The index of the document to retrieve.
        Returns:
            The filename, data and tensors for generating documents' digests.
        """
        fname = self.files[index]
        data = self.loader(fname)
        document = self.parser(data)
        tensors = self.tensorizer(document)
        return fname, data, tensors

    def __len__(self):
        """Returns the number of files in the dataset.

        Returns:
            The number of files in the dataset.
        """
        return len(self.files)


@dataclass
class GSDataset(torch.utils.data.Dataset):
    files: List[str]
    loader: BaseLoader
    parser: BaseGSParser
    scorer: GSScorer
    boundary_encoder: BoundaryEncoder
    tokenizer: Recursor
    tensorizer: Recursor
    grapher: GSGrapher
    eval: bool = False
    preprocess: Optional[Recursor] = None

    """A Dataset for training the graph summarization module.    
    
    Attributes:
        files: The files included in the dataset.
        loader: The `BaseLoader` returning a `Data` object.
        preprocess: The text normalization routine.
        scorer: The GSScorer generating the target of the
            graph summarization module.
        boundary_encoder: The sentence level positional encoder.
        tokenizer: The tokenizer.
        tensorizer: The routine for generating tokens' tensors.
        grapher: The GSGrapher generating the source graph for
            training the content ranking module.
        eval: Whether the dataset is used for evaluation (if eval=True),
            default is training (eval=False).
    """

    def __getitem__(self, index):
        """Retrieves source and target for training the graph summarization module.
        
        Args:
            index: The index of the document to retrieve.
        Returns:
            The source and target for training the graph summarization module.
        """
        file = self.files[index]
        data = self.loader(file)
        sections, summary, digest = self.parser(data)

        boundaries = self.boundary_encoder(sections, digest)
        if self.preprocess is not None:
            sections, summary = self.preprocess((sections, summary))
        target = self.scorer(summary, sections)

        target = select(target, digest)
        target = torch.tensor([val for sub in target for val in sub])

        digest = select(sections, digest)
        digest = self.tokenizer(digest)
        tensors = self.tensorizer(digest)
        graph = self.grapher(digest, tensors, boundaries)

        if self.eval:
            return graph, summary, digest
        return graph, target

    def __len__(self):
        """Returns the number of files in the dataset.

        Returns:
            The number of files in the dataset.
        """
        return len(self.files)
