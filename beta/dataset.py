import json
import os
import string
from dataclasses import dataclass
from typing import Tuple, List

import torch.utils.data
import torchtext.vocab

from grapher import BoundaryEncoder, GSGrapher
from ourtypes import Data, BaseLoader, Document, Summary, CRScoreTensors, DocumentTensors, Digest, Token, Section, \
    BaseDocumentParser, BaseSummaryParser, BaseCRParser, Title, BaseTitleParser, BaseSectionParser, BaseDigestParser, \
    BaseGSParser
from preprocessing import Recursor, PreprocessingPipeline, Remover, lowerize, SimpleTokenizer, GloVeTensorizer, \
    GloVeWrapper
from scorer import CRScorer, GSScorer


def load_json(file_path: str) -> Data:
    with open(file_path, 'r', encoding='utf-8') as fp:
        document = json.load(fp)
    return document


@dataclass
class DocumentDataset(torch.utils.data.Dataset):
    files: List[str]
    loader: BaseLoader = load_json

    def __getitem__(self, index: int) -> Data:
        return self.loader(self.files[index])

    def __len__(self) -> int:
        return len(self.files)


@dataclass
class SectionParser:
    sections: str = 'sections'

    def parse(self, data: Data) -> List[Section]:
        return data[self.sections]

    def __call__(self, data: Data) -> List[Section]:
        return self.parse(data)


@dataclass
class TitleParser:
    titles: str = 'section_names'

    def parse(self, data: Data) -> List[Title]:
        return data[self.titles]

    def __call__(self, data: Data) -> List[Title]:
        return self.parse(data)


@dataclass
class DocumentParser:
    title_parser: BaseTitleParser = TitleParser()
    section_parser: BaseSectionParser = SectionParser()

    def parse(self, data: Data) -> Document:
        titles = self.title_parser(data)
        sections = self.section_parser(data)
        document = [(title, section) for title, section in zip(titles, sections)]
        return document

    def __call__(self, data: Data) -> Document:
        return self.parse(data)


@dataclass
class SummaryParser:
    summary: str = 'abstract_text'

    def parse(self, data: Data) -> Summary:
        return data[self.summary]

    def __call__(self, data: Data) -> Summary:
        return self.parse(data)


@dataclass
class CRParser:
    document_parser: BaseDocumentParser = DocumentParser()
    summary_parser: BaseSummaryParser = SummaryParser()

    def parse(self, data: Data) -> Tuple[Document, Summary]:
        document = self.document_parser(data)
        summary = self.summary_parser(data)
        return document, summary

    def __call__(self, data: Data) -> Tuple[Document, Summary]:
        return self.parse(data)


@dataclass
class CRDataset(torch.utils.data.Dataset):
    files: List[str]
    loader: BaseLoader = load_json
    parser: BaseCRParser = CRParser()
    preprocess: Recursor = Recursor(PreprocessingPipeline((Remover(), lowerize)))
    scorer: CRScorer = CRScorer()
    datafier: Recursor = Recursor(PreprocessingPipeline((SimpleTokenizer(), GloVeTensorizer())))

    def __getitem__(self, index: int) -> Tuple[DocumentTensors, CRScoreTensors]:
        data = self.loader(self.files[index])
        document, summary = self.parser(data)
        document, summary = self.preprocess((document, summary))
        scores = self.scorer(document, summary)
        document = self.datafier(document)
        return document, scores

    def __len__(self):
        return len(self.files)


@dataclass
class DigestableDataset(torch.utils.data.Dataset):
    files: List[str]
    loader: BaseLoader = load_json
    parser: BaseDocumentParser = DocumentParser()
    preprocess: Recursor = Recursor(PreprocessingPipeline((Remover(), lowerize, SimpleTokenizer(), GloVeTensorizer())))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index) -> Tuple[str, Data, DocumentTensors]:
        fname = self.files[index]
        data = self.loader(fname)
        document = self.parser(data)
        tensors = self.preprocess(document)
        return fname, data, tensors


@dataclass
class DigestParser:
    digest: str = 'digest'

    def parse(self, data: Data) -> Digest:
        digest = data[self.digest]
        output = {}
        for key, vals in digest.items():
            output[int(key)] = list(map(int, vals))
        return output

    def __call__(self, data: Data) -> Digest:
        return self.parse(data)


@dataclass
class GSParser:
    section_parser: BaseSectionParser = SectionParser()
    summary_parser: BaseSummaryParser = SummaryParser()
    digest_parser: BaseDigestParser = DigestParser()

    def parse(self, data: Data) -> Tuple[List[Section], Summary, Digest]:
        sections = self.section_parser(data)
        summary = self.summary_parser(data)
        digest = self.digest_parser(data)

        return sections, summary, digest

    def __call__(self, data: Data) -> Tuple[List[Section], Summary, Digest]:
        return self.parse(data)


def select(list_of_lists: List[List], digest: Digest):
    output = []
    for section, sentences in digest.items():
        inter = []
        for sentence in sentences:
            inter.append(list_of_lists[section][sentence])
        output.append(inter)
    return output

@dataclass
class GSDataset(torch.utils.data.Dataset):
    files: List[str]
    return_fname: bool = False
    loader: BaseLoader = load_json
    parser: BaseGSParser = GSParser()
    scorer: GSScorer = GSScorer()
    boundary_encoder: BoundaryEncoder = BoundaryEncoder()
    preprocess: Recursor = Recursor(PreprocessingPipeline((Remover(), lowerize)))
    tokenizer: Recursor = Recursor(SimpleTokenizer())
    tensorizer: Recursor = Recursor(GloVeWrapper())
    grapher: GSGrapher = GSGrapher()

    def __getitem__(self, index):
        file = self.files[index]
        data = self.loader(file)
        sections, summary, digest = self.parser(data)

        boundaries = self.boundary_encoder(sections, digest)
        sections, summary = self.preprocess((sections, summary))
        target = self.scorer(sections, summary)

        target = select(target, digest)
        target = torch.tensor([val for sub in target for val in sub])

        sections = select(sections, digest)
        sections = self.tokenizer(sections)
        tensors = self.tensorizer(sections)
        graph = self.grapher(sections, tensors, boundaries)

        if self.return_fname:
            return file, graph, target, summary, sections
        return graph, target, summary, sections

    def __len__(self):
        return len(self.files)


def digest_path(file: str) -> str:
    file = file.split('\\')
    root, article = '\\'.join(file[:-1]), file[-1]
    return f"{root}\\digests\\{article}"


def document_to_gs_dataset(dataset: DocumentDataset) -> GSDataset:
    files = dataset.files
    files = [(file, digest_path(file)) for file in files]
    files = [file for file in files if os.path.isfile(file[1])]
    return GSDataset(files)
