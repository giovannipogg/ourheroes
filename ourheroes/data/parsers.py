"""Module implementing parsers to convert from the `Data` to a suitable format.

The classes defined in this module are implemented by the
datasets for converting a document loaded in the `Data` format
to the objects they need.
"""

from dataclasses import dataclass
from typing import Tuple, Generic, List

from ourheroes.data.types import Data, Document, Summary, Digest, Section, Parsed, BaseTitleParser, BaseSectionParser, \
    BaseDocumentParser, BaseSummaryParser, BaseDigestParser


@dataclass
class DataParser(Generic[Parsed]):
    field: str

    """Generic utility class to parse a field from data.
    
    The class is intended for generalization purposes:
    although in the PubMed and Arxiv datasets documents
    are `json` files so that retrieval of individual
    fields could be performed directly by key, the
    dataset classes can generalize to other encodings
    by defining new ad hoc parsers.
    
    
    Attributes:
        field: The field to be parsed.
    """

    def parse(self, data: Data) -> Parsed:
        """Returns `data[self.field]`.
        
        Args:
            data: The data to be parsed i.e a dictionary with string keys.
            
        Returns:
            The value `data[self.field]` of generic type `Parsed`.
        """
        return data[self.field]

    def __call__(self, data: Data) -> Parsed:
        """Returns `data[self.field]`, equivalent to calling self.parse(data).
        
        Args:
            data: The data to be parsed i.e a dictionary with string keys.
            
        Returns:
            The value `data[self.field]` of generic type `Parsed`.
        """
        return self.parse(data)


@dataclass
class DocumentParser:
    title_parser: BaseTitleParser
    section_parser: BaseSectionParser

    """Parses a document from `Data` to `Document` type.
    
    The class is callable and implemented by datasets to
    retrieve from the document in the `Data` format its
    `Document` type representation, i.e. a `List[Tuple[
    Title, Section]]=List[str, List[str]]`
    
    Attributes:
        title_parser: The parser returning the sections' titles.
        section_parser: The parser returning the sections' text.
    """

    def parse(self, data: Data) -> Document:
        """Parses `data` of type `Data` into a document of type `Document`.
        
        Args:
            data: The data to be parsed i.e a dictionary with string keys.
            
        Returns:
            The document as type `Document=List[Tuple[Title, Section]]=List[str, List[str]]`.
        """
        titles = self.title_parser(data)
        sections = self.section_parser(data)
        document = [(title, section) for title, section in zip(titles, sections)]
        return document

    def __call__(self, data: Data) -> Document:
        """Parses `data` of type `Data` into a document of type `Document`, equivalent to
        self.parse(data).
        
        Args:
            data: The data to be parsed i.e a dictionary with string keys.
            
        Returns:
            The document as type `Document=List[Tuple[Title, Section]]=List[str, List[str]]`.
        """
        return self.parse(data)


@dataclass
class CRParser:
    document_parser: BaseDocumentParser
    summary_parser: BaseSummaryParser

    """Parses a document from `Data` to a `Tuple[Document, Summary]`.
    
    The class is implemented by the `CRDataset`, which is used for
    training the content ranking module, thus extracting from a 
    `Data` object the `Document` and `Summary` parts.
    
    Attributes:
        document_parser: The parser returning the document text.
        summary_parser: The parser returning the summary text.
    """

    def parse(self, data: Data) -> Tuple[Document, Summary]:
        """Parses `data` of type `Data` into a `Tuple[Document, Summary]`.
        
        Args:
            data: The data to be parsed i.e a dictionary with string keys.
            
        Returns:
            The `Tuple[Document, Summary]` where
            `Document=List[Tuple[Title, Section]]=List[str, List[str]]`
            and `Summary=List[str]`.
        """
        document = self.document_parser(data)
        summary = self.summary_parser(data)
        return document, summary

    def __call__(self, data: Data) -> Tuple[Document, Summary]:
        """Parses `data` of type `Data` into a `Tuple[Document, Summary]`,
        equivalent to self.parse(data).
        
        Args:
            data: The data to be parsed i.e a dictionary with string keys.
            
        Returns:
            The `Tuple[Document, Summary]` where
            `Document=List[Tuple[Title, Section]]=List[str, List[str]]`
            and `Summary=List[str]`.
        """
        return self.parse(data)


@dataclass
class DigestParser:
    digest: str

    """Parses the digest from data.
    
    The class performs the conversion from `str` to `int`
    necessitated by the original `json` formatting of the document.
    
    Attributes:
        digest: The key of the digest within the data.
    """

    def parse(self, data: Data) -> Digest:
        """Retrieves the digest from `data`.
        
        Args:
            data: The data to be parsed i.e a dictionary with string keys.
            
        Returns:
            The digest in the `Digest` format i.e. `Dict[int, List[int]]`.
        """
        digest = data[self.digest]
        output = {}
        for key, values in digest.items():
            output[int(key)] = list(map(int, values))
        return output

    def __call__(self, data: Data) -> Digest:
        """Retrieves the digest from `data`, equivalent to self.parse(data).
        
        Args:
            data: The data to be parsed i.e a dictionary with string keys.
            
        Returns:
            The digest in the `Digest` format i.e. `Dict[int, List[int]]`.
        """
        return self.parse(data)


@dataclass
class GSParser:
    section_parser: BaseSectionParser
    summary_parser: BaseSummaryParser
    digest_parser: BaseDigestParser

    """Parses the sections, summary and digest from data.
    
    The class is implemented by the GSDataset class and
    retrieves the necessary for generating the source and
    target variables for training the graph summarization
    module.
    
    Attributes:
        section_parser: The parser retrieving sections.
        summary_parser: The parser retrieving the summary.
        digest_parser: The parser retrieving the digest.
    """

    def parse(self, data: Data) -> Tuple[List[Section], Summary, Digest]:
        """Retrieves sections, summary and digest from data.
        
        Args:
            data: The data to be parsed i.e a dictionary with string keys.
            
        Returns:
            The `Tuple[List[Section], Summary, Digest]=
            Tuple[List[List[str]], List[str], Dict[int, List[int]]]`
            retrieved from `data`.
        """
        sections = self.section_parser(data)
        summary = self.summary_parser(data)
        digest = self.digest_parser(data)

        return sections, summary, digest

    def __call__(self, data: Data) -> Tuple[List[Section], Summary, Digest]:
        """Retrieves sections, summary and digest from data, equivalent to self.parse(data).
        
        Args:
            data: The data to be parsed i.e a dictionary with string keys.
            
        Returns:
            The `Tuple[List[Section], Summary, Digest]=
            Tuple[List[List[str]], List[str], Dict[int, List[int]]]`
            retrieved from `data`.
        """
        return self.parse(data)
