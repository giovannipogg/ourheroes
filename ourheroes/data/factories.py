"""Module implementing factory methods for default parsers and datasets.

Functions in this module have default arguments intended to work
with documents as formatted in the PubMed and Arxiv datasets,
and return the default instance of the object they create.
"""

from typing import List

from ourheroes.data.datasets import CRDataset, DigestableDataset, GSDataset, DocumentDataset
from ourheroes.data.graphing.grapher import GSGrapher
from ourheroes.data.parsers import DataParser, DocumentParser, CRParser, DigestParser, GSParser
from ourheroes.data.preprocessing.boundary_encoder import BoundaryEncoder
from ourheroes.data.preprocessing.pipeline import PreprocessingPipeline
from ourheroes.data.preprocessing.recursor import Recursor
from ourheroes.data.preprocessing.factories import default_tokenizer, default_sentence_tensorizer, \
    default_token_tensorizer
from ourheroes.data.scoring.factories import default_cr_scorer, default_gs_scorer
from ourheroes.data.types import Title, Section, Summary, FilePath, BaseLoader
from ourheroes.data.utils import load_json
from ourheroes.inspection.utils import get_not_graphed
from ourheroes.inspection.vocabs import get_representable


def default_title_parser(titles_field: str = 'section_names') -> DataParser[List[Title]]:
    """Creates a `DataParser` parsing the sections titles.
    
    Args:
        titles_field: The name of the sections titles field within the expected data.
        
    Returns:
        A DataParser parsing the sections' titles.
    """
    return DataParser(titles_field)


def default_section_parser(sections_field: str = 'sections') -> DataParser[List[Section]]:
    """Creates a `DataParser` parsing the sections text.
    
    Args:
        sections_field: The name of the sections field within the expected data.
        
    Returns:
        A DataParser parsing the sections' literals.
    """
    return DataParser(sections_field)


def default_summary_parser(summary_field: str = 'abstract_text') -> DataParser[Summary]:
    """Creates a `DataParser` parsing the summary text.
    
    Args:
        summary_field: The name of the summary field within the expected data.
        
    Returns:
        A DataParser parsing the summary literal.
    """
    return DataParser(summary_field)


def default_document_parser(titles_field: str = 'section_names', sections_field: str = 'sections'
                            ) -> DocumentParser:
    """Creates a `DocumentParser` parsing the document.
    
    Args:
        titles_field: The name of the sections' titles field within the expected data.
        sections_field: The name of the sections field within the expected data.
        
    Returns:
        A DocumentParser parsing the document.
    """
    title_parser = default_title_parser(titles_field)
    section_parser = default_section_parser(sections_field)
    return DocumentParser(title_parser, section_parser)


def default_cr_parser(titles_field: str = 'section_names', sections_field: str = 'sections',
                      summary_field: str = 'abstract_text') -> CRParser:
    """Creates a `CRParser` parsing the document and its summary.
    
    Args:
        titles_field: The name of the sections' titles field within the expected data.
        sections_field: The name of the sections field within the expected data.
        summary_field: The name of the summary field within the expected data.
        
    Returns:
        A CRParser parsing the document and its summary.
    """
    document_parser = default_document_parser(titles_field, sections_field)
    summary_parser = default_summary_parser(summary_field)
    return CRParser(document_parser, summary_parser)


def default_digest_parser(digest_field: str = 'digest') -> DigestParser:
    """Creates a `DigestParser` parsing the digest of a document.
    
    Args:
        digest_field: The name of the digest field within the expected data.
        
    Returns:
        A DigestParser parsing the document digest.
    """
    return DigestParser(digest_field)


def default_gs_parser(sections_field: str = 'sections', summary_field: str = 'abstract_text',
                      digest_field: str = 'digest') -> GSParser:
    """Creates a `GSParser` parsing the sections, summary and digest of a document.
    
    Args:
        sections_field: The name of the sections field within the expected data.
        summary_field: The name of the summary field within the expected data.
        digest_field: The name of the digest field within the expected data.
        
    Returns:
        A GSParser parsing the sections, summary and digest of a document.
    """
    section_parser = default_section_parser(sections_field)
    summary_parser = default_summary_parser(summary_field)
    digest_parser = default_digest_parser(digest_field)
    return GSParser(section_parser, summary_parser, digest_parser)


def default_cr_tensorizer() -> Recursor:
    """Creates a `Recursor` object for mapping sentences to tensors.
        
    Returns:
        A Recursor object for mapping `Sequence` to tensor, i.e.
        conversion of possibly nested sequences of normalized `Sentence=str` to the
        same nesting and type sequences of `torch.Tensor` in the shape
        `(number_of_words, embedding_dimension)`.
    """
    vocab = get_representable()
    pipeline = PreprocessingPipeline(
        (default_tokenizer(vocab), default_sentence_tensorizer())
    )
    return Recursor(pipeline)


def default_digestable_preprocess() -> Recursor:
    """Creates a `Recursor` object for converting a document to a digestable format.
        
    Returns:
        A Recursor object for `Sequence`, i.e.
        normalization and conversion of possibly nested sequences of `Sentence=str`
        to the same nesting and type sequences of `torch.Tensor` in the shape
        `(number_of_words, embedding_dimension)`.
    """
    vocab = get_representable()
    pipeline = PreprocessingPipeline(
        (default_tokenizer(vocab), default_sentence_tensorizer())
    )
    return Recursor(pipeline)


def default_document_dataset(files: List[FilePath], loader: BaseLoader = load_json) -> DocumentDataset:
    """Creates a `DocumentDataset` object given `files`.
    
    Args:
        files: The files in the dataset.
        loader: The BaseLoader implemented by the dataset.
        
    Returns:
        A DocumentDataset accessing `files` by means of `loader`.
    """
    return DocumentDataset(files, loader)


def default_cr_dataset(files: List[FilePath], loader: BaseLoader = load_json) -> CRDataset:
    """Creates a `CRDataset` object with default attributes given `files`.
    
    Args:
        files: The files in the dataset.
        loader: The BaseLoader implemented by the dataset.
        
    Returns:
        A CRDataset accessing `files` by means of `loader`, and
        implementing the default parser, preprocessing and scoring.
    """
    parser = default_cr_parser()
    scorer = default_cr_scorer()
    tensorizer = default_cr_tensorizer()
    return CRDataset(files, loader, parser, scorer, tensorizer)


def default_digestable_dataset(files: List[FilePath], loader: BaseLoader = load_json) -> DigestableDataset:
    """Creates a `DigestableDataset` object with default attributes given `files`.
    
    Args:
        files: The files in the dataset.
        loader: The BaseLoader implemented by the dataset.
        
    Returns:
        A DigestableDataset accessing `files` by means of `loader`, and
        implementing the default parser and preprocessing.
    """
    parser = default_document_parser()
    preprocess = default_digestable_preprocess()
    return DigestableDataset(files, loader, parser, preprocess)


def default_gs_dataset(files: List[FilePath], loader: BaseLoader = load_json, val: bool = False) -> DigestableDataset:
    """Creates a `GSDataset` object with default attributes given `files`.
    
    Args:
        files: The files in the dataset.
        loader: The BaseLoader implemented by the dataset.
        val: Whether the dataset is used for evaluation.
        
    Returns:
        A GSDataset accessing `files` by means of `loader`, and
        implementing the default parser, preprocessing and feature extraction.
    """
    parser = default_gs_parser()
    scorer = default_gs_scorer()
    vocab = get_representable()
    not_graphed = get_not_graphed()
    boundary_encoder = BoundaryEncoder()
    tokenizer = Recursor(default_tokenizer(vocab))
    tensorizer = Recursor(default_token_tensorizer())
    grapher = GSGrapher(not_graphed)
    return GSDataset(
        files, loader, parser, scorer, boundary_encoder, tokenizer, tensorizer, grapher, val
    )
