"""Module implementing factory functions used by the preprocessing modules.

The functions in this module return a default instance of the object they create.
"""

from typing import Optional

import nltk.tokenize
import torchtext

from ourheroes.data.preprocessing.tensorizers import SentenceTensorizer, TokenTensorizer
from ourheroes.data.preprocessing.tokenizers import NLTKTokenizer
from ourheroes.data.types import Token, Tokenizer, Vocab


def default_tokenizer(vocab: Optional[Vocab] = None) -> Tokenizer:
    """Creates the default Tokenizer.

    Returns:
        The nltk word_tokenize function.
    """
    tokenize = nltk.tokenize.word_tokenize
    return NLTKTokenizer(tokenize, vocab)


def default_token_tensorizer() -> TokenTensorizer:
    """Creates the default TokenTensorizer.
    
    Returns:
        The TokenTensorizerObject implementing
        torchtext.vocab.GloVe.
    """
    return TokenTensorizer(torchtext.vocab.GloVe())


def default_sentence_tensorizer(start_of_sentence: Token = '<s>',
                                end_of_sentence: Token = '</s>') -> SentenceTensorizer:
    """Creates the default SentenceTensorizer.
    Args:
        start_of_sentence: the start-of-sentence token or None to disable.
        end_of_sentence: the end-of-sentence token or None to disable.
        
    Returns:
        The SentenceTensorizer object implementing
        the default TokenTensorizer.
    """
    token_tensorizer = default_token_tensorizer()
    return SentenceTensorizer(token_tensorizer, start_of_sentence, end_of_sentence)
